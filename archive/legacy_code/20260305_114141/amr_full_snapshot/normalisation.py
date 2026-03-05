"""Normalisation utilities using background publication volume.

This module helps adjust AMR counts (per pathogen, antibiotic, and time interval)
by the background number of papers about the pathogen and the antibiotic across
time. It provides:

- Simple fractions relative to pathogen-only and antibiotic-only volumes
- An expected-count model using the product of marginals (pathogen × antibiotic)
  within each pair over intervals, returning residuals (log space) and z-scores

Expected usage (inside a notebook):

    from amr.normalisation import normalise_by_background
    out = normalise_by_background(amr_df, pathogen_year_df, antibiotic_year_df)

    # Columns added:
    #   bg_pathogen_n, bg_antibiotic_n,
    #   frac_among_pathogen, frac_among_antibiotic, geo_frac,
    #   expected_bg, resid_log_bg, z_bg
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import warnings


@dataclass
class BackgroundNormaliseConfig:
    """Configuration for normalisation by background volumes.

    - alpha: Laplace smoothing for marginal probabilities/denominators
    - tiny: small constant to avoid division by zero
    """

    alpha: float = 1.0
    tiny: float = 1e-9


METRIC_COLS = ["expected_bg", "resid_log_bg", "z_bg"]


def _compute_pairwise_expected(
    df: pd.DataFrame,
    pair_cols: Tuple[str, str],
    bg_pathogen_col: str,
    bg_antibiotic_col: str,
    count_col: str,
    cfg: BackgroundNormaliseConfig,
) -> pd.DataFrame:
    """For each (pathogen, antibiotic) pair, compute expected counts per interval based on
    product-of-marginals across that pair's intervals, then residuals and z-scores.
    """
    pcol, acol = pair_cols
    group_cols = [pcol, acol]
    work = df.copy()
    n_intervals = work.groupby(group_cols)[count_col].transform("size")

    # Totals over intervals for each pair.
    work["pair_total"] = work.groupby(group_cols)[count_col].transform("sum")
    work["bgp_total"] = work.groupby(group_cols)[bg_pathogen_col].transform("sum")
    work["bga_total"] = work.groupby(group_cols)[bg_antibiotic_col].transform("sum")

    # Marginal probabilities within each pair over intervals (with smoothing).
    work["p_path_int"] = (work[bg_pathogen_col] + cfg.alpha) / (
        work["bgp_total"] + cfg.alpha * n_intervals
    )
    work["p_ab_int"] = (work[bg_antibiotic_col] + cfg.alpha) / (
        work["bga_total"] + cfg.alpha * n_intervals
    )

    # Unnormalised joint under independence
    work["q_int"] = work["p_path_int"] * work["p_ab_int"]

    # Normalise q_int across intervals for each pair so expected sums to pair_total
    q_sum = work.groupby([pcol, acol])["q_int"].transform("sum") + cfg.tiny
    work["expected_bg"] = work["pair_total"] * (work["q_int"] / q_sum)

    # Residuals
    work["resid_log_bg"] = np.log1p(work[count_col]) - np.log1p(work["expected_bg"])
    # Poisson-style z-score
    work["z_bg"] = (work[count_col] - work["expected_bg"]) / np.sqrt(np.maximum(work["expected_bg"], cfg.tiny))

    return work


def _normalise_level_name(level: str) -> str:
    return str(level).strip().lower().replace(" ", "_")


def _resolve_antibiotic_level_column(level: str) -> str:
    level_map = {
        "antibiotic": "Antibiotic",
        "class": "Class",
        "simplified_category": "Simplified categories",
        "simplified_categories": "Simplified categories",
        "high_level_category": "High level categories",
        "high_level_categories": "High level categories",
        "origin": "Origin",
    }
    return level_map.get(_normalise_level_name(level), level)


def _prepare_year_counts(year_df: pd.DataFrame) -> pd.DataFrame:
    """Normalise a yearly background dataframe to integer years and numeric counts."""
    work = year_df.copy()
    work["year"] = pd.to_numeric(work["year"], errors="coerce").astype("Int64")
    work["count"] = pd.to_numeric(work["count"], errors="coerce").fillna(0.0)
    return work.dropna(subset=["year"]).astype({"year": int})


def _expand_interval_years(work: pd.DataFrame) -> pd.DataFrame:
    """Explode interval rows into one row per covered year."""
    exploded = work[["_row_id", "pathogen", "antibiotic", "start_year", "end_year"]].copy()
    exploded["year"] = [
        list(range(int(start_year), int(end_year) + 1))
        for start_year, end_year in zip(exploded["start_year"], exploded["end_year"])
    ]
    return exploded.explode("year").astype({"year": int})


def recompute_resid_log_bg_by_antibiotic_level(
    amr_norm_df: pd.DataFrame,
    *,
    antibiotic_level: str = "antibiotic",
    antibiotic_meta_df: pd.DataFrame | None = None,
    count_col: str = "count",
    cfg: BackgroundNormaliseConfig | None = None,
    error_on_unmapped: bool = True,
    validate_bg_consistency: bool = True,
) -> pd.DataFrame:
    """
    Recompute background-normalised residuals at a chosen antibiotic partition level.

    This function expects a dataframe that already contains interval-level background
    columns (`bg_pathogen_n`, `bg_antibiotic_n`) such as `amr_df_normalised.parquet`.
    It aggregates rows to the requested antibiotic granularity, then recomputes:
    `frac_among_*`, `geo_frac`, `expected_bg`, `resid_log_bg`, and `z_bg`.
    """
    if cfg is None:
        cfg = BackgroundNormaliseConfig()

    required = {
        "pathogen",
        "antibiotic",
        "start_date",
        "end_date",
        "bg_pathogen_n",
        "bg_antibiotic_n",
        count_col,
    }
    missing = required - set(amr_norm_df.columns)
    if missing:
        raise KeyError(f"amr_norm_df missing required columns: {missing}")

    work = amr_norm_df.copy()
    work["antibiotic"] = work["antibiotic"].astype(str)
    level_key = _normalise_level_name(antibiotic_level)

    if level_key == "antibiotic":
        work["_ab_group"] = work["antibiotic"]
    else:
        if antibiotic_meta_df is None:
            raise ValueError(
                f"antibiotic_level='{antibiotic_level}' requires antibiotic_meta_df mapping."
            )
        if "Antibiotic" not in antibiotic_meta_df.columns:
            available = ", ".join(sorted(antibiotic_meta_df.columns))
            raise KeyError(
                "antibiotic_meta_df must include an 'Antibiotic' column. "
                f"Available columns: {available}"
            )
        level_col = _resolve_antibiotic_level_column(antibiotic_level)
        if level_col not in antibiotic_meta_df.columns:
            available = ", ".join(sorted(antibiotic_meta_df.columns))
            raise KeyError(
                f"antibiotic_level '{antibiotic_level}' not found in metadata. "
                f"Expected column '{level_col}'. Available columns: {available}"
            )

        meta = antibiotic_meta_df[["Antibiotic", level_col]].copy()
        meta["Antibiotic"] = meta["Antibiotic"].astype(str)
        conflicting = (
            meta.dropna(subset=[level_col])
            .groupby("Antibiotic")[level_col]
            .nunique()
        )
        conflicting = conflicting[conflicting > 1]
        if not conflicting.empty:
            antibiotics = ", ".join(conflicting.index.astype(str).tolist()[:10])
            extra = "" if len(conflicting) <= 10 else f" ... (+{len(conflicting) - 10} more)"
            raise ValueError(
                f"Conflicting metadata values found for level '{antibiotic_level}' "
                f"for {len(conflicting)} antibiotics: {antibiotics}{extra}"
            )
        meta = meta.drop_duplicates(subset=["Antibiotic"], keep="first")
        mapper = dict(zip(meta["Antibiotic"], meta[level_col]))
        mapped = work["antibiotic"].map(mapper)
        if error_on_unmapped:
            missing = sorted(work.loc[mapped.isna(), "antibiotic"].astype(str).unique().tolist())
            if missing:
                preview = ", ".join(missing[:20])
                extra = "" if len(missing) <= 20 else f" ... (+{len(missing)-20} more)"
                raise ValueError(
                    f"{len(missing)} antibiotics are missing from metadata for level '{antibiotic_level}': "
                    f"{preview}{extra}"
                )
        work["_ab_group"] = mapped.fillna(work["antibiotic"])

    group_cols = ["pathogen", "_ab_group", "start_date", "end_date"]
    if validate_bg_consistency:
        # bg_pathogen_n should be identical within (pathogen, interval).
        bg_check = work.groupby(["pathogen", "start_date", "end_date"])["bg_pathogen_n"].nunique(dropna=False)
        inconsistent = int((bg_check > 1).sum())
        if inconsistent > 0:
            warnings.warn(
                "Inconsistent bg_pathogen_n values detected within (pathogen, interval). "
                f"Found {inconsistent} inconsistent groups. Aggregation will use max(bg_pathogen_n).",
                RuntimeWarning,
            )
    agg = (
        work.groupby(group_cols, as_index=False)
        .agg(
            **{
                count_col: (count_col, "sum"),
                "bg_antibiotic_n": ("bg_antibiotic_n", "sum"),
                # Pathogen background should be interval-level; keep one representative.
                "bg_pathogen_n": ("bg_pathogen_n", "max"),
            }
        )
    )
    agg = agg.rename(columns={"_ab_group": "antibiotic"})

    # Simple fractions (smoothed)
    agg["frac_among_pathogen"] = (agg[count_col] + cfg.alpha) / (agg["bg_pathogen_n"] + cfg.alpha)
    agg["frac_among_antibiotic"] = (agg[count_col] + cfg.alpha) / (agg["bg_antibiotic_n"] + cfg.alpha)
    agg["geo_frac"] = np.sqrt(agg["frac_among_pathogen"] * agg["frac_among_antibiotic"])

    # Expected counts and residuals at the chosen antibiotic level
    pairwise = _compute_pairwise_expected(
        agg,
        pair_cols=("pathogen", "antibiotic"),
        bg_pathogen_col="bg_pathogen_n",
        bg_antibiotic_col="bg_antibiotic_n",
        count_col=count_col,
        cfg=cfg,
    )
    agg[METRIC_COLS] = pairwise[METRIC_COLS]
    agg["antibiotic_level"] = level_key

    return agg


def normalise_by_background(
    amr_df: pd.DataFrame,
    pathogen_year_df: pd.DataFrame,
    antibiotic_year_df: pd.DataFrame,
    *,
    count_col: str = "count",
    cfg: BackgroundNormaliseConfig | None = None,
) -> pd.DataFrame:
    """Add background-normalised metrics to AMR interval counts.

    Inputs
    - amr_df: rows per (pathogen, antibiotic, start_date, end_date, count)
    - pathogen_year_df: columns ['pathogen','year','count'] (counts per year for pathogen)
    - antibiotic_year_df: columns ['antibiotic','year','count'] (counts per year for antibiotic)

    Returns amr_df with added columns:
    - bg_pathogen_n: background papers for pathogen within the interval
    - bg_antibiotic_n: background papers for antibiotic within the interval
    - frac_among_pathogen: count / bg_pathogen_n (smoothed)
    - frac_among_antibiotic: count / bg_antibiotic_n (smoothed)
    - geo_frac: geometric mean of the two fractions
    - expected_bg: expected AMR count per interval from product-of-marginals across intervals for the pair
    - resid_log_bg: log residual (log1p(observed) - log1p(expected_bg))
    - z_bg: (observed - expected_bg) / sqrt(expected_bg)
    """
    if cfg is None:
        cfg = BackgroundNormaliseConfig()

    required_cols = {"pathogen", "antibiotic", "start_date", "end_date", count_col}
    missing = required_cols - set(amr_df.columns)
    if missing:
        raise KeyError(f"amr_df missing required columns: {missing}")

    for df, cols in (
        (pathogen_year_df, {"pathogen", "year", "count"}),
        (antibiotic_year_df, {"antibiotic", "year", "count"}),
    ):
        if not cols.issubset(df.columns):
            raise KeyError(f"yearly df missing required columns: expected {cols}, got {set(df.columns)}")

    work = amr_df.copy()
    work["start_year"] = pd.to_datetime(work["start_date"]).dt.year.astype(int)
    work["end_year"] = pd.to_datetime(work["end_date"]).dt.year.astype(int)

    # Pre-aggregate yearly background counts per (entity, year).
    p_year = _prepare_year_counts(pathogen_year_df)
    a_year = _prepare_year_counts(antibiotic_year_df)

    work["_row_id"] = np.arange(len(work), dtype=int)
    exploded = _expand_interval_years(work)

    # Join pathogen-year counts
    p_join = exploded.merge(p_year, on=["pathogen", "year"], how="left")
    bg_path = p_join.groupby("_row_id")["count"].sum().rename("bg_pathogen_n")

    # Join antibiotic-year counts
    a_join = exploded.merge(a_year, on=["antibiotic", "year"], how="left")
    bg_ab = a_join.groupby("_row_id")["count"].sum().rename("bg_antibiotic_n")

    # Map back to rows
    work = work.merge(bg_path, on="_row_id", how="left").merge(bg_ab, on="_row_id", how="left").drop(columns=["_row_id"])
    work["bg_pathogen_n"] = work["bg_pathogen_n"].fillna(0.0)
    work["bg_antibiotic_n"] = work["bg_antibiotic_n"].fillna(0.0)

    # Simple fractions (smoothed)
    work["frac_among_pathogen"] = (work[count_col] + cfg.alpha) / (work["bg_pathogen_n"] + cfg.alpha)
    work["frac_among_antibiotic"] = (work[count_col] + cfg.alpha) / (work["bg_antibiotic_n"] + cfg.alpha)
    work["geo_frac"] = np.sqrt(work["frac_among_pathogen"] * work["frac_among_antibiotic"])  # symmetric summary

    # Expected counts via product-of-marginals across intervals (pairwise)
    pairwise = _compute_pairwise_expected(
        work,
        pair_cols=("pathogen", "antibiotic"),
        bg_pathogen_col="bg_pathogen_n",
        bg_antibiotic_col="bg_antibiotic_n",
        count_col=count_col,
        cfg=cfg,
    )

    work[METRIC_COLS] = pairwise[METRIC_COLS]

    return work


__all__ = [
    "BackgroundNormaliseConfig",
    "normalise_by_background",
    "recompute_resid_log_bg_by_antibiotic_level",
]
