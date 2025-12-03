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


@dataclass
class BackgroundNormaliseConfig:
    """Configuration for normalisation by background volumes.

    - alpha: Laplace smoothing for marginal probabilities/denominators
    - tiny: small constant to avoid division by zero
    """

    alpha: float = 1.0
    tiny: float = 1e-9


def _aggregate_years_over_interval(year_df: pd.DataFrame, key_col: str, start_year: int, end_year: int) -> pd.Series:
    """Return a Series of background counts per key within [start_year, end_year].

    year_df must have columns: key_col, 'year', 'count'.
    """
    if year_df.empty:
        return pd.Series(dtype=float)
    mask = (year_df["year"] >= start_year) & (year_df["year"] <= end_year)
    sub = year_df.loc[mask]
    return sub.groupby(key_col)["count"].sum()


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

    # Totals over intervals for each pair (observed AMR positives)
    totals = df.groupby([pcol, acol])[count_col].sum().rename("pair_total")

    # Background totals across intervals for the pair
    bgp_tot = df.groupby([pcol, acol])[bg_pathogen_col].sum().rename("bgp_total")
    bga_tot = df.groupby([pcol, acol])[bg_antibiotic_col].sum().rename("bga_total")

    work = df.merge(totals, on=[pcol, acol], how="left").merge(bgp_tot, on=[pcol, acol], how="left").merge(bga_tot, on=[pcol, acol], how="left")

    # Marginal probabilities within each pair over intervals (with smoothing)
    work["p_path_int"] = (work[bg_pathogen_col] + cfg.alpha) / (work["bgp_total"] + cfg.alpha * work.groupby([pcol, acol])[bg_pathogen_col].transform("size"))
    work["p_ab_int"] = (work[bg_antibiotic_col] + cfg.alpha) / (work["bga_total"] + cfg.alpha * work.groupby([pcol, acol])[bg_antibiotic_col].transform("size"))

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

    # Pre-aggregate yearly background counts per (entity, year)
    p_year = pathogen_year_df.copy()
    p_year["year"] = pd.to_numeric(p_year["year"], errors="coerce").astype("Int64")
    p_year = p_year.dropna(subset=["year"]).astype({"year": int})

    a_year = antibiotic_year_df.copy()
    a_year["year"] = pd.to_numeric(a_year["year"], errors="coerce").astype("Int64")
    a_year = a_year.dropna(subset=["year"]).astype({"year": int})

    # Compute background counts per interval row (vectorised via merge-expansion)
    # Expand per-row into years, then aggregate – safe for typical intervals (<= 10 years)
    def _row_years(s, e):
        return list(range(int(s), int(e) + 1))

    exploded = work[["pathogen", "antibiotic", "start_year", "end_year"]].copy()
    exploded["year"] = [
        _row_years(s, e) for s, e in zip(exploded["start_year"], exploded["end_year"])
    ]
    exploded = exploded.explode("year").astype({"year": int})
    exploded["_row_id"] = np.arange(len(exploded), dtype=int)

    # Join pathogen-year counts
    p_join = exploded.merge(p_year, on=["pathogen", "year"], how="left")
    p_join["count"] = p_join["count"].fillna(0.0)
    bg_path = p_join.groupby(["_row_id"])['count'].sum().rename("bg_pathogen_n")

    # Join antibiotic-year counts
    a_join = exploded.merge(a_year, on=["antibiotic", "year"], how="left")
    a_join["count"] = a_join["count"].fillna(0.0)
    bg_ab = a_join.groupby(["_row_id"])['count'].sum().rename("bg_antibiotic_n")

    # Map back to rows
    work["_row_id"] = np.arange(len(work), dtype=int)
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

    cols_to_keep = [
        "expected_bg",
        "resid_log_bg",
        "z_bg",
    ]
    work = work.join(pairwise[cols_to_keep])

    return work


__all__ = [
    "BackgroundNormaliseConfig",
    "normalise_by_background",
]

