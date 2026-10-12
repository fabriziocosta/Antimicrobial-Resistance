"""Pipelines for yearly scientific-interest modeling and ranking."""

from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Any, Sequence
import warnings

from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from amr.io import atomic_write_parquet
from amr.plot_labels import format_pathogen_antibiotic_label
from amr.si.core import (
    AMR_DRIVER_COARSE_CATEGORY_COLORS,
    AMR_DRIVER_COARSE_CATEGORY_ORDER,
    AMR_DRIVER_CATEGORY_COLORS,
    build_yearly_category_counts,
    fit_predict_yearly_si_loess,
    fit_global_pooled_si_model,
    plot_yearly_category_driver_counts,
    plot_yearly_si_pair,
    rank_yearly_pairs_by_future_si_shift,
    select_interesting_yearly_rank_cases,
    summarize_si_change_drivers,
    _materialize_pair_counts,
)
from amr.validation import clean_background_counts
from amr.pipeline_config import cfg, cfg_path

DEFAULT_DRIVER_REQUIRED_COLUMNS = {
    "pathogen",
    "antibiotic",
    "year",
    "unified_decision",
    "amr_primary_category",
    "amr_target_pair_relation_supported",
}
DEFAULT_DRIVER_CATEGORY_COARSENING_PATH = Path(__file__).resolve().parents[2] / "assets" / "amr_driver_category_coarsening.csv"

DEFAULT_INTERESTING_SUMMARY_COLUMNS = [
    "pathogen",
    "antibiotic",
    "interest_score",
    "si_change",
    "current_predicted_count",
    "future_predicted_count",
    "si_loess_residual_variance_mean",
    "post_fit_si_abs_pct_error_mean",
]

DEFAULT_DRIVER_SUMMARY_COLUMNS = [
    "pathogen",
    "antibiotic",
    "interest_score",
    "si_change",
    "current_predicted_count",
    "future_predicted_count",
    "driver_recent_total",
    "driver_count_change",
    "driver_dominant_category",
    "driver_fastest_growing_category",
    "driver_weak_signal_share",
    "driver_target_pair_supported_share",
]

DEFAULT_DECREASING_SI_SUMMARY_COLUMNS = [
    "pathogen",
    "antibiotic",
    "reference_year",
    "future_year",
    "current_predicted_SI",
    "future_predicted_SI",
    "si_change",
    "current_predicted_count",
    "future_predicted_count",
    "si_loess_residual_variance_mean",
    "post_fit_si_abs_pct_error_mean",
]

DEFAULT_SMOOTHED_SI_SHAPE_SUMMARY_COLUMNS = [
    "pathogen",
    "antibiotic",
    "shape_start_year",
    "shape_midpoint_year",
    "curve_fit_end_year",
    "shape_end_year",
    "start_predicted_SI",
    "midpoint_predicted_SI",
    "end_predicted_SI",
    "endpoint_balance_abs_diff",
    "convex_u_score",
    "concave_cap_score",
    "si_quadratic_coef",
    "si_quadratic_r2",
    "si_change",
    "si_loess_residual_variance_mean",
]


def slugify_analysis_level(analysis_level: str) -> str:
    """Return the suffix used for output files/directories for an antibiotic analysis level."""
    slug = (
        str(analysis_level)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
    )
    return "antibiotic" if slug in {"", "antibiotic"} else slug


def build_si_analysis_paths(data_dir: str | Path, analysis_level: str = "antibiotic") -> dict[str, Path]:
    """Build SI output paths under the shared generated-output directory."""
    data_dir = Path(data_dir)
    output_dir = data_dir / "outputs"
    parquet_dir = output_dir / "parquets"
    figure_dir = output_dir / "figures"
    slug = slugify_analysis_level(analysis_level)
    is_antibiotic = slug == "antibiotic"
    return {
        "output_dir": output_dir,
        "parquet_dir": parquet_dir,
        "yearly_si": parquet_dir / ("yearly_si.parquet" if is_antibiotic else f"yearly_si_{slug}.parquet"),
        "rank_output": parquet_dir
        / ("yearly_si_ranked_future.parquet" if is_antibiotic else f"yearly_si_ranked_future_{slug}.parquet"),
        "interesting_output": parquet_dir
        / (
            "yearly_si_ranked_future_interesting.parquet"
            if is_antibiotic
            else f"yearly_si_ranked_future_interesting_{slug}.parquet"
        ),
        "driver_output": parquet_dir
        / (
            "yearly_si_driver_explanations.parquet"
            if is_antibiotic
            else f"yearly_si_driver_explanations_{slug}.parquet"
        ),
        "figure_dir": figure_dir,
        "interesting_plot_dir": figure_dir
        / ("interesting_yearly_si_plots" if is_antibiotic else f"interesting_yearly_si_plots_{slug}"),
        "driver_plot_dir": figure_dir
        / (
            "interesting_yearly_si_driver_plots"
            if is_antibiotic
            else f"interesting_yearly_si_driver_plots_{slug}"
        ),
    }


def load_yearly_si_inputs(
    *,
    yearly_screened_path: str | Path,
    pathogen_background_path: str | Path,
    antibiotic_background_path: str | Path,
    unified_decisions_path: str | Path | None = None,
    driver_required_cols: set[str] | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load canonical SI inputs and return an empty driver frame when category columns are unavailable."""
    yearly_screened_df = pd.read_parquet(yearly_screened_path)
    pathogen_bg_df = pd.read_parquet(pathogen_background_path)
    antibiotic_bg_df = pd.read_parquet(antibiotic_background_path)

    required_cols = DEFAULT_DRIVER_REQUIRED_COLUMNS if driver_required_cols is None else set(driver_required_cols)
    unified_decisions_df = pd.DataFrame()
    if unified_decisions_path and Path(unified_decisions_path).exists():
        candidate_unified_df = pd.read_parquet(unified_decisions_path)
        missing_driver_cols = required_cols - set(candidate_unified_df.columns)
        if missing_driver_cols:
            if verbose:
                print(
                    "Skipping SI driver classification analysis because "
                    f"{unified_decisions_path} is missing columns: {sorted(missing_driver_cols)}"
                )
        else:
            unified_decisions_df = candidate_unified_df
    elif verbose and unified_decisions_path:
        print(f"Skipping SI driver classification analysis because {unified_decisions_path} was not found.")

    if verbose:
        print(f"yearly_screened rows: {len(yearly_screened_df):,}")
        if {"pathogen", "antibiotic", "year"}.issubset(yearly_screened_df.columns):
            years = pd.to_numeric(yearly_screened_df["year"], errors="coerce").dropna()
            year_span = f"{int(years.min())} - {int(years.max())}" if not years.empty else "none"
            print(f"yearly_screened unique pathogens: {yearly_screened_df['pathogen'].nunique():,}")
            print(f"yearly_screened unique antibiotics: {yearly_screened_df['antibiotic'].nunique():,}")
            print(f"yearly_screened year range: {year_span}")
        print(f"pathogen bg rows: {len(pathogen_bg_df):,}")
        print(f"antibiotic bg rows: {len(antibiotic_bg_df):,}")
        print(f"unified decision rows available for driver analysis: {len(unified_decisions_df):,}")
    return yearly_screened_df, pathogen_bg_df, antibiotic_bg_df, unified_decisions_df


def _coerce_pmid_items(value: object) -> list[object]:
    if isinstance(value, (str, bytes)) or value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if hasattr(value, "tolist"):
        items = value.tolist()
        return items if isinstance(items, list) else []
    return []


def _unique_pmids_from_lists(values: pd.Series) -> list[object]:
    seen: set[str] = set()
    out: list[object] = []
    for value in values:
        for pmid in _coerce_pmid_items(value):
            key = str(pmid)
            if key not in seen:
                seen.add(key)
                out.append(pmid)
    return out


def _resolve_coarsening_column(requested_level: str, lookup_columns: list[str]) -> str | None:
    requested = str(requested_level).strip()
    if requested.lower() in {"", "antibiotic"}:
        return None
    column_by_lower = {col.lower(): col for col in lookup_columns}
    if requested.lower() not in column_by_lower:
        raise ValueError(
            f"ANTIBIOTIC_ANALYSIS_LEVEL={requested_level!r} is not available. "
            f"Use 'antibiotic' or one of: {lookup_columns}"
        )
    return column_by_lower[requested.lower()]


def build_antibiotic_group_mapping(
    coarsening_path: str | Path,
    requested_level: str,
) -> tuple[pd.DataFrame | None, str]:
    """Build a one-to-one antibiotic-to-analysis-level mapping, or ``None`` for antibiotic level."""
    lookup_df = pd.read_csv(coarsening_path)
    group_col = _resolve_coarsening_column(requested_level, lookup_df.columns.tolist())
    if group_col is None:
        return None, "antibiotic"
    if "Antibiotic" not in lookup_df.columns:
        raise KeyError(f"{coarsening_path} must contain an 'Antibiotic' column.")

    mapping_df = lookup_df[["Antibiotic", group_col]].copy()
    mapping_df = mapping_df.rename(columns={"Antibiotic": "source_antibiotic", group_col: "analysis_antibiotic"})
    for column in ("source_antibiotic", "analysis_antibiotic"):
        mapping_df[column] = mapping_df[column].fillna("").astype(str).str.strip()
    mapping_df = mapping_df.replace("", pd.NA)
    mapping_df = mapping_df.dropna(subset=["source_antibiotic", "analysis_antibiotic"])

    duplicate_mapping_df = (
        mapping_df.groupby("source_antibiotic")["analysis_antibiotic"]
        .nunique()
        .reset_index(name="n_groups")
        .query("n_groups > 1")
    )
    if not duplicate_mapping_df.empty:
        raise ValueError(
            f"Some antibiotics map to multiple {group_col!r} values. Fix the coarsening lookup before modeling."
        )
    return mapping_df.drop_duplicates(subset=["source_antibiotic"]).reset_index(drop=True), group_col


def prepare_antibiotic_analysis_level(
    *,
    yearly_screened_df: pd.DataFrame,
    antibiotic_background_df: pd.DataFrame,
    unified_decisions_df: pd.DataFrame,
    coarsening_path: str | Path,
    analysis_level: str = "antibiotic",
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Optionally aggregate AMR, background, and driver inputs to an antibiotic coarsening level."""
    mapping_df, analysis_column = build_antibiotic_group_mapping(coarsening_path, analysis_level)
    if mapping_df is None:
        summary = {
            "analysis_column": "antibiotic",
            "analysis_level": "antibiotic",
            "mapping_rows": None,
            "analysis_values": int(yearly_screened_df["antibiotic"].nunique()) if "antibiotic" in yearly_screened_df else 0,
        }
        if verbose:
            print("Antibiotic analysis level: individual antibiotic.")
            print(f"model yearly rows: {len(yearly_screened_df):,}")
            print(f"model pathogen-antibiotic rows: {yearly_screened_df[['pathogen', 'antibiotic']].drop_duplicates().shape[0]:,}")
            print(f"model antibiotic values: {summary['analysis_values']:,}")
            print(f"model antibiotic background rows: {len(antibiotic_background_df):,}")
            print(f"model unified decision rows: {len(unified_decisions_df):,}")
        return yearly_screened_df.copy(), antibiotic_background_df.copy(), unified_decisions_df.copy(), summary

    if verbose:
        print(
            f"Antibiotic analysis level: {analysis_column!r}. "
            f"Mapped {len(mapping_df):,} antibiotics into {mapping_df['analysis_antibiotic'].nunique():,} groups."
        )

    yearly_mapped_df = yearly_screened_df.merge(
        mapping_df,
        left_on="antibiotic",
        right_on="source_antibiotic",
        how="left",
    )
    amr_coverage = float(yearly_mapped_df["analysis_antibiotic"].notna().mean())
    yearly_mapped_df = yearly_mapped_df.dropna(subset=["analysis_antibiotic"]).copy()
    yearly_mapped_df["pmids"] = yearly_mapped_df["pmids"].apply(_coerce_pmid_items)
    yearly_screened_model_df = (
        yearly_mapped_df.groupby(["pathogen", "analysis_antibiotic", "year"], as_index=False)
        .agg(pmids=("pmids", _unique_pmids_from_lists))
        .rename(columns={"analysis_antibiotic": "antibiotic"})
    )
    yearly_screened_model_df["count"] = yearly_screened_model_df["pmids"].apply(len)

    antibiotic_bg_source_df = clean_background_counts(antibiotic_background_df, key_col="antibiotic")
    antibiotic_bg_mapped_df = antibiotic_bg_source_df.merge(
        mapping_df,
        left_on="antibiotic",
        right_on="source_antibiotic",
        how="left",
    )
    bg_coverage = float(antibiotic_bg_mapped_df["analysis_antibiotic"].notna().mean())
    antibiotic_bg_model_df = (
        antibiotic_bg_mapped_df.dropna(subset=["analysis_antibiotic"])
        .groupby(["analysis_antibiotic", "year"], as_index=False)["count"]
        .sum()
        .rename(columns={"analysis_antibiotic": "antibiotic"})
    )

    driver_coverage: float | None = None
    if unified_decisions_df.empty:
        unified_decisions_model_df = unified_decisions_df.copy()
    else:
        unified_mapped_df = unified_decisions_df.merge(
            mapping_df,
            left_on="antibiotic",
            right_on="source_antibiotic",
            how="left",
        )
        driver_coverage = float(unified_mapped_df["analysis_antibiotic"].notna().mean())
        unified_decisions_model_df = unified_mapped_df.dropna(subset=["analysis_antibiotic"]).copy()
        unified_decisions_model_df["antibiotic"] = unified_decisions_model_df["analysis_antibiotic"]
        if "pmid" in unified_decisions_model_df.columns:
            dedupe_cols = ["pathogen", "antibiotic", "year", "pmid", "unified_decision", "amr_primary_category"]
            dedupe_cols = [col for col in dedupe_cols if col in unified_decisions_model_df.columns]
            unified_decisions_model_df = unified_decisions_model_df.drop_duplicates(subset=dedupe_cols, keep="first")
        drop_cols = [col for col in ["source_antibiotic", "analysis_antibiotic"] if col in unified_decisions_model_df.columns]
        unified_decisions_model_df = unified_decisions_model_df.drop(columns=drop_cols)

    summary = {
        "analysis_column": analysis_column,
        "analysis_level": analysis_level,
        "mapping_rows": int(len(mapping_df)),
        "analysis_values": int(yearly_screened_model_df["antibiotic"].nunique()),
        "amr_mapping_coverage": amr_coverage,
        "background_mapping_coverage": bg_coverage,
        "driver_mapping_coverage": driver_coverage,
    }
    if verbose:
        print(f"AMR yearly mapping coverage: {100 * amr_coverage:.2f}%")
        print(f"Antibiotic background mapping coverage: {100 * bg_coverage:.2f}%")
        if driver_coverage is not None:
            print(f"Unified decision mapping coverage: {100 * driver_coverage:.2f}%")
        print(f"model yearly rows: {len(yearly_screened_model_df):,}")
        print(f"model pathogen-antibiotic rows: {yearly_screened_model_df[['pathogen', 'antibiotic']].drop_duplicates().shape[0]:,}")
        print(f"model antibiotic values: {yearly_screened_model_df['antibiotic'].nunique():,}")
        print(f"model antibiotic background rows: {len(antibiotic_bg_model_df):,}")
        print(f"model unified decision rows: {len(unified_decisions_model_df):,}")
    return yearly_screened_model_df, antibiotic_bg_model_df, unified_decisions_model_df, summary


def build_yearly_si_dataset(
    yearly_amr_df: pd.DataFrame,
    pathogen_background_df: pd.DataFrame,
    antibiotic_background_df: pd.DataFrame,
    *,
    alpha: float = 1.0,
    min_amr_pair_publication_count: float = 2.0,
    output_path: str | Path | None = "yearly_si.parquet",
    warn_on_dropped_rows: bool = True,
) -> tuple[pd.DataFrame, Any, list[str]]:
    """Build yearly SI rows by merging yearly AMR counts with background marginals."""
    required_amr = {"pathogen", "antibiotic", "year", "count"}
    missing_amr = required_amr - set(yearly_amr_df.columns)
    if missing_amr:
        raise KeyError(f"yearly_amr_df missing required columns: {sorted(missing_amr)}")

    required_path = {"pathogen", "year", "count"}
    missing_path = required_path - set(pathogen_background_df.columns)
    if missing_path:
        raise KeyError(f"pathogen_background_df missing required columns: {sorted(missing_path)}")

    required_ab = {"antibiotic", "year", "count"}
    missing_ab = required_ab - set(antibiotic_background_df.columns)
    if missing_ab:
        raise KeyError(f"antibiotic_background_df missing required columns: {sorted(missing_ab)}")

    work = yearly_amr_df.copy()
    path_bg = clean_background_counts(pathogen_background_df, key_col="pathogen").rename(
        columns={"count": "bg_pathogen_n"}
    )
    ab_bg = clean_background_counts(antibiotic_background_df, key_col="antibiotic").rename(
        columns={"count": "bg_antibiotic_n"}
    )

    work["year"] = pd.to_numeric(work["year"], errors="coerce").astype("Int64")
    path_bg["year"] = pd.to_numeric(path_bg["year"], errors="coerce").astype("Int64")
    ab_bg["year"] = pd.to_numeric(ab_bg["year"], errors="coerce").astype("Int64")

    merged_raw = work.merge(
        path_bg[["pathogen", "year", "bg_pathogen_n"]],
        on=["pathogen", "year"],
        how="left",
    ).merge(
        ab_bg[["antibiotic", "year", "bg_antibiotic_n"]],
        on=["antibiotic", "year"],
        how="left",
    )
    coverage_report = pd.DataFrame(
        [
            {"metric": "input_rows", "value": int(len(merged_raw))},
            {"metric": "missing_year_rows", "value": int(merged_raw["year"].isna().sum())},
            {"metric": "missing_count_rows", "value": int(merged_raw["count"].isna().sum())},
            {"metric": "missing_pathogen_background_rows", "value": int(merged_raw["bg_pathogen_n"].isna().sum())},
            {"metric": "missing_antibiotic_background_rows", "value": int(merged_raw["bg_antibiotic_n"].isna().sum())},
        ]
    )
    merged = merged_raw.dropna(subset=["year", "count", "bg_pathogen_n", "bg_antibiotic_n"]).copy()
    dropped_rows = int(len(merged_raw) - len(merged))
    coverage_report.loc[len(coverage_report)] = {"metric": "dropped_rows", "value": dropped_rows}
    coverage_report.loc[len(coverage_report)] = {"metric": "kept_rows", "value": int(len(merged))}
    if warn_on_dropped_rows and dropped_rows:
        warnings.warn(
            f"Dropped {dropped_rows} yearly AMR row(s) while building SI because year/count/background data were missing.",
            RuntimeWarning,
            stacklevel=2,
        )
    merged["year"] = merged["year"].astype(int)
    merged["count"] = pd.to_numeric(merged["count"], errors="coerce")
    merged["bg_pathogen_n"] = pd.to_numeric(merged["bg_pathogen_n"], errors="coerce")
    merged["bg_antibiotic_n"] = pd.to_numeric(merged["bg_antibiotic_n"], errors="coerce")
    merged = merged.dropna(subset=["count", "bg_pathogen_n", "bg_antibiotic_n"]).reset_index(drop=True)

    merged["count_for_model"] = _materialize_pair_counts(
        merged["count"],
        min_amr_pair_publication_count,
    )
    merged["log_count"] = np.log10(1.0 + merged["count_for_model"])
    merged["log_bg_pathogen_n"] = np.log10(1.0 + merged["bg_pathogen_n"])
    merged["log_bg_antibiotic_n"] = np.log10(1.0 + merged["bg_antibiotic_n"])

    model_input_mask = merged["log_count"].notna()
    if not model_input_mask.any():
        raise ValueError(
            "No yearly AMR pair counts meet min_amr_pair_publication_count="
            f"{float(min_amr_pair_publication_count):g}."
        )
    modeled_df, pooled_model, design_columns = fit_global_pooled_si_model(
        merged.loc[model_input_mask].copy(),
        alpha=alpha,
    )
    si_df = merged.copy()
    for column in ["expected_log_count_global", "expected_count_global", "SI"]:
        si_df[column] = np.nan
    si_df.loc[model_input_mask, ["expected_log_count_global", "expected_count_global", "SI"]] = modeled_df[
        ["expected_log_count_global", "expected_count_global", "SI"]
    ].to_numpy()
    si_df["min_amr_pair_publication_count"] = float(min_amr_pair_publication_count)
    si_df.attrs["coverage_report"] = coverage_report.to_dict(orient="records")
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        atomic_write_parquet(si_df, output_path, index=False)
    return si_df, pooled_model, design_columns


def recompute_pair_si_from_counts(
    si_reference_df: pd.DataFrame,
    yearly_pair_counts_df: pd.DataFrame,
    pathogen_background_df: pd.DataFrame,
    antibiotic_background_df: pd.DataFrame,
    pathogen: str,
    antibiotic: str,
    *,
    min_amr_pair_publication_count: float = 1.0,
) -> pd.DataFrame:
    """Recompute one pair's SI by refitting the pooled model with its available rows.

    The reference SI table supplies the existing eligible pair-year training rows.
    The requested pair is then added from the supplied count table, using the
    supplied yearly count threshold. This is an exploratory pair-specific refit and
    does not modify or reproduce the canonical SI output exactly.
    """
    required_si = {
        "pathogen", "antibiotic", "year", "count", "count_for_model",
        "bg_pathogen_n", "bg_antibiotic_n",
    }
    missing_si = required_si - set(si_reference_df.columns)
    if missing_si:
        raise KeyError(f"si_reference_df missing required columns: {sorted(missing_si)}")
    required_counts = {"pathogen", "antibiotic", "year", "count"}
    missing_counts = required_counts - set(yearly_pair_counts_df.columns)
    if missing_counts:
        raise KeyError(f"yearly_pair_counts_df missing required columns: {sorted(missing_counts)}")

    threshold = float(min_amr_pair_publication_count)
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("min_amr_pair_publication_count must be a finite non-negative number.")

    pair_mask = (
        yearly_pair_counts_df["pathogen"].astype(str).eq(str(pathogen))
        & yearly_pair_counts_df["antibiotic"].astype(str).eq(str(antibiotic))
    )
    target = yearly_pair_counts_df.loc[pair_mask, ["pathogen", "antibiotic", "year", "count"]].copy()
    if target.empty:
        raise ValueError(
            f"No screened source counts found for pathogen={pathogen!r}, antibiotic={antibiotic!r}. "
            "Refresh collection/screening first; SI cannot be recomputed from absent pair counts."
        )
    target["year"] = pd.to_numeric(target["year"], errors="coerce")
    target["count"] = pd.to_numeric(target["count"], errors="coerce")
    target = (
        target.dropna(subset=["year", "count"])
        .assign(year=lambda frame: frame["year"].astype(int))
        .groupby(["pathogen", "antibiotic", "year"], as_index=False)["count"]
        .sum()
    )

    path_bg = clean_background_counts(pathogen_background_df, key_col="pathogen").rename(
        columns={"count": "bg_pathogen_n"}
    )
    ab_bg = clean_background_counts(antibiotic_background_df, key_col="antibiotic").rename(
        columns={"count": "bg_antibiotic_n"}
    )
    target = target.merge(path_bg, on=["pathogen", "year"], how="left").merge(
        ab_bg, on=["antibiotic", "year"], how="left"
    )
    missing_background = target["bg_pathogen_n"].isna() | target["bg_antibiotic_n"].isna()
    if missing_background.all():
        raise ValueError(
            f"No complete yearly background counts found for pathogen={pathogen!r}, "
            f"antibiotic={antibiotic!r}."
        )
    target = target.loc[~missing_background].copy()
    target["count_for_model"] = _materialize_pair_counts(target["count"], threshold)
    target = target.dropna(subset=["count_for_model"]).copy()
    if target.empty:
        raise ValueError(
            f"No screened years meet min_amr_pair_publication_count={threshold:g} "
            f"for pathogen={pathogen!r}, antibiotic={antibiotic!r}."
        )

    target["log_count"] = np.log10(1.0 + target["count_for_model"].astype(float))
    target["log_bg_pathogen_n"] = np.log10(1.0 + target["bg_pathogen_n"].astype(float))
    target["log_bg_antibiotic_n"] = np.log10(1.0 + target["bg_antibiotic_n"].astype(float))

    target_pair = (
        si_reference_df["pathogen"].astype(str).eq(str(pathogen))
        & si_reference_df["antibiotic"].astype(str).eq(str(antibiotic))
    )
    reference = si_reference_df.loc[
        si_reference_df["count_for_model"].notna() & ~target_pair,
        [
            "pathogen", "antibiotic", "year", "count", "count_for_model", "log_count",
            "log_bg_pathogen_n", "log_bg_antibiotic_n",
        ],
    ].copy()
    if reference.empty:
        raise ValueError(
            "No eligible reference SI rows are available to refit the pooled baseline. "
            "Run notebook 05 first to create the reference SI dataset."
        )
    fit_input = pd.concat(
        [reference, target[reference.columns]],
        ignore_index=True,
    )
    scored, _, _ = fit_global_pooled_si_model(fit_input)
    target_scored = scored.loc[
        scored["pathogen"].astype(str).eq(str(pathogen))
        & scored["antibiotic"].astype(str).eq(str(antibiotic))
    ].copy()
    target_scored["min_amr_pair_publication_count"] = threshold
    return target_scored.sort_values("year").reset_index(drop=True)


def recompute_pair_si_from_available_sources(
    si_reference_df: pd.DataFrame,
    pathogen: str,
    antibiotic: str,
    *,
    pipeline_config: dict[str, Any],
    project_root: str | Path,
    pathogen_background_df: pd.DataFrame,
    antibiotic_background_df: pd.DataFrame,
    min_amr_pair_publication_count: float = 1.0,
    system1_probability_threshold: float | None = None,
    llm_include_decisions: Sequence[str] | None = None,
    allow_candidate_proxy: bool = True,
) -> dict[str, Any]:
    """Recover and score one pair from the deepest available cached AMR source.

    The helper prefers stored screened counts. If those are absent, it rebuilds
    pair-specific screened counts from candidate PMID/year rows and the configured
    cached prediction source. If yearly candidate rows are absent, it can rebuild
    them from candidate PMID links plus cached PubMed publication years. If candidate
    counts are the only evidence left, it can return an explicitly unvalidated
    candidate-literature proxy when ``allow_candidate_proxy`` is true.
    """
    from amr.screening import build_yearly_screened_pairs

    root = Path(project_root)
    analysis_level = str(cfg(pipeline_config, "si.antibiotic_analysis_level", "antibiotic")).strip()
    if analysis_level.casefold() not in {"", "antibiotic"}:
        raise ValueError(
            "Pair-specific recovery currently supports individual-antibiotic SI only; "
            f"configured analysis level is {analysis_level!r}."
        )

    def read_optional(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame(columns=columns or [])
        frame = pd.read_parquet(path)
        return frame[[column for column in (columns or frame.columns.tolist()) if column in frame.columns]].copy()

    def select_pair(frame: pd.DataFrame) -> pd.DataFrame:
        if not {"pathogen", "antibiotic"}.issubset(frame.columns):
            return frame.iloc[0:0].copy()
        mask = (
            frame["pathogen"].astype(str).str.casefold().eq(str(pathogen).casefold())
            & frame["antibiotic"].astype(str).str.casefold().eq(str(antibiotic).casefold())
        )
        return frame.loc[mask].copy()

    paths = {
        key: cfg_path(pipeline_config, f"paths.{key}", root)
        for key in (
            "yearly_screened", "yearly_candidates", "yearly_pair_pmids",
            "pubmed_metadata_cache", "system1_decisions", "unified_decisions",
        )
    }
    stored_screened = select_pair(
        read_optional(paths["yearly_screened"], ["pathogen", "antibiotic", "year", "count", "pmids"])
    )
    candidate = select_pair(
        read_optional(paths["yearly_candidates"], ["pathogen", "antibiotic", "year", "count", "pmids"])
    )
    candidate_method = "candidate_amr_pair_year_counts.parquet"

    if candidate.empty or "pmids" not in candidate.columns:
        pair_links = select_pair(
            read_optional(paths["yearly_pair_pmids"], ["pathogen", "antibiotic", "pmid"])
        )
        if not pair_links.empty:
            metadata = read_optional(paths["pubmed_metadata_cache"], ["pmid", "pub_year"])
            if not metadata.empty:
                pair_links["pmid"] = pair_links["pmid"].astype(str)
                metadata["pmid"] = metadata["pmid"].astype(str)
                metadata["pub_year"] = pd.to_numeric(metadata["pub_year"], errors="coerce")
                linked = pair_links.merge(metadata.dropna(subset=["pub_year"]), on="pmid", how="inner")
                linked["year"] = linked["pub_year"].astype(int)
                linked = linked.drop_duplicates(["pathogen", "antibiotic", "year", "pmid"])
                linked_candidate = (
                    linked.groupby(["pathogen", "antibiotic", "year"], as_index=False)
                    .agg(count=("pmid", "nunique"), pmids=("pmid", lambda values: sorted(set(values))))
                )
                if not linked_candidate.empty:
                    candidate = linked_candidate
                    candidate_method = "candidate PMID links + PubMed publication-year cache"

    source_name = str(cfg(pipeline_config, "screening.active_source", "llm")).strip().lower()
    if source_name not in {"llm", "system1"}:
        raise ValueError("screening.active_source must be 'llm' or 'system1'.")
    run_id = str(cfg(pipeline_config, "system1.run_id", "jev-default"))
    threshold = (
        float(cfg(pipeline_config, "screening.system1_probability_threshold", 0.5))
        if system1_probability_threshold is None
        else float(system1_probability_threshold)
    )
    include_decisions = tuple(
        cfg(pipeline_config, "llm.screened_include_decisions", ["strict", "yes"])
        if llm_include_decisions is None
        else llm_include_decisions
    )

    rebuilt_screened = pd.DataFrame(columns=["pathogen", "antibiotic", "year", "count", "pmids"])
    screening_overridden = system1_probability_threshold is not None or llm_include_decisions is not None
    if screening_overridden and (
        (source_name == "system1" and llm_include_decisions is not None)
        or (source_name == "llm" and system1_probability_threshold is not None)
    ):
        raise ValueError("Set only the override that matches screening.active_source.")
    rebuild_attempted = screening_overridden or stored_screened.empty
    if rebuild_attempted and not candidate.empty:
        decision_key = "system1_decisions" if source_name == "system1" else "unified_decisions"
        prediction_columns = (
            ["pmid", "pathogen", "antibiotic", "year", "system1_p_yes", "system1_status", "system1_run_id"]
            if source_name == "system1"
            else ["pmid", "pathogen", "antibiotic", "year", "unified_decision"]
        )
        predictions = select_pair(read_optional(paths[decision_key], prediction_columns))
        required_prediction_columns = (
            {"pmid", "pathogen", "antibiotic", "year", "system1_p_yes", "system1_status"}
            if source_name == "system1"
            else {"pmid", "pathogen", "antibiotic", "year", "unified_decision"}
        )
        if not predictions.empty and "pmids" in candidate.columns and required_prediction_columns.issubset(predictions.columns):
            rebuild_kwargs: dict[str, Any] = {"prediction_source": source_name, "output_path": None}
            if source_name == "system1":
                rebuild_kwargs.update(probability_threshold=threshold, system1_run_id=run_id)
            else:
                rebuild_kwargs.update(include_decisions=include_decisions)
            rebuilt_screened = build_yearly_screened_pairs(candidate, predictions, **rebuild_kwargs)

    if screening_overridden:
        selected_counts = rebuilt_screened
        selected_source = f"recomputed from candidate records using {source_name} override"
    elif not stored_screened.empty:
        selected_counts = stored_screened
        selected_source = "stored yearly_amr_screened_pairs.parquet"
    else:
        selected_counts = rebuilt_screened
        selected_source = f"rebuilt from {candidate_method} and cached {source_name} predictions"

    diagnostic = pd.DataFrame(columns=["year", "candidate_count", "stored_screened_count", "recomputed_screened_count"])
    yearly_frames: list[pd.DataFrame] = []
    for frame, name in (
        (candidate, "candidate_count"),
        (stored_screened, "stored_screened_count"),
        (rebuilt_screened, "recomputed_screened_count"),
    ):
        if not frame.empty:
            counts = frame.groupby("year", as_index=False)["count"].sum().rename(columns={"count": name})
            yearly_frames.append(counts)
    for frame in yearly_frames:
        diagnostic = diagnostic.merge(frame, on="year", how="outer") if not diagnostic.empty else frame
    if not diagnostic.empty:
        diagnostic = diagnostic.sort_values("year")
        for column in ["candidate_count", "stored_screened_count", "recomputed_screened_count"]:
            if column not in diagnostic:
                diagnostic[column] = 0 if column != "recomputed_screened_count" or rebuild_attempted else pd.NA
        diagnostic[["candidate_count", "stored_screened_count"]] = diagnostic[["candidate_count", "stored_screened_count"]].fillna(0)
        if rebuild_attempted:
            diagnostic["recomputed_screened_count"] = diagnostic["recomputed_screened_count"].fillna(0)
        diagnostic = diagnostic[["year", "candidate_count", "stored_screened_count", "recomputed_screened_count"]]

    data_label = "AMR pair"
    data_interpretation = "screened_amr"
    if selected_counts.empty and allow_candidate_proxy and not candidate.empty:
        selected_counts = candidate
        selected_source = f"candidate-only proxy from {candidate_method}; no AMR-screened counts recovered"
        data_label = "candidate pair"
        data_interpretation = "candidate_proxy"

    if selected_counts.empty:
        return {
            "pair_si_df": pd.DataFrame(),
            "availability_by_year": diagnostic,
            "source": selected_source,
            "candidate_source": candidate_method if not candidate.empty else "no candidate PMID/year records found",
            "data_label": data_label,
            "data_interpretation": data_interpretation,
            "message": (
                f"No AMR-accepted yearly counts are available for {pathogen!r} — {antibiotic!r}. "
                "The availability table shows candidate counts and any recoverable screened counts. "
                "If candidate counts exist but cannot support a fit, review the availability table "
                "and cached predictions. Raw candidate mentions are not AMR-screened observations."
            ),
        }

    pathogen_bg_path = cfg_path(pipeline_config, "paths.pathogen_background", root)
    antibiotic_bg_path = cfg_path(pipeline_config, "paths.antibiotic_background", root)
    if not pathogen_background_df.empty and not antibiotic_background_df.empty:
        pair_si = recompute_pair_si_from_counts(
            si_reference_df,
            selected_counts,
            pathogen_background_df,
            antibiotic_background_df,
            str(selected_counts.iloc[0]["pathogen"]),
            str(selected_counts.iloc[0]["antibiotic"]),
            min_amr_pair_publication_count=min_amr_pair_publication_count,
        )
    else:
        raise ValueError(
            "Pathogen and antibiotic background tables are required to recompute SI "
            f"(expected {pathogen_bg_path} and {antibiotic_bg_path})."
        )
    return {
        "pair_si_df": pair_si,
        "availability_by_year": diagnostic,
        "source": selected_source,
        "candidate_source": candidate_method if not candidate.empty else "no candidate PMID/year records found",
        "data_label": data_label,
        "data_interpretation": data_interpretation,
        "message": None,
    }


def filter_model_inputs_by_average_amr_publications(
    *,
    yearly_screened_model_df: pd.DataFrame,
    unified_decisions_model_df: pd.DataFrame,
    fit_start_year: int,
    fit_end_year: int,
    publication_filter_end_year: int | None = None,
    min_avg_amr_publications_per_year: float | None = None,
    min_total_amr_publications: float | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Keep model inputs above an average or total publication-window AMR threshold.

    ``min_total_amr_publications`` takes precedence for callers that want a total-count
    threshold. ``publication_filter_end_year`` defaults to ``fit_end_year`` so existing
    callers retain the historical behavior. The average threshold remains available;
    when neither threshold is supplied, the historical default of 5 publications per year
    is used.
    """
    required_cols = {"pathogen", "antibiotic", "year", "count"}
    missing = required_cols - set(yearly_screened_model_df.columns)
    if missing:
        raise KeyError(f"yearly_screened_model_df missing required columns: {sorted(missing)}")
    if fit_end_year < fit_start_year:
        raise ValueError("fit_end_year must be >= fit_start_year.")
    if publication_filter_end_year is None:
        publication_filter_end_year = fit_end_year
    if publication_filter_end_year < fit_start_year:
        raise ValueError("publication_filter_end_year must be >= fit_start_year.")
    if min_avg_amr_publications_per_year is not None and min_total_amr_publications is not None:
        raise ValueError("Set only one of min_avg_amr_publications_per_year or min_total_amr_publications.")
    if min_avg_amr_publications_per_year is None and min_total_amr_publications is None:
        min_avg_amr_publications_per_year = 5.0

    publication_filter_year_count = int(publication_filter_end_year) - int(fit_start_year) + 1
    publication_filter_window_df = yearly_screened_model_df.loc[
        yearly_screened_model_df["year"].between(int(fit_start_year), int(publication_filter_end_year))
    ]
    pair_publication_rates = (
        publication_filter_window_df.groupby(["pathogen", "antibiotic"], as_index=False)["count"]
        .sum()
        .rename(columns={"count": "fit_window_amr_publications"})
    )
    pair_publication_rates["avg_amr_publications_per_year"] = (
        pair_publication_rates["fit_window_amr_publications"] / publication_filter_year_count
    )

    if min_total_amr_publications is not None:
        eligible_pairs = pair_publication_rates.loc[
            pair_publication_rates["fit_window_amr_publications"] >= float(min_total_amr_publications),
            ["pathogen", "antibiotic"],
        ]
        threshold_description = (
            f">= {float(min_total_amr_publications):g} total publications"
        )
    else:
        assert min_avg_amr_publications_per_year is not None
        eligible_pairs = pair_publication_rates.loc[
            pair_publication_rates["avg_amr_publications_per_year"] >= float(min_avg_amr_publications_per_year),
            ["pathogen", "antibiotic"],
        ]
        threshold_description = (
            f">= {float(min_avg_amr_publications_per_year):g} publications/year on average"
        )
    pair_count_before = yearly_screened_model_df[["pathogen", "antibiotic"]].drop_duplicates().shape[0]
    filtered_yearly_df = yearly_screened_model_df.merge(
        eligible_pairs,
        on=["pathogen", "antibiotic"],
        how="inner",
    )
    if unified_decisions_model_df.empty:
        filtered_unified_df = unified_decisions_model_df.copy()
    else:
        filtered_unified_df = unified_decisions_model_df.merge(
            eligible_pairs,
            on=["pathogen", "antibiotic"],
            how="inner",
        )
    pair_count_after = filtered_yearly_df[["pathogen", "antibiotic"]].drop_duplicates().shape[0]

    if verbose:
        print(
            f"AMR publication filter: kept {pair_count_after:,} / {pair_count_before:,} pairs "
            f"with {threshold_description} "
            f"from {fit_start_year} to {publication_filter_end_year}."
        )
    return filtered_yearly_df, filtered_unified_df, pair_publication_rates


def summarize_pair_publication_rates(
    pair_publication_rates: pd.DataFrame,
    *,
    head: int | None = 30,
) -> pd.DataFrame:
    """Return pair publication rates sorted by average fit-window AMR volume."""
    out = pair_publication_rates.sort_values(
        "avg_amr_publications_per_year",
        ascending=False,
    )
    return out.head(head) if head is not None else out


def print_si_dataset_summary(si_df: pd.DataFrame) -> None:
    """Print a short yearly SI dataset summary for notebooks."""
    print(f"SI rows: {len(si_df):,}")
    print(f"SI pairs: {si_df[['pathogen', 'antibiotic']].drop_duplicates().shape[0]:,}")
    if "pathogen" in si_df:
        print(f"SI pathogens: {si_df['pathogen'].nunique():,}")


def summarize_pathogen_coverage(
    *,
    source_df: pd.DataFrame,
    modeled_df: pd.DataFrame,
    source_label: str = "source",
    modeled_label: str = "modeled",
    print_summary: bool = True,
) -> pd.DataFrame:
    """Summarize which pathogens survive from a source table into a modeled table."""
    for label, df in [(source_label, source_df), (modeled_label, modeled_df)]:
        if "pathogen" not in df.columns:
            raise KeyError(f"{label} dataframe is missing required column: 'pathogen'")

    def _pathogen_stats(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        grouped = df.groupby("pathogen", dropna=False).size().reset_index(name=f"{prefix}_rows")
        if "count" in df.columns:
            counts = (
                df.assign(count=pd.to_numeric(df["count"], errors="coerce").fillna(0))
                .groupby("pathogen", dropna=False)["count"]
                .sum()
                .reset_index(name=f"{prefix}_publications")
            )
            grouped = grouped.merge(counts, on="pathogen", how="left")
        if {"pathogen", "antibiotic"}.issubset(df.columns):
            pairs = (
                df[["pathogen", "antibiotic"]]
                .drop_duplicates()
                .groupby("pathogen", dropna=False)
                .size()
                .reset_index(name=f"{prefix}_pairs")
            )
            grouped = grouped.merge(pairs, on="pathogen", how="left")
        return grouped

    source_stats = _pathogen_stats(source_df, "source")
    modeled_stats = _pathogen_stats(modeled_df, "modeled")
    summary_df = source_stats.merge(modeled_stats, on="pathogen", how="outer")
    summary_df["in_source"] = summary_df["source_rows"].notna()
    summary_df["in_modeled"] = summary_df["modeled_rows"].notna()

    for col in summary_df.columns:
        if col.endswith(("_rows", "_pairs", "_publications")):
            summary_df[col] = summary_df[col].fillna(0)

    summary_df["status"] = "present_in_both"
    summary_df.loc[summary_df["in_source"] & ~summary_df["in_modeled"], "status"] = "missing_from_modeled"
    summary_df.loc[~summary_df["in_source"] & summary_df["in_modeled"], "status"] = "only_in_modeled"
    summary_df = summary_df.sort_values(["status", "pathogen"]).reset_index(drop=True)

    if print_summary:
        source_pathogens = int(summary_df["in_source"].sum())
        modeled_pathogens = int(summary_df["in_modeled"].sum())
        missing = summary_df.loc[summary_df["status"].eq("missing_from_modeled"), "pathogen"].astype(str).tolist()
        extra = summary_df.loc[summary_df["status"].eq("only_in_modeled"), "pathogen"].astype(str).tolist()
        print(f"{source_label} pathogens: {source_pathogens:,}")
        print(f"{modeled_label} pathogens: {modeled_pathogens:,}")
        if missing:
            print(f"Pathogens present in {source_label} but absent from {modeled_label}: {len(missing):,}")
            print(", ".join(missing))
        if extra:
            print(f"Pathogens present in {modeled_label} but absent from {source_label}: {len(extra):,}")
            print(", ".join(extra))
    return summary_df


def rank_future_high_interest_pairs(
    si_df: pd.DataFrame,
    *,
    fit_start_year: int = 1970,
    fit_end_year: int | None = None,
    num_future_years: int = 20,
    curve_fit_end_year: int | None = None,
    post_fit_validation_end_year: int | None = None,
    min_current_predicted_count: float = 10.0,
    min_current_amr_count: float = 10.0,
    min_future_predicted_count: float = 10.0,
    min_si_change: float = 0.1,
    min_post_fit_observed_years: int = 1,
    max_post_fit_si_abs_pct_error_mean: float = 40.0,
    max_si_loess_residual_variance_mean: float | None = None,
    fallback_top_n: int = 0,
    rank_output_path: str | Path = "yearly_si_ranked_future.parquet",
    interesting_output_path: str | Path | None = "yearly_si_ranked_future_interesting.parquet",
    loess_frac: float = 0.6,
    loess_degree: int = 1,
    tail_damping_years: Any = None,
    min_amr_pair_publication_count: float = 2.0,
    min_num_years_with_valid_counts: int | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank future SI shifts, then apply reliability-aware interesting-case selection."""
    rank_df = rank_yearly_pairs_by_future_si_shift(
        si_df,
        fit_start_year=fit_start_year,
        fit_end_year=fit_end_year,
        curve_fit_end_year=curve_fit_end_year,
        post_fit_validation_end_year=post_fit_validation_end_year,
        num_future_years=num_future_years,
        save_path=str(rank_output_path),
        loess_frac=loess_frac,
        loess_degree=loess_degree,
        tail_damping_years=tail_damping_years,
        min_amr_pair_publication_count=min_amr_pair_publication_count,
        min_num_years_with_valid_counts=min_num_years_with_valid_counts,
        verbose=verbose,
    )
    interesting_df = select_interesting_yearly_rank_cases(
        rank_df,
        min_current_predicted_count=min_current_predicted_count,
        min_current_amr_count=min_current_amr_count,
        min_future_predicted_count=min_future_predicted_count,
        min_si_change=min_si_change,
        min_post_fit_observed_years=min_post_fit_observed_years,
        max_post_fit_si_abs_pct_error_mean=max_post_fit_si_abs_pct_error_mean,
        max_si_loess_residual_variance_mean=max_si_loess_residual_variance_mean,
        fallback_top_n=fallback_top_n,
    )
    if interesting_output_path:
        Path(interesting_output_path).parent.mkdir(parents=True, exist_ok=True)
        atomic_write_parquet(interesting_df, interesting_output_path, index=False)
    return rank_df, interesting_df


def build_si_driver_analysis(
    *,
    interesting_df: pd.DataFrame,
    unified_decisions_df: pd.DataFrame,
    fit_end_year: int,
    include_decisions: tuple[str, ...] | list[str] | set[str] = ("strict", "yes"),
    lookback_years: int = 5,
    output_path: str | Path | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build yearly category counts and merge driver summaries into interesting pairs."""
    category_counts_df = build_yearly_category_counts(
        unified_decisions_df,
        include_decisions=include_decisions,
    )
    driver_summary_df = summarize_si_change_drivers(
        interesting_df,
        unified_decisions_df,
        fit_end_year=fit_end_year,
        include_decisions=include_decisions,
        lookback_years=lookback_years,
    )
    if driver_summary_df.empty:
        interesting_with_drivers_df = interesting_df.copy()
        if verbose:
            print("No SI driver classification summaries were produced.")
    else:
        interesting_with_drivers_df = interesting_df.merge(
            driver_summary_df,
            on=["pathogen", "antibiotic"],
            how="left",
        )
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            atomic_write_parquet(interesting_with_drivers_df, output_path, index=False)
        if verbose:
            print(f"Saved SI driver explanations to {output_path}")
    return category_counts_df, interesting_with_drivers_df


def _display_figure(figure: Any) -> None:
    try:
        from IPython.display import display
    except ImportError:
        warnings.warn(
            "display_plots=True requires IPython; plots were saved but not displayed inline.",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    display(figure)


def _load_driver_category_coarsening(
    category_coarsening_path: str | Path | None,
    category_coarsening_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Load fine-to-coarse AMR driver category mapping."""
    if category_coarsening_df is not None:
        lookup_df = category_coarsening_df.copy()
    else:
        lookup_path = DEFAULT_DRIVER_CATEGORY_COARSENING_PATH if category_coarsening_path is None else Path(category_coarsening_path)
        lookup_df = pd.read_csv(lookup_path)
    required = {"amr_primary_category", "amr_coarse_category"}
    missing = required - set(lookup_df.columns)
    if missing:
        raise KeyError(f"category coarsening lookup missing required columns: {sorted(missing)}")
    lookup_df = lookup_df.copy()
    for column in ("amr_primary_category", "amr_coarse_category"):
        lookup_df[column] = lookup_df[column].fillna("").astype(str).str.strip()
    lookup_df = lookup_df.loc[
        lookup_df["amr_primary_category"].ne("") & lookup_df["amr_coarse_category"].ne("")
    ]
    conflicting = lookup_df.groupby("amr_primary_category")["amr_coarse_category"].nunique()
    conflicting = conflicting[conflicting > 1]
    if not conflicting.empty:
        preview = ", ".join(conflicting.index.astype(str).tolist()[:10])
        extra = "" if len(conflicting) <= 10 else f" ... (+{len(conflicting) - 10} more)"
        raise ValueError(
            "Conflicting coarse categories found for "
            f"{len(conflicting)} primary AMR categories: {preview}{extra}"
        )
    lookup_df = lookup_df.drop_duplicates(subset=["amr_primary_category"], keep="first")
    return lookup_df


def coarsen_category_counts(
    category_counts_df: pd.DataFrame,
    *,
    category_coarsening_path: str | Path | None = None,
    category_coarsening_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    """Map fine AMR driver categories to coarse groups and aggregate counts."""
    required = {"pathogen", "antibiotic", "year", "amr_primary_category", "count"}
    missing = required - set(category_counts_df.columns)
    if missing:
        raise KeyError(f"category_counts_df missing required columns: {sorted(missing)}")
    lookup_df = _load_driver_category_coarsening(category_coarsening_path, category_coarsening_df)
    work = category_counts_df.copy()
    work["amr_primary_category"] = work["amr_primary_category"].astype(str).str.strip()
    work = work.merge(lookup_df, on="amr_primary_category", how="left")
    work["amr_primary_category"] = work["amr_coarse_category"].fillna(work["amr_primary_category"])
    grouped_df = (
        work.groupby(["pathogen", "antibiotic", "year", "amr_primary_category"], as_index=False)["count"]
        .sum()
        .sort_values(["pathogen", "antibiotic", "year", "amr_primary_category"])
        .reset_index(drop=True)
    )

    order_df = lookup_df[["amr_coarse_category"]].drop_duplicates().copy()
    if "coarse_order" in lookup_df.columns:
        order_df = (
            lookup_df[["amr_coarse_category", "coarse_order"]]
            .drop_duplicates(subset=["amr_coarse_category"])
            .assign(coarse_order=lambda df: pd.to_numeric(df["coarse_order"], errors="coerce"))
            .sort_values(["coarse_order", "amr_coarse_category"], na_position="last")
        )
    category_order = order_df["amr_coarse_category"].dropna().astype(str).tolist()
    category_order = category_order or AMR_DRIVER_COARSE_CATEGORY_ORDER

    category_colors = AMR_DRIVER_COARSE_CATEGORY_COLORS.copy()
    if "coarse_color" in lookup_df.columns:
        color_df = lookup_df[["amr_coarse_category", "coarse_color"]].dropna().drop_duplicates(
            subset=["amr_coarse_category"], keep="first"
        )
        category_colors.update(dict(color_df.itertuples(index=False, name=None)))
    return grouped_df, category_order, category_colors


def save_category_driver_plots(
    *,
    category_counts_df: pd.DataFrame,
    si_df: pd.DataFrame | None = None,
    interesting_df: pd.DataFrame,
    output_dir: str | Path,
    start_year: int | None = None,
    end_year: int | None = None,
    fit_end_year: int,
    max_plots: int = 25,
    figure_format: str = "pdf",
    driver_category_level: str = "original",
    category_coarsening_path: str | Path | None = None,
    category_coarsening_df: pd.DataFrame | None = None,
    display_plots: bool = False,
    max_display_plots: int | None = None,
    verbose: bool = True,
) -> int:
    """Save category driver count plots for the top interesting rows."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_format = str(figure_format).strip().lstrip(".") or "pdf"
    saved_plots = 0
    displayed_plots = 0
    if category_counts_df.empty:
        if verbose:
            print("No category count curves available to plot.")
        return saved_plots
    driver_category_level = str(driver_category_level).strip().lower()
    if driver_category_level in {"original", "fine", "primary"}:
        plot_category_counts_df = category_counts_df
        category_order = None
        category_colors = None
        category_label = "AMR category drivers"
        missing_information_category = None
        filename_suffix = "drivers"
    elif driver_category_level in {"coarse", "coarsened"}:
        plot_category_counts_df, category_order, category_colors = coarsen_category_counts(
            category_counts_df,
            category_coarsening_path=category_coarsening_path,
            category_coarsening_df=category_coarsening_df,
        )
        category_label = "Coarse AMR category drivers"
        missing_information_category = "Unsupported or missing evidence"
        filename_suffix = "coarse_drivers"
    else:
        raise ValueError("driver_category_level must be one of: 'original', 'fine', 'primary', 'coarse', 'coarsened'.")

    for row in interesting_df.head(max_plots).itertuples(index=False):
        try:
            result = plot_yearly_category_driver_counts(
                category_counts_df=plot_category_counts_df,
                pathogen=row.pathogen,
                antibiotic=row.antibiotic,
                si_df=si_df,
                start_year=start_year,
                end_year=end_year,
                fit_end_year=fit_end_year,
                category_order=category_order,
                category_colors=category_colors,
                category_label=category_label,
                **({} if missing_information_category is None else {"missing_information_category": missing_information_category}),
            )
        except ValueError:
            continue
        pathogen_slug = str(row.pathogen).replace("/", "_").replace(" ", "_")
        antibiotic_slug = str(row.antibiotic).replace("/", "_").replace(" ", "_")
        out_path = output_dir / f"{pathogen_slug}__{antibiotic_slug}__{filename_suffix}.{figure_format}"
        result["figure"].savefig(out_path, bbox_inches="tight")
        if display_plots and (max_display_plots is None or displayed_plots < max_display_plots):
            _display_figure(result["figure"])
            displayed_plots += 1
        plt.close(result["figure"])
        saved_plots += 1

    if verbose:
        print(f"Saved {saved_plots} {driver_category_level} category driver plots to {output_dir}")
        if display_plots:
            print(f"Displayed {displayed_plots} category driver plots inline.")
    return saved_plots


def save_category_driver_pie_grid(
    *,
    category_counts_df: pd.DataFrame,
    si_df: pd.DataFrame | None = None,
    interesting_df: pd.DataFrame,
    output_path: str | Path,
    num_columns: int = 5,
    driver_category_level: str = "original",
    category_coarsening_path: str | Path | None = None,
    category_coarsening_df: pd.DataFrame | None = None,
    verbose: bool = True,
) -> int:
    """Save one-page pie-chart grid of all-year driver fractions for each pair."""
    if int(num_columns) < 1:
        raise ValueError("num_columns must be >= 1.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if category_counts_df.empty or interesting_df.empty:
        if verbose:
            print("No category driver pairs available for the pie-chart grid.")
        return 0

    driver_category_level = str(driver_category_level).strip().lower()
    if driver_category_level in {"original", "fine", "primary"}:
        plot_category_counts_df = category_counts_df
        category_order = None
        category_colors = AMR_DRIVER_CATEGORY_COLORS
        missing_information_category = None
    elif driver_category_level in {"coarse", "coarsened"}:
        plot_category_counts_df, category_order, category_colors = coarsen_category_counts(
            category_counts_df,
            category_coarsening_path=category_coarsening_path,
            category_coarsening_df=category_coarsening_df,
        )
        missing_information_category = "Unsupported or missing evidence"
    else:
        raise ValueError("driver_category_level must be one of: 'original', 'fine', 'primary', 'coarse', 'coarsened'.")

    pair_totals: list[tuple[str, pd.Series]] = []
    for row in interesting_df.itertuples(index=False):
        try:
            result = plot_yearly_category_driver_counts(
                category_counts_df=plot_category_counts_df,
                pathogen=row.pathogen,
                antibiotic=row.antibiotic,
                si_df=si_df,
                category_order=category_order,
                category_colors=category_colors,
                **(
                    {}
                    if missing_information_category is None
                    else {"missing_information_category": missing_information_category}
                ),
            )
        except ValueError:
            continue

        totals = pd.to_numeric(result["count_pivot"].sum(axis=0), errors="coerce").fillna(0.0)
        totals = totals.loc[totals.gt(0)]
        plt.close(result["figure"])
        if totals.empty:
            continue
        label = format_pathogen_antibiotic_label(row.pathogen, row.antibiotic).replace("<br>", "\n")
        pair_totals.append((label, totals))

    if not pair_totals:
        if verbose:
            print("No non-zero category driver totals available for the pie-chart grid.")
        return 0

    num_columns = int(num_columns)
    num_rows = int(ceil(len(pair_totals) / num_columns))
    fig, axes = plt.subplots(
        num_rows,
        num_columns,
        figsize=(3.2 * num_columns, 3.0 * num_rows),
        squeeze=False,
    )
    present_categories = []
    for _, totals in pair_totals:
        for category in totals.index:
            if category not in present_categories:
                present_categories.append(category)
    if category_order is not None:
        ordered_categories = [category for category in category_order if category in present_categories]
        ordered_categories.extend(category for category in present_categories if category not in ordered_categories)
    else:
        ordered_categories = present_categories

    fallback_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    category_color_values = {
        category: category_colors.get(
            category,
            fallback_colors[idx % len(fallback_colors)] if fallback_colors else "#999999",
        )
        for idx, category in enumerate(ordered_categories)
    }
    for index, (label, totals) in enumerate(pair_totals):
        ax = axes.flat[index]
        values = totals.reindex(ordered_categories, fill_value=0.0)
        values = values.loc[values.gt(0)]
        ax.pie(
            values.to_numpy(dtype=float),
            colors=[category_color_values[category] for category in values.index],
            startangle=90,
            counterclock=False,
            autopct=lambda pct: f"{pct:.0f}%" if pct >= 4 else "",
            textprops={"fontsize": 7},
            wedgeprops={"linewidth": 0.5, "edgecolor": "white"},
        )
        ax.set_title(f"{label}\n(n={int(values.sum()):,})", fontsize=9)

    for ax in axes.flat[len(pair_totals) :]:
        ax.axis("off")

    legend_handles = [
        Patch(facecolor=category_color_values[category], edgecolor="none", label=category)
        for category in ordered_categories
    ]
    fig.suptitle(
        f"All-year AMR driver composition by pair ({len(pair_totals):,} pairs)",
        fontsize=14,
        y=0.995,
    )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=min(num_columns, max(len(legend_handles), 1)),
        frameon=False,
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    if verbose:
        print(f"Saved all-year category driver pie grid for {len(pair_totals):,} pairs to {output_path}")
    return len(pair_totals)


def save_interesting_si_plots(
    *,
    si_df: pd.DataFrame,
    interesting_df: pd.DataFrame,
    output_dir: str | Path,
    fit_start_year: int,
    fit_end_year: int,
    num_future_years: int,
    curve_fit_end_year: int | None = None,
    pathogen_background_df: pd.DataFrame | None = None,
    antibiotic_background_df: pd.DataFrame | None = None,
    max_plots: int = 25,
    loess_frac: float = 0.6,
    loess_degree: int = 1,
    tail_damping_years: Any = None,
    min_amr_pair_publication_count: float = 2.0,
    min_num_years_with_valid_counts: int | None = None,
    fill_missing_amr_counts_with_zero: bool = False,
    figure_format: str = "pdf",
    display_plots: bool = False,
    max_display_plots: int | None = None,
    verbose: bool = True,
) -> int:
    """Save SI forecast plots for the top interesting rows."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_format = str(figure_format).strip().lstrip(".") or "pdf"
    saved_plots = 0
    skipped_plots = 0
    displayed_plots = 0
    plot_fit_end_year = fit_end_year if curve_fit_end_year is None else int(curve_fit_end_year)
    for row in interesting_df.head(max_plots).itertuples(index=False):
        try:
            result = plot_yearly_si_pair(
                si_df=si_df,
                pathogen=row.pathogen,
                antibiotic=row.antibiotic,
                fit_start_year=fit_start_year,
                fit_end_year=plot_fit_end_year,
                num_additional_years=num_future_years,
                loess_frac=loess_frac,
                loess_degree=loess_degree,
                tail_damping_years=tail_damping_years,
                min_amr_pair_publication_count=min_amr_pair_publication_count,
                min_num_years_with_valid_counts=min_num_years_with_valid_counts,
                fill_missing_amr_counts_with_zero=fill_missing_amr_counts_with_zero,
                pathogen_background_df=pathogen_background_df,
                antibiotic_background_df=antibiotic_background_df,
            )
        except ValueError as exc:
            if "years with valid AMR pair counts" not in str(exc):
                raise
            skipped_plots += 1
            if verbose:
                print(f"Skipping {row.pathogen} vs {row.antibiotic}: {exc}")
            continue
        pathogen_slug = str(row.pathogen).replace("/", "_").replace(" ", "_")
        antibiotic_slug = str(row.antibiotic).replace("/", "_").replace(" ", "_")
        out_path = output_dir / f"{pathogen_slug}__{antibiotic_slug}.{figure_format}"
        result["figure"].savefig(out_path, bbox_inches="tight")
        if display_plots and (max_display_plots is None or displayed_plots < max_display_plots):
            _display_figure(result["figure"])
            displayed_plots += 1
        plt.close(result["figure"])
        saved_plots += 1

    if verbose:
        print(f"Saved {saved_plots} plots to {output_dir}")
        if skipped_plots:
            print(f"Skipped {skipped_plots} plots because they lacked enough valid yearly AMR counts.")
        if display_plots:
            print(f"Displayed {displayed_plots} plots inline.")
    return saved_plots


def _slugify_plot_name(value: object) -> str:
    return (
        str(value)
        .strip()
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
    )


def select_decreasing_smoothed_si_trends(
    rank_df: pd.DataFrame,
    *,
    output_path: str | Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Return ranked pairs whose smoothed SI decreases from reference to future year."""
    required_cols = {"status", "si_change"}
    missing = required_cols - set(rank_df.columns)
    if missing:
        raise KeyError(f"rank_df is missing required columns: {sorted(missing)}")

    out = (
        rank_df.loc[
            rank_df["status"].eq("ok")
            & pd.to_numeric(rank_df["si_change"], errors="coerce").lt(0)
        ]
        .copy()
        .sort_values("si_change", ascending=True)
        .reset_index(drop=True)
    )
    if output_path:
        atomic_write_parquet(out, output_path, index=False)
        if verbose:
            print(f"Saved decreasing smoothed SI table: {len(out):,} rows -> {output_path}")
    return out


def summarize_smoothed_si_shapes(
    rank_df: pd.DataFrame,
    si_df: pd.DataFrame,
    *,
    fit_start_year: int,
    curve_fit_end_year: int,
    num_future_years: int,
    loess_frac: float = 0.6,
    loess_degree: int = 1,
    tail_damping_years: Any = None,
    min_amr_pair_publication_count: float = 2.0,
    min_num_years_with_valid_counts: int | None = None,
) -> pd.DataFrame:
    """Score smoothed SI curves for U-shaped and inverted-U shape."""
    rows: list[dict[str, Any]] = []
    curve_end_year = int(curve_fit_end_year) + int(num_future_years)
    for row in rank_df.loc[rank_df["status"].eq("ok")].itertuples(index=False):
        try:
            result = fit_predict_yearly_si_loess(
                si_df,
                pathogen=row.pathogen,
                antibiotic=row.antibiotic,
                fit_start_year=fit_start_year,
                fit_end_year=curve_fit_end_year,
                num_additional_years=num_future_years,
                loess_frac=loess_frac,
                loess_degree=loess_degree,
                tail_damping_years=tail_damping_years,
                min_amr_pair_publication_count=min_amr_pair_publication_count,
                min_num_years_with_valid_counts=min_num_years_with_valid_counts,
            )
        except Exception as exc:
            rows.append(
                {
                    "pathogen": row.pathogen,
                    "antibiotic": row.antibiotic,
                    "shape_status": "failed",
                    "shape_error": str(exc),
                }
            )
            continue

        pred_df = result["pred_df"].loc[
            lambda df: df["year"].between(int(fit_start_year), curve_end_year)
        ].copy()
        if pred_df["year"].nunique() < 3:
            continue

        shape_start_year = int(pred_df["year"].min())
        shape_end_year = int(pred_df["year"].max())
        midpoint_year = int(round((shape_start_year + shape_end_year) / 2))
        midpoint_idx = (pred_df["year"] - midpoint_year).abs().idxmin()

        x = pred_df["year"].to_numpy(dtype=float) - float(midpoint_year)
        y = pred_df["predicted_SI"].to_numpy(dtype=float)
        quadratic_coef, linear_coef, intercept = np.polyfit(x, y, deg=2)
        fitted_y = (quadratic_coef * x * x) + (linear_coef * x) + intercept
        total_ss = float(np.square(y - y.mean()).sum())
        residual_ss = float(np.square(y - fitted_y).sum())
        quadratic_r2 = np.nan if total_ss <= 0 else 1.0 - (residual_ss / total_ss)

        start_si = float(pred_df.loc[pred_df["year"].eq(shape_start_year), "predicted_SI"].iloc[0])
        end_si = float(pred_df.loc[pred_df["year"].eq(shape_end_year), "predicted_SI"].iloc[0])
        midpoint_si = float(pred_df.loc[midpoint_idx, "predicted_SI"])
        endpoint_min = float(min(start_si, end_si))
        endpoint_max = float(max(start_si, end_si))
        # Require both sides of the curve to support the shape. Comparing the
        # midpoint only with the endpoint mean mislabels monotone curves whose
        # quadratic fit has a small amount of flattening.
        convex_u_score = endpoint_min - midpoint_si
        concave_cap_score = midpoint_si - endpoint_max

        rows.append(
            {
                "pathogen": row.pathogen,
                "antibiotic": row.antibiotic,
                "shape_start_year": shape_start_year,
                "curve_fit_end_year": int(curve_fit_end_year),
                "shape_end_year": shape_end_year,
                "shape_midpoint_year": int(pred_df.loc[midpoint_idx, "year"]),
                "start_predicted_SI": start_si,
                "midpoint_predicted_SI": midpoint_si,
                "end_predicted_SI": end_si,
                "endpoint_balance_abs_diff": float(abs(end_si - start_si)),
                "convex_u_score": convex_u_score,
                "concave_cap_score": concave_cap_score,
                "shape_strength": float(max(convex_u_score, concave_cap_score, 0.0)),
                "si_quadratic_coef": float(quadratic_coef),
                "si_quadratic_r2": quadratic_r2,
                "current_predicted_SI": getattr(row, "current_predicted_SI", np.nan),
                "future_predicted_SI": getattr(row, "future_predicted_SI", np.nan),
                "si_change": getattr(row, "si_change", np.nan),
                "si_loess_residual_variance_mean": getattr(row, "si_loess_residual_variance_mean", np.nan),
                "shape_status": "ok",
                "shape_error": None,
            }
        )
    return pd.DataFrame(rows)


def select_smoothed_si_shape_cases(
    shape_df: pd.DataFrame,
    *,
    top_n: int = 10,
    convex_output_path: str | Path | None = None,
    concave_output_path: str | Path | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return top U-shaped and inverted-U smoothed SI curves."""
    ok_shape_df = shape_df.loc[shape_df["shape_status"].eq("ok")].copy()
    convex_df = (
        ok_shape_df.loc[
            pd.to_numeric(ok_shape_df["convex_u_score"], errors="coerce").gt(0)
            & pd.to_numeric(ok_shape_df["si_quadratic_coef"], errors="coerce").gt(0)
        ]
        .sort_values(["convex_u_score", "si_quadratic_r2"], ascending=[False, False])
        .head(int(top_n))
        .reset_index(drop=True)
    )
    concave_df = (
        ok_shape_df.loc[
            pd.to_numeric(ok_shape_df["concave_cap_score"], errors="coerce").gt(0)
            & pd.to_numeric(ok_shape_df["si_quadratic_coef"], errors="coerce").lt(0)
        ]
        .sort_values(["concave_cap_score", "si_quadratic_r2"], ascending=[False, False])
        .head(int(top_n))
        .reset_index(drop=True)
    )
    if convex_output_path:
        atomic_write_parquet(convex_df, convex_output_path, index=False)
        if verbose:
            print(f"Saved convex U-shaped smoothed SI table: {len(convex_df):,} rows -> {convex_output_path}")
    if concave_output_path:
        atomic_write_parquet(concave_df, concave_output_path, index=False)
        if verbose:
            print(f"Saved concave inverted-U smoothed SI table: {len(concave_df):,} rows -> {concave_output_path}")
    return convex_df, concave_df


def save_smoothed_si_screen_plots(
    *,
    screen_df: pd.DataFrame,
    si_df: pd.DataFrame,
    output_dir: str | Path,
    fit_start_year: int,
    curve_fit_end_year: int,
    num_future_years: int,
    pathogen_background_df: pd.DataFrame | None = None,
    antibiotic_background_df: pd.DataFrame | None = None,
    max_plots: int = 10,
    loess_frac: float = 0.6,
    loess_degree: int = 1,
    tail_damping_years: Any = None,
    min_amr_pair_publication_count: float = 2.0,
    min_num_years_with_valid_counts: int | None = None,
    fill_missing_amr_counts_with_zero: bool = False,
    figure_format: str = "pdf",
    display_plots: bool = False,
    verbose: bool = True,
) -> int:
    """Save and optionally display SI plots for a shape-screen table."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_format = str(figure_format).strip().lstrip(".") or "pdf"
    saved_plots = 0
    skipped_plots = 0
    for row in screen_df.head(max_plots).itertuples(index=False):
        try:
            result = plot_yearly_si_pair(
                si_df=si_df,
                pathogen=row.pathogen,
                antibiotic=row.antibiotic,
                fit_start_year=fit_start_year,
                fit_end_year=curve_fit_end_year,
                num_additional_years=num_future_years,
                loess_frac=loess_frac,
                loess_degree=loess_degree,
                tail_damping_years=tail_damping_years,
                min_amr_pair_publication_count=min_amr_pair_publication_count,
                min_num_years_with_valid_counts=min_num_years_with_valid_counts,
                fill_missing_amr_counts_with_zero=fill_missing_amr_counts_with_zero,
                pathogen_background_df=pathogen_background_df,
                antibiotic_background_df=antibiotic_background_df,
            )
        except ValueError as exc:
            if "years with valid AMR pair counts" not in str(exc):
                raise
            skipped_plots += 1
            if verbose:
                print(f"Skipping {row.pathogen} vs {row.antibiotic}: {exc}")
            continue
        plot_path = output_dir / f"{_slugify_plot_name(row.pathogen)}__{_slugify_plot_name(row.antibiotic)}.{figure_format}"
        result["figure"].savefig(plot_path, bbox_inches="tight")
        if display_plots:
            _display_figure(result["figure"])
        plt.close(result["figure"])
        saved_plots += 1

    if verbose:
        print(f"Saved {saved_plots} smoothed SI screen plots to {output_dir}")
        if skipped_plots:
            print(f"Skipped {skipped_plots} smoothed SI plots because they lacked enough valid yearly AMR counts.")
    return saved_plots


def select_existing_columns(df: pd.DataFrame, columns: Sequence[str], *, head: int | None = None) -> pd.DataFrame:
    """Return the requested columns that exist, optionally limited to the first rows."""
    out = df[[col for col in columns if col in df.columns]]
    return out.head(head) if head is not None else out


__all__ = [
    "build_antibiotic_group_mapping",
    "coarsen_category_counts",
    "build_si_analysis_paths",
    "build_si_driver_analysis",
    "build_yearly_si_dataset",
    "DEFAULT_DRIVER_CATEGORY_COARSENING_PATH",
    "DEFAULT_DRIVER_SUMMARY_COLUMNS",
    "DEFAULT_DECREASING_SI_SUMMARY_COLUMNS",
    "DEFAULT_INTERESTING_SUMMARY_COLUMNS",
    "DEFAULT_SMOOTHED_SI_SHAPE_SUMMARY_COLUMNS",
    "filter_model_inputs_by_average_amr_publications",
    "load_yearly_si_inputs",
    "print_si_dataset_summary",
    "prepare_antibiotic_analysis_level",
    "recompute_pair_si_from_counts",
    "recompute_pair_si_from_available_sources",
    "rank_future_high_interest_pairs",
    "save_category_driver_plots",
    "save_category_driver_pie_grid",
    "save_interesting_si_plots",
    "save_smoothed_si_screen_plots",
    "select_decreasing_smoothed_si_trends",
    "select_existing_columns",
    "select_smoothed_si_shape_cases",
    "slugify_analysis_level",
    "summarize_smoothed_si_shapes",
    "summarize_pair_publication_rates",
    "summarize_pathogen_coverage",
]
