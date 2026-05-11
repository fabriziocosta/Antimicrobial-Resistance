"""Pipelines for yearly scientific-interest modeling and ranking."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from amr.io import atomic_write_parquet
from amr.si.core import (
    build_yearly_category_counts,
    fit_global_pooled_si_model,
    plot_yearly_category_driver_counts,
    plot_yearly_si_pair,
    rank_yearly_pairs_by_future_si_shift,
    select_interesting_yearly_rank_cases,
    summarize_si_change_drivers,
)

DEFAULT_DRIVER_REQUIRED_COLUMNS = {
    "pathogen",
    "antibiotic",
    "year",
    "unified_decision",
    "amr_primary_category",
    "amr_pair_centrality",
    "amr_needs_manual_review",
}

DEFAULT_INTERESTING_SUMMARY_COLUMNS = [
    "pathogen",
    "antibiotic",
    "interest_score",
    "si_change",
    "current_predicted_count",
    "future_predicted_count",
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
    "driver_manual_review_share",
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
    mapping_df["source_antibiotic"] = mapping_df["source_antibiotic"].astype(str).str.strip()
    mapping_df["analysis_antibiotic"] = mapping_df["analysis_antibiotic"].astype(str).str.strip()
    mapping_df = mapping_df.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
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

    antibiotic_bg_mapped_df = antibiotic_background_df.merge(
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
    path_bg = pathogen_background_df.rename(columns={"count": "bg_pathogen_n"}).copy()
    ab_bg = antibiotic_background_df.rename(columns={"count": "bg_antibiotic_n"}).copy()

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

    merged["log_count"] = np.log10(1.0 + merged["count"])
    merged["log_bg_pathogen_n"] = np.log10(1.0 + merged["bg_pathogen_n"])
    merged["log_bg_antibiotic_n"] = np.log10(1.0 + merged["bg_antibiotic_n"])

    si_df, pooled_model, design_columns = fit_global_pooled_si_model(merged, alpha=alpha)
    si_df.attrs["coverage_report"] = coverage_report.to_dict(orient="records")
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        atomic_write_parquet(si_df, output_path, index=False)
    return si_df, pooled_model, design_columns


def filter_model_inputs_by_average_amr_publications(
    *,
    yearly_screened_model_df: pd.DataFrame,
    unified_decisions_model_df: pd.DataFrame,
    fit_start_year: int,
    fit_end_year: int,
    min_avg_amr_publications_per_year: float = 5.0,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Keep model inputs whose pair-level AMR count averages above a fit-window threshold."""
    required_cols = {"pathogen", "antibiotic", "year", "count"}
    missing = required_cols - set(yearly_screened_model_df.columns)
    if missing:
        raise KeyError(f"yearly_screened_model_df missing required columns: {sorted(missing)}")
    if fit_end_year < fit_start_year:
        raise ValueError("fit_end_year must be >= fit_start_year.")

    fit_year_count = int(fit_end_year) - int(fit_start_year) + 1
    fit_window_df = yearly_screened_model_df.loc[
        yearly_screened_model_df["year"].between(int(fit_start_year), int(fit_end_year))
    ]
    pair_publication_rates = (
        fit_window_df.groupby(["pathogen", "antibiotic"], as_index=False)["count"]
        .sum()
        .rename(columns={"count": "fit_window_amr_publications"})
    )
    pair_publication_rates["avg_amr_publications_per_year"] = (
        pair_publication_rates["fit_window_amr_publications"] / fit_year_count
    )

    eligible_pairs = pair_publication_rates.loc[
        pair_publication_rates["avg_amr_publications_per_year"] >= float(min_avg_amr_publications_per_year),
        ["pathogen", "antibiotic"],
    ]
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
            f"with >= {min_avg_amr_publications_per_year:g} publications/year on average "
            f"from {fit_start_year} to {fit_end_year}."
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


def rank_future_high_interest_pairs(
    si_df: pd.DataFrame,
    *,
    fit_start_year: int = 1970,
    fit_end_year: int | None = None,
    num_future_years: int = 20,
    min_current_predicted_count: float = 10.0,
    min_future_predicted_count: float = 10.0,
    min_si_change: float = 0.1,
    min_post_fit_observed_years: int = 1,
    max_post_fit_si_abs_pct_error_mean: float = 40.0,
    fallback_top_n: int = 0,
    rank_output_path: str | Path = "yearly_si_ranked_future.parquet",
    interesting_output_path: str | Path | None = "yearly_si_ranked_future_interesting.parquet",
    loess_frac: float = 0.6,
    loess_degree: int = 1,
    tail_damping_years: Any = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank future SI shifts, then apply reliability-aware interesting-case selection."""
    rank_df = rank_yearly_pairs_by_future_si_shift(
        si_df,
        fit_start_year=fit_start_year,
        fit_end_year=fit_end_year,
        num_future_years=num_future_years,
        save_path=str(rank_output_path),
        loess_frac=loess_frac,
        loess_degree=loess_degree,
        tail_damping_years=tail_damping_years,
        verbose=verbose,
    )
    interesting_df = select_interesting_yearly_rank_cases(
        rank_df,
        min_current_predicted_count=min_current_predicted_count,
        min_future_predicted_count=min_future_predicted_count,
        min_si_change=min_si_change,
        min_post_fit_observed_years=min_post_fit_observed_years,
        max_post_fit_si_abs_pct_error_mean=max_post_fit_si_abs_pct_error_mean,
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
    include_decisions: tuple[str, ...] | list[str] | set[str] = ("yes", "uncertain"),
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

    for row in interesting_df.head(max_plots).itertuples(index=False):
        try:
            result = plot_yearly_category_driver_counts(
                category_counts_df=category_counts_df,
                pathogen=row.pathogen,
                antibiotic=row.antibiotic,
                si_df=si_df,
                start_year=start_year,
                end_year=end_year,
                fit_end_year=fit_end_year,
            )
        except ValueError:
            continue
        pathogen_slug = str(row.pathogen).replace("/", "_").replace(" ", "_")
        antibiotic_slug = str(row.antibiotic).replace("/", "_").replace(" ", "_")
        out_path = output_dir / f"{pathogen_slug}__{antibiotic_slug}__drivers.{figure_format}"
        result["figure"].savefig(out_path, bbox_inches="tight")
        if display_plots and (max_display_plots is None or displayed_plots < max_display_plots):
            _display_figure(result["figure"])
            displayed_plots += 1
        plt.close(result["figure"])
        saved_plots += 1

    if verbose:
        print(f"Saved {saved_plots} category driver plots to {output_dir}")
        if display_plots:
            print(f"Displayed {displayed_plots} category driver plots inline.")
    return saved_plots


def save_interesting_si_plots(
    *,
    si_df: pd.DataFrame,
    interesting_df: pd.DataFrame,
    output_dir: str | Path,
    fit_start_year: int,
    fit_end_year: int,
    num_future_years: int,
    pathogen_background_df: pd.DataFrame | None = None,
    antibiotic_background_df: pd.DataFrame | None = None,
    max_plots: int = 25,
    loess_frac: float = 0.6,
    loess_degree: int = 1,
    tail_damping_years: Any = None,
    fill_missing_amr_counts_with_zero: bool = True,
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
    displayed_plots = 0
    for row in interesting_df.head(max_plots).itertuples(index=False):
        result = plot_yearly_si_pair(
            si_df=si_df,
            pathogen=row.pathogen,
            antibiotic=row.antibiotic,
            fit_start_year=fit_start_year,
            fit_end_year=fit_end_year,
            num_additional_years=num_future_years,
            loess_frac=loess_frac,
            loess_degree=loess_degree,
            tail_damping_years=tail_damping_years,
            fill_missing_amr_counts_with_zero=fill_missing_amr_counts_with_zero,
            pathogen_background_df=pathogen_background_df,
            antibiotic_background_df=antibiotic_background_df,
        )
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
        if display_plots:
            print(f"Displayed {displayed_plots} plots inline.")
    return saved_plots


def select_existing_columns(df: pd.DataFrame, columns: Sequence[str], *, head: int | None = None) -> pd.DataFrame:
    """Return the requested columns that exist, optionally limited to the first rows."""
    out = df[[col for col in columns if col in df.columns]]
    return out.head(head) if head is not None else out


__all__ = [
    "build_antibiotic_group_mapping",
    "build_si_analysis_paths",
    "build_si_driver_analysis",
    "build_yearly_si_dataset",
    "DEFAULT_DRIVER_SUMMARY_COLUMNS",
    "DEFAULT_INTERESTING_SUMMARY_COLUMNS",
    "filter_model_inputs_by_average_amr_publications",
    "load_yearly_si_inputs",
    "print_si_dataset_summary",
    "prepare_antibiotic_analysis_level",
    "rank_future_high_interest_pairs",
    "save_category_driver_plots",
    "save_interesting_si_plots",
    "select_existing_columns",
    "slugify_analysis_level",
    "summarize_pair_publication_rates",
]
