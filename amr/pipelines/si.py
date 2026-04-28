"""Pipelines for yearly scientific-interest modeling and ranking."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd

from amr.io import atomic_write_parquet
from amr.si.core import (
    fit_global_pooled_si_model,
    rank_yearly_pairs_by_future_si_shift,
    select_interesting_yearly_rank_cases,
)


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


__all__ = [
    "build_yearly_si_dataset",
    "rank_future_high_interest_pairs",
]
