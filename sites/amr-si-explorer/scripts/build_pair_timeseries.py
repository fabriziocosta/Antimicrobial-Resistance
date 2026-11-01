#!/usr/bin/env python3
"""Precompute plot_single_pair_si series for every pair in the web scatter plot.

Run from any working directory:

    python sites/amr-si-explorer/scripts/build_pair_timeseries.py

This writes only static, browser-ready data. The Site does not fit models at
runtime; it renders the saved observed, predicted, and count tables.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from typing import Any

import pandas as pd


def find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "amr").is_dir() and (parent / "outputs").is_dir():
            return parent
    raise RuntimeError("Could not locate the AMR repository root from this script.")


ROOT = find_project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from amr.pipeline_config import cfg, cfg_path, load_pipeline_config  # noqa: E402
from amr.pipelines.si import build_si_analysis_paths  # noqa: E402
from amr.si.core import fit_predict_yearly_si_loess  # noqa: E402


def json_value(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def records_for(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    return [
        {column: json_value(row[column]) for column in columns}
        for _, row in frame[columns].iterrows()
    ]


def main() -> None:
    config = load_pipeline_config(project_root=ROOT)
    analysis_level = cfg(config, "si.antibiotic_analysis_level", "antibiotic")
    si_path = build_si_analysis_paths(ROOT, analysis_level)["yearly_si"]
    plotted_pairs_path = ROOT / "outputs/figures/future_si_vs_current_si_quadrant.csv"
    out_path = ROOT / "sites/amr-si-explorer/dist/pair_timeseries.json"

    if not si_path.exists():
        raise FileNotFoundError(f"Missing yearly SI data: {si_path}")
    if not plotted_pairs_path.exists():
        raise FileNotFoundError(f"Missing plotted pair table: {plotted_pairs_path}")

    si_df = pd.read_parquet(si_path)
    plotted_pairs = pd.read_csv(plotted_pairs_path)[["pathogen", "antibiotic"]].drop_duplicates()
    fit_start_year = int(cfg(config, "si.fit_start_year"))
    fit_end_year = int(cfg(config, "si.curve_fit_end_year"))
    forecast_years = int(cfg(config, "si.num_future_years"))
    min_count = float(cfg(config, "si.min_amr_pair_publication_count"))
    min_valid_years = int(cfg(config, "si.min_num_years_with_valid_counts"))
    loess_frac = float(cfg(config, "si.loess_frac"))
    loess_degree = int(cfg(config, "si.loess_degree"))
    tail_damping_years = cfg(config, "si.tail_damping_years")

    pair_rows: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    observed_columns = [
        "year", "SI", "count", "count_for_model", "bg_pathogen_n", "bg_antibiotic_n"
    ]
    prediction_columns = [
        "year", "predicted_SI", "lower_1std", "upper_1std",
        "predicted_count_loess", "predicted_count_loess_lower_1std",
        "predicted_count_loess_upper_1std", "expected_count_baseline",
    ]

    for pair in plotted_pairs.itertuples(index=False):
        pathogen, antibiotic = str(pair.pathogen), str(pair.antibiotic)
        try:
            pair_si = si_df.loc[
                si_df["pathogen"].eq(pathogen) & si_df["antibiotic"].eq(antibiotic)
            ]
            available_fit_years = int(
                pair_si.loc[
                    pair_si["year"].between(fit_start_year, fit_end_year)
                    & pair_si["count_for_model"].notna(),
                    "year",
                ].nunique()
            )
            # plot_single_pair_si's recovery path reduces the configured
            # minimum to the actual number of available years for a recovered
            # pair. Apply that same rule here so sparse pairs in the scatter
            # plot still have a precomputed chart when at least 3 points exist.
            pair_min_valid_years = min(min_valid_years, available_fit_years)
            if pair_min_valid_years < 3:
                raise ValueError(f"Need at least 3 valid training years; found {available_fit_years}.")
            result = fit_predict_yearly_si_loess(
                si_df=si_df,
                pathogen=pathogen,
                antibiotic=antibiotic,
                fit_start_year=fit_start_year,
                fit_end_year=fit_end_year,
                num_additional_years=forecast_years,
                loess_frac=loess_frac,
                loess_degree=loess_degree,
                tail_damping_years=tail_damping_years,
                min_amr_pair_publication_count=min_count,
                min_num_years_with_valid_counts=pair_min_valid_years,
            )
            pair_df = result["pair_df"].sort_values("year").copy()
            pred_df = result["pred_df"].sort_values("year").copy()

            # The Matplotlib source plot shows observed SI for each calendar
            # year through the latest observed pair year. Nulls preserve gaps.
            observed_years = pd.DataFrame({
                "year": range(int(pair_df["year"].min()), int(pair_df["year"].max()) + 1)
            }).merge(pair_df[observed_columns], on="year", how="left")

            # Reproduce plot_yearly_si_pair's annual count table. Background
            # counts are already embedded in the yearly SI table; fill years
            # without a pair row from the same pathogen/antibiotic's maxima.
            count_years = pd.DataFrame({
                "year": range(int(pair_df["year"].min()), int(pair_df["year"].max()) + 1)
            }).merge(
                pair_df[["year", "count", "count_for_model", "bg_pathogen_n", "bg_antibiotic_n"]],
                on="year",
                how="left",
                indicator="pair_year_present",
            )
            count_years["pair_year_present"] = count_years["pair_year_present"].eq("both")
            pathogen_background = (
                si_df.loc[si_df["pathogen"].eq(pathogen), ["year", "bg_pathogen_n"]]
                .groupby("year", as_index=False)["bg_pathogen_n"].max()
                .rename(columns={"bg_pathogen_n": "pathogen_fill"})
            )
            antibiotic_background = (
                si_df.loc[si_df["antibiotic"].eq(antibiotic), ["year", "bg_antibiotic_n"]]
                .groupby("year", as_index=False)["bg_antibiotic_n"].max()
                .rename(columns={"bg_antibiotic_n": "antibiotic_fill"})
            )
            count_years = count_years.merge(pathogen_background, on="year", how="left").merge(
                antibiotic_background, on="year", how="left"
            )
            count_years["bg_pathogen_n"] = count_years["bg_pathogen_n"].fillna(count_years["pathogen_fill"])
            count_years["bg_antibiotic_n"] = count_years["bg_antibiotic_n"].fillna(count_years["antibiotic_fill"])
            count_years = count_years.drop(columns=["pathogen_fill", "antibiotic_fill"])
            for source, target in [
                ("count_for_model", "amr_count_plot"),
                ("bg_pathogen_n", "pathogen_count_plot"),
                ("bg_antibiotic_n", "antibiotic_count_plot"),
            ]:
                count_years[target] = pd.to_numeric(count_years[source], errors="coerce").clip(lower=1.0)

            pair_rows.append({
                "pathogen": pathogen,
                "antibiotic": antibiotic,
                "fit_start_year": fit_start_year,
                "fit_end_year": fit_end_year,
                "latest_available_year": result["latest_available_year"],
                "prediction_end_year": int(pred_df["year"].max()),
                "forecast_years": forecast_years,
                "min_amr_pair_publication_count": min_count,
                "min_num_years_with_valid_counts": min_valid_years,
                "min_valid_years_used": pair_min_valid_years,
                "loess_frac": loess_frac,
                "loess_degree": loess_degree,
                "tail_damping_years": tail_damping_years,
                "total_amr_pair_publications": float(pd.to_numeric(pair_df["count"], errors="coerce").fillna(0).sum()),
                "observed": records_for(observed_years, observed_columns),
                "predicted": records_for(pred_df, prediction_columns),
                "counts": records_for(count_years, [
                    "year", "count", "count_for_model", "pair_year_present",
                    "bg_pathogen_n", "bg_antibiotic_n", "amr_count_plot",
                    "pathogen_count_plot", "antibiotic_count_plot",
                ]),
            })
        except Exception as exc:  # retain failures as explicit, inspectable records
            failed.append({"pathogen": pathogen, "antibiotic": antibiotic, "error": str(exc)})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": str(si_path.relative_to(ROOT)),
        "method": "amr.si.core.fit_predict_yearly_si_loess used by plot_single_pair_si",
        "fit_settings": {
            "fit_start_year": fit_start_year,
            "fit_end_year": fit_end_year,
            "forecast_years": forecast_years,
            "min_amr_pair_publication_count": min_count,
            "min_num_years_with_valid_counts": min_valid_years,
            "loess_frac": loess_frac,
            "loess_degree": loess_degree,
            "tail_damping_years": tail_damping_years,
        },
        "count": len(pair_rows),
        "records": pair_rows,
        "failed_count": len(failed),
        "failed": failed,
    }
    out_path.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    print(f"Wrote {len(pair_rows)} precomputed pair plots to {out_path.relative_to(ROOT)}")
    if failed:
        print(f"Pairs unavailable for a single-pair plot: {len(failed)}")


if __name__ == "__main__":
    main()
