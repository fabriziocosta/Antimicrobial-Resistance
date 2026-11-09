#!/usr/bin/env python3
"""Build the explorer's plotted-pair bundle from resistance-only SI outputs."""

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
from amr.screening.calibration import validate_resistance_screen_provenance  # noqa: E402


def json_value(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def sorted_pmids(value: Any) -> list[str]:
    if value is None:
        return []
    values = value.tolist() if hasattr(value, "tolist") else value
    if not isinstance(values, (list, tuple, set)):
        try:
            if pd.isna(values):
                return []
        except (TypeError, ValueError):
            pass
        values = [values]
    return sorted(
        {str(pmid).strip().removesuffix(".0") for pmid in values if str(pmid).strip()},
        key=lambda pmid: (not pmid.isdigit(), int(pmid) if pmid.isdigit() else pmid),
    )


def main() -> None:
    config = load_pipeline_config(project_root=ROOT)
    calibration_status = str(cfg(config, "system1.threshold_calibration_status", "pending"))
    if calibration_status not in {"validated", "provisional_uncalibrated"}:
        raise RuntimeError("Explorer pair data require a validated threshold or explicit provisional_uncalibrated mode.")
    screened_path = cfg_path(config, "paths.yearly_screened", ROOT)
    provenance = validate_resistance_screen_provenance(
        screened_path.with_suffix(".provenance.json"),
        run_id=str(cfg(config, "system1.run_id", "")),
        question_version=str(cfg(config, "system1.question_version", "")),
        probability_threshold=float(cfg(config, "screening.system1_probability_threshold", 0.5)),
        target_precision=float(cfg(config, "system1.target_precision", 0.90)),
        allow_provisional=calibration_status == "provisional_uncalibrated",
    )
    plot_path = ROOT / "outputs/figures/future_si_vs_current_si_quadrant.csv"
    if not plot_path.exists():
        raise FileNotFoundError(f"Missing current SI ranking: {plot_path}. Run notebooks 05–06 first.")
    if not screened_path.exists():
        raise FileNotFoundError(f"Missing resistance-only screened-pair table: {screened_path}.")

    plot = pd.read_csv(plot_path)
    required = {
        "pathogen",
        "antibiotic",
        "future_predicted_SI",
        "future_minus_latest_smoothed_SI",
        "total_publications",
    }
    missing = required - set(plot.columns)
    if missing:
        raise KeyError(f"The SI ranking is missing required fields: {sorted(missing)}")
    plot["pathogen"] = plot["pathogen"].astype(str).str.strip()
    plot["antibiotic"] = plot["antibiotic"].astype(str).str.strip()
    if plot.duplicated(["pathogen", "antibiotic"]).any():
        raise ValueError("The current SI ranking contains duplicate pathogen–antibiotic pairs.")

    annual = pd.read_parquet(screened_path, columns=["pathogen", "antibiotic", "year", "pmids"])
    annual["pathogen"] = annual["pathogen"].astype(str).str.strip()
    annual["antibiotic"] = annual["antibiotic"].astype(str).str.strip()
    annual["year"] = pd.to_numeric(annual["year"], errors="coerce")
    annual["count"] = annual["pmids"].map(lambda values: len(sorted_pmids(values)))
    annual_totals = annual.groupby(["pathogen", "antibiotic"], as_index=False)["count"].sum()
    consistency = plot.merge(annual_totals, on=["pathogen", "antibiotic"], how="left")
    reported = pd.to_numeric(consistency.get("total_publications"), errors="coerce")
    actual = pd.to_numeric(consistency["count"], errors="coerce")
    if reported.isna().any() or actual.isna().any() or not reported.eq(actual).all():
        mismatched = consistency.loc[reported.ne(actual), ["pathogen", "antibiotic"]].head(10)
        raise ValueError(
            "The SI ranking totals do not match the configured resistance-positive screen for "
            f"{len(consistency.loc[reported.ne(actual)]):,} plotted pairs. Sample: "
            f"{mismatched.to_dict(orient='records')}"
        )
    annual_2025 = annual.loc[annual["year"].eq(2025)]
    pmids_by_pair = {
        (str(row.pathogen), str(row.antibiotic)): sorted_pmids(row.pmids)
        for row in annual_2025.itertuples(index=False)
    }

    records = []
    for row in plot.to_dict(orient="records"):
        record = {key: json_value(value) for key, value in row.items()}
        pair = (str(row["pathogen"]), str(row["antibiotic"]))
        record["pmids_2025"] = pmids_by_pair.get(pair, [])
        records.append(record)

    out_path = ROOT / "sites/amr-si-explorer/dist/pairs.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "count": len(records),
        "screening_provenance": {
            "run_id": provenance["run_id"],
            "question_version": provenance["question_version"],
            "probability_threshold": provenance["probability_threshold"],
            "target_precision": provenance["target_precision"],
            "holdout_precision": provenance["holdout_precision"],
            "threshold_calibration_status": provenance["threshold_calibration_status"],
            "provisional_note": provenance.get("provisional_note"),
        },
        "records": records,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"Wrote {out_path.relative_to(ROOT)}: {len(records):,} resistance-focused pairs.")
    print(f"Attached {sum(len(row['pmids_2025']) for row in records):,} 2025 pair–PMID links from the {calibration_status} screen.")


if __name__ == "__main__":
    main()
