#!/usr/bin/env python3
"""Build the static pair-by-year PMID table used by the AMR SI Explorer."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import yaml


def find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "amr").is_dir() and (parent / "outputs").is_dir():
            return parent
    raise RuntimeError("Could not locate the AMR repository root from this script.")


def as_pmids(value: Any) -> list[str]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        values = value.tolist()
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        try:
            if pd.isna(value):
                return []
        except (TypeError, ValueError):
            pass
        values = list(value)
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        if raw is None or pd.isna(raw):
            continue
        pmid = str(raw).strip()
        if pmid.endswith(".0"):
            pmid = pmid[:-2]
        if pmid and pmid not in seen:
            seen.add(pmid)
            result.append(pmid)
    return result


def main() -> None:
    root = find_project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    dist = root / "sites/amr-si-explorer/dist"
    pair_path = dist / "pairs.json"
    annual_path = root / "outputs/parquets/yearly_amr_screened_pairs.parquet"
    decisions_path = root / "outputs/parquets/yearly_amr_system1_decisions.parquet"
    metadata_path = root / "outputs/parquets/pubmed_pmid_metadata.parquet"
    out_path = dist / "pair_pmids_by_year.json"

    pair_records = json.loads(pair_path.read_text(encoding="utf-8"))["records"]
    pair_keys = {(str(row["pathogen"]), str(row["antibiotic"])) for row in pair_records}
    annual = pd.read_parquet(
        annual_path,
        columns=["pathogen", "antibiotic", "year", "pmids"],
    )
    annual = annual.loc[
        [(str(pathogen), str(antibiotic)) in pair_keys for pathogen, antibiotic in zip(annual.pathogen, annual.antibiotic)]
    ].copy()

    config = yaml.safe_load((root / "outputs/pipeline_config.yaml").read_text(encoding="utf-8"))
    system1_config = config.get("system1", {})
    calibration_status = system1_config.get("threshold_calibration_status")
    if calibration_status not in {"validated", "provisional_uncalibrated"}:
        raise RuntimeError("Explorer PMID data require a validated threshold or explicit provisional_uncalibrated mode.")
    from amr.screening.calibration import validate_resistance_screen_provenance

    provenance = validate_resistance_screen_provenance(
        annual_path.with_suffix(".provenance.json"),
        run_id=str(system1_config.get("run_id", "")),
        question_version=str(system1_config.get("question_version", "")),
        probability_threshold=float(config.get("screening", {}).get("system1_probability_threshold", 0.5)),
        target_precision=float(system1_config.get("target_precision", 0.90)),
        allow_provisional=calibration_status == "provisional_uncalibrated",
    )
    decisions = pd.read_parquet(
        decisions_path,
        columns=["pmid", "pathogen", "antibiotic", "system1_run_id", "system1_p_yes", "system1_status"],
    )
    if system1_config.get("run_filter", True):
        decisions = decisions.loc[decisions["system1_run_id"] == system1_config.get("run_id")]
    decisions = decisions.loc[decisions["system1_status"] == "ok"]
    decisions["pmid"] = decisions["pmid"].map(lambda value: str(value).removesuffix(".0"))
    confidence_by_key = {
        (str(row.pathogen), str(row.antibiotic), str(row.pmid)): float(row.system1_p_yes)
        for row in decisions.itertuples(index=False)
        if pd.notna(row.system1_p_yes)
    }
    metadata = pd.read_parquet(metadata_path, columns=["pmid", "pub_date"])
    metadata["pmid"] = metadata["pmid"].map(lambda value: str(value).removesuffix(".0"))
    metadata_by_pmid = {
        str(row.pmid): row.pub_date.strftime("%Y-%m-%d") if pd.notna(row.pub_date) else None
        for row in metadata.itertuples(index=False)
    }

    by_pair: dict[tuple[str, str], dict[str, list[list[Any]]]] = {pair: {} for pair in pair_keys}
    for row in annual.itertuples(index=False):
        pair = (str(row.pathogen), str(row.antibiotic))
        papers = []
        for pmid in as_pmids(row.pmids):
            pub_date = metadata_by_pmid.get(pmid)
            confidence = confidence_by_key.get((pair[0], pair[1], pmid))
            # Compact tuples keep the browser payload small: PMID, Jev confidence, publication date.
            papers.append([pmid, confidence, pub_date])
        by_pair[pair][str(int(row.year))] = papers

    # Protect the existing 2025 list from accidental source drift.
    mismatches = []
    for row in pair_records:
        pair = (str(row["pathogen"]), str(row["antibiotic"]))
        if [paper[0] for paper in by_pair[pair].get("2025", [])] != [str(value) for value in row.get("pmids_2025", [])]:
            mismatches.append(pair)
    if mismatches:
        raise ValueError(f"Yearly PMID source differs from the existing 2025 lists for {len(mismatches)} pairs.")

    years = [int(year) for mapping in by_pair.values() for year in mapping]
    payload = {
        "source": "outputs/parquets/yearly_amr_screened_pairs.parquet + outputs/parquets/yearly_amr_system1_decisions.parquet + outputs/parquets/pubmed_pmid_metadata.parquet",
        "screening_provenance": {
            "run_id": provenance["run_id"],
            "question_version": provenance["question_version"],
            "probability_threshold": provenance["probability_threshold"],
            "target_precision": provenance["target_precision"],
            "holdout_precision": provenance["holdout_precision"],
            "threshold_calibration_status": provenance["threshold_calibration_status"],
            "provisional_note": provenance.get("provisional_note"),
        },
        "year_min": min(years) if years else None,
        "year_max": max(years) if years else None,
        "count": len(pair_records),
        "pair_year_count": sum(len(mapping) for mapping in by_pair.values()),
        "records": [
            {
                "pathogen": str(row["pathogen"]),
                "antibiotic": str(row["antibiotic"]),
                "pmids_by_year": by_pair[(str(row["pathogen"]), str(row["antibiotic"]))],
            }
            for row in pair_records
        ],
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(
        f"Wrote {out_path.relative_to(root)}: {payload['count']:,} pairs, "
        f"{payload['pair_year_count']:,} pair-years, years {payload['year_min']}–{payload['year_max']}"
    )


if __name__ == "__main__":
    main()
