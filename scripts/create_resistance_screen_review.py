#!/usr/bin/env python3
"""Run the resistance-only Jev question on a review sample and export labels."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from amr.pipeline_config import cfg, cfg_path, load_pipeline_config  # noqa: E402
from amr.screening import prepare_yearly_candidate_pmids  # noqa: E402
from amr.screening.calibration import build_resistance_review_sample  # noqa: E402


def main() -> None:
    config = load_pipeline_config(project_root=ROOT)
    decisions_path = cfg_path(config, "paths.system1_decisions", ROOT)
    abstracts_path = cfg_path(config, "paths.abstract_cache", ROOT)
    candidates_path = cfg_path(config, "paths.yearly_candidates", ROOT)
    sample_path = ROOT / "outputs" / "review" / "resistance_screen_review_sample.csv"
    run_id = str(cfg(config, "system1.run_id"))
    question_version = str(cfg(config, "system1.question_version"))
    model = str(cfg(config, "system1.model"))
    decisions = pd.read_parquet(decisions_path)
    candidate_source = pd.read_parquet(candidates_path)
    candidate_keys = prepare_yearly_candidate_pmids(candidate_source)
    candidate_keys = candidate_keys.loc[
        candidate_keys["year"].between(
            int(cfg(config, "collection.slice_start_year")),
            int(cfg(config, "collection.slice_end_year")),
        ),
        ["pmid", "pathogen", "antibiotic", "year"],
    ].drop_duplicates()
    current_run = decisions.loc[
        decisions["system1_run_id"].astype(str).eq(run_id)
        & decisions["system1_question_version"].astype(str).eq(question_version)
    ].merge(candidate_keys, on=["pmid", "pathogen", "antibiotic", "year"], how="inner")
    key_cols = ["pmid", "pathogen", "antibiotic", "year"]
    candidate_set = set(map(tuple, candidate_keys[key_cols].astype(str).to_numpy()))
    run_set = set(map(tuple, current_run[key_cols].astype(str).to_numpy()))
    if candidate_set != run_set:
        raise RuntimeError(
            f"Current resistance run coverage mismatch: {len(candidate_set - run_set):,} missing and "
            f"{len(run_set - candidate_set):,} unexpected candidate keys."
        )
    if current_run["system1_status"].astype(str).eq("error").any():
        raise RuntimeError("Current resistance run contains failed records; complete the run before calibration.")
    sample = build_resistance_review_sample(current_run, run_id=run_id)
    abstracts = pd.read_parquet(abstracts_path)
    sample = sample.merge(abstracts[["pmid", "title", "abstract"]], on="pmid", how="left")
    sample["screen_run_id"] = run_id
    sample["screen_question_version"] = question_version
    sample["screen_model"] = model
    sample["resistance_demonstrated"] = ""
    sample["reviewer_notes"] = ""
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(sample_path, index=False)
    print(f"Saved {len(sample):,} resistance-only scored review records from {len(current_run):,} current run records and {len(candidate_keys):,} current candidate keys to {sample_path}")
    print("Label resistance_demonstrated as yes/no using the title and abstract and the exact target pair.")


if __name__ == "__main__":
    main()
