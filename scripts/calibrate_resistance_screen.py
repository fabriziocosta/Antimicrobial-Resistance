#!/usr/bin/env python3
"""Calibrate the resistance-only Jev cutoff from human-reviewed records."""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from amr.pipeline_config import cfg, load_pipeline_config  # noqa: E402
from amr.screening.calibration import calibrate_resistance_threshold  # noqa: E402


def main() -> None:
    config = load_pipeline_config(project_root=ROOT)
    review_path = ROOT / "outputs" / "review" / "resistance_screen_review_sample.csv"
    config_path = ROOT / "outputs" / "pipeline_config.yaml"
    reviewed = pd.read_csv(review_path, keep_default_na=False)
    expected_run = str(cfg(config, "system1.run_id"))
    expected_question = str(cfg(config, "system1.question_version"))
    if not reviewed.get("screen_run_id", pd.Series(dtype=str)).astype(str).eq(expected_run).all():
        raise ValueError("Review sample run ID does not match the active System 1 run.")
    if not reviewed.get("screen_question_version", pd.Series(dtype=str)).astype(str).eq(expected_question).all():
        raise ValueError("Review sample question version does not match the active System 1 question.")
    report = calibrate_resistance_threshold(
        reviewed,
        target_precision=float(cfg(config, "system1.target_precision", 0.90)),
    )
    config_text = config_path.read_text()
    threshold_pattern = r"(?m)^  system1_probability_threshold: .*?$"
    status_pattern = r"(?m)^  threshold_calibration_status: .*?$"
    if not re.search(threshold_pattern, config_text) or not re.search(status_pattern, config_text):
        raise RuntimeError("Could not find the System 1 calibration settings in pipeline_config.yaml.")
    config_text = re.sub(threshold_pattern, f"  system1_probability_threshold: {report['threshold']:.6f}", config_text)
    config_text = re.sub(status_pattern, "  threshold_calibration_status: validated", config_text)
    config_path.write_text(config_text)
    report_path = review_path.with_name("resistance_screen_calibration.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Validated threshold written to {config_path}; report saved to {report_path}.")


if __name__ == "__main__":
    main()
