"""Human-review sampling and probability-threshold calibration for System 1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REVIEW_SCORE_BINS = [0.0, 0.50, 0.65, 0.75, 0.85, 0.95, 1.0000001]
REVIEW_SAMPLE_ALLOCATION = [20, 30, 40, 60, 70, 80]


def build_resistance_review_sample(
    decisions_df: pd.DataFrame,
    *,
    run_id: str,
    sample_size: int = 300,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Select a weighted score-stratified review sample from a Jev run.

    The current decision scores are used to stratify only; labels remain human
    supplied. The sample is split before review into calibration and holdout sets.
    """
    required = {"pmid", "pathogen", "antibiotic", "year", "system1_run_id", "system1_p_yes", "system1_status"}
    missing = required - set(decisions_df.columns)
    if missing:
        raise KeyError(f"decisions_df missing required columns: {sorted(missing)}")
    if sample_size != sum(REVIEW_SAMPLE_ALLOCATION):
        raise ValueError(f"sample_size must be {sum(REVIEW_SAMPLE_ALLOCATION)} for the configured strata.")

    work = decisions_df.loc[
        decisions_df["system1_run_id"].astype(str).eq(str(run_id))
        & decisions_df["system1_status"].astype(str).eq("ok")
    ].copy()
    work["system1_p_yes"] = pd.to_numeric(work["system1_p_yes"], errors="coerce")
    work = work.dropna(subset=["system1_p_yes"]).drop_duplicates(
        subset=["pmid", "pathogen", "antibiotic", "year"], keep="last"
    )
    if work.empty:
        raise ValueError(f"No successful Jev decisions found for run {run_id!r}.")
    work["score_stratum"] = pd.cut(
        work["system1_p_yes"], bins=REVIEW_SCORE_BINS, right=False, include_lowest=True
    )
    rng = np.random.default_rng(random_seed)
    pieces: list[pd.DataFrame] = []
    for stratum, requested in zip(work["score_stratum"].cat.categories, REVIEW_SAMPLE_ALLOCATION):
        candidates = work.loc[work["score_stratum"].eq(stratum)].copy()
        if candidates.empty:
            continue
        take = min(int(requested), len(candidates))
        selected_idx = rng.choice(candidates.index.to_numpy(), size=take, replace=False)
        selected = candidates.loc[selected_idx].copy()
        selected["stratum_population"] = len(candidates)
        selected["stratum_sample"] = take
        selected["sampling_weight"] = len(candidates) / take
        pieces.append(selected)
    sample = pd.concat(pieces, ignore_index=True)
    if len(sample) < sample_size:
        raise ValueError(f"Only {len(sample)} review candidates were available; {sample_size} were requested.")
    sample = sample.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    sample["review_split"] = "calibration"
    for _, group in sample.groupby("score_stratum", observed=True):
        idx = group.sample(frac=1, random_state=random_seed).index
        sample.loc[idx[: max(1, round(len(idx) / 3))], "review_split"] = "holdout"
    split_sizes = sample.groupby(["score_stratum", "review_split"], observed=True)["pmid"].transform("size")
    sample["sampling_weight"] = sample["stratum_population"] / split_sizes
    sample["resistance_demonstrated"] = ""
    sample["reviewer_notes"] = ""
    return sample


def calibrate_resistance_threshold(
    reviewed_df: pd.DataFrame,
    *,
    target_precision: float = 0.90,
    min_holdout_accepted: int = 20,
) -> dict[str, Any]:
    """Choose maximum-coverage threshold on calibration rows and test on holdout."""
    required = {"system1_p_yes", "review_split", "sampling_weight", "resistance_demonstrated"}
    missing = required - set(reviewed_df.columns)
    if missing:
        raise KeyError(f"reviewed_df missing required columns: {sorted(missing)}")
    work = reviewed_df.copy()
    labels = work["resistance_demonstrated"].astype(str).str.strip().str.lower().map(
        {"yes": 1.0, "true": 1.0, "1": 1.0, "no": 0.0, "false": 0.0, "0": 0.0}
    )
    work["label"] = labels
    work["system1_p_yes"] = pd.to_numeric(work["system1_p_yes"], errors="coerce")
    work["sampling_weight"] = pd.to_numeric(work["sampling_weight"], errors="coerce")
    if work["label"].isna().any() or work[["system1_p_yes", "sampling_weight"]].isna().any().any():
        raise ValueError("Every row must have a yes/no resistance label, score, and sampling weight.")
    calibration = work.loc[work["review_split"].eq("calibration")]
    holdout = work.loc[work["review_split"].eq("holdout")]
    if calibration.empty or holdout.empty:
        raise ValueError("Both calibration and holdout review rows are required.")

    options = []
    for threshold in sorted(calibration["system1_p_yes"].unique()):
        accepted = calibration.loc[calibration["system1_p_yes"].ge(threshold)]
        if accepted.empty:
            continue
        precision = float(np.average(accepted["label"], weights=accepted["sampling_weight"]))
        if precision >= target_precision:
            options.append((len(accepted), float(threshold), precision))
    if not options:
        raise ValueError(f"No threshold achieved {target_precision:.0%} weighted precision on calibration rows.")
    _, threshold, calibration_precision = max(options, key=lambda row: (row[0], -row[1]))
    held = holdout.loc[holdout["system1_p_yes"].ge(threshold)]
    if len(held) < int(min_holdout_accepted):
        raise ValueError(
            f"Only {len(held)} held-out records meet threshold {threshold:.4f}; "
            f"at least {min_holdout_accepted} are required."
        )
    holdout_precision = float(np.average(held["label"], weights=held["sampling_weight"]))
    if holdout_precision < target_precision:
        raise ValueError(
            f"Held-out weighted precision {holdout_precision:.1%} is below target {target_precision:.1%}."
        )
    return {
        "threshold": threshold,
        "target_precision": target_precision,
        "calibration_precision": calibration_precision,
        "holdout_precision": holdout_precision,
        "calibration_accepted_n": int(calibration["system1_p_yes"].ge(threshold).sum()),
        "holdout_accepted_n": int(len(held)),
        "calibration_total_n": int(len(calibration)),
        "holdout_total_n": int(len(holdout)),
    }


def write_screening_provenance(path: str | Path, provenance: dict[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return out


def validate_resistance_screen_provenance(
    path: str | Path,
    *,
    run_id: str,
    question_version: str,
    probability_threshold: float,
    target_precision: float = 0.90,
    allow_provisional: bool = False,
) -> dict[str, Any]:
    """Fail closed unless a screened dataset records a validated resistance run."""
    provenance_path = Path(path)
    if not provenance_path.exists():
        raise FileNotFoundError(
            f"Missing resistance-screen provenance: {provenance_path}. "
            "Run the resistance-only screening and screened-dataset workflow first."
        )
    provenance = json.loads(provenance_path.read_text())
    status = str(provenance.get("threshold_calibration_status", ""))
    provisional = status == "provisional_uncalibrated"
    if provisional and not allow_provisional:
        raise ValueError(
            "Screened dataset is explicitly provisional and uncalibrated; pass allow_provisional=True "
            "only for exploratory outputs."
        )
    if status != "validated" and not provisional:
        raise ValueError(f"Unsupported resistance-screen calibration status: {status!r}.")
    expected = {
        "screening_source": "system1",
        "run_id": str(run_id),
        "question_version": str(question_version),
        "probability_threshold": float(probability_threshold),
        "threshold_calibration_status": status,
    }
    mismatches = {
        key: (provenance.get(key), value)
        for key, value in expected.items()
        if provenance.get(key) != value
    }
    precision_value = provenance.get("holdout_precision")
    precision = float(precision_value) if precision_value is not None else -1.0
    if not provisional and precision < float(target_precision):
        mismatches["holdout_precision"] = (precision, f">={float(target_precision):.0%}")
    if mismatches:
        details = "; ".join(f"{key}={actual!r} (expected {expected_value!r})" for key, (actual, expected_value) in mismatches.items())
        raise ValueError(f"Screened dataset does not match a validated resistance-only run: {details}")
    return provenance


__all__ = [
    "REVIEW_SCORE_BINS",
    "REVIEW_SAMPLE_ALLOCATION",
    "build_resistance_review_sample",
    "calibrate_resistance_threshold",
    "write_screening_provenance",
    "validate_resistance_screen_provenance",
]
