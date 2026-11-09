import unittest

import pandas as pd

from amr.screening.core import DEFAULT_SYSTEM_PROMPT, UNIFIED_AMR_INTEREST_SYSTEM_PROMPT
from amr.screening.calibration import (
    build_resistance_review_sample,
    calibrate_resistance_threshold,
    validate_resistance_screen_provenance,
)
from amr.screening.system1 import SYSTEM1_AMR_SUPPORT_QUESTION, SYSTEM1_QUESTION_VERSION


class TestResistanceOnlyScreen(unittest.TestCase):
    def test_llm_prompts_exclude_non_resistance_evidence(self):
        for prompt in (DEFAULT_SYSTEM_PROMPT, UNIFIED_AMR_INTEREST_SYSTEM_PROMPT):
            lowered = prompt.lower()
            for phrase in (
                "exact target",
                "susceptibility testing",
                "reduced susceptibility",
                "mic",
                "resistance gene",
                "class",
                "another pathogen",
            ):
                self.assertIn(phrase, lowered)

    def test_jev_question_requires_exact_pair_resistance(self):
        self.assertEqual(SYSTEM1_QUESTION_VERSION, "explicit-pair-resistance-noul-v1")
        question = SYSTEM1_AMR_SUPPORT_QUESTION.lower()
        for phrase in (
            "explicitly reports",
            "resistant isolates",
            "exact pathogen-antibiotic pair",
            "susceptibility testing",
            "resistance gene",
            "class",
        ):
            self.assertIn(phrase, question)

    def test_review_sample_is_stratified_weighted_and_split(self):
        rows = []
        for i in range(2400):
            score = (i % 100) / 100
            rows.append({
                "pmid": str(i),
                "pathogen": "E. coli",
                "antibiotic": "Ciprofloxacin",
                "year": 2000 + i % 25,
                "system1_run_id": "old-run",
                "system1_p_yes": score,
                "system1_status": "ok",
            })
        sample = build_resistance_review_sample(pd.DataFrame(rows), run_id="old-run")
        self.assertEqual(len(sample), 300)
        self.assertEqual(set(sample["review_split"]), {"calibration", "holdout"})
        self.assertTrue(sample["sampling_weight"].gt(0).all())
        self.assertTrue(sample["resistance_demonstrated"].eq("").all())
        represented_population = sample.groupby(
            ["score_stratum", "review_split"], observed=True
        ).apply(lambda group: (group["sampling_weight"].sum(), group["stratum_population"].iloc[0]))
        self.assertTrue(all(abs(weight_sum - population) < 1e-9 for weight_sum, population in represented_population))

    def test_threshold_requires_heldout_precision_target(self):
        rows = []
        for split in ("calibration", "holdout"):
            for i in range(30):
                score = 0.95 if i < 20 else 0.60
                label = "yes" if i < 20 else "no"
                rows.append({
                    "system1_p_yes": score,
                    "review_split": split,
                    "sampling_weight": 1.0,
                    "resistance_demonstrated": label,
                })
        report = calibrate_resistance_threshold(pd.DataFrame(rows))
        self.assertEqual(report["threshold"], 0.95)
        self.assertEqual(report["holdout_precision"], 1.0)
        rows[-1]["resistance_demonstrated"] = "no"
        failed = pd.DataFrame(rows)
        failed.loc[(failed.review_split == "holdout") & (failed.system1_p_yes == 0.95), "resistance_demonstrated"] = "no"
        with self.assertRaisesRegex(ValueError, "below target"):
            calibrate_resistance_threshold(failed)

    def test_provenance_rejects_legacy_or_unvalidated_screen(self):
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "screen.provenance.json"
            path.write_text(json.dumps({
                "screening_source": "system1",
                "run_id": "new-run",
                "question_version": "new-question",
                "probability_threshold": 0.88,
                "threshold_calibration_status": "validated",
                "holdout_precision": 0.92,
            }))
            valid = validate_resistance_screen_provenance(
                path,
                run_id="new-run",
                question_version="new-question",
                probability_threshold=0.88,
            )
            self.assertEqual(valid["holdout_precision"], 0.92)
            with self.assertRaisesRegex(ValueError, "does not match"):
                validate_resistance_screen_provenance(
                    path,
                    run_id="old-run",
                    question_version="new-question",
                    probability_threshold=0.88,
                )

    def test_provisional_provenance_requires_explicit_exploratory_mode(self):
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "screen.provenance.json"
            path.write_text(json.dumps({
                "screening_source": "system1",
                "run_id": "new-run",
                "question_version": "new-question",
                "probability_threshold": 0.75,
                "threshold_calibration_status": "provisional_uncalibrated",
                "holdout_precision": None,
                "target_precision": 0.90,
            }))
            with self.assertRaisesRegex(ValueError, "explicitly provisional"):
                validate_resistance_screen_provenance(
                    path,
                    run_id="new-run",
                    question_version="new-question",
                    probability_threshold=0.75,
                )
            accepted = validate_resistance_screen_provenance(
                path,
                run_id="new-run",
                question_version="new-question",
                probability_threshold=0.75,
                allow_provisional=True,
            )
            self.assertIsNone(accepted["holdout_precision"])


if __name__ == "__main__":
    unittest.main()
