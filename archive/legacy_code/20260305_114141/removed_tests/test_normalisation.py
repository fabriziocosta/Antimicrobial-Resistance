import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "amr" / "normalisation.py"
SPEC = importlib.util.spec_from_file_location("amr_normalisation_under_test", MODULE_PATH)
normalisation = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = normalisation
SPEC.loader.exec_module(normalisation)


class NormalisationTests(unittest.TestCase):
    def test_missing_antibiotic_metadata_raises(self) -> None:
        amr_norm_df = pd.DataFrame(
            {
                "pathogen": ["p1", "p1"],
                "antibiotic": ["a1", "a2"],
                "start_date": ["2000-01-01", "2000-01-01"],
                "end_date": ["2000-12-31", "2000-12-31"],
                "bg_pathogen_n": [10, 10],
                "bg_antibiotic_n": [3, 4],
                "count": [1, 2],
            }
        )
        antibiotic_meta_df = pd.DataFrame(
            {
                "Antibiotic": ["a1", "a2"],
                "Class": ["c1", None],
            }
        )

        with self.assertRaisesRegex(ValueError, "missing from metadata"):
            normalisation.recompute_resid_log_bg_by_antibiotic_level(
                amr_norm_df,
                antibiotic_level="class",
                antibiotic_meta_df=antibiotic_meta_df,
            )

    def test_conflicting_antibiotic_metadata_raises(self) -> None:
        amr_norm_df = pd.DataFrame(
            {
                "pathogen": ["p1"],
                "antibiotic": ["a1"],
                "start_date": ["2000-01-01"],
                "end_date": ["2000-12-31"],
                "bg_pathogen_n": [10],
                "bg_antibiotic_n": [3],
                "count": [1],
            }
        )
        antibiotic_meta_df = pd.DataFrame(
            {
                "Antibiotic": ["a1", "a1"],
                "Class": ["c1", "c2"],
            }
        )

        with self.assertRaisesRegex(ValueError, "Conflicting metadata values"):
            normalisation.recompute_resid_log_bg_by_antibiotic_level(
                amr_norm_df,
                antibiotic_level="class",
                antibiotic_meta_df=antibiotic_meta_df,
            )

    def test_normalise_by_background_keeps_pairwise_metrics_aligned(self) -> None:
        amr_df = pd.DataFrame(
            {
                "pathogen": ["p1", "p1", "p2"],
                "antibiotic": ["a1", "a1", "a2"],
                "start_date": ["2000-01-01", "2001-01-01", "2000-01-01"],
                "end_date": ["2000-12-31", "2001-12-31", "2001-12-31"],
                "count": [1, 2, 3],
            }
        )
        pathogen_year_df = pd.DataFrame(
            {
                "pathogen": ["p1", "p1", "p2", "p2"],
                "year": [2000, 2001, 2000, 2001],
                "count": [10, 20, 30, 40],
            }
        )
        antibiotic_year_df = pd.DataFrame(
            {
                "antibiotic": ["a1", "a1", "a2", "a2"],
                "year": [2000, 2001, 2000, 2001],
                "count": [5, 6, 7, 8],
            }
        )

        out = normalisation.normalise_by_background(amr_df, pathogen_year_df, antibiotic_year_df)

        self.assertEqual(out.loc[0, "bg_pathogen_n"], 10)
        self.assertEqual(out.loc[1, "bg_pathogen_n"], 20)
        self.assertEqual(out.loc[2, "bg_pathogen_n"], 70)
        self.assertEqual(out.loc[0, "bg_antibiotic_n"], 5)
        self.assertEqual(out.loc[1, "bg_antibiotic_n"], 6)
        self.assertEqual(out.loc[2, "bg_antibiotic_n"], 15)
        self.assertAlmostEqual(out.loc[:1, "expected_bg"].sum(), 3.0)
        self.assertAlmostEqual(out.loc[2, "expected_bg"], 3.0)


if __name__ == "__main__":
    unittest.main()
