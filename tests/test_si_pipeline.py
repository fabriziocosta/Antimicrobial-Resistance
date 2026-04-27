import unittest
from tempfile import NamedTemporaryFile

import pandas as pd

from amr.scientific_interest import (
    build_yearly_category_counts,
    summarize_si_change_drivers,
)
from amr.pipelines.scientific_interest import (
    build_yearly_si_dataset,
    rank_future_high_interest_pairs,
)


class ScientificInterestPipelineTests(unittest.TestCase):
    def _build_synthetic_yearly_frames(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        years = list(range(2000, 2011))
        rows = []
        for year in years:
            rows.append(
                {
                    "pathogen": "p1",
                    "antibiotic": "a1",
                    "year": year,
                    "count": 10 + (year - 2000) * 2,
                }
            )
            rows.append(
                {
                    "pathogen": "p2",
                    "antibiotic": "a2",
                    "year": year,
                    "count": 1 if year % 2 == 0 else 0,
                }
            )
        yearly_amr_df = pd.DataFrame(rows)

        path_bg_rows = []
        ab_bg_rows = []
        for year in years:
            path_bg_rows.extend(
                [
                    {"pathogen": "p1", "year": year, "count": 300 + (year - 2000) * 8},
                    {"pathogen": "p2", "year": year, "count": 40 + (year - 2000)},
                ]
            )
            ab_bg_rows.extend(
                [
                    {"antibiotic": "a1", "year": year, "count": 250 + (year - 2000) * 7},
                    {"antibiotic": "a2", "year": year, "count": 35 + (year - 2000)},
                ]
            )
        return yearly_amr_df, pd.DataFrame(path_bg_rows), pd.DataFrame(ab_bg_rows)

    def test_build_yearly_si_dataset_has_required_non_null_columns(self) -> None:
        yearly_amr_df, pathogen_bg_df, antibiotic_bg_df = self._build_synthetic_yearly_frames()
        si_df, _, _ = build_yearly_si_dataset(
            yearly_amr_df,
            pathogen_bg_df,
            antibiotic_bg_df,
            output_path=None,
        )
        self.assertIn("expected_log_count_global", si_df.columns)
        self.assertIn("SI", si_df.columns)
        self.assertFalse(si_df["expected_log_count_global"].isna().any())
        self.assertFalse(si_df["SI"].isna().any())

    def test_rank_future_high_interest_pairs_filters_low_support_pairs(self) -> None:
        yearly_amr_df, pathogen_bg_df, antibiotic_bg_df = self._build_synthetic_yearly_frames()
        si_df, _, _ = build_yearly_si_dataset(
            yearly_amr_df,
            pathogen_bg_df,
            antibiotic_bg_df,
            output_path=None,
        )
        with NamedTemporaryFile(suffix=".parquet") as tmp_rank:
            rank_df, interesting_df = rank_future_high_interest_pairs(
                si_df,
                fit_start_year=2000,
                fit_end_year=2008,
                num_future_years=2,
                min_current_predicted_count=5.0,
                min_future_predicted_count=5.0,
                min_si_change=-10.0,
                min_post_fit_observed_years=1,
                max_post_fit_si_abs_pct_error_mean=1000.0,
                fallback_top_n=0,
                rank_output_path=tmp_rank.name,
                interesting_output_path=None,
                verbose=False,
            )
        self.assertTrue((rank_df["status"] == "ok").any())
        if not interesting_df.empty:
            kept_pairs = set(zip(interesting_df["pathogen"], interesting_df["antibiotic"]))
            self.assertIn(("p1", "a1"), kept_pairs)
            self.assertNotIn(("p2", "a2"), kept_pairs)

    def test_build_yearly_category_counts_filters_and_aggregates(self) -> None:
        unified_df = pd.DataFrame(
            [
                {
                    "pathogen": "p1",
                    "antibiotic": "a1",
                    "year": 2022,
                    "unified_decision": "YES",
                    "amr_primary_category": "RESISTANCE_MECHANISM",
                },
                {
                    "pathogen": "p1",
                    "antibiotic": "a1",
                    "year": 2022,
                    "unified_decision": "uncertain",
                    "amr_primary_category": "RESISTANCE_MECHANISM",
                },
                {
                    "pathogen": "p1",
                    "antibiotic": "a1",
                    "year": 2022,
                    "unified_decision": "no",
                    "amr_primary_category": "TREATMENT_EFFECTIVENESS",
                },
                {
                    "pathogen": "p1",
                    "antibiotic": "a1",
                    "year": 2023,
                    "unified_decision": "yes",
                    "amr_primary_category": None,
                },
            ]
        )

        out = build_yearly_category_counts(unified_df, include_decisions=("yes", "uncertain"))

        mechanism = out.loc[out["amr_primary_category"] == "RESISTANCE_MECHANISM"]
        weak = out.loc[out["amr_primary_category"] == "CO_OCCURRENCE_OR_WEAK_SIGNAL"]
        self.assertEqual(int(mechanism.iloc[0]["count"]), 2)
        self.assertEqual(int(weak.iloc[0]["count"]), 1)
        self.assertNotIn("TREATMENT_EFFECTIVENESS", set(out["amr_primary_category"]))

    def test_summarize_si_change_drivers_reports_mix_and_growth(self) -> None:
        interesting_df = pd.DataFrame(
            [
                {
                    "pathogen": "p1",
                    "antibiotic": "a1",
                    "si_change": 0.8,
                }
            ]
        )
        unified_df = pd.DataFrame(
            [
                {
                    "pathogen": "p1",
                    "antibiotic": "a1",
                    "year": 2020,
                    "unified_decision": "yes",
                    "amr_primary_category": "RESISTANCE_PREVALENCE_SURVEILLANCE",
                    "amr_pair_centrality": "central",
                    "amr_needs_manual_review": False,
                },
                {
                    "pathogen": "p1",
                    "antibiotic": "a1",
                    "year": 2023,
                    "unified_decision": "yes",
                    "amr_primary_category": "RESISTANCE_MECHANISM",
                    "amr_pair_centrality": "central",
                    "amr_needs_manual_review": False,
                },
                {
                    "pathogen": "p1",
                    "antibiotic": "a1",
                    "year": 2023,
                    "unified_decision": "uncertain",
                    "amr_primary_category": "RESISTANCE_MECHANISM",
                    "amr_pair_centrality": "partial",
                    "amr_needs_manual_review": True,
                },
                {
                    "pathogen": "p1",
                    "antibiotic": "a1",
                    "year": 2024,
                    "unified_decision": "yes",
                    "amr_primary_category": "CO_OCCURRENCE_OR_WEAK_SIGNAL",
                    "amr_pair_centrality": "incidental",
                    "amr_needs_manual_review": True,
                },
                {
                    "pathogen": "p1",
                    "antibiotic": "a1",
                    "year": 2024,
                    "unified_decision": "no",
                    "amr_primary_category": "TREATMENT_EFFECTIVENESS",
                    "amr_pair_centrality": "central",
                    "amr_needs_manual_review": False,
                },
            ]
        )

        out = summarize_si_change_drivers(
            interesting_df,
            unified_df,
            fit_end_year=2024,
            lookback_years=3,
        )

        self.assertEqual(len(out), 1)
        row = out.iloc[0]
        self.assertEqual(row["driver_recent_total"], 3)
        self.assertEqual(row["driver_previous_total"], 1)
        self.assertEqual(row["driver_dominant_category"], "RESISTANCE_MECHANISM")
        self.assertEqual(row["driver_fastest_growing_category"], "RESISTANCE_MECHANISM")
        self.assertAlmostEqual(row["driver_weak_signal_share"], 1 / 3)
        self.assertAlmostEqual(row["driver_manual_review_share"], 2 / 3)
        self.assertEqual(row["driver_central_count"], 1)
        self.assertEqual(row["driver_partial_count"], 1)
        self.assertEqual(row["driver_incidental_count"], 1)

    def test_driver_helpers_return_empty_for_missing_category_data(self) -> None:
        interesting_df = pd.DataFrame([{"pathogen": "p1", "antibiotic": "a1"}])
        unified_df = pd.DataFrame(
            [
                {
                    "pathogen": "p1",
                    "antibiotic": "a1",
                    "year": 2024,
                    "unified_decision": "yes",
                }
            ]
        )

        counts = build_yearly_category_counts(unified_df)
        summary = summarize_si_change_drivers(interesting_df, unified_df, fit_end_year=2024)

        self.assertTrue(counts.empty)
        self.assertTrue(summary.empty)


if __name__ == "__main__":
    unittest.main()
