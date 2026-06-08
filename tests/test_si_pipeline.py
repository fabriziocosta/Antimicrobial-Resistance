import unittest
import warnings
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import pandas as pd

from amr.si import (
    build_yearly_category_counts,
    fit_predict_yearly_si_loess,
    plot_yearly_category_driver_counts,
    plot_yearly_si_pair,
    rank_yearly_pairs_by_future_si_shift,
    select_interesting_yearly_rank_cases,
    summarize_si_change_drivers,
)
from amr.si.core import AMR_DRIVER_CATEGORY_COLORS, MISSING_LLM_INFORMATION_CATEGORY
from amr.pipelines.si import (
    build_si_analysis_paths,
    build_yearly_si_dataset,
    coarsen_category_counts,
    filter_model_inputs_by_average_amr_publications,
    rank_future_high_interest_pairs,
    save_category_driver_plots,
)


class ScientificInterestPipelineTests(unittest.TestCase):
    def test_analysis_paths_use_format_neutral_plot_directories(self) -> None:
        paths = build_si_analysis_paths("/tmp/amr", "antibiotic")
        self.assertEqual(paths["output_dir"].as_posix(), "/tmp/amr/outputs")
        self.assertEqual(paths["parquet_dir"].as_posix(), "/tmp/amr/outputs/parquets")
        self.assertEqual(paths["yearly_si"].as_posix(), "/tmp/amr/outputs/parquets/yearly_si.parquet")
        self.assertEqual(paths["figure_dir"].as_posix(), "/tmp/amr/outputs/figures")
        self.assertEqual(paths["interesting_plot_dir"].parent.as_posix(), "/tmp/amr/outputs/figures")
        self.assertEqual(paths["driver_plot_dir"].parent.as_posix(), "/tmp/amr/outputs/figures")
        self.assertEqual(paths["interesting_plot_dir"].name, "interesting_yearly_si_plots")
        self.assertEqual(paths["driver_plot_dir"].name, "interesting_yearly_si_driver_plots")

        class_paths = build_si_analysis_paths("/tmp/amr", "Class")
        self.assertEqual(class_paths["yearly_si"].as_posix(), "/tmp/amr/outputs/parquets/yearly_si_class.parquet")
        self.assertEqual(class_paths["figure_dir"].as_posix(), "/tmp/amr/outputs/figures")
        self.assertEqual(class_paths["interesting_plot_dir"].parent.as_posix(), "/tmp/amr/outputs/figures")
        self.assertEqual(class_paths["driver_plot_dir"].parent.as_posix(), "/tmp/amr/outputs/figures")
        self.assertEqual(class_paths["interesting_plot_dir"].name, "interesting_yearly_si_plots_class")
        self.assertEqual(class_paths["driver_plot_dir"].name, "interesting_yearly_si_driver_plots_class")

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
        self.assertIn("coverage_report", si_df.attrs)

    def test_filter_model_inputs_by_average_amr_publications_keeps_high_volume_pairs(self) -> None:
        yearly_df = pd.DataFrame(
            [
                {"pathogen": "p1", "antibiotic": "a1", "year": 2020, "count": 10},
                {"pathogen": "p1", "antibiotic": "a1", "year": 2021, "count": 10},
                {"pathogen": "p2", "antibiotic": "a2", "year": 2020, "count": 1},
                {"pathogen": "p2", "antibiotic": "a2", "year": 2021, "count": 1},
            ]
        )
        unified_df = pd.DataFrame(
            [
                {"pathogen": "p1", "antibiotic": "a1", "year": 2020, "pmid": "1"},
                {"pathogen": "p2", "antibiotic": "a2", "year": 2020, "pmid": "2"},
            ]
        )

        filtered_yearly_df, filtered_unified_df, rates_df = filter_model_inputs_by_average_amr_publications(
            yearly_screened_model_df=yearly_df,
            unified_decisions_model_df=unified_df,
            fit_start_year=2020,
            fit_end_year=2021,
            min_avg_amr_publications_per_year=5.0,
            verbose=False,
        )

        self.assertEqual(set(filtered_yearly_df["pathogen"]), {"p1"})
        self.assertEqual(set(filtered_unified_df["pathogen"]), {"p1"})
        high_rate = rates_df.loc[rates_df["pathogen"] == "p1", "avg_amr_publications_per_year"].iloc[0]
        low_rate = rates_df.loc[rates_df["pathogen"] == "p2", "avg_amr_publications_per_year"].iloc[0]
        self.assertEqual(float(high_rate), 10.0)
        self.assertEqual(float(low_rate), 1.0)

    def test_rank_output_includes_latest_count_fields_for_interactive_plots(self) -> None:
        yearly_amr_df, pathogen_bg_df, antibiotic_bg_df = self._build_synthetic_yearly_frames()
        si_df, _, _ = build_yearly_si_dataset(
            yearly_amr_df,
            pathogen_bg_df,
            antibiotic_bg_df,
            output_path=None,
        )
        with NamedTemporaryFile(suffix=".parquet") as tmp_rank:
            rank_df = rank_yearly_pairs_by_future_si_shift(
                si_df,
                fit_start_year=2000,
                fit_end_year=2008,
                num_future_years=2,
                save_path=tmp_rank.name,
                verbose=False,
            )
            saved_rank_df = pd.read_parquet(tmp_rank.name)

        for col in [
            "latest_available_count",
            "latest_available_predicted_SI",
            "n_count_error_years",
            "post_fit_count_abs_pct_error_mean",
            "si_loess_residual_variance_mean",
            "si_loess_residual_std",
        ]:
            self.assertIn(col, rank_df.columns)
            self.assertIn(col, saved_rank_df.columns)

    def test_rank_uses_separate_validation_and_curve_fit_windows(self) -> None:
        si_df = pd.DataFrame(
            [
                {
                    "pathogen": "p1",
                    "antibiotic": "a1",
                    "year": year,
                    "count": 10 + (year - 2000),
                    "bg_pathogen_n": 100 + year - 2000,
                    "bg_antibiotic_n": 80 + year - 2000,
                    "expected_count_global": 8.0 + 0.2 * (year - 2000),
                    "expected_log_count_global": 1.0 + 0.01 * (year - 2000),
                    "SI": 0.2 + 0.03 * (year - 2000),
                }
                for year in range(2000, 2027)
            ]
        )

        rank_df = rank_yearly_pairs_by_future_si_shift(
            si_df,
            fit_start_year=2000,
            fit_end_year=2023,
            curve_fit_end_year=2025,
            post_fit_validation_end_year=2025,
            num_future_years=2,
            save_path=None,
            verbose=False,
        )
        row = rank_df.loc[rank_df["status"] == "ok"].iloc[0]
        curve_result = fit_predict_yearly_si_loess(
            si_df,
            pathogen="p1",
            antibiotic="a1",
            fit_start_year=2000,
            fit_end_year=2025,
            num_additional_years=2,
        )
        curve_pred_df = curve_result["pred_df"].set_index("year")
        expected_si_change = (
            float(curve_pred_df.loc[2027, "predicted_SI"])
            - float(curve_pred_df.loc[2025, "predicted_SI"])
        )
        curve_fit_comparison_df = curve_result["detailed_comparison_df"].query("year >= 2000 and year <= 2025")
        expected_variance = float((curve_fit_comparison_df["prediction_error"] ** 2).mean())

        self.assertEqual(int(row["reference_year"]), 2025)
        self.assertEqual(int(row["future_year"]), 2027)
        self.assertEqual(int(row["curve_fit_end_year"]), 2025)
        self.assertEqual(int(row["post_fit_validation_end_year"]), 2025)
        self.assertEqual(int(row["raw_latest_available_year"]), 2026)
        self.assertEqual(int(row["latest_available_year"]), 2025)
        self.assertEqual(int(row["n_post_fit_observed_years"]), 2)
        self.assertEqual(int(row["latest_post_fit_observed_year"]), 2025)
        self.assertAlmostEqual(float(row["si_change"]), expected_si_change)
        self.assertAlmostEqual(float(row["si_loess_residual_variance_mean"]), expected_variance)

    def test_interesting_selection_can_filter_noisy_si_loess_curves(self) -> None:
        rank_df = pd.DataFrame(
            [
                {
                    "pathogen": "p1",
                    "antibiotic": "a1",
                    "status": "ok",
                    "current_predicted_count": 20.0,
                    "future_predicted_count": 30.0,
                    "future_predicted_SI": 1.0,
                    "si_change": 0.5,
                    "n_post_fit_observed_years": 1,
                    "post_fit_si_abs_error_mean": 0.1,
                    "post_fit_si_abs_pct_error_mean": 5.0,
                    "si_loess_residual_variance_mean": 0.05,
                },
                {
                    "pathogen": "p2",
                    "antibiotic": "a2",
                    "status": "ok",
                    "current_predicted_count": 20.0,
                    "future_predicted_count": 30.0,
                    "future_predicted_SI": 2.0,
                    "si_change": 0.5,
                    "n_post_fit_observed_years": 1,
                    "post_fit_si_abs_error_mean": 0.1,
                    "post_fit_si_abs_pct_error_mean": 5.0,
                    "si_loess_residual_variance_mean": 1.5,
                },
            ]
        )

        selected = select_interesting_yearly_rank_cases(
            rank_df,
            min_current_predicted_count=1.0,
            min_future_predicted_count=1.0,
            min_si_change=0.0,
            min_post_fit_observed_years=1,
            max_post_fit_si_abs_error_mean=1.0,
            max_post_fit_si_abs_pct_error_mean=100.0,
            max_si_loess_residual_variance_mean=0.1,
            fallback_top_n=0,
        )

        self.assertEqual(set(zip(selected["pathogen"], selected["antibiotic"])), {("p1", "a1")})
        self.assertTrue(selected["passes_si_loess_residual_variance"].all())

    def test_yearly_si_plot_includes_zero_reference_line(self) -> None:
        yearly_amr_df, pathogen_bg_df, antibiotic_bg_df = self._build_synthetic_yearly_frames()
        si_df, _, _ = build_yearly_si_dataset(
            yearly_amr_df,
            pathogen_bg_df,
            antibiotic_bg_df,
            output_path=None,
        )

        result = plot_yearly_si_pair(
            si_df,
            pathogen="p1",
            antibiotic="a1",
            fit_start_year=2000,
            fit_end_year=2008,
            num_additional_years=2,
        )
        zero_lines = [
            line
            for line in result["ax_si"].lines
            if line.get_linestyle() == ":" and all(float(y) == 0.0 for y in line.get_ydata())
        ]
        plt.close(result["figure"])

        self.assertEqual(len(zero_lines), 1)

    def test_yearly_si_plot_can_fill_missing_pair_year_counts_with_zero(self) -> None:
        years = list(range(2000, 2005))
        yearly_amr_df = pd.DataFrame(
            [
                {"pathogen": "p1", "antibiotic": "a1", "year": year, "count": 10 + year - 2000}
                for year in years
                if year != 2002
            ]
            + [
                {"pathogen": "p1", "antibiotic": "a2", "year": year, "count": 5}
                for year in years
            ]
            + [
                {"pathogen": "p2", "antibiotic": "a1", "year": year, "count": 4}
                for year in years
            ]
        )
        pathogen_bg_df = pd.DataFrame(
            [
                {"pathogen": pathogen, "year": year, "count": 100 + year - 2000}
                for pathogen in ["p1", "p2"]
                for year in years
            ]
        )
        antibiotic_bg_df = pd.DataFrame(
            [
                {"antibiotic": antibiotic, "year": year, "count": 80 + year - 2000}
                for antibiotic in ["a1", "a2"]
                for year in years
            ]
        )
        si_df, _, _ = build_yearly_si_dataset(
            yearly_amr_df,
            pathogen_bg_df,
            antibiotic_bg_df,
            output_path=None,
        )

        result = plot_yearly_si_pair(
            si_df,
            pathogen="p1",
            antibiotic="a1",
            fit_start_year=2000,
            fit_end_year=2004,
            num_additional_years=0,
            pathogen_background_df=pathogen_bg_df,
            antibiotic_background_df=antibiotic_bg_df,
        )
        count_plot_df = result["count_plot_df"]
        plt.close(result["figure"])

        self.assertEqual(count_plot_df["year"].tolist(), years)
        missing_year_row = count_plot_df.loc[count_plot_df["year"] == 2002].iloc[0]
        self.assertEqual(float(missing_year_row["count"]), 0.0)
        self.assertEqual(float(missing_year_row["bg_pathogen_n"]), 102.0)
        self.assertEqual(float(missing_year_row["bg_antibiotic_n"]), 82.0)

    def test_build_yearly_si_dataset_warns_when_background_rows_are_dropped(self) -> None:
        yearly_amr_df = pd.DataFrame(
            [{"pathogen": "p1", "antibiotic": "a1", "year": 2020, "count": 3}]
        )
        pathogen_bg_df = pd.DataFrame(columns=["pathogen", "year", "count"])
        antibiotic_bg_df = pd.DataFrame([{"antibiotic": "a1", "year": 2020, "count": 10}])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with self.assertRaises(ValueError):
                build_yearly_si_dataset(
                    yearly_amr_df,
                    pathogen_bg_df,
                    antibiotic_bg_df,
                    output_path=None,
                )

        self.assertTrue(any("Dropped 1 yearly AMR row" in str(item.message) for item in caught))

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
        ok_degrees = set(rank_df.loc[rank_df["status"] == "ok", "loess_degree"].astype(int))
        self.assertEqual(ok_degrees, {1})
        if not interesting_df.empty:
            kept_pairs = set(zip(interesting_df["pathogen"], interesting_df["antibiotic"]))
            self.assertIn(("p1", "a1"), kept_pairs)
            self.assertNotIn(("p2", "a2"), kept_pairs)

    def test_quadratic_loess_is_rejected(self) -> None:
        yearly_amr_df, pathogen_bg_df, antibiotic_bg_df = self._build_synthetic_yearly_frames()
        si_df, _, _ = build_yearly_si_dataset(
            yearly_amr_df,
            pathogen_bg_df,
            antibiotic_bg_df,
            output_path=None,
        )

        with self.assertRaisesRegex(ValueError, "Only linear LOESS"):
            fit_predict_yearly_si_loess(
                si_df,
                pathogen="p1",
                antibiotic="a1",
                fit_start_year=2000,
                fit_end_year=2008,
                loess_degree=2,
            )

    def test_si_prediction_baseline_ignores_post_fit_baseline_collapse(self) -> None:
        si_df = pd.DataFrame(
            [
                {
                    "pathogen": "p1",
                    "antibiotic": "a1",
                    "year": year,
                    "count": 10 + year - 2000,
                    "bg_pathogen_n": 100 + year - 2000,
                    "bg_antibiotic_n": 80 + year - 2000,
                    "expected_count_global": 10.0,
                    "expected_log_count_global": 1.0 + 0.1 * (year - 2000),
                    "SI": 0.0,
                }
                for year in range(2000, 2005)
            ]
            + [
                {
                    "pathogen": "p1",
                    "antibiotic": "a1",
                    "year": 2005,
                    "count": 15,
                    "bg_pathogen_n": 10,
                    "bg_antibiotic_n": 8,
                    "expected_count_global": 0.1,
                    "expected_log_count_global": -1.0,
                    "SI": 2.2,
                }
            ]
        )

        result = fit_predict_yearly_si_loess(
            si_df,
            pathogen="p1",
            antibiotic="a1",
            fit_start_year=2000,
            fit_end_year=2004,
            num_additional_years=0,
            loess_frac=0.6,
        )
        pred_2005 = result["pred_df"].loc[result["pred_df"]["year"] == 2005].iloc[0]

        self.assertGreater(float(pred_2005["expected_log_count_baseline"]), 1.3)
        self.assertLess(float(pred_2005["predicted_SI"]), 0.5)

    def test_build_yearly_category_counts_filters_and_aggregates(self) -> None:
        unified_df = pd.DataFrame(
            [
                {
                    "pathogen": "p1",
                    "antibiotic": "a1",
                    "year": 2022,
                    "unified_decision": "YES",
                    "amr_primary_category": "MECHANISMS_OF_RESISTANCE_GENETICS_LAB_STUDIES",
                },
                {
                    "pathogen": "p1",
                    "antibiotic": "a1",
                    "year": 2022,
                    "unified_decision": "uncertain",
                    "amr_primary_category": "MECHANISMS_OF_RESISTANCE_GENETICS_LAB_STUDIES",
                },
                {
                    "pathogen": "p1",
                    "antibiotic": "a1",
                    "year": 2022,
                    "unified_decision": "no",
                    "amr_primary_category": "CLINICAL_OUTBREAKS_REPORTS",
                },
                {
                    "pathogen": "p1",
                    "antibiotic": "a1",
                    "year": 2023,
                    "unified_decision": "yes",
                    "amr_primary_category": None,
                },
                {
                    "pathogen": "p1",
                    "antibiotic": "a1",
                    "year": 2024,
                    "unified_decision": "uncertain",
                    "amr_primary_category": "NONE",
                    "unified_rationale": "LLM failure: OpenAI completion failed: Error code: 429",
                },
            ]
        )

        out = build_yearly_category_counts(unified_df, include_decisions=("yes", "uncertain"))

        mechanism = out.loc[out["amr_primary_category"] == "MECHANISMS_OF_RESISTANCE_GENETICS_LAB_STUDIES"]
        missing = out.loc[out["amr_primary_category"] == MISSING_LLM_INFORMATION_CATEGORY]
        self.assertEqual(int(mechanism.iloc[0]["count"]), 2)
        self.assertEqual(int(missing["count"].sum()), 2)
        self.assertNotIn("NONE", set(out["amr_primary_category"]))
        self.assertNotIn("CLINICAL_OUTBREAKS_REPORTS", set(out["amr_primary_category"]))

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
                    "amr_primary_category": "SURVEILLANCE",
                    "amr_target_pair_relation_supported": True,
                },
                {
                    "pathogen": "p1",
                    "antibiotic": "a1",
                    "year": 2023,
                    "unified_decision": "yes",
                    "amr_primary_category": "MECHANISMS_OF_RESISTANCE_GENETICS_LAB_STUDIES",
                    "amr_target_pair_relation_supported": True,
                },
                {
                    "pathogen": "p1",
                    "antibiotic": "a1",
                    "year": 2023,
                    "unified_decision": "uncertain",
                    "amr_primary_category": "MECHANISMS_OF_RESISTANCE_GENETICS_LAB_STUDIES",
                    "amr_target_pair_relation_supported": True,
                },
                {
                    "pathogen": "p1",
                    "antibiotic": "a1",
                    "year": 2024,
                    "unified_decision": "yes",
                    "amr_primary_category": "NONE",
                    "amr_target_pair_relation_supported": False,
                },
                {
                    "pathogen": "p1",
                    "antibiotic": "a1",
                    "year": 2024,
                    "unified_decision": "no",
                    "amr_primary_category": "CLINICAL_OUTBREAKS_REPORTS",
                    "amr_target_pair_relation_supported": True,
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
        self.assertEqual(row["driver_dominant_category"], "MECHANISMS_OF_RESISTANCE_GENETICS_LAB_STUDIES")
        self.assertEqual(row["driver_fastest_growing_category"], "MECHANISMS_OF_RESISTANCE_GENETICS_LAB_STUDIES")
        self.assertAlmostEqual(row["driver_fastest_growing_category_change"], 2 / 3)
        self.assertAlmostEqual(row["driver_weak_signal_share"], 1 / 3)
        self.assertAlmostEqual(row["driver_target_pair_supported_share"], 2 / 3)
        self.assertEqual(row["driver_target_pair_supported_count"], 2)
        self.assertEqual(row["driver_target_pair_unsupported_count"], 1)

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

    def test_category_driver_plot_uses_fixed_category_colors(self) -> None:
        category_counts_df = pd.DataFrame(
            [
                {
                    "pathogen": "p1",
                    "antibiotic": "a1",
                    "year": 2022,
                    "amr_primary_category": "MECHANISMS_OF_RESISTANCE_GENETICS_LAB_STUDIES",
                    "count": 2,
                },
                {
                    "pathogen": "p1",
                    "antibiotic": "a1",
                    "year": 2022,
                    "amr_primary_category": MISSING_LLM_INFORMATION_CATEGORY,
                    "count": 1,
                },
                {
                    "pathogen": "p2",
                    "antibiotic": "a2",
                    "year": 2022,
                    "amr_primary_category": "CLINICAL_OUTBREAKS_REPORTS",
                    "count": 1,
                },
            ]
        )

        result_with_mechanism = plot_yearly_category_driver_counts(category_counts_df, "p1", "a1")
        result_without_mechanism = plot_yearly_category_driver_counts(category_counts_df, "p2", "a2")

        try:
            treatment_rgba = to_rgba(AMR_DRIVER_CATEGORY_COLORS["CLINICAL_OUTBREAKS_REPORTS"])
            missing_rgba = to_rgba(AMR_DRIVER_CATEGORY_COLORS[MISSING_LLM_INFORMATION_CATEGORY])
            missing_index_with_mechanism = list(result_with_mechanism["pivot"].columns).index(
                MISSING_LLM_INFORMATION_CATEGORY
            )
            treatment_index_without_mechanism = list(result_without_mechanism["pivot"].columns).index("CLINICAL_OUTBREAKS_REPORTS")
            self.assertEqual(
                result_with_mechanism["ax"].containers[missing_index_with_mechanism].patches[0].get_facecolor(),
                missing_rgba,
            )
            self.assertEqual(
                result_without_mechanism["ax"].containers[treatment_index_without_mechanism].patches[0].get_facecolor(),
                treatment_rgba,
            )
        finally:
            plt.close(result_with_mechanism["figure"])
            plt.close(result_without_mechanism["figure"])

    def test_category_driver_plot_keeps_all_missing_llm_years_gray(self) -> None:
        category_counts_df = pd.DataFrame(
            [
                {
                    "pathogen": "p1",
                    "antibiotic": "a1",
                    "year": 2020,
                    "amr_primary_category": MISSING_LLM_INFORMATION_CATEGORY,
                    "count": 10,
                }
            ]
        )

        result = plot_yearly_category_driver_counts(category_counts_df, "p1", "a1")

        try:
            missing_rgba = to_rgba(AMR_DRIVER_CATEGORY_COLORS[MISSING_LLM_INFORMATION_CATEGORY])
            missing_index = list(result["pivot"].columns).index(MISSING_LLM_INFORMATION_CATEGORY)
            self.assertAlmostEqual(float(result["pivot"].loc[2020, MISSING_LLM_INFORMATION_CATEGORY]), 1.0)
            self.assertEqual(
                result["ax"].containers[missing_index].patches[0].get_facecolor(),
                missing_rgba,
            )
        finally:
            plt.close(result["figure"])

    def test_category_driver_plot_fills_unclassified_observed_counts_from_si_df(self) -> None:
        category_counts_df = pd.DataFrame(
            [
                {
                    "pathogen": "p1",
                    "antibiotic": "a1",
                    "year": 2021,
                    "amr_primary_category": "MECHANISMS_OF_RESISTANCE_GENETICS_LAB_STUDIES",
                    "count": 2,
                },
                {
                    "pathogen": "p1",
                    "antibiotic": "a1",
                    "year": 2022,
                    "amr_primary_category": "MECHANISMS_OF_RESISTANCE_GENETICS_LAB_STUDIES",
                    "count": 3,
                },
            ]
        )
        si_df = pd.DataFrame(
            [
                {"pathogen": "p1", "antibiotic": "a1", "year": 2020, "SI": 0.1, "count": 5},
                {"pathogen": "p1", "antibiotic": "a1", "year": 2021, "SI": 0.2, "count": 2},
                {"pathogen": "p1", "antibiotic": "a1", "year": 2022, "SI": 0.3, "count": 5},
            ]
        )

        result = plot_yearly_category_driver_counts(
            category_counts_df,
            "p1",
            "a1",
            si_df=si_df,
            start_year=2020,
            end_year=2022,
        )

        try:
            missing_rgba = to_rgba(AMR_DRIVER_CATEGORY_COLORS[MISSING_LLM_INFORMATION_CATEGORY])
            missing_index = list(result["pivot"].columns).index(MISSING_LLM_INFORMATION_CATEGORY)
            self.assertAlmostEqual(float(result["pivot"].loc[2020, MISSING_LLM_INFORMATION_CATEGORY]), 1.0)
            self.assertAlmostEqual(float(result["pivot"].loc[2022, MISSING_LLM_INFORMATION_CATEGORY]), 2 / 5)
            self.assertEqual(float(result["count_pivot"].loc[2020, MISSING_LLM_INFORMATION_CATEGORY]), 5.0)
            self.assertEqual(float(result["count_pivot"].loc[2022, MISSING_LLM_INFORMATION_CATEGORY]), 2.0)
            self.assertEqual(
                result["ax"].containers[missing_index].patches[0].get_facecolor(),
                missing_rgba,
            )
        finally:
            plt.close(result["figure"])

    def test_category_driver_plot_can_use_fixed_year_window(self) -> None:
        category_counts_df = pd.DataFrame(
            [
                {
                    "pathogen": "p1",
                    "antibiotic": "a1",
                    "year": 2002,
                    "amr_primary_category": "MECHANISMS_OF_RESISTANCE_GENETICS_LAB_STUDIES",
                    "count": 2,
                },
                {
                    "pathogen": "p1",
                    "antibiotic": "a1",
                    "year": 2004,
                    "amr_primary_category": "CLINICAL_OUTBREAKS_REPORTS",
                    "count": 1,
                },
            ]
        )

        result = plot_yearly_category_driver_counts(
            category_counts_df,
            "p1",
            "a1",
            start_year=2000,
            end_year=2005,
            fit_end_year=2003,
        )
        try:
            self.assertEqual(result["pivot"].index.tolist(), [2000, 2001, 2002, 2003, 2004, 2005])
            self.assertEqual(int(result["pivot"].loc[2000].sum()), 0)
            self.assertEqual(int(result["pivot"].loc[2003].sum()), 0)
            self.assertAlmostEqual(float(result["pivot"].loc[2002].sum()), 1.0)
            self.assertEqual(int(result["count_pivot"].loc[2002].sum()), 2)
            self.assertEqual(result["ax"].get_ylabel(), "Share of LLM-classified AMR publications")
            fit_lines = [line for line in result["ax"].lines if line.get_linestyle() == "--"]
            self.assertEqual(len(fit_lines), 1)
            self.assertAlmostEqual(float(fit_lines[0].get_xdata()[0]), 3.5)
        finally:
            plt.close(result["figure"])

    def test_category_driver_plot_can_overlay_observed_si(self) -> None:
        category_counts_df = pd.DataFrame(
            [
                {
                    "pathogen": "p1",
                    "antibiotic": "a1",
                    "year": 2002,
                    "amr_primary_category": "MECHANISMS_OF_RESISTANCE_GENETICS_LAB_STUDIES",
                    "count": 2,
                },
                {
                    "pathogen": "p1",
                    "antibiotic": "a1",
                    "year": 2004,
                    "amr_primary_category": "CLINICAL_OUTBREAKS_REPORTS",
                    "count": 1,
                },
            ]
        )
        si_df = pd.DataFrame(
            [
                {"pathogen": "p1", "antibiotic": "a1", "year": 2002, "SI": 0.25},
                {"pathogen": "p1", "antibiotic": "a1", "year": 2004, "SI": 0.75},
                {"pathogen": "p1", "antibiotic": "a2", "year": 2004, "SI": 2.0},
            ]
        )

        result = plot_yearly_category_driver_counts(
            category_counts_df,
            "p1",
            "a1",
            si_df=si_df,
            start_year=2002,
            end_year=2004,
        )
        try:
            self.assertIsNotNone(result["si_ax"])
            si_lines = result["si_ax"].lines
            self.assertEqual(len(si_lines), 1)
            self.assertEqual(si_lines[0].get_label(), "Observed SI")
            self.assertEqual(si_lines[0].get_xdata().tolist(), [0, 2])
            self.assertEqual(si_lines[0].get_ydata().tolist(), [0.25, 0.75])
            self.assertEqual(result["si_ax"].get_ylabel(), "Scientific interest (SI)")
        finally:
            plt.close(result["figure"])

    def test_coarsen_category_counts_groups_fine_driver_categories(self) -> None:
        category_counts_df = pd.DataFrame(
            [
                {
                    "pathogen": "p1",
                    "antibiotic": "a1",
                    "year": 2022,
                    "amr_primary_category": "SURVEILLANCE",
                    "count": 2,
                },
                {
                    "pathogen": "p1",
                    "antibiotic": "a1",
                    "year": 2022,
                    "amr_primary_category": "MECHANISMS_OF_RESISTANCE_GENETICS_LAB_STUDIES",
                    "count": 3,
                },
                {
                    "pathogen": "p1",
                    "antibiotic": "a1",
                    "year": 2022,
                    "amr_primary_category": MISSING_LLM_INFORMATION_CATEGORY,
                    "count": 1,
                },
            ]
        )

        coarse_df, category_order, category_colors = coarsen_category_counts(category_counts_df)

        self.assertIn("Surveillance", category_order)
        self.assertIn("Unsupported or missing evidence", category_colors)
        surveillance_count = coarse_df.loc[
            coarse_df["amr_primary_category"].eq("Surveillance"),
            "count",
        ].item()
        mechanism_count = coarse_df.loc[
            coarse_df["amr_primary_category"].eq("Mechanisms genetics and lab studies"),
            "count",
        ].item()
        weak_count = coarse_df.loc[
            coarse_df["amr_primary_category"].eq("Unsupported or missing evidence"),
            "count",
        ].item()
        self.assertEqual(int(surveillance_count), 2)
        self.assertEqual(int(mechanism_count), 3)
        self.assertEqual(int(weak_count), 1)

    def test_category_driver_plot_can_fill_missing_counts_into_coarse_group(self) -> None:
        category_counts_df = pd.DataFrame(
            [
                {
                    "pathogen": "p1",
                    "antibiotic": "a1",
                    "year": 2022,
                    "amr_primary_category": "MECHANISMS_OF_RESISTANCE_GENETICS_LAB_STUDIES",
                    "count": 2,
                }
            ]
        )
        si_df = pd.DataFrame(
            [{"pathogen": "p1", "antibiotic": "a1", "year": 2022, "SI": 0.1, "count": 5}]
        )
        coarse_df, category_order, category_colors = coarsen_category_counts(category_counts_df)

        result = plot_yearly_category_driver_counts(
            coarse_df,
            "p1",
            "a1",
            si_df=si_df,
            category_order=category_order,
            category_colors=category_colors,
            missing_information_category="Unsupported or missing evidence",
        )
        try:
            self.assertAlmostEqual(float(result["pivot"].loc[2022, "Mechanisms genetics and lab studies"]), 2 / 5)
            self.assertAlmostEqual(float(result["pivot"].loc[2022, "Unsupported or missing evidence"]), 3 / 5)
            self.assertNotIn(MISSING_LLM_INFORMATION_CATEGORY, result["pivot"].columns)
        finally:
            plt.close(result["figure"])

    def test_save_category_driver_plots_can_use_coarse_categories(self) -> None:
        category_counts_df = pd.DataFrame(
            [
                {
                    "pathogen": "p1",
                    "antibiotic": "a1",
                    "year": 2022,
                    "amr_primary_category": "SURVEILLANCE",
                    "count": 2,
                },
                {
                    "pathogen": "p1",
                    "antibiotic": "a1",
                    "year": 2022,
                    "amr_primary_category": "MECHANISMS_OF_RESISTANCE_GENETICS_LAB_STUDIES",
                    "count": 3,
                },
            ]
        )
        interesting_df = pd.DataFrame([{"pathogen": "p1", "antibiotic": "a1"}])

        with TemporaryDirectory() as tmpdir:
            saved = save_category_driver_plots(
                category_counts_df=category_counts_df,
                interesting_df=interesting_df,
                output_dir=tmpdir,
                fit_end_year=2022,
                max_plots=1,
                figure_format="png",
                driver_category_level="coarse",
                verbose=False,
            )

            self.assertEqual(saved, 1)
            self.assertTrue((Path(tmpdir) / "p1__a1__coarse_drivers.png").exists())


if __name__ == "__main__":
    unittest.main()
