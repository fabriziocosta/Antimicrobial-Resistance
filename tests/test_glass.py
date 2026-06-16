from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

from amr.glass import load_and_clean_glass_list
from amr.pipelines.glass_si import (
    build_glass_match_table,
    build_novel_prediction_table,
    load_or_build_interactive_rank,
    plot_future_si_vs_si_shift,
    plot_latest_si_vs_future_si,
)
from amr.si import plot_interactive_latest_observed_count_vs_future_si_shift


class GlassCleaningTests(unittest.TestCase):
    def test_load_and_clean_glass_list_standardizes_names_and_saves(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            glass_path = tmp / "glass.xlsx"
            pd.DataFrame(
                [
                    {"species": "E. coli", "antibiotics": "Co-trimoxazole"},
                    {"species": "Salmonella spp", "antibiotics": "cefotaxime"},
                ]
            ).to_excel(glass_path, sheet_name="clean", index=False)
            bacteria_path = tmp / "bacteria.csv"
            pd.DataFrame(
                [
                    {"bacteria synonym": "salmonella enterica", "bacteria name": "Salmonella enterica"},
                ]
            ).to_csv(bacteria_path, index=False)
            antibiotic_synonym_path = tmp / "antibiotic_synonyms.csv"
            pd.DataFrame(
                [
                    {"antibiotic synonym": "cefotaxime", "antibiotic name": "Cefotaxime"},
                ]
            ).to_csv(antibiotic_synonym_path, index=False)
            coarsening_path = tmp / "coarsening.csv"
            pd.DataFrame(
                [
                    {
                        "Antibiotic": "Cefotaxime",
                        "Class": "Third-generation-cephalosporins",
                        "Simplified categories": "Cephalosporins",
                        "High level categories": "Beta lactam and related",
                        "Origin": "Naturally occurring/modified",
                    },
                    {
                        "Antibiotic": "Sulfamethoxazole/trimethoprim",
                        "Class": "Sulfonamide-trimethoprim-combinations",
                        "Simplified categories": "Sulfonamides/Trimethoprim derivatives and their combinations",
                        "High level categories": "Sulfonamides/Trimethoprim derivatives and their combinations",
                        "Origin": "Synthetic",
                    },
                ]
            ).to_csv(coarsening_path, index=False)
            output_path = tmp / "cleaned.csv"

            cleaned_df = load_and_clean_glass_list(
                glass_path,
                bacteria_synonym_path=bacteria_path,
                antibiotic_synonym_path=antibiotic_synonym_path,
                antibiotic_coarsening_path=coarsening_path,
                output_path=output_path,
                pathogen_universe=["Escherichia coli", "Salmonella enterica"],
            )

            self.assertTrue(output_path.exists())
            pairs = set(zip(cleaned_df["pathogen"], cleaned_df["antibiotic"]))
            self.assertIn(("Escherichia coli", "Sulfamethoxazole/trimethoprim"), pairs)
            self.assertIn(("Salmonella enterica", "Cefotaxime"), pairs)
            self.assertFalse(cleaned_df["Class"].isna().any())

    def test_interactive_rank_cache_rebuilds_when_curve_fit_endpoint_changes(self) -> None:
        with TemporaryDirectory() as tmpdir:
            rank_path = Path(tmpdir) / "rank.parquet"
            cached_df = pd.DataFrame(
                [
                    {
                        "pathogen": "p1",
                        "antibiotic": "a1",
                        "fit_start_year": 2000,
                        "fit_end_year": 2023,
                        "curve_fit_end_year": 2023,
                        "post_fit_validation_end_year": 2025,
                        "num_future_years": 2,
                        "latest_available_count": 10,
                        "latest_available_predicted_SI": 0.1,
                        "post_fit_count_abs_pct_error_mean": 5.0,
                        "n_count_error_years": 2,
                    }
                ]
            )
            cached_df.to_parquet(rank_path, index=False)
            rebuilt_df = cached_df.copy()
            rebuilt_df["curve_fit_end_year"] = 2025

            with patch("amr.pipelines.glass_si.rank_yearly_pairs_by_future_si_shift", return_value=rebuilt_df) as mocked_rank:
                rank_df = load_or_build_interactive_rank(
                    pd.DataFrame({"pathogen": ["p1"], "antibiotic": ["a1"], "year": [2025], "SI": [0.1]}),
                    rank_path,
                    fit_start_year=2000,
                    fit_end_year=2023,
                    curve_fit_end_year=2025,
                    post_fit_validation_end_year=2025,
                    num_future_years=2,
                    verbose=False,
                )

            mocked_rank.assert_called_once()
            self.assertEqual(int(rank_df["curve_fit_end_year"].iloc[0]), 2025)

    def test_glass_si_tables_and_scatter_plot_use_shared_visual_encoding(self) -> None:
        plot_df = pd.DataFrame(
            [
                {
                    "pathogen": "Escherichia coli",
                    "antibiotic": "Cefotaxime",
                    "latest_available_year": 2023,
                    "future_year": 2025,
                    "latest_available_count": 30,
                    "future_predicted_count": 42,
                    "latest_smoothed_SI": 0.1,
                    "future_predicted_SI": 0.22,
                    "future_minus_latest_smoothed_SI": 0.12,
                    "post_fit_count_abs_pct_error_mean": 4.0,
                    "n_count_error_years": 3,
                    "is_glass_pair": True,
                    "glass_group_column": "antibiotic",
                },
                {
                    "pathogen": "Klebsiella pneumoniae",
                    "antibiotic": "Colistin",
                    "latest_available_year": 2023,
                    "future_year": 2025,
                    "latest_available_count": 12,
                    "future_predicted_count": 15,
                    "latest_smoothed_SI": -0.05,
                    "future_predicted_SI": 0.08,
                    "future_minus_latest_smoothed_SI": 0.13,
                    "post_fit_count_abs_pct_error_mean": 8.0,
                    "n_count_error_years": 3,
                    "is_glass_pair": False,
                    "glass_group_column": "antibiotic",
                },
            ]
        )

        glass_table = build_glass_match_table(plot_df)
        novel_table = build_novel_prediction_table(plot_df)
        self.assertEqual(glass_table["pathogen"].tolist(), ["Escherichia coli"])
        self.assertEqual(novel_table["pathogen"].tolist(), ["Klebsiella pneumoniae"])

        with TemporaryDirectory() as tmpdir, patch("plotly.graph_objs._figure.Figure.write_image") as mock_write_image:
            pdf_path = Path(tmpdir) / "future_si_vs_si_shift.pdf"
            result = plot_future_si_vs_si_shift(
                plot_df,
                min_count_abs_pct_error_mean=0,
                max_count_abs_pct_error_mean=10,
                display_name=True,
                max_text_labels=2,
                show=False,
                pdf_path=pdf_path,
            )
            mock_write_image.assert_called_once_with(str(pdf_path), format="pdf", width=1400, height=900)
        fig = result["figure"]
        marker = fig.data[0].marker
        self.assertEqual(list(marker.line.color), ["#000000", "#4b5563"])
        self.assertEqual(list(marker.line.width), [3.2, 1.0])
        self.assertEqual(list(marker.size), [13, 10])
        self.assertEqual(len(fig.layout.shapes), 2)
        self.assertTrue(all(shape.type == "line" for shape in fig.layout.shapes))
        self.assertTrue(any(shape.y0 == 0 and shape.y1 == 0 for shape in fig.layout.shapes))
        self.assertTrue(any(shape.x0 == 0 and shape.x1 == 0 for shape in fig.layout.shapes))

    def test_latest_si_vs_future_si_plot_uses_latest_and_projected_si_axes(self) -> None:
        plot_df = pd.DataFrame(
            [
                {
                    "pathogen": "Escherichia coli",
                    "antibiotic": "Cefotaxime",
                    "latest_available_year": 2023,
                    "future_year": 2025,
                    "latest_available_count": 30,
                    "future_predicted_count": 42,
                    "latest_smoothed_SI": 0.1,
                    "future_predicted_SI": 0.22,
                    "future_minus_latest_smoothed_SI": 0.12,
                    "post_fit_count_abs_pct_error_mean": 4.0,
                    "n_count_error_years": 3,
                    "is_glass_pair": True,
                    "glass_group_column": "antibiotic",
                }
            ]
        )

        result = plot_latest_si_vs_future_si(plot_df, show=False)

        fig = result["figure"]
        self.assertEqual(fig.data[0].x[0], 0.1)
        self.assertEqual(fig.data[0].y[0], 0.22)
        self.assertEqual(fig.layout.xaxis.title.text, "Smoothed SI at latest available year")
        self.assertEqual(fig.layout.yaxis.title.text, "Predicted future SI")
        self.assertEqual(len(fig.layout.shapes), 3)
        self.assertTrue(any(shape.xref == "x domain" and shape.yref == "y domain" for shape in fig.layout.shapes))

    def test_latest_count_shift_plot_can_export_pdf(self) -> None:
        rank_df = pd.DataFrame(
            [
                {
                    "pathogen": "Escherichia coli",
                    "antibiotic": "Cefotaxime",
                    "latest_available_year": 2023,
                    "future_year": 2025,
                    "latest_available_count": 30,
                    "latest_available_predicted_SI": 0.1,
                    "future_predicted_SI": 0.22,
                    "future_predicted_count": 42,
                    "post_fit_count_abs_pct_error_mean": 4.0,
                    "n_count_error_years": 3,
                    "status": "ok",
                }
            ]
        )

        with TemporaryDirectory() as tmpdir, patch("plotly.graph_objs._figure.Figure.write_image") as mock_write_image:
            pdf_path = Path(tmpdir) / "latest_observed_count_vs_future_si_shift.pdf"
            result = plot_interactive_latest_observed_count_vs_future_si_shift(
                rank_df,
                min_last_available_count=1,
                min_count_abs_pct_error_mean=0,
                max_count_abs_pct_error_mean=10,
                show=False,
                pdf_path=pdf_path,
            )

        mock_write_image.assert_called_once_with(str(pdf_path), format="pdf", width=1400, height=900)
        self.assertEqual(result["pdf_path"], str(pdf_path))

    def test_latest_count_shift_plot_uses_predictions_when_observed_metrics_are_unavailable(self) -> None:
        rank_df = pd.DataFrame(
            [
                {
                    "pathogen": "Escherichia coli",
                    "antibiotic": "Cefotaxime",
                    "latest_available_year": 2025,
                    "future_year": 2028,
                    "latest_available_count": float("nan"),
                    "latest_available_predicted_count": 30,
                    "latest_available_predicted_SI": 0.1,
                    "future_predicted_SI": 0.22,
                    "future_predicted_count": 42,
                    "post_fit_count_abs_pct_error_mean": float("nan"),
                    "n_count_error_years": 0,
                    "status": "ok",
                }
            ]
        )

        result = plot_interactive_latest_observed_count_vs_future_si_shift(
            rank_df,
            min_last_available_count=10,
            min_count_abs_pct_error_mean=0,
            max_count_abs_pct_error_mean=10,
            show=False,
        )

        self.assertEqual(len(result["plot_df"]), 1)
        self.assertEqual(result["plot_df"].iloc[0]["latest_available_count"], 30)
        self.assertEqual(result["plot_df"].iloc[0]["latest_count_source"], "predicted")
        self.assertEqual(result["predicted_count_fallback_rows"], 1)
        self.assertEqual(result["missing_count_error_rows"], 1)

    def test_latest_count_shift_plot_pdf_export_failure_is_nonfatal(self) -> None:
        rank_df = pd.DataFrame(
            [
                {
                    "pathogen": "Escherichia coli",
                    "antibiotic": "Cefotaxime",
                    "latest_available_year": 2023,
                    "future_year": 2025,
                    "latest_available_count": 30,
                    "latest_available_predicted_SI": 0.1,
                    "future_predicted_SI": 0.22,
                    "future_predicted_count": 42,
                    "post_fit_count_abs_pct_error_mean": 4.0,
                    "n_count_error_years": 3,
                    "status": "ok",
                }
            ]
        )

        with (
            TemporaryDirectory() as tmpdir,
            patch("plotly.graph_objs._figure.Figure.write_image", side_effect=RuntimeError("browser failed")),
            self.assertWarns(RuntimeWarning),
        ):
            result = plot_interactive_latest_observed_count_vs_future_si_shift(
                rank_df,
                min_last_available_count=1,
                min_count_abs_pct_error_mean=0,
                max_count_abs_pct_error_mean=10,
                show=False,
                pdf_path=Path(tmpdir) / "plot.pdf",
            )

        self.assertIsNone(result["pdf_path"])
        self.assertIn("browser failed", result["pdf_error"])


if __name__ == "__main__":
    unittest.main()
