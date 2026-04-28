from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from amr.glass import load_and_clean_glass_list
from amr.pipelines.glass_si import (
    build_glass_match_table,
    build_novel_prediction_table,
    plot_future_si_vs_si_shift,
)


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

        result = plot_future_si_vs_si_shift(
            plot_df,
            min_count_abs_pct_error_mean=0,
            max_count_abs_pct_error_mean=10,
            display_name=True,
            max_text_labels=2,
            show=False,
        )
        fig = result["figure"]
        marker = fig.data[0].marker
        self.assertEqual(list(marker.line.color), ["#000000", "#4b5563"])
        self.assertEqual(list(marker.line.width), [3.2, 1.0])
        self.assertEqual(list(marker.size), [13, 10])
        self.assertEqual(len(fig.layout.shapes), 2)
        self.assertTrue(all(shape.type == "line" for shape in fig.layout.shapes))
        self.assertTrue(any(shape.y0 == 0 and shape.y1 == 0 for shape in fig.layout.shapes))
        self.assertTrue(any(shape.x0 == 0 and shape.x1 == 0 for shape in fig.layout.shapes))


if __name__ == "__main__":
    unittest.main()
