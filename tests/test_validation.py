import unittest

import pandas as pd

from amr.validation import (
    clean_background_counts,
    clean_interval_results,
    clean_pubmed_metadata_df,
    clean_yearly_pair_counts,
)


class ValidationTests(unittest.TestCase):
    def test_interval_results_recompute_counts_and_drop_bad_shape(self) -> None:
        source = pd.DataFrame(
            [
                {
                    "pathogen": " Salmonella enterica ",
                    "antibiotic": "Amikacin",
                    "start_date": "2020/01/01",
                    "end_date": "2024/12/31",
                    "pmids": ["111", "nan", None, "PMID:222", "111"],
                    "count": 99,
                    "venue_histogram": {"Journal": 2},
                },
                {
                    "pathogen": "nan",
                    "antibiotic": "Amikacin",
                    "start_date": "2020/01/01",
                    "end_date": "2024/12/31",
                    "pmids": ["333"],
                    "count": 1,
                },
                {
                    "pathogen": "p1",
                    "antibiotic": "a1",
                    "start_date": "2025/01/01",
                    "end_date": "2024/12/31",
                    "pmids": ["444"],
                    "count": 1,
                },
            ]
        )

        out = clean_interval_results(source)

        self.assertEqual(len(out), 1)
        self.assertEqual(out.loc[0, "pathogen"], "Salmonella enterica")
        self.assertEqual(out.loc[0, "pmids"], ["111", "222"])
        self.assertEqual(out.loc[0, "count"], 2)

    def test_metadata_requires_valid_pmid_and_year(self) -> None:
        source = pd.DataFrame(
            {
                "pmid": ["111", "nan", "222", "333"],
                "pub_date": ["2020-01-01", "2021-01-01", pd.NaT, "1800-01-01"],
                "pub_year": [pd.NA, 2021, pd.NA, 1800],
                "venue": ["", "Bad", "Missing date", None],
            }
        )

        out = clean_pubmed_metadata_df(source)

        self.assertEqual(out["pmid"].tolist(), ["111"])
        self.assertEqual(int(out.loc[0, "pub_year"]), 2020)
        self.assertEqual(out.loc[0, "venue"], "Unknown")

    def test_yearly_counts_clean_pmids_and_recompute_count(self) -> None:
        source = pd.DataFrame(
            {
                "pathogen": ["p1", "p2"],
                "antibiotic": ["a1", "a2"],
                "year": [2020, 2020],
                "count": [99, 1],
                "pmids": [["111", "nan", "112"], []],
            }
        )

        out = clean_yearly_pair_counts(source)

        self.assertEqual(len(out), 1)
        self.assertEqual(out.loc[0, "count"], 2)
        self.assertEqual(out.loc[0, "pmids"], ["111", "112"])

    def test_background_counts_drop_invalid_keys_and_clip_counts(self) -> None:
        source = pd.DataFrame(
            {
                "pathogen": ["p1", "nan", "p2"],
                "year": [2020, 2020, 2050],
                "count": [-1, 3, 4],
            }
        )

        out = clean_background_counts(source, key_col="pathogen")

        self.assertEqual(len(out), 1)
        self.assertEqual(out.loc[0, "pathogen"], "p1")
        self.assertEqual(out.loc[0, "count"], 0)


if __name__ == "__main__":
    unittest.main()
