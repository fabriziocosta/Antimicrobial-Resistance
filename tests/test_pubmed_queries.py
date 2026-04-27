import unittest

import pandas as pd

from amr.pubmed.queries import (
    build_antibiotic_pathogen_query,
    build_antibiotic_query,
    build_pathogen_tiab_query,
    expand_antibiotic_synonyms,
)


class PubMedQueryTests(unittest.TestCase):
    def test_build_pair_query_uses_bacteria_and_antibiotic_synonyms(self) -> None:
        bacteria_synonyms = pd.DataFrame(
            {
                "bacteria name": ["Escherichia coli", "Escherichia coli"],
                "bacteria synonym": ["E. coli", "Bacterium coli"],
            }
        )
        antibiotic_synonyms = pd.DataFrame(
            {
                "antibiotic name": ["Colistin", "Colistin"],
                "antibiotic synonym": ["Colistin", "Polymyxin E"],
            }
        )

        query = build_antibiotic_pathogen_query(
            "Colistin",
            "Escherichia coli",
            bacteria_synonyms,
            antibiotic_synonym_df=antibiotic_synonyms,
            extra_clauses=["resistance[Title/Abstract]"],
            min_date="2020/01/01",
            max_date="2024/12/31",
        )

        self.assertIn('"Escherichia coli"[Abstract]', query)
        self.assertIn('"E. coli"[Abstract]', query)
        self.assertIn('"Bacterium coli"[Abstract]', query)
        self.assertIn('"Colistin"[Abstract]', query)
        self.assertIn('"Polymyxin E"[Abstract]', query)
        self.assertIn("resistance[Title/Abstract]", query)
        self.assertIn('"2020/01/01"[Date - Publication] : "2024/12/31"[Date - Publication]', query)

    def test_build_antibiotic_query_expands_split_variants(self) -> None:
        antibiotic_synonyms = pd.DataFrame(
            {
                "antibiotic name": ["Amoxicillin/clavulanic-acid"],
                "antibiotic synonym": ["Amoxicillin-clavulanic-acid"],
            }
        )

        query = build_antibiotic_query(
            "Amoxicillin/clavulanic-acid",
            antibiotic_synonyms,
            field="Title/Abstract",
        )

        self.assertIn('"Amoxicillin/clavulanic-acid"[Title/Abstract]', query)
        self.assertIn('"Amoxicillin-clavulanic-acid"[Title/Abstract]', query)
        self.assertIn('"Amoxicillin"[Title/Abstract]', query)
        self.assertIn('"clavulanic"[Title/Abstract]', query)
        self.assertIn('"acid"[Title/Abstract]', query)

    def test_antibiotic_synonym_expansion_falls_back_to_canonical(self) -> None:
        self.assertEqual(expand_antibiotic_synonyms("Amikacin", None), ["Amikacin"])

    def test_pathogen_query_uses_canonical_name_when_synonyms_are_missing(self) -> None:
        bacteria_synonyms = pd.DataFrame(columns=["bacteria name", "bacteria synonym"])

        query = build_pathogen_tiab_query("Salmonella enterica", bacteria_synonyms)

        self.assertEqual(query, '("Salmonella enterica"[Title/Abstract])')


if __name__ == "__main__":
    unittest.main()
