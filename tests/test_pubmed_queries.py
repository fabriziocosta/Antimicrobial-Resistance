import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd
import pytest

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

    def test_query_terms_are_sanitized_before_quoting(self) -> None:
        bacteria_synonyms = pd.DataFrame(
            {
                "bacteria name": ['Bug "bad"[x]'],
                "bacteria synonym": ['Alias "bad"[y]'],
            }
        )
        antibiotic_synonyms = pd.DataFrame(
            {
                "antibiotic name": ['Drug "bad"[x]'],
                "antibiotic synonym": ['Other "bad"[y]'],
            }
        )

        query = build_antibiotic_pathogen_query(
            'Drug "bad"[x]',
            'Bug "bad"[x]',
            bacteria_synonyms,
            antibiotic_synonym_df=antibiotic_synonyms,
        )

        self.assertIn('"Drug bad x"[Abstract]', query)
        self.assertIn('"Other bad y"[Abstract]', query)
        self.assertIn('"Bug bad x"[Abstract]', query)
        self.assertIn('"Alias bad y"[Abstract]', query)
        self.assertNotIn('"bad"[x]', query)

    def test_metadata_refresh_fetches_missing_pmids_as_one_total_job(self) -> None:
        pytest.importorskip("Bio")
        from amr.pubmed.yearly import refresh_pubmed_metadata_cache

        calls = []

        def fake_fetch(pmids, *, email, batch_size, on_batch_complete=None):
            calls.append({"pmids": list(pmids), "email": email, "batch_size": batch_size})
            total_batches = 2
            batches = [list(pmids)[:2], list(pmids)[2:]]
            for batch_idx, batch in enumerate(batches, start=1):
                date_map = {pmid: pd.Timestamp(f"202{batch_idx}-01-01") for pmid in batch}
                venue_map = {pmid: f"Journal {batch_idx}" for pmid in batch}
                if on_batch_complete is not None:
                    on_batch_complete(batch_idx, total_batches, batch, date_map, venue_map)
            return {}, {}

        with TemporaryDirectory() as tmpdir, patch("amr.pubmed.yearly.fetch_pub_metadata", side_effect=fake_fetch):
            cache_path = Path(tmpdir) / "metadata.parquet"
            out = refresh_pubmed_metadata_cache(
                ["101", "102", "103"],
                cache_path=cache_path,
                email="person@example.com",
                batch_size=2,
                checkpoint_every_batches=1,
            )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["pmids"], ["101", "102", "103"])
        self.assertEqual(calls[0]["batch_size"], 2)
        self.assertEqual(set(out["pmid"]), {"101", "102", "103"})

    def test_background_count_metadata_is_consolidated_across_windows(self) -> None:
        pytest.importorskip("Bio")
        from amr.pubmed import intervals

        update_calls = []

        def fake_search(term_clause, *, min_date, max_date, email):
            return f"query:{term_clause}:{min_date}:{max_date}", 2, "webenv", "query_key"

        def fake_retrieve(query, webenv, query_key, count, batch_size=5000):
            return ["101", "102"] if "bug1" in query else ["201", "202"]

        def fake_update(pmids, metadata_cache, email, batch_size=200):
            update_calls.append({"pmids": list(pmids), "batch_size": batch_size})
            pmid_years = {"101": 2020, "102": 2021, "201": 2020, "202": 2021}
            for pmid in pmids:
                year = pmid_years[str(pmid)]
                metadata_cache[pmid] = {"date": pd.Timestamp(year=year, month=1, day=1), "venue": "Journal"}

        with TemporaryDirectory() as tmpdir:
            with (
                patch("amr.pubmed.intervals.search_term_clause", side_effect=fake_search),
                patch("amr.pubmed.intervals.retrieve_pmids", side_effect=fake_retrieve),
                patch("amr.pubmed.intervals.update_metadata_cache", side_effect=fake_update),
            ):
                out = intervals._count_pmids_per_year_resumable(
                    entity_values=["bug1", "bug2"],
                    entity_col="pathogen",
                    clause_builder=lambda value: value,
                    output_path=str(Path(tmpdir) / "background.parquet"),
                    start_year=2020,
                    end_year=2021,
                    email="person@example.com",
                    save_every=200,
                    progress_every=0,
                    fetch_threshold=10,
                    metadata_batch_size=2,
                )

        self.assertEqual(len(update_calls), 1)
        self.assertEqual(
            update_calls[0],
            {"pmids": ["101", "102", "201", "202"], "batch_size": 2},
        )
        self.assertEqual(int(out["count"].sum()), 4)
        self.assertEqual(set(out["pathogen"]), {"bug1", "bug2"})


if __name__ == "__main__":
    unittest.main()
