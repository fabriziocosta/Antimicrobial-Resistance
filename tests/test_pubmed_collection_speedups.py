from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from amr.pubmed.intervals import _collect_pair_raw_results, fetch_pmids_intervals
from amr.pubmed.yearly import load_pubmed_metadata_cache


def test_partial_cache_queries_only_the_missing_interval() -> None:
    missing_entry = {
        "start": "2025/01/01",
        "end": "2026/12/31",
        "start_dt": datetime(2025, 1, 1),
        "end_dt": datetime(2026, 12, 31),
    }
    with (
        patch("amr.pubmed.intervals.fetch_pmids_by_date_and_keywords", return_value=["123"]) as fetch,
        patch("amr.pubmed.intervals.search_pmids") as search,
    ):
        result = _collect_pair_raw_results(
            pathogen="p1",
            antibiotic="a1",
            missing_entries=[missing_entry],
            bacteria_synonym_df=pd.DataFrame(),
            antibiotic_synonym_df=None,
            email="person@example.com",
            min_date="2000/01/01",
            max_date="2026/12/31",
            extra_keywords=None,
            full_fetch_threshold=9500,
            retrieve_batch_size=9000,
            partial_interval_max=1,
            all_intervals_missing=False,
            verbose=False,
            search_recovery_attempts=3,
            search_recovery_delay_seconds=0,
        )

    assert result["mode"] == "interval"
    assert result["entries"][0]["pmids"] == ["123"]
    fetch.assert_called_once()
    search.assert_not_called()


def test_interval_collection_persists_metadata_for_the_next_stage() -> None:
    with TemporaryDirectory() as tmpdir:
        interval_path = Path(tmpdir) / "intervals.parquet"
        metadata_path = Path(tmpdir) / "metadata.parquet"

        def fake_update(pmids, metadata_cache, *, email, batch_size, verbose):
            for pmid in pmids:
                metadata_cache[str(pmid)] = {
                    "date": datetime(2020, 1, 1),
                    "venue": "Journal",
                }

        with (
            patch(
                "amr.pubmed.intervals.search_pmids",
                return_value=("query", 1, "webenv", "query_key"),
            ),
            patch("amr.pubmed.intervals.retrieve_pmids", return_value=["123"]),
            patch("amr.pubmed.intervals.update_metadata_cache", side_effect=fake_update),
        ):
            result = fetch_pmids_intervals(
                pathogens=["p1"],
                antibiotics=["a1"],
                bacteria_synonym_df=pd.DataFrame(),
                min_date="2020/01/01",
                max_date="2020/12/31",
                parquet_path=str(interval_path),
                metadata_cache_path=str(metadata_path),
                max_workers=1,
                verbose=0,
            )

        metadata = load_pubmed_metadata_cache(metadata_path)

    assert len(result) == 1
    assert set(metadata["pmid"]) == {"123"}
    assert metadata.iloc[0]["pub_year"] == 2020

