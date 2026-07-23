from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from amr.pubmed.intervals import fetch_pmids_intervals


def test_interval_collection_reports_scoped_progress_and_hides_request_noise(capsys):
    with TemporaryDirectory() as tmpdir, patch(
        "amr.pubmed.intervals.search_pmids",
        return_value=("query", 0, None, None),
    ) as search:
        result = fetch_pmids_intervals(
            pathogens=["p1", "p2"],
            antibiotics=["a1", "a2"],
            bacteria_synonym_df=pd.DataFrame(),
            min_date="2020/01/01",
            max_date="2021/12/31",
            parquet_path=f"{tmpdir}/intervals.parquet",
            verbose=1,
            progress_every=1,
        )

    output = capsys.readouterr().out
    assert "pathogens 0/2 (0.0%)" in output
    assert "current p1" in output
    assert "antibiotics 1/2 (50.0%)" in output
    assert "pairs 4/4 (100.0%)" in output
    assert "pair-years 8/8 (100.0%)" in output
    assert "[search_pmids]" not in output
    assert len(result) == 4
    assert all(call.kwargs["verbose"] is False for call in search.call_args_list)


def test_interval_collection_resumes_entire_cache_without_searching():
    with TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/intervals.parquet"
        with patch("amr.pubmed.intervals.search_pmids", return_value=("query", 0, None, None)):
            first = fetch_pmids_intervals(
                pathogens=["p1", "p2"],
                antibiotics=["a1", "a2"],
                bacteria_synonym_df=pd.DataFrame(),
                min_date="2020/01/01",
                max_date="2021/12/31",
                parquet_path=path,
                verbose=0,
            )

        with patch("amr.pubmed.intervals.search_pmids") as search:
            resumed = fetch_pmids_intervals(
                pathogens=["p1", "p2"],
                antibiotics=["a1", "a2"],
                bacteria_synonym_df=pd.DataFrame(),
                min_date="2020/01/01",
                max_date="2021/12/31",
                parquet_path=path,
                verbose=0,
            )

        assert search.call_count == 0
        assert len(resumed) == len(first) == 4


def test_interval_collection_recovers_failed_pair_search(capsys):
    with TemporaryDirectory() as tmpdir, patch(
        "amr.pubmed.intervals.search_pmids",
        side_effect=[
            ("query", None, None, None),
            ("query", None, None, None),
            ("query", 0, None, None),
        ],
    ) as search, patch("amr.pubmed.intervals.time.sleep") as sleep:
        result = fetch_pmids_intervals(
            pathogens=["p1"],
            antibiotics=["a1"],
            bacteria_synonym_df=pd.DataFrame(),
            min_date="2020/01/01",
            max_date="2021/12/31",
            parquet_path=f"{tmpdir}/intervals.parquet",
            verbose=1,
            search_recovery_attempts=3,
            search_recovery_delay_seconds=0.01,
        )

    output = capsys.readouterr().out
    assert "PubMed recovery" in output
    assert "search round 1/3 failed" in output
    assert "search round 2/3 failed" in output
    assert search.call_count == 3
    assert [call.args[0] for call in sleep.call_args_list] == [0.01, 0.02]
    assert len(result) == 1
