from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

import pandas as pd

from amr.pubmed.search import _retry_delay, search_pmids, search_term_clause


def _entrez_handle():
    handle = MagicMock()
    handle.close = MagicMock()
    return handle


@patch("amr.pubmed.search.time.sleep")
@patch("amr.pubmed.search.Entrez.read")
@patch("amr.pubmed.search.Entrez.esearch")
def test_search_term_clause_retries_incomplete_history(esearch, read, _sleep):
    esearch.side_effect = [_entrez_handle(), _entrez_handle()]
    read.side_effect = [
        {"Count": "2"},
        {"Count": "2", "WebEnv": "env", "QueryKey": "1"},
    ]

    _, count, webenv, query_key = search_term_clause(
        "resistance[Title/Abstract]",
        min_date="2000/01/01",
        max_date="2026/12/31",
        email="test@example.com",
    )

    assert (count, webenv, query_key) == (2, "env", "1")
    assert esearch.call_count == 2


@patch("amr.pubmed.search.build_antibiotic_pathogen_query", return_value="query")
@patch("amr.pubmed.search.time.sleep")
@patch("amr.pubmed.search.Entrez.read")
@patch("amr.pubmed.search.Entrez.esearch")
def test_search_pmids_retries_incomplete_history(esearch, read, _sleep, _build_query):
    esearch.side_effect = [_entrez_handle(), _entrez_handle()]
    read.side_effect = [
        {"Count": "1", "WebEnv": "env"},
        {"Count": "1", "WebEnv": "env", "QueryKey": "1"},
    ]

    _, count, webenv, query_key = search_pmids(
        "Minocycline",
        "Leptospira interrogans",
        pd.DataFrame(),
        extra_clauses=None,
        min_date="2000/01/01",
        max_date="2026/12/31",
        email="test@example.com",
    )

    assert (count, webenv, query_key) == (1, "env", "1")
    assert esearch.call_count == 2


def test_http_429_uses_retry_after_and_longer_backoff():
    error = HTTPError(
        "https://eutils.ncbi.nlm.nih.gov/",
        429,
        "Too Many Requests",
        {"Retry-After": "47"},
        None,
    )

    assert _retry_delay(error, 0) == 47
    assert _retry_delay(error, 1) == 60
