"""Compatibility layer for legacy imports from amr.data_collection.

This module forwards to modular subpackages under:
- amr.pubmed (queries/search/cache)
- amr.features (venues)
- amr.pipelines (collection orchestrators)
- amr.visualization (plotting)

Existing notebooks can keep using amr.data_collection; a deprecation
warning is emitted once per process.
"""

from __future__ import annotations

import warnings

from amr.config import (
    EUTILS_RETRIES as RETRIES,
    EUTILS_BACKOFF_FACTOR as BACKOFF_FACTOR,
    EUTILS_RATE_LIMIT_DELAY as RATE_LIMIT_DELAY,
)
from amr.pubmed.queries import (
    prepare_antibiotic_pubmed_query,
    prepare_antibiotic_pubmed_query_tiab,
    expand_synonyms,
    build_pathogen_tiab_query,
    build_antibiotic_pathogen_query,
)
from amr.pubmed.search import (
    parse_pubdate,
    esearch_count_for_year,
    search_pmids,
    retrieve_pmids,
    fetch_pub_metadata,
)
from amr.pubmed.cache import (
    merge_with_existing,
    update_metadata_cache,
    get_metadata,
)
from amr.features.venues import add_venue_vectors
from amr.visualization import plot_venue_vector
from amr.pipelines.collection import (
    fetch_pmids_by_date_and_keywords,
    fetch_pmids_intervals,
    count_pmids_per_year_for_pathogens,
    count_pmids_per_year_for_antibiotics,
)

_warned = False


def _maybe_warn():
    global _warned
    if not _warned:
        warnings.warn(
            "amr.data_collection is deprecated; use amr.pubmed.*, amr.features.venues, "
            "amr.pipelines.collection, and amr.visualization instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        _warned = True


_maybe_warn()


__all__ = [
    "RETRIES",
    "BACKOFF_FACTOR",
    "RATE_LIMIT_DELAY",
    # queries
    "prepare_antibiotic_pubmed_query",
    "prepare_antibiotic_pubmed_query_tiab",
    "expand_synonyms",
    "build_pathogen_tiab_query",
    "build_antibiotic_pathogen_query",
    # search
    "parse_pubdate",
    "esearch_count_for_year",
    "search_pmids",
    "retrieve_pmids",
    "fetch_pub_metadata",
    # cache
    "merge_with_existing",
    "update_metadata_cache",
    "get_metadata",
    # features/visualization
    "add_venue_vectors",
    "plot_venue_vector",
    # pipelines
    "fetch_pmids_by_date_and_keywords",
    "fetch_pmids_intervals",
    "count_pmids_per_year_for_pathogens",
    "count_pmids_per_year_for_antibiotics",
]

