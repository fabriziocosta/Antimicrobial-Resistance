"""PubMed query builders and retrieval helpers."""

from .queries import (
    prepare_antibiotic_pubmed_query,
    prepare_antibiotic_pubmed_query_tiab,
    expand_synonyms,
    build_pathogen_tiab_query,
    build_antibiotic_pathogen_query,
)
from .search import (
    parse_pubdate,
    esearch_count_for_year,
    search_pmids,
    retrieve_pmids,
    fetch_pub_metadata,
)
from .cache import (
    update_metadata_cache,
    get_metadata,
    merge_with_existing,
)

__all__ = [
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
    "update_metadata_cache",
    "get_metadata",
    "merge_with_existing",
]

