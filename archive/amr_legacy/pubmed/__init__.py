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
from .yearly import (
    explode_positive_pair_pmids,
    load_pubmed_metadata_cache,
    save_pubmed_metadata_cache,
    refresh_pubmed_metadata_cache,
    attach_pubmed_metadata,
    aggregate_pair_counts_by_year,
    summarize_metadata_coverage,
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
    # yearly helpers
    "explode_positive_pair_pmids",
    "load_pubmed_metadata_cache",
    "save_pubmed_metadata_cache",
    "refresh_pubmed_metadata_cache",
    "attach_pubmed_metadata",
    "aggregate_pair_counts_by_year",
    "summarize_metadata_coverage",
]
