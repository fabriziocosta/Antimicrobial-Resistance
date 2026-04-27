"""PubMed query builders and retrieval helpers."""

__all__ = [
    # queries
    "prepare_antibiotic_pubmed_query",
    "prepare_antibiotic_pubmed_query_tiab",
    "expand_synonyms",
    "expand_antibiotic_synonyms",
    "build_antibiotic_query",
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

_QUERY_EXPORTS = {
    "prepare_antibiotic_pubmed_query",
    "prepare_antibiotic_pubmed_query_tiab",
    "expand_synonyms",
    "expand_antibiotic_synonyms",
    "build_antibiotic_query",
    "build_pathogen_tiab_query",
    "build_antibiotic_pathogen_query",
}
_SEARCH_EXPORTS = {
    "parse_pubdate",
    "esearch_count_for_year",
    "search_pmids",
    "retrieve_pmids",
    "fetch_pub_metadata",
}
_CACHE_EXPORTS = {"update_metadata_cache", "get_metadata", "merge_with_existing"}
_YEARLY_EXPORTS = {
    "explode_positive_pair_pmids",
    "load_pubmed_metadata_cache",
    "save_pubmed_metadata_cache",
    "refresh_pubmed_metadata_cache",
    "attach_pubmed_metadata",
    "aggregate_pair_counts_by_year",
    "summarize_metadata_coverage",
}


def __getattr__(name: str):
    """Load retrieval dependencies only when retrieval helpers are requested."""
    from importlib import import_module

    if name in _QUERY_EXPORTS:
        queries = import_module(".queries", __name__)
        return getattr(queries, name)
    if name in _SEARCH_EXPORTS:
        search = import_module(".search", __name__)
        return getattr(search, name)
    if name in _CACHE_EXPORTS:
        cache = import_module(".cache", __name__)
        return getattr(cache, name)
    if name in _YEARLY_EXPORTS:
        yearly = import_module(".yearly", __name__)
        return getattr(yearly, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
