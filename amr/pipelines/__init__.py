"""Pipeline entry points for yearly collection and scientific-interest workflows."""

__all__ = [
    "count_pmids_per_year_for_antibiotics",
    "count_pmids_per_year_for_antibiotics_resumable",
    "count_pmids_per_year_for_pathogens",
    "count_pmids_per_year_for_pathogens_resumable",
    "fetch_pmids_by_date_and_keywords",
    "fetch_pmids_intervals",
    "collect_yearly_amr_dataset",
    "collect_yearly_background_counts",
    "build_yearly_si_dataset",
    "rank_future_high_interest_pairs",
]

_COLLECTION_EXPORTS = {
    "count_pmids_per_year_for_antibiotics",
    "count_pmids_per_year_for_antibiotics_resumable",
    "count_pmids_per_year_for_pathogens",
    "count_pmids_per_year_for_pathogens_resumable",
    "fetch_pmids_by_date_and_keywords",
    "fetch_pmids_intervals",
}
_YEARLY_COLLECTION_EXPORTS = {
    "collect_yearly_amr_dataset",
    "collect_yearly_background_counts",
}
_SI_EXPORTS = {"build_yearly_si_dataset", "rank_future_high_interest_pairs"}


def __getattr__(name: str):
    from importlib import import_module

    if name in _COLLECTION_EXPORTS:
        collection = import_module(".collection", __name__)
        return getattr(collection, name)
    if name in _YEARLY_COLLECTION_EXPORTS:
        yearly_collection = import_module(".yearly_collection", __name__)
        return getattr(yearly_collection, name)
    if name in _SI_EXPORTS:
        scientific_interest = import_module(".scientific_interest", __name__)
        return getattr(scientific_interest, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
