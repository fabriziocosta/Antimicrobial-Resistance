"""Pipeline entry points for yearly collection and scientific-interest workflows."""

from .collection import (
    count_pmids_per_year_for_antibiotics,
    count_pmids_per_year_for_antibiotics_resumable,
    count_pmids_per_year_for_pathogens,
    count_pmids_per_year_for_pathogens_resumable,
    fetch_pmids_by_date_and_keywords,
    fetch_pmids_intervals,
)
from .scientific_interest import (
    build_yearly_si_dataset,
    rank_future_high_interest_pairs,
)
from .yearly_collection import (
    collect_yearly_amr_dataset,
    collect_yearly_background_counts,
)

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
