"""Utilities for antimicrobial resistance analysis notebooks."""

from . import pubmed, features, pipelines, debiasing, visualization, predictive

# Re-export common entry points for convenience
from .pipelines.collection import (
    fetch_pmids_intervals,
    fetch_pmids_by_date_and_keywords,
    count_pmids_per_year_for_pathogens,
    count_pmids_per_year_for_antibiotics,
)
from .pipelines.yearly_collection import (
    collect_yearly_amr_dataset,
    collect_yearly_background_counts,
)
from .pipelines.scientific_interest import (
    build_yearly_si_dataset,
    rank_future_high_interest_pairs,
)
from .pubmed.queries import (
    build_pathogen_tiab_query,
    prepare_antibiotic_pubmed_query_tiab,
)
from .pubmed.search import esearch_count_for_year
from .features.venues import add_venue_vectors
__all__ = [
    "pubmed",
    "features",
    "pipelines",
    "debiasing",
    "visualization",
    "predictive",
    # common functions
    "fetch_pmids_intervals",
    "fetch_pmids_by_date_and_keywords",
    "count_pmids_per_year_for_pathogens",
    "count_pmids_per_year_for_antibiotics",
    "collect_yearly_amr_dataset",
    "collect_yearly_background_counts",
    "build_yearly_si_dataset",
    "rank_future_high_interest_pairs",
    "build_pathogen_tiab_query",
    "prepare_antibiotic_pubmed_query_tiab",
    "esearch_count_for_year",
    "add_venue_vectors",
]
