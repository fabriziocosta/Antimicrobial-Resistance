"""Utilities for antimicrobial resistance analysis notebooks."""

from . import pubmed, features, pipelines, debiasing, visualization, predictive

# Re-export common entry points for convenience
from .pipelines.collection import (
    fetch_pmids_intervals,
    fetch_pmids_by_date_and_keywords,
    count_pmids_per_year_for_pathogens,
    count_pmids_per_year_for_antibiotics,
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
    "build_pathogen_tiab_query",
    "prepare_antibiotic_pubmed_query_tiab",
    "esearch_count_for_year",
    "add_venue_vectors",
]
