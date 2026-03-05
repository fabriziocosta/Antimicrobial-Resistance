"""Utilities for antimicrobial resistance analysis notebooks."""

from . import pubmed, pipelines

from .pipelines.yearly_collection import (
    collect_yearly_amr_dataset,
    collect_yearly_background_counts,
)
from .pipelines.scientific_interest import (
    build_yearly_si_dataset,
    rank_future_high_interest_pairs,
)

__all__ = [
    "pubmed",
    "pipelines",
    # active pipeline functions
    "collect_yearly_amr_dataset",
    "collect_yearly_background_counts",
    "build_yearly_si_dataset",
    "rank_future_high_interest_pairs",
]
