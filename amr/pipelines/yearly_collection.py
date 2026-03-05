"""Yearly AMR collection pipelines built on top of interval PubMed retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from .collection import (
    count_pmids_per_year_for_antibiotics_resumable,
    count_pmids_per_year_for_pathogens_resumable,
    fetch_pmids_intervals,
)
from ..pubmed.yearly import (
    aggregate_pair_counts_by_year,
    attach_pubmed_metadata,
    explode_positive_pair_pmids,
    refresh_pubmed_metadata_cache,
)


def collect_yearly_background_counts(
    *,
    pathogens: Sequence[str],
    antibiotics: Sequence[str],
    bacteria_synonym_df: pd.DataFrame,
    start_year: int,
    end_year: int,
    email: str,
    pathogen_output_path: str | Path = "pathogen_tiab_counts.parquet",
    antibiotic_output_path: str | Path = "antibiotic_tiab_counts.parquet",
    save_every: int = 200,
    progress_every: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collect yearly background publication counts for pathogens and antibiotics."""
    pathogen_counts_df = count_pmids_per_year_for_pathogens_resumable(
        pathogens=pathogens,
        bacteria_synonym_df=bacteria_synonym_df,
        start_year=start_year,
        end_year=end_year,
        email=email,
        output_path=str(pathogen_output_path),
        save_every=save_every,
        progress_every=progress_every,
    )
    antibiotic_counts_df = count_pmids_per_year_for_antibiotics_resumable(
        antibiotics=antibiotics,
        start_year=start_year,
        end_year=end_year,
        email=email,
        output_path=str(antibiotic_output_path),
        save_every=save_every,
        progress_every=progress_every,
    )
    return pathogen_counts_df, antibiotic_counts_df


def collect_yearly_amr_dataset(
    *,
    pathogens: Sequence[str],
    antibiotics: Sequence[str],
    bacteria_synonym_df: pd.DataFrame,
    min_date: str,
    max_date: str,
    email: str,
    interval_year_gap: int = 5,
    intervals_output_path: str | Path = "pmids_results.parquet",
    yearly_pair_pmid_output_path: str | Path = "yearly_pair_pmid.parquet",
    metadata_cache_path: str | Path = "pubmed_pmid_metadata.parquet",
    output_path: str | Path = "yearly_amr_pairs.parquet",
    save_every: int = 50,
    extra_keywords: Sequence[str] | None = None,
    full_fetch_threshold: int = 2000,
    summary_batch_size: int = 200,
    metadata_batch_size: int = 200,
    metadata_checkpoint_every_batches: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the canonical yearly pair dataset from interval PMIDs and publication dates.

    Returns ``(yearly_df, pair_pmid_df, enriched_pair_pmid_df)``.
    """
    interval_df = fetch_pmids_intervals(
        pathogens=pathogens,
        antibiotics=antibiotics,
        bacteria_synonym_df=bacteria_synonym_df,
        min_date=min_date,
        max_date=max_date,
        year_gap=interval_year_gap,
        email=email,
        parquet_path=str(intervals_output_path),
        save_every=save_every,
        extra_keywords=extra_keywords,
        full_fetch_threshold=full_fetch_threshold,
        summary_batch_size=summary_batch_size,
    )

    pair_pmid_df = explode_positive_pair_pmids(interval_df)
    Path(yearly_pair_pmid_output_path).parent.mkdir(parents=True, exist_ok=True)
    pair_pmid_df.to_parquet(yearly_pair_pmid_output_path, index=False)

    metadata_df = refresh_pubmed_metadata_cache(
        pair_pmid_df["pmid"].astype(str).tolist(),
        cache_path=metadata_cache_path,
        email=email,
        batch_size=metadata_batch_size,
        checkpoint_every_batches=metadata_checkpoint_every_batches,
    )
    enriched_df = attach_pubmed_metadata(pair_pmid_df, metadata_df)
    yearly_df = aggregate_pair_counts_by_year(enriched_df)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    yearly_df.to_parquet(output_path, index=False)
    return yearly_df, pair_pmid_df, enriched_df


__all__ = [
    "collect_yearly_background_counts",
    "collect_yearly_amr_dataset",
]
