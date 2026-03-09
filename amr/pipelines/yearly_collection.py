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

def _merge_pair_pmid_frames(existing_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    if existing_df.empty:
        return new_df.copy()
    if new_df.empty:
        return existing_df.copy()
    merged = pd.concat([existing_df, new_df], ignore_index=True)
    key_cols = ["pathogen", "antibiotic", "pmid"]
    return merged.drop_duplicates(subset=key_cols, keep="last").reset_index(drop=True)


def _as_pair_pmid_keyframe(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    for col in ["pathogen", "antibiotic", "pmid"]:
        work[col] = work[col].astype(str)
    return work[["pathogen", "antibiotic", "pmid"]].drop_duplicates().reset_index(drop=True)


def _extract_new_pair_pmids(existing_pair_df: pd.DataFrame, current_pair_df: pd.DataFrame) -> pd.DataFrame:
    """Return only pair-PMID rows not present in existing storage."""
    existing_keys = _as_pair_pmid_keyframe(existing_pair_df)
    current_keys = _as_pair_pmid_keyframe(current_pair_df)
    if existing_keys.empty:
        return current_pair_df.copy()
    merged = current_keys.merge(
        existing_keys,
        on=["pathogen", "antibiotic", "pmid"],
        how="left",
        indicator=True,
    )
    new_keys = merged.loc[merged["_merge"] == "left_only", ["pathogen", "antibiotic", "pmid"]]
    if new_keys.empty:
        return current_pair_df.iloc[0:0].copy()
    return current_pair_df.merge(new_keys, on=["pathogen", "antibiotic", "pmid"], how="inner")


def _merge_yearly_counts_incremental(existing_df: pd.DataFrame, delta_df: pd.DataFrame) -> pd.DataFrame:
    """Incrementally merge yearly pair counts by unioning PMID lists per key."""
    if existing_df.empty:
        out = delta_df.copy()
        out["pmids"] = out["pmids"].apply(lambda v: v if isinstance(v, list) else [])
        out["count"] = out["pmids"].apply(len)
        return out.sort_values(["pathogen", "antibiotic", "year"]).reset_index(drop=True)
    if delta_df.empty:
        return existing_df.copy().sort_values(["pathogen", "antibiotic", "year"]).reset_index(drop=True)

    key_cols = ["pathogen", "antibiotic", "year"]
    work_existing = existing_df.copy()
    work_delta = delta_df.copy()
    work_existing["year"] = pd.to_numeric(work_existing["year"], errors="coerce").astype("Int64")
    work_delta["year"] = pd.to_numeric(work_delta["year"], errors="coerce").astype("Int64")
    work_existing = work_existing.dropna(subset=["year"]).copy()
    work_delta = work_delta.dropna(subset=["year"]).copy()
    work_existing["year"] = work_existing["year"].astype(int)
    work_delta["year"] = work_delta["year"].astype(int)
    work_existing["pmids"] = work_existing["pmids"].apply(lambda v: v if isinstance(v, list) else [])
    work_delta["pmids"] = work_delta["pmids"].apply(lambda v: v if isinstance(v, list) else [])
    combined = pd.concat([work_existing, work_delta], ignore_index=True)
    merged = (
        combined.groupby(key_cols, as_index=False)
        .agg(
            pmids=("pmids", lambda rows: sorted({str(p) for row in rows for p in row})),
            count=("count", "max"),
        )
        .sort_values(key_cols)
        .reset_index(drop=True)
    )
    merged["count"] = merged["pmids"].apply(len)
    return merged


def build_collection_preflight_summary(
    *,
    pathogens: Sequence[str],
    antibiotics: Sequence[str],
    min_date: str,
    max_date: str,
    interval_year_gap: int,
    intervals_output_path: str | Path,
    yearly_pair_pmid_path: str | Path,
    yearly_output_path: str | Path,
    pathogen_bg_path: str | Path,
    antibiotic_bg_path: str | Path,
    background_start_year: int,
    background_end_year: int,
) -> dict[str, object]:
    """Summarize cached yearly-collection artifacts for a given run window."""
    overall_start_dt = pd.to_datetime(min_date, format="%Y/%m/%d")
    overall_end_dt = pd.to_datetime(max_date, format="%Y/%m/%d")

    intervals: list[tuple[str, str]] = []
    current = overall_start_dt
    while current <= overall_end_dt:
        end = pd.Timestamp(min(current.year + interval_year_gap - 1, overall_end_dt.year), 12, 31)
        if end > overall_end_dt:
            end = overall_end_dt
        intervals.append((current.strftime("%Y/%m/%d"), end.strftime("%Y/%m/%d")))
        current = pd.Timestamp(end.year + 1, 1, 1)

    expected_interval_rows = len(pathogens) * len(antibiotics) * len(intervals)

    interval_rows = 0
    interval_unique_pairs = 0
    interval_unique_pathogens = 0
    interval_unique_antibiotics = 0
    interval_missing_rows = expected_interval_rows
    intervals_output_path = Path(intervals_output_path)
    if intervals_output_path.exists():
        interval_df = pd.read_parquet(
            intervals_output_path,
            columns=["pathogen", "antibiotic", "start_date", "end_date"],
        )
        target_intervals = pd.DataFrame(intervals, columns=["start_date", "end_date"])
        interval_df = interval_df.merge(target_intervals, on=["start_date", "end_date"], how="inner")
        interval_rows = len(interval_df)
        interval_unique_pairs = interval_df[["pathogen", "antibiotic"]].drop_duplicates().shape[0]
        interval_unique_pathogens = interval_df["pathogen"].astype(str).nunique()
        interval_unique_antibiotics = interval_df["antibiotic"].astype(str).nunique()
        interval_missing_rows = max(expected_interval_rows - interval_rows, 0)

    yearly_rows = 0
    yearly_year_min = None
    yearly_year_max = None
    yearly_output_path = Path(yearly_output_path)
    if yearly_output_path.exists():
        yearly_df = pd.read_parquet(yearly_output_path, columns=["year"])
        yearly_rows = len(yearly_df)
        if not yearly_df.empty:
            years = pd.to_numeric(yearly_df["year"], errors="coerce").dropna().astype(int)
            if not years.empty:
                yearly_year_min = int(years.min())
                yearly_year_max = int(years.max())

    yearly_pair_pmid_rows = 0
    yearly_pair_pmid_path = Path(yearly_pair_pmid_path)
    if yearly_pair_pmid_path.exists():
        yearly_pair_pmid_rows = len(pd.read_parquet(yearly_pair_pmid_path, columns=["pmid"]))

    expected_pathogen_bg = len(pathogens) * (background_end_year - background_start_year + 1)
    pathogen_bg_rows = 0
    pathogen_bg_path = Path(pathogen_bg_path)
    if pathogen_bg_path.exists():
        pathogen_bg_rows = len(pd.read_parquet(pathogen_bg_path, columns=["pathogen", "year"]))

    expected_antibiotic_bg = len(antibiotics) * (background_end_year - background_start_year + 1)
    antibiotic_bg_rows = 0
    antibiotic_bg_path = Path(antibiotic_bg_path)
    if antibiotic_bg_path.exists():
        antibiotic_bg_rows = len(pd.read_parquet(antibiotic_bg_path, columns=["antibiotic", "year"]))

    return {
        "intervals": intervals,
        "expected_interval_rows": expected_interval_rows,
        "cached_interval_rows": interval_rows,
        "missing_interval_rows": interval_missing_rows,
        "cached_interval_pct": (interval_rows / expected_interval_rows * 100) if expected_interval_rows else 100.0,
        "cached_unique_pairs": interval_unique_pairs,
        "cached_unique_pathogens": interval_unique_pathogens,
        "cached_unique_antibiotics": interval_unique_antibiotics,
        "yearly_rows": yearly_rows,
        "yearly_year_min": yearly_year_min,
        "yearly_year_max": yearly_year_max,
        "yearly_pair_pmid_rows": yearly_pair_pmid_rows,
        "pathogen_bg_rows": pathogen_bg_rows,
        "expected_pathogen_bg_rows": expected_pathogen_bg,
        "antibiotic_bg_rows": antibiotic_bg_rows,
        "expected_antibiotic_bg_rows": expected_antibiotic_bg,
    }


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
    collection_verbose: int = 1,
    collection_progress_every: int = 200,
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
        verbose=collection_verbose,
        progress_every=collection_progress_every,
    )

    pair_pmid_df_raw = explode_positive_pair_pmids(interval_df)
    if Path(yearly_pair_pmid_output_path).exists():
        existing_pair_pmid_df = pd.read_parquet(yearly_pair_pmid_output_path)
    else:
        existing_pair_pmid_df = pd.DataFrame(columns=["pathogen", "antibiotic", "start_date", "end_date", "pmid"])

    new_pair_pmid_df = _extract_new_pair_pmids(existing_pair_pmid_df, pair_pmid_df_raw)
    pair_pmid_df = _merge_pair_pmid_frames(existing_pair_pmid_df, new_pair_pmid_df)
    Path(yearly_pair_pmid_output_path).parent.mkdir(parents=True, exist_ok=True)
    pair_pmid_df.to_parquet(yearly_pair_pmid_output_path, index=False)

    metadata_df = refresh_pubmed_metadata_cache(
        new_pair_pmid_df["pmid"].astype(str).tolist() if not new_pair_pmid_df.empty else [],
        cache_path=metadata_cache_path,
        email=email,
        batch_size=metadata_batch_size,
        checkpoint_every_batches=metadata_checkpoint_every_batches,
    )
    enriched_df = (
        attach_pubmed_metadata(new_pair_pmid_df, metadata_df)
        if not new_pair_pmid_df.empty
        else pd.DataFrame(columns=["pathogen", "antibiotic", "start_date", "end_date", "pmid", "pub_date", "pub_year", "venue"])
    )
    yearly_delta_df = (
        aggregate_pair_counts_by_year(enriched_df)
        if not enriched_df.empty
        else pd.DataFrame(columns=["pathogen", "antibiotic", "year", "count", "pmids"])
    )
    if Path(output_path).exists():
        existing_yearly_df = pd.read_parquet(output_path)
    else:
        existing_yearly_df = pd.DataFrame(columns=["pathogen", "antibiotic", "year", "count", "pmids"])
    yearly_df = _merge_yearly_counts_incremental(existing_yearly_df, yearly_delta_df)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    yearly_df.to_parquet(output_path, index=False)
    return yearly_df, pair_pmid_df, enriched_df


__all__ = [
    "build_collection_preflight_summary",
    "collect_yearly_background_counts",
    "collect_yearly_amr_dataset",
]
