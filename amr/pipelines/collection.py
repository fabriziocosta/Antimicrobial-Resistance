"""Yearly AMR collection pipelines built on top of interval PubMed retrieval."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Sequence

import pandas as pd

from amr.io import atomic_write_parquet
from amr.validation import clean_background_counts, clean_pair_pmid_df, clean_yearly_pair_counts
from amr.pubmed.intervals import (
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
    existing_df = clean_pair_pmid_df(existing_df) if not existing_df.empty else existing_df.copy()
    new_df = clean_pair_pmid_df(new_df) if not new_df.empty else new_df.copy()
    if existing_df.empty:
        return new_df.copy()
    if new_df.empty:
        return existing_df.copy()
    merged = pd.concat([existing_df, new_df], ignore_index=True)
    key_cols = ["pathogen", "antibiotic", "pmid"]
    return merged.drop_duplicates(subset=key_cols, keep="last").reset_index(drop=True)


def _as_pair_pmid_keyframe(df: pd.DataFrame) -> pd.DataFrame:
    work = clean_pair_pmid_df(df) if not df.empty else df.copy()
    if work.empty:
        return pd.DataFrame(columns=["pathogen", "antibiotic", "pmid"])
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
        out = clean_yearly_pair_counts(delta_df)
        return out.sort_values(["pathogen", "antibiotic", "year"]).reset_index(drop=True)
    if delta_df.empty:
        return clean_yearly_pair_counts(existing_df).sort_values(["pathogen", "antibiotic", "year"]).reset_index(drop=True)

    key_cols = ["pathogen", "antibiotic", "year"]
    work_existing = clean_yearly_pair_counts(existing_df)
    work_delta = clean_yearly_pair_counts(delta_df)
    work_existing["year"] = pd.to_numeric(work_existing["year"], errors="coerce").astype("Int64")
    work_delta["year"] = pd.to_numeric(work_delta["year"], errors="coerce").astype("Int64")
    work_existing = work_existing.dropna(subset=["year"]).copy()
    work_delta = work_delta.dropna(subset=["year"]).copy()
    work_existing["year"] = work_existing["year"].astype(int)
    work_delta["year"] = work_delta["year"].astype(int)
    work_existing["pmids"] = work_existing["pmids"].apply(lambda v: v if isinstance(v, list) else [])
    work_delta["pmids"] = work_delta["pmids"].apply(lambda v: v if isinstance(v, list) else [])
    combined = pd.concat([work_existing, work_delta], ignore_index=True)
    agg_spec: dict[str, tuple[str, object]] = {
        "pmids": ("pmids", lambda rows: sorted({str(p) for row in rows for p in row})),
        "count": ("count", "max"),
    }
    for col in ["collection_run_min_date", "collection_run_max_date"]:
        if col in combined.columns:
            agg_spec[col] = (col, lambda values: ";".join(sorted({str(v) for v in values if pd.notna(v)})))
    merged = combined.groupby(key_cols, as_index=False).agg(**agg_spec).sort_values(key_cols).reset_index(drop=True)
    merged["count"] = merged["pmids"].apply(len)
    return merged


def _build_interval_windows(min_date: str, max_date: str, interval_year_gap: int) -> list[tuple[str, str]]:
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
    return intervals


def _summarize_interval_cache(
    *,
    interval_df: pd.DataFrame,
    intervals: list[tuple[str, str]],
    pathogens: Sequence[str],
    antibiotics: Sequence[str],
) -> dict[str, object]:
    pathogen_values = {str(value) for value in pathogens}
    antibiotic_values = {str(value) for value in antibiotics}
    pathogen_count = len(pathogen_values)
    antibiotic_count = len(antibiotic_values)
    expected_rows = pathogen_count * antibiotic_count * len(intervals)
    if interval_df.empty or not intervals:
        return {
            "intervals": intervals,
            "expected_rows": expected_rows,
            "cached_rows": 0,
            "missing_rows": expected_rows,
            "cached_pct": (0.0 if expected_rows else 100.0),
            "unique_pairs": 0,
            "unique_pathogens": 0,
            "unique_antibiotics": 0,
        }

    target_intervals = pd.DataFrame(intervals, columns=["start_date", "end_date"])
    work = interval_df.copy()
    work["pathogen"] = work["pathogen"].astype(str)
    work["antibiotic"] = work["antibiotic"].astype(str)
    scoped = work.loc[
        work["pathogen"].isin(pathogen_values) & work["antibiotic"].isin(antibiotic_values)
    ].merge(target_intervals, on=["start_date", "end_date"], how="inner")
    cached_rows = len(scoped)
    return {
        "intervals": intervals,
        "expected_rows": expected_rows,
        "cached_rows": cached_rows,
        "missing_rows": max(expected_rows - cached_rows, 0),
        "cached_pct": (cached_rows / expected_rows * 100) if expected_rows else 100.0,
        "unique_pairs": scoped[["pathogen", "antibiotic"]].drop_duplicates().shape[0],
        "unique_pathogens": scoped["pathogen"].astype(str).nunique(),
        "unique_antibiotics": scoped["antibiotic"].astype(str).nunique(),
    }


def _summarize_background_cache(
    *,
    df: pd.DataFrame,
    key_col: str,
    entities: Sequence[str],
    start_year: int,
    end_year: int,
) -> dict[str, object]:
    entity_values = {str(value) for value in entities}
    entity_count = len(entity_values)
    expected_rows = entity_count * (end_year - start_year + 1)
    if df.empty:
        return {
            "cached_rows": 0,
            "expected_rows": expected_rows,
            "missing_rows": expected_rows,
            "cached_pct": (0.0 if expected_rows else 100.0),
            "years": (start_year, end_year),
            "warning": None,
        }

    work = df.copy()
    work[key_col] = work[key_col].astype(str)
    work["year"] = pd.to_numeric(work["year"], errors="coerce")
    work = work.dropna(subset=["year"]).copy()
    work["year"] = work["year"].astype(int)
    scoped = work.loc[
        work[key_col].isin(entity_values) & work["year"].between(start_year, end_year),
        [key_col, "year"],
    ].drop_duplicates()
    cached_rows = len(scoped)
    warning = None
    if cached_rows > expected_rows:
        warning = (
            f"cached rows exceed expected rows for {start_year}-{end_year}; "
            "the cache likely includes duplicates or was built for a different entity set."
        )
    return {
        "cached_rows": cached_rows,
        "expected_rows": expected_rows,
        "missing_rows": max(expected_rows - cached_rows, 0),
        "cached_pct": (cached_rows / expected_rows * 100) if expected_rows else 100.0,
        "years": (start_year, end_year),
        "warning": warning,
    }


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
    total_min_date: str | None = None,
    total_max_date: str | None = None,
    total_background_start_year: int | None = None,
    total_background_end_year: int | None = None,
) -> dict[str, object]:
    """Summarize cached yearly-collection artifacts for a given run window."""
    run_intervals = _build_interval_windows(min_date, max_date, interval_year_gap)
    total_min_date = total_min_date or min_date
    total_max_date = total_max_date or max_date
    total_background_start_year = (
        background_start_year if total_background_start_year is None else total_background_start_year
    )
    total_background_end_year = (
        background_end_year if total_background_end_year is None else total_background_end_year
    )

    intervals_output_path = Path(intervals_output_path)
    interval_df = pd.DataFrame(columns=["pathogen", "antibiotic", "start_date", "end_date"])
    if intervals_output_path.exists():
        interval_df = pd.read_parquet(
            intervals_output_path,
            columns=["pathogen", "antibiotic", "start_date", "end_date"],
        )
    run_interval_summary = _summarize_interval_cache(
        interval_df=interval_df,
        intervals=run_intervals,
        pathogens=pathogens,
        antibiotics=antibiotics,
    )
    total_interval_summary = _summarize_interval_cache(
        interval_df=interval_df,
        intervals=_build_interval_windows(total_min_date, total_max_date, interval_year_gap),
        pathogens=pathogens,
        antibiotics=antibiotics,
    )

    yearly_rows = 0
    yearly_year_min = None
    yearly_year_max = None
    yearly_rows_in_run_window = 0
    yearly_rows_in_total_window = 0
    yearly_output_path = Path(yearly_output_path)
    if yearly_output_path.exists():
        yearly_df = pd.read_parquet(yearly_output_path, columns=["year"])
        yearly_rows = len(yearly_df)
        if not yearly_df.empty:
            years = pd.to_numeric(yearly_df["year"], errors="coerce").dropna().astype(int)
            if not years.empty:
                yearly_year_min = int(years.min())
                yearly_year_max = int(years.max())
                run_year_start = datetime.strptime(min_date, "%Y/%m/%d").year
                run_year_end = datetime.strptime(max_date, "%Y/%m/%d").year
                total_year_start = datetime.strptime(total_min_date, "%Y/%m/%d").year
                total_year_end = datetime.strptime(total_max_date, "%Y/%m/%d").year
                yearly_rows_in_run_window = int(years.between(run_year_start, run_year_end).sum())
                yearly_rows_in_total_window = int(years.between(total_year_start, total_year_end).sum())

    yearly_pair_pmid_rows = 0
    yearly_pair_pmid_rows_in_run_window = 0
    yearly_pair_pmid_rows_in_total_window = 0
    yearly_pair_pmid_path = Path(yearly_pair_pmid_path)
    if yearly_pair_pmid_path.exists():
        pair_df = pd.read_parquet(yearly_pair_pmid_path, columns=["pmid", "start_date", "end_date"])
        yearly_pair_pmid_rows = len(pair_df)
        if not pair_df.empty:
            pair_df = pair_df.copy()
            pair_df["start_year"] = pd.to_datetime(pair_df["start_date"], errors="coerce").dt.year
            pair_df["end_year"] = pd.to_datetime(pair_df["end_date"], errors="coerce").dt.year
            run_year_start = datetime.strptime(min_date, "%Y/%m/%d").year
            run_year_end = datetime.strptime(max_date, "%Y/%m/%d").year
            total_year_start = datetime.strptime(total_min_date, "%Y/%m/%d").year
            total_year_end = datetime.strptime(total_max_date, "%Y/%m/%d").year
            yearly_pair_pmid_rows_in_run_window = int(
                (
                    pair_df["start_year"].ge(run_year_start).fillna(False)
                    & pair_df["end_year"].le(run_year_end).fillna(False)
                ).sum()
            )
            yearly_pair_pmid_rows_in_total_window = int(
                (
                    pair_df["start_year"].ge(total_year_start).fillna(False)
                    & pair_df["end_year"].le(total_year_end).fillna(False)
                ).sum()
            )

    pathogen_bg_path = Path(pathogen_bg_path)
    pathogen_bg_df = pd.DataFrame(columns=["pathogen", "year"])
    if pathogen_bg_path.exists():
        pathogen_bg_df = pd.read_parquet(pathogen_bg_path, columns=["pathogen", "year"])

    antibiotic_bg_path = Path(antibiotic_bg_path)
    antibiotic_bg_df = pd.DataFrame(columns=["antibiotic", "year"])
    if antibiotic_bg_path.exists():
        antibiotic_bg_df = pd.read_parquet(antibiotic_bg_path, columns=["antibiotic", "year"])

    run_pathogen_bg_summary = _summarize_background_cache(
        df=pathogen_bg_df,
        key_col="pathogen",
        entities=pathogens,
        start_year=background_start_year,
        end_year=background_end_year,
    )
    total_pathogen_bg_summary = _summarize_background_cache(
        df=pathogen_bg_df,
        key_col="pathogen",
        entities=pathogens,
        start_year=total_background_start_year,
        end_year=total_background_end_year,
    )
    run_antibiotic_bg_summary = _summarize_background_cache(
        df=antibiotic_bg_df,
        key_col="antibiotic",
        entities=antibiotics,
        start_year=background_start_year,
        end_year=background_end_year,
    )
    total_antibiotic_bg_summary = _summarize_background_cache(
        df=antibiotic_bg_df,
        key_col="antibiotic",
        entities=antibiotics,
        start_year=total_background_start_year,
        end_year=total_background_end_year,
    )

    return {
        "run_window": {
            "min_date": min_date,
            "max_date": max_date,
            "intervals": run_interval_summary["intervals"],
            "interval_cache": run_interval_summary,
            "yearly_rows": yearly_rows_in_run_window,
            "yearly_pair_pmid_rows": yearly_pair_pmid_rows_in_run_window,
            "pathogen_bg_cache": run_pathogen_bg_summary,
            "antibiotic_bg_cache": run_antibiotic_bg_summary,
        },
        "total_window": {
            "min_date": total_min_date,
            "max_date": total_max_date,
            "intervals": total_interval_summary["intervals"],
            "interval_cache": total_interval_summary,
            "yearly_rows": yearly_rows_in_total_window,
            "yearly_pair_pmid_rows": yearly_pair_pmid_rows_in_total_window,
            "pathogen_bg_cache": total_pathogen_bg_summary,
            "antibiotic_bg_cache": total_antibiotic_bg_summary,
        },
        "yearly_rows": yearly_rows,
        "yearly_year_min": yearly_year_min,
        "yearly_year_max": yearly_year_max,
        "yearly_pair_pmid_rows": yearly_pair_pmid_rows,
    }


def collect_yearly_background_counts(
    *,
    pathogens: Sequence[str],
    antibiotics: Sequence[str],
    bacteria_synonym_df: pd.DataFrame,
    antibiotic_synonym_df: pd.DataFrame | None = None,
    start_year: int,
    end_year: int,
    email: str,
    pathogen_output_path: str | Path = "pathogen_tiab_counts.parquet",
    antibiotic_output_path: str | Path = "antibiotic_tiab_counts.parquet",
    save_every: int = 200,
    progress_every: int = 50,
    fetch_threshold: int = 9500,
    metadata_batch_size: int = 200,
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
        fetch_threshold=fetch_threshold,
        metadata_batch_size=metadata_batch_size,
    )
    pathogen_counts_df = clean_background_counts(pathogen_counts_df, key_col="pathogen")
    atomic_write_parquet(pathogen_counts_df, pathogen_output_path, index=False)
    antibiotic_counts_df = count_pmids_per_year_for_antibiotics_resumable(
        antibiotics=antibiotics,
        antibiotic_synonym_df=antibiotic_synonym_df,
        start_year=start_year,
        end_year=end_year,
        email=email,
        output_path=str(antibiotic_output_path),
        save_every=save_every,
        progress_every=progress_every,
        fetch_threshold=fetch_threshold,
        metadata_batch_size=metadata_batch_size,
    )
    antibiotic_counts_df = clean_background_counts(antibiotic_counts_df, key_col="antibiotic")
    atomic_write_parquet(antibiotic_counts_df, antibiotic_output_path, index=False)
    return pathogen_counts_df, antibiotic_counts_df


def collect_yearly_amr_dataset(
    *,
    pathogens: Sequence[str],
    antibiotics: Sequence[str],
    bacteria_synonym_df: pd.DataFrame,
    antibiotic_synonym_df: pd.DataFrame | None = None,
    min_date: str,
    max_date: str,
    email: str,
    interval_year_gap: int = 5,
    intervals_output_path: str | Path = "pmids_results.parquet",
    yearly_pair_pmid_output_path: str | Path = "candidate_amr_pair_pmid_links.parquet",
    metadata_cache_path: str | Path = "pubmed_pmid_metadata.parquet",
    output_path: str | Path = "candidate_amr_pair_year_counts.parquet",
    save_every: int = 50,
    extra_keywords: Sequence[str] | None = None,
    full_fetch_threshold: int = 9500,
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
        antibiotic_synonym_df=antibiotic_synonym_df,
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
        existing_pair_pmid_df = clean_pair_pmid_df(pd.read_parquet(yearly_pair_pmid_output_path))
    else:
        existing_pair_pmid_df = pd.DataFrame(columns=["pathogen", "antibiotic", "start_date", "end_date", "pmid"])

    new_pair_pmid_df = _extract_new_pair_pmids(existing_pair_pmid_df, pair_pmid_df_raw)
    if not new_pair_pmid_df.empty:
        new_pair_pmid_df = new_pair_pmid_df.copy()
        new_pair_pmid_df["collection_run_min_date"] = min_date
        new_pair_pmid_df["collection_run_max_date"] = max_date
    pair_pmid_df = _merge_pair_pmid_frames(existing_pair_pmid_df, new_pair_pmid_df)
    Path(yearly_pair_pmid_output_path).parent.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(pair_pmid_df, yearly_pair_pmid_output_path, index=False)

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
    if not yearly_delta_df.empty:
        yearly_delta_df = yearly_delta_df.copy()
        yearly_delta_df["collection_run_min_date"] = min_date
        yearly_delta_df["collection_run_max_date"] = max_date
    if Path(output_path).exists():
        existing_yearly_df = clean_yearly_pair_counts(pd.read_parquet(output_path))
    else:
        existing_yearly_df = pd.DataFrame(columns=["pathogen", "antibiotic", "year", "count", "pmids"])
    yearly_df = _merge_yearly_counts_incremental(existing_yearly_df, yearly_delta_df)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(yearly_df, output_path, index=False)
    return yearly_df, pair_pmid_df, enriched_df


__all__ = [
    "build_collection_preflight_summary",
    "collect_yearly_background_counts",
    "collect_yearly_amr_dataset",
]
