"""High-level data collection pipelines for PubMed AMR queries."""

from __future__ import annotations

import json
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from amr.io import atomic_write_parquet
from amr.validation import clean_background_counts, clean_interval_results, normalize_entity_key
from amr.pubmed.queries import (
    build_antibiotic_query,
    build_pathogen_tiab_query,
)
from amr.pubmed.search import (
    esearch_count_for_year,
    search_term_clause,
    search_pmids,
    retrieve_pmids,
)
from amr.pubmed.cache import (
    update_metadata_cache,
    get_metadata,
)
from amr.pubmed.pmids import normalize_pmids
from amr.pubmed.yearly import (
    load_pubmed_metadata_cache,
    metadata_dataframe_to_dict,
    metadata_dict_to_dataframe,
    save_pubmed_metadata_cache,
)


def _build_interval_cache_index(existing_df: pd.DataFrame) -> dict[tuple[str, str, str, str], int]:
    """Index saved interval counts once for constant-time resume checks."""
    if existing_df.empty:
        return {}
    required = ["pathogen", "antibiotic", "start_date", "end_date", "count"]
    missing = set(required) - set(existing_df.columns)
    if missing:
        raise KeyError(f"existing interval cache missing required columns: {sorted(missing)}")
    return {
        (str(row.pathogen), str(row.antibiotic), str(row.start_date), str(row.end_date)): int(row.count)
        for row in existing_df[required].itertuples(index=False)
    }


def count_pmids_per_year_for_pathogens(
    pathogens: Sequence[str],
    bacteria_synonym_df: pd.DataFrame,
    *,
    start_year: int | None = None,
    end_year: int | None = None,
    email: str = "your_email@example.com",
) -> pd.DataFrame:
    """Return a DataFrame with yearly PMID counts for each pathogen (title/abstract match, synonym-aware).

    Columns: pathogen, year, count
    """
    # Coerce to list to support numpy arrays/Series without ambiguous truth-value errors
    pathogens = list(pathogens)
    if len(pathogens) == 0:
        return pd.DataFrame(columns=["pathogen", "year", "count"])
    if start_year is None:
        start_year = 1950
    if end_year is None:
        end_year = datetime.now().year
    rows = []
    total_tasks = len(pathogens) * (end_year - start_year + 1)
    done = 0
    for pathogen in pathogens:
        clause = build_pathogen_tiab_query(pathogen, bacteria_synonym_df)
        for year in range(start_year, end_year + 1):
            cnt = esearch_count_for_year(clause, year, email=email)
            if cnt is None:
                raise RuntimeError(f"PubMed count failed for pathogen={pathogen!r}, year={year}.")
            rows.append({"pathogen": pathogen, "year": year, "count": cnt})
            done += 1
            if done % 50 == 0:
                print(f"Progress: {done}/{total_tasks} year-queries done")
    return pd.DataFrame(rows)


def count_pmids_per_year_for_antibiotics(
    antibiotics: Sequence[str],
    antibiotic_synonym_df: pd.DataFrame | None = None,
    *,
    start_year: int | None = None,
    end_year: int | None = None,
    email: str = "your_email@example.com",
) -> pd.DataFrame:
    """Return a DataFrame with yearly PMID counts for each antibiotic (title/abstract match, dash/slash variants).

    Columns: antibiotic, year, count
    """
    antibiotics = list(antibiotics)
    if len(antibiotics) == 0:
        return pd.DataFrame(columns=["antibiotic", "year", "count"])
    if start_year is None:
        start_year = 1950
    if end_year is None:
        end_year = datetime.now().year
    rows = []
    total_tasks = len(antibiotics) * (end_year - start_year + 1)
    done = 0
    for ab in antibiotics:
        clause = build_antibiotic_query(ab, antibiotic_synonym_df, field="Title/Abstract")
        for year in range(start_year, end_year + 1):
            cnt = esearch_count_for_year(clause, year, email=email)
            if cnt is None:
                raise RuntimeError(f"PubMed count failed for antibiotic={ab!r}, year={year}.")
            rows.append({"antibiotic": ab, "year": year, "count": cnt})
            done += 1
            if done % 50 == 0:
                print(f"Progress: {done}/{total_tasks} year-queries done")
    return pd.DataFrame(rows)


def fetch_pmids_by_date_and_keywords(
    min_date: str,
    max_date: str,
    antibiotic: str,
    pathogen: str,
    bacteria_synonym_df: pd.DataFrame,
    antibiotic_synonym_df: pd.DataFrame | None = None,
    email: str = "your_email@example.com",
    batch_size: int = 9000,
    extra_clauses: Optional[Sequence[str]] = None,
    verbose: bool = True,
) -> list[str]:
    """Retrieve PMIDs within a date range; splits queries that exceed PubMed limits."""
    query, count, webenv, query_key = search_pmids(
        antibiotic,
        pathogen,
        bacteria_synonym_df,
        antibiotic_synonym_df=antibiotic_synonym_df,
        extra_clauses=extra_clauses,
        min_date=min_date,
        max_date=max_date,
        email=email,
        verbose=verbose,
    )
    if count is None:
        raise RuntimeError(
            f"PubMed search failed for pathogen={pathogen!r}, antibiotic={antibiotic!r}, "
            f"window={min_date}..{max_date}."
        )
    if count == 0:
        return []
    if count > 9999:
        start_dt = datetime.strptime(min_date, "%Y/%m/%d")
        end_dt = datetime.strptime(max_date, "%Y/%m/%d")
        mid_year = (start_dt.year + end_dt.year) // 2
        mid_dt = datetime(mid_year, 12, 31)

        left_pmids = fetch_pmids_by_date_and_keywords(
            min_date=min_date,
            max_date=mid_dt.strftime("%Y/%m/%d"),
            antibiotic=antibiotic,
            pathogen=pathogen,
            bacteria_synonym_df=bacteria_synonym_df,
            antibiotic_synonym_df=antibiotic_synonym_df,
            email=email,
            batch_size=batch_size,
            extra_clauses=extra_clauses,
            verbose=verbose,
        )
        right_pmids = fetch_pmids_by_date_and_keywords(
            min_date=(mid_dt + timedelta(days=1)).strftime("%Y/%m/%d"),
            max_date=max_date,
            antibiotic=antibiotic,
            pathogen=pathogen,
            bacteria_synonym_df=bacteria_synonym_df,
            antibiotic_synonym_df=antibiotic_synonym_df,
            email=email,
            batch_size=batch_size,
            extra_clauses=extra_clauses,
            verbose=verbose,
        )
        return left_pmids + right_pmids
    return retrieve_pmids(query, webenv, query_key, count, batch_size=batch_size, verbose=verbose)


def _collect_pair_raw_results(
    *,
    pathogen: str,
    antibiotic: str,
    missing_entries: list[dict[str, object]],
    bacteria_synonym_df: pd.DataFrame,
    antibiotic_synonym_df: pd.DataFrame | None,
    email: str,
    min_date: str,
    max_date: str,
    extra_keywords: Optional[Sequence[str]],
    full_fetch_threshold: int,
    retrieve_batch_size: int,
    partial_interval_max: int,
    all_intervals_missing: bool,
    verbose: bool,
    search_recovery_attempts: int,
    search_recovery_delay_seconds: float,
) -> dict[str, object]:
    """Collect PMIDs for one pair without touching shared caches or files.

    Workers return raw PMIDs. Metadata lookup, materialization, and persistence stay
    in the caller so that duplicate PMIDs are fetched once and Parquet has one writer.
    """
    def _fetch_interval(entry: dict[str, object]) -> list[str]:
        last_error: RuntimeError | None = None
        for recovery_attempt in range(1, search_recovery_attempts + 1):
            try:
                return fetch_pmids_by_date_and_keywords(
                    min_date=str(entry["start"]),
                    max_date=str(entry["end"]),
                    antibiotic=antibiotic,
                    pathogen=pathogen,
                    bacteria_synonym_df=bacteria_synonym_df,
                    antibiotic_synonym_df=antibiotic_synonym_df,
                    email=email,
                    batch_size=retrieve_batch_size,
                    extra_clauses=extra_keywords,
                    verbose=verbose,
                )
            except RuntimeError as exc:
                last_error = exc
                if recovery_attempt == search_recovery_attempts:
                    raise
                recovery_delay = search_recovery_delay_seconds * (2 ** (recovery_attempt - 1))
                print(
                    "PubMed recovery | "
                    f"pathogen={pathogen} | antibiotic={antibiotic} | "
                    f"interval={entry['start']}..{entry['end']} | "
                    f"search round {recovery_attempt}/{search_recovery_attempts} failed; "
                    f"waiting {recovery_delay:g}s before retrying",
                    flush=True,
                )
                time.sleep(recovery_delay)
        raise last_error or RuntimeError("PubMed interval retrieval failed.")

    if not all_intervals_missing and len(missing_entries) <= partial_interval_max:
        interval_rows = []
        for entry in missing_entries:
            pmids = _fetch_interval(entry)
            interval_rows.append({
                "start": entry["start"],
                "end": entry["end"],
                "start_dt": entry["start_dt"],
                "end_dt": entry["end_dt"],
                "pmids": normalize_pmids(pmids),
            })
        return {
            "pathogen": pathogen,
            "antibiotic": antibiotic,
            "mode": "interval",
            "entries": interval_rows,
        }

    query = ""
    total_count = None
    webenv = None
    query_key = None
    for recovery_attempt in range(1, search_recovery_attempts + 1):
        query, total_count, webenv, query_key = search_pmids(
            antibiotic,
            pathogen,
            bacteria_synonym_df,
            antibiotic_synonym_df=antibiotic_synonym_df,
            extra_clauses=extra_keywords,
            min_date=min_date,
            max_date=max_date,
            email=email,
            verbose=verbose,
        )
        if total_count is not None:
            break
        if recovery_attempt < search_recovery_attempts:
            recovery_delay = search_recovery_delay_seconds * (2 ** (recovery_attempt - 1))
            print(
                "PubMed recovery | "
                f"pathogen={pathogen} | antibiotic={antibiotic} | "
                f"search round {recovery_attempt}/{search_recovery_attempts} failed; "
                f"waiting {recovery_delay:g}s before retrying",
                flush=True,
            )
            time.sleep(recovery_delay)

    if total_count is None:
        raise RuntimeError(
            f"PubMed search failed for pathogen={pathogen!r}, antibiotic={antibiotic!r}, "
            f"window={min_date}..{max_date} after {search_recovery_attempts} recovery rounds."
        )

    if total_count == 0:
        return {
            "pathogen": pathogen,
            "antibiotic": antibiotic,
            "mode": "zero",
            "entries": [
                {
                    "start": entry["start"],
                    "end": entry["end"],
                    "start_dt": entry["start_dt"],
                    "end_dt": entry["end_dt"],
                    "pmids": [],
                }
                for entry in missing_entries
            ],
        }

    if total_count <= full_fetch_threshold:
        pmids_all = retrieve_pmids(
            query,
            webenv,
            query_key,
            total_count,
            batch_size=retrieve_batch_size,
            verbose=verbose,
        )
        return {
            "pathogen": pathogen,
            "antibiotic": antibiotic,
            "mode": "full",
            "pmids": normalize_pmids(pmids_all),
            "entries": [
                {
                    "start": entry["start"],
                    "end": entry["end"],
                    "start_dt": entry["start_dt"],
                    "end_dt": entry["end_dt"],
                }
                for entry in missing_entries
            ],
        }

    interval_rows = []
    for entry in missing_entries:
        pmids = _fetch_interval(entry)
        interval_rows.append({
            "start": entry["start"],
            "end": entry["end"],
            "start_dt": entry["start_dt"],
            "end_dt": entry["end_dt"],
            "pmids": normalize_pmids(pmids),
        })
    return {
        "pathogen": pathogen,
        "antibiotic": antibiotic,
        "mode": "interval",
        "entries": interval_rows,
    }


def _materialize_pair_results(
    raw_result: dict[str, object],
    *,
    metadata_cache: dict[str, dict[str, object]],
    email: str,
    metadata_batch_size: int,
    overall_start_dt: datetime,
    overall_end_dt: datetime,
    verbose: bool,
) -> tuple[list[dict], int]:
    """Fetch shared metadata once, then turn raw pair PMIDs into interval rows."""
    mode = str(raw_result["mode"])
    entries = list(raw_result["entries"])
    if mode == "zero":
        return [
            {
                "pathogen": raw_result["pathogen"],
                "antibiotic": raw_result["antibiotic"],
                "start_date": entry["start"],
                "end_date": entry["end"],
                "pmids": [],
                "count": 0,
                "venue_histogram": json.dumps({}),
            }
            for entry in entries
        ], 0

    if mode == "full":
        entry_pmids = {id(entry): normalize_pmids(raw_result.get("pmids", [])) for entry in entries}
        all_pmids = normalize_pmids(raw_result.get("pmids", []))
    else:
        entry_pmids = {id(entry): normalize_pmids(entry.get("pmids", [])) for entry in entries}
        all_pmids = normalize_pmids(pmid for pmids in entry_pmids.values() for pmid in pmids)

    before = len(metadata_cache)
    update_metadata_cache(
        all_pmids,
        metadata_cache,
        email=email,
        batch_size=metadata_batch_size,
        verbose=verbose,
    )
    metadata_added = max(0, len(metadata_cache) - before)

    rows = []
    for entry in entries:
        interval_pmids = []
        venue_counter = Counter()
        candidate_pmids = entry_pmids[id(entry)]
        for pmid in candidate_pmids:
            meta = get_metadata(pmid, metadata_cache)
            pub_date = meta["date"]
            if mode == "full":
                if not pub_date or not (overall_start_dt <= pub_date <= overall_end_dt):
                    continue
            elif pub_date and not (entry["start_dt"] <= pub_date <= entry["end_dt"]):
                continue
            interval_pmids.append(pmid)
            venue_counter[meta["venue"] or "Unknown"] += 1
        rows.append(
            {
                "pathogen": raw_result["pathogen"],
                "antibiotic": raw_result["antibiotic"],
                "start_date": entry["start"],
                "end_date": entry["end"],
                "pmids": interval_pmids,
                "count": len(interval_pmids),
                "venue_histogram": json.dumps(dict(venue_counter)),
            }
        )
    return rows, metadata_added




def fetch_pmids_intervals(
    pathogens: Sequence[str],
    antibiotics: Sequence[str],
    bacteria_synonym_df: pd.DataFrame,
    min_date: str,
    max_date: str,
    antibiotic_synonym_df: pd.DataFrame | None = None,
    year_gap: int = 5,
    email: str = "your_email@example.com",
    parquet_path: str = "pmids_results.parquet",
    save_every: int = 500,
    extra_keywords: Optional[Sequence[str]] = None,
    full_fetch_threshold: int = 9500,
    summary_batch_size: int = 1000,
    retrieve_batch_size: int = 9000,
    metadata_cache_path: str | Path | None = None,
    max_workers: int = 4,
    max_pending_futures: int | None = None,
    partial_interval_max: int = 1,
    verbose: int = 1,
    progress_every: int = 200,
    search_recovery_attempts: int = 3,
    search_recovery_delay_seconds: float = 30.0,
) -> pd.DataFrame:
    """Query PubMed intervals with bounded pair parallelism and resumable writes.

    Pair retrieval is network-bound, so workers collect raw PMIDs independently.
    Metadata lookup and all Parquet writes remain in the caller thread, which avoids
    duplicate metadata requests and concurrent-file corruption.
    """
    pathogens = list(pathogens)
    antibiotics = list(antibiotics)
    parquet_path = Path(parquet_path)
    checkpoint_dir = Path(f"{parquet_path}.checkpoints")
    checkpoint_paths = sorted(checkpoint_dir.glob("part-*.parquet")) if checkpoint_dir.exists() else []

    cached_frames = []
    if parquet_path.exists():
        cached_frames.append(pd.read_parquet(parquet_path))
    for checkpoint_path in checkpoint_paths:
        cached_frames.append(pd.read_parquet(checkpoint_path))
    if cached_frames:
        existing_df = clean_interval_results(pd.concat(cached_frames, ignore_index=True))
    else:
        existing_df = pd.DataFrame(
            columns=[
                "pathogen",
                "antibiotic",
                "start_date",
                "end_date",
                "pmids",
                "count",
                "venue_histogram",
            ]
        )
    cache_index = _build_interval_cache_index(existing_df)

    metadata_cache: dict[str, dict[str, object]] = {}
    if metadata_cache_path is not None and Path(metadata_cache_path).exists():
        metadata_cache = metadata_dataframe_to_dict(load_pubmed_metadata_cache(metadata_cache_path))

    new_results: list[dict] = []
    metadata_dirty = False
    data_dirty = bool(checkpoint_paths)
    checkpoint_counter = len(checkpoint_paths)
    overall_start_dt = datetime.strptime(min_date, "%Y/%m/%d")
    overall_end_dt = datetime.strptime(max_date, "%Y/%m/%d")

    intervals = []
    current = overall_start_dt
    while current <= overall_end_dt:
        end = datetime(min(current.year + year_gap - 1, overall_end_dt.year), 12, 31)
        if end > overall_end_dt:
            end = overall_end_dt
        intervals.append((current.strftime("%Y/%m/%d"), end.strftime("%Y/%m/%d")))
        current = datetime(end.year + 1, 1, 1)

    total_tasks = len(pathogens) * len(antibiotics)
    total_pathogens = len(pathogens)
    total_antibiotics = len(antibiotics)
    total_years = overall_end_dt.year - overall_start_dt.year + 1
    progress_every = max(1, int(progress_every))
    max_workers = max(1, int(max_workers))
    max_pending = max_workers * 2 if max_pending_futures is None else max(max_workers, int(max_pending_futures))
    partial_interval_max = max(0, int(partial_interval_max))
    search_recovery_attempts = max(1, int(search_recovery_attempts))
    search_recovery_delay_seconds = max(0.0, float(search_recovery_delay_seconds))
    event_counter = 0
    event_stats = {
        "cached": 0,
        "zero": 0,
        "full_fetch_partitioned": 0,
        "partial_interval_fetch": 0,
        "new": 0,
    }
    low_level_verbose = verbose >= 2
    antibiotic_status_every = max(1, progress_every // max(len(intervals), 1))

    def _log_event(kind: str, message: str) -> None:
        nonlocal event_counter
        event_counter += 1
        event_stats[kind] = event_stats.get(kind, 0) + 1
        if verbose >= 2:
            print(message)

    def _flush_checkpoint(*, force: bool = False) -> None:
        nonlocal existing_df, new_results, metadata_dirty, data_dirty, checkpoint_counter
        should_write_data = bool(new_results) and (
            force or not save_every or len(new_results) >= max(1, int(save_every))
        )
        if should_write_data:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            chunk_df = clean_interval_results(pd.DataFrame(new_results))
            chunk_path = checkpoint_dir / f"part-{checkpoint_counter:08d}.parquet"
            atomic_write_parquet(chunk_df, chunk_path, index=False)
            checkpoint_counter += 1
            existing_df = pd.concat([existing_df, chunk_df], ignore_index=True)
            for row in chunk_df[
                ["pathogen", "antibiotic", "start_date", "end_date", "count"]
            ].itertuples(index=False):
                cache_index[(str(row.pathogen), str(row.antibiotic), str(row.start_date), str(row.end_date))] = int(row.count)
            new_results = []
            data_dirty = True

        if metadata_cache_path is not None and metadata_dirty and (force or should_write_data):
            save_pubmed_metadata_cache(
                metadata_dict_to_dataframe(metadata_cache),
                metadata_cache_path,
            )
            metadata_dirty = False

    def _iter_pair_tasks():
        task_counter = 0
        for pathogen_idx, pathogen in enumerate(pathogens, start=1):
            for antibiotic_idx, antibiotic in enumerate(antibiotics, start=1):
                if antibiotic_idx == 1 or (antibiotic_idx - 1) % antibiotic_status_every == 0:
                    if verbose >= 1:
                        completed_pathogens = pathogen_idx - 1
                        completed_antibiotics = antibiotic_idx - 1
                        completed_pairs = completed_pathogens * total_antibiotics + completed_antibiotics
                        completed_pair_years = completed_pairs * total_years
                        total_pair_years = total_tasks * total_years
                        print(
                            "Collection status | "
                            f"pathogens {completed_pathogens}/{total_pathogens} "
                            f"({100 * completed_pathogens / max(total_pathogens, 1):.1f}%) | "
                            f"current {pathogen} | "
                            f"antibiotics {completed_antibiotics}/{total_antibiotics} "
                            f"({100 * completed_antibiotics / max(total_antibiotics, 1):.1f}%) | "
                            f"pairs {completed_pairs}/{total_tasks} "
                            f"({100 * completed_pairs / max(total_tasks, 1):.1f}%) | "
                            f"pair-years {completed_pair_years}/{total_pair_years} "
                            f"({100 * completed_pair_years / max(total_pair_years, 1):.1f}%)",
                            flush=True,
                        )
                task_counter += 1
                task_progress = task_counter / max(total_tasks, 1) * 100
                missing_entries = []
                for interval_start, interval_end in intervals:
                    start_dt = datetime.strptime(interval_start, "%Y/%m/%d")
                    end_dt = datetime.strptime(interval_end, "%Y/%m/%d")
                    cache_key = (str(pathogen), str(antibiotic), interval_start, interval_end)
                    if cache_key in cache_index:
                        _log_event(
                            "cached",
                            f"[{task_counter:>3}/{total_tasks}] ({task_progress:5.1f}%) | "
                            f"{pathogen:<40} | {antibiotic:<20} | "
                            f"{interval_start} – {interval_end} | "
                            f"{cache_index[cache_key]:5} PMIDs (cached)",
                        )
                    else:
                        missing_entries.append(
                            {
                                "start": interval_start,
                                "end": interval_end,
                                "start_dt": start_dt,
                                "end_dt": end_dt,
                            }
                        )
                if missing_entries:
                    yield {
                        "task_counter": task_counter,
                        "pathogen": str(pathogen),
                        "antibiotic": str(antibiotic),
                        "missing_entries": missing_entries,
                        "all_intervals_missing": len(missing_entries) == len(intervals),
                    }

    def _worker_kwargs(task: dict[str, object]) -> dict[str, object]:
        return {
            "pathogen": task["pathogen"],
            "antibiotic": task["antibiotic"],
            "missing_entries": task["missing_entries"],
            "bacteria_synonym_df": bacteria_synonym_df,
            "antibiotic_synonym_df": antibiotic_synonym_df,
            "email": email,
            "min_date": min_date,
            "max_date": max_date,
            "extra_keywords": extra_keywords,
            "full_fetch_threshold": full_fetch_threshold,
            "retrieve_batch_size": retrieve_batch_size,
            "partial_interval_max": partial_interval_max,
            "all_intervals_missing": task["all_intervals_missing"],
            "verbose": low_level_verbose,
            "search_recovery_attempts": search_recovery_attempts,
            "search_recovery_delay_seconds": search_recovery_delay_seconds,
        }

    def _consume(raw_result: dict[str, object]) -> None:
        nonlocal metadata_dirty
        rows, metadata_added = _materialize_pair_results(
            raw_result,
            metadata_cache=metadata_cache,
            email=email,
            metadata_batch_size=summary_batch_size,
            overall_start_dt=overall_start_dt,
            overall_end_dt=overall_end_dt,
            verbose=low_level_verbose,
        )
        metadata_dirty = metadata_dirty or metadata_added > 0
        mode = str(raw_result["mode"])
        kind = {
            "zero": "zero",
            "full": "full_fetch_partitioned",
            "interval": "partial_interval_fetch" if len(rows) < len(intervals) else "new",
        }[mode]
        for row in rows:
            new_results.append(row)
            _log_event(
                kind,
                f"{raw_result['pathogen']} | {raw_result['antibiotic']} | "
                f"{row['start_date']} – {row['end_date']} | "
                f"{row['count']:5} PMIDs ({kind})",
            )
        _flush_checkpoint()

    task_iter = iter(_iter_pair_tasks())
    pending = set()
    try:
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="pubmed-pair") as executor:
            exhausted = False
            while pending or not exhausted:
                while not exhausted and len(pending) < max_pending:
                    try:
                        task = next(task_iter)
                    except StopIteration:
                        exhausted = True
                        break
                    pending.add(executor.submit(_collect_pair_raw_results, **_worker_kwargs(task)))
                if not pending:
                    continue
                completed, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    _consume(future.result())
    except BaseException:
        _flush_checkpoint(force=True)
        raise

    _flush_checkpoint(force=True)
    if data_dirty:
        existing_df = clean_interval_results(existing_df)
        atomic_write_parquet(existing_df, parquet_path, index=False)
        for checkpoint_path in sorted(checkpoint_dir.glob("part-*.parquet")):
            checkpoint_path.unlink()
        try:
            checkpoint_dir.rmdir()
        except OSError:
            pass

    if verbose >= 1:
        completed_pair_years = total_tasks * total_years
        print(
            "Collection status | "
            f"pathogens {total_pathogens}/{total_pathogens} (100.0%) | "
            f"antibiotics {total_antibiotics}/{total_antibiotics} (100.0%) | "
            f"pairs {total_tasks}/{total_tasks} (100.0%) | "
            f"pair-years {completed_pair_years}/{completed_pair_years} (100.0%)",
            flush=True,
        )
        print(
            "Collection summary: "
            f"events={event_counter}, "
            f"cached={event_stats.get('cached', 0)}, "
            f"zero={event_stats.get('zero', 0)}, "
            f"full_fetch={event_stats.get('full_fetch_partitioned', 0)}, "
            f"partial_fetch={event_stats.get('partial_interval_fetch', 0)}, "
            f"new={event_stats.get('new', 0)}"
        )

    pathogen_values = {value for value in (normalize_entity_key(value) for value in pathogens) if value}
    antibiotic_values = {value for value in (normalize_entity_key(value) for value in antibiotics) if value}
    target_intervals = pd.DataFrame(intervals, columns=["start_date", "end_date"])
    scoped_df = existing_df.copy()
    scoped_df["pathogen"] = scoped_df["pathogen"].astype(str)
    scoped_df["antibiotic"] = scoped_df["antibiotic"].astype(str)
    scoped_df = scoped_df.loc[
        scoped_df["pathogen"].isin(pathogen_values) & scoped_df["antibiotic"].isin(antibiotic_values)
    ].merge(target_intervals, on=["start_date", "end_date"], how="inner")
    return scoped_df.reset_index(drop=True)


__all__ = [
    "count_pmids_per_year_for_pathogens",
    "count_pmids_per_year_for_antibiotics",
    "fetch_pmids_by_date_and_keywords",
    "fetch_pmids_intervals",
]


def _year_count_rows(entity_col: str, entity_value: str, start_year: int, end_year: int, year_to_count: dict[int, int]) -> list[dict]:
    return [
        {entity_col: entity_value, "year": year, "count": int(year_to_count.get(year, 0))}
        for year in range(start_year, end_year + 1)
    ]


def _count_pmids_per_year_via_windows(
    *,
    entity_value: str,
    entity_col: str,
    clause: str,
    start_year: int,
    end_year: int,
    email: str,
    metadata_cache: dict[str, dict],
    fetch_threshold: int,
    metadata_batch_size: int,
) -> list[dict]:
    """Count yearly PMIDs by recursively querying date windows and partitioning locally when feasible."""
    if start_year > end_year:
        return []

    min_date = f"{start_year}/01/01"
    max_date = f"{end_year}/12/31"
    _, total_count, webenv, query_key = search_term_clause(
        clause,
        min_date=min_date,
        max_date=max_date,
        email=email,
    )

    if total_count is None:
        raise RuntimeError(
            f"PubMed background-count search failed for {entity_col}={entity_value!r}, "
            f"years={start_year}-{end_year}; no zero rows were written."
        )

    if total_count == 0:
        return _year_count_rows(entity_col, entity_value, start_year, end_year, {})

    if start_year == end_year:
        return [{entity_col: entity_value, "year": start_year, "count": int(total_count)}]

    if total_count <= fetch_threshold and webenv and query_key:
        query = f'({clause}) AND ("{min_date}"[Date - Publication] : "{max_date}"[Date - Publication])'
        pmids_all = list(dict.fromkeys(retrieve_pmids(query, webenv, query_key, total_count)))
        update_metadata_cache(pmids_all, metadata_cache, email=email, batch_size=metadata_batch_size)
        year_to_count: dict[int, int] = {}
        for pmid in pmids_all:
            meta = get_metadata(pmid, metadata_cache)
            pub_date = meta["date"]
            if not pub_date:
                continue
            pub_year = int(pub_date.year)
            if start_year <= pub_year <= end_year:
                year_to_count[pub_year] = year_to_count.get(pub_year, 0) + 1
        return _year_count_rows(entity_col, entity_value, start_year, end_year, year_to_count)

    mid_year = (start_year + end_year) // 2
    left_rows = _count_pmids_per_year_via_windows(
        entity_value=entity_value,
        entity_col=entity_col,
        clause=clause,
        start_year=start_year,
        end_year=mid_year,
        email=email,
        metadata_cache=metadata_cache,
        fetch_threshold=fetch_threshold,
        metadata_batch_size=metadata_batch_size,
    )
    right_rows = _count_pmids_per_year_via_windows(
        entity_value=entity_value,
        entity_col=entity_col,
        clause=clause,
        start_year=mid_year + 1,
        end_year=end_year,
        email=email,
        metadata_cache=metadata_cache,
        fetch_threshold=fetch_threshold,
        metadata_batch_size=metadata_batch_size,
    )
    return left_rows + right_rows


def _plan_year_counts_via_windows(
    *,
    entity_value: str,
    entity_col: str,
    clause: str,
    start_year: int,
    end_year: int,
    email: str,
    fetch_threshold: int,
) -> tuple[list[dict], list[dict]]:
    """Plan yearly background counts, deferring metadata fetches for multi-year windows."""
    if start_year > end_year:
        return [], []

    min_date = f"{start_year}/01/01"
    max_date = f"{end_year}/12/31"
    _, total_count, webenv, query_key = search_term_clause(
        clause,
        min_date=min_date,
        max_date=max_date,
        email=email,
    )

    if total_count is None:
        raise RuntimeError(
            f"PubMed background-count search failed for {entity_col}={entity_value!r}, "
            f"years={start_year}-{end_year}; no zero rows were written."
        )

    if total_count == 0:
        return _year_count_rows(entity_col, entity_value, start_year, end_year, {}), []

    if start_year == end_year:
        return [{entity_col: entity_value, "year": start_year, "count": int(total_count)}], []

    if total_count <= fetch_threshold and webenv and query_key:
        query = f'({clause}) AND ("{min_date}"[Date - Publication] : "{max_date}"[Date - Publication])'
        pmids_all = list(dict.fromkeys(retrieve_pmids(query, webenv, query_key, total_count)))
        return [], [
            {
                "entity_value": entity_value,
                "entity_col": entity_col,
                "start_year": start_year,
                "end_year": end_year,
                "pmids": pmids_all,
            }
        ]

    mid_year = (start_year + end_year) // 2
    left_rows, left_windows = _plan_year_counts_via_windows(
        entity_value=entity_value,
        entity_col=entity_col,
        clause=clause,
        start_year=start_year,
        end_year=mid_year,
        email=email,
        fetch_threshold=fetch_threshold,
    )
    right_rows, right_windows = _plan_year_counts_via_windows(
        entity_value=entity_value,
        entity_col=entity_col,
        clause=clause,
        start_year=mid_year + 1,
        end_year=end_year,
        email=email,
        fetch_threshold=fetch_threshold,
    )
    return left_rows + right_rows, left_windows + right_windows


def _year_rows_from_planned_metadata_windows(
    windows: list[dict],
    *,
    metadata_cache: dict[str, dict],
) -> list[dict]:
    rows: list[dict] = []
    for window in windows:
        entity_col = str(window["entity_col"])
        entity_value = str(window["entity_value"])
        start_year = int(window["start_year"])
        end_year = int(window["end_year"])
        year_to_count: dict[int, int] = {}
        for pmid in normalize_pmids(window["pmids"]):
            meta = get_metadata(pmid, metadata_cache)
            pub_date = meta["date"]
            if not pub_date:
                continue
            pub_year = int(pub_date.year)
            if start_year <= pub_year <= end_year:
                year_to_count[pub_year] = year_to_count.get(pub_year, 0) + 1
        rows.extend(_year_count_rows(entity_col, entity_value, start_year, end_year, year_to_count))
    return rows


def _count_pmids_per_year_resumable(
    *,
    entity_values: Sequence[str],
    entity_col: str,
    clause_builder,
    output_path: str,
    start_year: int,
    end_year: int,
    email: str,
    save_every: int,
    progress_every: int,
    fetch_threshold: int,
    metadata_batch_size: int,
) -> pd.DataFrame:
    import os

    entity_values = [str(v) for v in entity_values]
    if len(entity_values) == 0:
        return pd.DataFrame(columns=[entity_col, "year", "count"])

    if os.path.exists(output_path):
        try:
            existing = pd.read_parquet(output_path)
        except Exception:
            try:
                existing = pd.read_csv(output_path)
            except Exception:
                existing = pd.DataFrame(columns=[entity_col, "year", "count"])
    else:
        existing = pd.DataFrame(columns=[entity_col, "year", "count"])

    if not {entity_col, "year", "count"}.issubset(existing.columns):
        existing = pd.DataFrame(columns=[entity_col, "year", "count"])

    existing = clean_background_counts(existing, key_col=entity_col)

    tasks: list[tuple[str, int, int]] = []
    years = list(range(start_year, end_year + 1))
    for entity_value in entity_values:
        completed_years = set(
            existing.loc[existing[entity_col] == entity_value, "year"].dropna().astype(int).tolist()
        )
        missing_years = [year for year in years if year not in completed_years]
        if not missing_years:
            continue
        window_start = missing_years[0]
        prev_year = missing_years[0]
        for year in missing_years[1:]:
            if year == prev_year + 1:
                prev_year = year
                continue
            tasks.append((entity_value, window_start, prev_year))
            window_start = year
            prev_year = year
        tasks.append((entity_value, window_start, prev_year))

    total = len(tasks)
    if total == 0:
        return existing.sort_values([entity_col, "year"]).reset_index(drop=True)

    direct_results: list[dict] = []
    metadata_windows: list[dict] = []
    metadata_cache: dict[str, dict] = {}

    import time
    from collections import deque

    recent_progress = deque()
    RECENT_WINDOW_SEC = 900

    try:
        for idx, (entity_value, window_start, window_end) in enumerate(tasks, start=1):
            clause = clause_builder(entity_value)
            window_rows, window_metadata_windows = _plan_year_counts_via_windows(
                entity_value=entity_value,
                entity_col=entity_col,
                clause=clause,
                start_year=window_start,
                end_year=window_end,
                email=email,
                fetch_threshold=fetch_threshold,
            )
            direct_results.extend(window_rows)
            metadata_windows.extend(window_metadata_windows)

            if progress_every and (idx % progress_every == 0):
                now = time.time()
                recent_progress.append((now, idx))
                cutoff = now - RECENT_WINDOW_SEC
                while len(recent_progress) > 1 and recent_progress[0][0] < cutoff:
                    recent_progress.popleft()
                eta_str = "[ETA --]"
                if len(recent_progress) >= 2:
                    dt = recent_progress[-1][0] - recent_progress[0][0]
                    dp = recent_progress[-1][1] - recent_progress[0][1]
                    if dt > 0 and dp > 0:
                        rate = dp / dt
                        remaining = max(0, total - idx)
                        eta_sec = remaining / rate if rate > 0 else 0
                        eta_h = int(eta_sec // 3600)
                        eta_m = int((eta_sec % 3600) // 60)
                        eta_str = f"[ETA {eta_h}h {eta_m} min]"
                print(
                    f"Progress: {idx}/{total} windows processed… {eta_str} | "
                    f"{entity_col}={entity_value} | years={window_start}-{window_end}"
                )

            if save_every and (len(direct_results) >= save_every):
                batch_df = pd.DataFrame(direct_results)
                combined = clean_background_counts(pd.concat([existing, batch_df], ignore_index=True), key_col=entity_col)
                atomic_write_parquet(combined, output_path, index=False)
                existing = combined
                direct_results = []
    except KeyboardInterrupt:
        print("Interrupted. Saving partial results…")
    finally:
        if direct_results:
            batch_df = pd.DataFrame(direct_results)
            combined = clean_background_counts(pd.concat([existing, batch_df], ignore_index=True), key_col=entity_col)
            try:
                atomic_write_parquet(combined, output_path, index=False)
            except Exception:
                combined.to_csv(output_path, index=False)
            existing = combined

    if metadata_windows:
        metadata_pmids = normalize_pmids(
            pmid
            for window in metadata_windows
            for pmid in window["pmids"]
        )
        total_window_pmids = sum(len(normalize_pmids(window["pmids"])) for window in metadata_windows)
        print(
            f"[background_metadata:{entity_col}] fetching metadata for {len(metadata_pmids)} unique PMIDs "
            f"from {total_window_pmids} window-PMID assignments across {len(metadata_windows)} window(s)."
        )
        update_metadata_cache(metadata_pmids, metadata_cache, email=email, batch_size=metadata_batch_size)
        metadata_results = _year_rows_from_planned_metadata_windows(metadata_windows, metadata_cache=metadata_cache)
        if metadata_results:
            batch_df = pd.DataFrame(metadata_results)
            combined = clean_background_counts(pd.concat([existing, batch_df], ignore_index=True), key_col=entity_col)
            atomic_write_parquet(combined, output_path, index=False)
            existing = combined

    return existing.sort_values([entity_col, "year"]).reset_index(drop=True)


def count_pmids_per_year_for_pathogens_resumable(
    pathogens: Sequence[str],
    bacteria_synonym_df: pd.DataFrame,
    *,
    start_year: int | None = None,
    end_year: int | None = None,
    email: str = "your_email@example.com",
    output_path: str = "pathogen_tiab_counts.parquet",
    save_every: int = 200,
    progress_every: int = 50,
    fetch_threshold: int = 9500,
    metadata_batch_size: int = 1000,
) -> pd.DataFrame:
    """Resumable variant of yearly PMID counts per pathogen (TIAB + synonyms).

    - Skips (pathogen, year) pairs already present in ``output_path``.
    - Periodically merges and saves results; safe to interrupt and resume.
    - Deduplicates on (pathogen, year) when saving.
    """
    pathogens = list(pathogens)
    if len(pathogens) == 0:
        return pd.DataFrame(columns=["pathogen", "year", "count"])

    if start_year is None:
        start_year = 1950
    if end_year is None:
        end_year = datetime.now().year

    return _count_pmids_per_year_resumable(
        entity_values=pathogens,
        entity_col="pathogen",
        clause_builder=lambda pathogen: build_pathogen_tiab_query(pathogen, bacteria_synonym_df),
        output_path=output_path,
        start_year=start_year,
        end_year=end_year,
        email=email,
        save_every=save_every,
        progress_every=progress_every,
        fetch_threshold=fetch_threshold,
        metadata_batch_size=metadata_batch_size,
    )


def count_pmids_per_year_for_antibiotics_resumable(
    antibiotics: Sequence[str],
    antibiotic_synonym_df: pd.DataFrame | None = None,
    *,
    start_year: int | None = None,
    end_year: int | None = None,
    email: str = "your_email@example.com",
    output_path: str = "antibiotic_tiab_counts.parquet",
    save_every: int = 200,
    progress_every: int = 50,
    fetch_threshold: int = 9500,
    metadata_batch_size: int = 1000,
) -> pd.DataFrame:
    """Resumable yearly PMID counts per antibiotic (Title/Abstract; synonym-aware).

    - Uses canonical antibiotic names plus spelling/synonym variants when provided.
    - Skips already-processed (antibiotic, year) pairs from ``output_path``.
    - Saves incrementally and resumes safely after interruption.
    """
    antibiotics = list(antibiotics)
    if len(antibiotics) == 0:
        return pd.DataFrame(columns=["antibiotic", "year", "count"])

    if start_year is None:
        start_year = 1950
    if end_year is None:
        end_year = datetime.now().year

    return _count_pmids_per_year_resumable(
        entity_values=antibiotics,
        entity_col="antibiotic",
        clause_builder=lambda antibiotic: build_antibiotic_query(
            antibiotic,
            antibiotic_synonym_df,
            field="Title/Abstract",
        ),
        output_path=output_path,
        start_year=start_year,
        end_year=end_year,
        email=email,
        save_every=save_every,
        progress_every=progress_every,
        fetch_threshold=fetch_threshold,
        metadata_batch_size=metadata_batch_size,
    )
