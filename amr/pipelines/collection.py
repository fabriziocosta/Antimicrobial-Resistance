"""High-level data collection pipelines for PubMed AMR queries."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta
from typing import Optional, Sequence

import pandas as pd

from amr.pubmed.queries import (
    build_pathogen_tiab_query,
    prepare_antibiotic_pubmed_query_tiab,
)
from amr.pubmed.search import (
    esearch_count_for_year,
    search_term_clause,
    search_pmids,
    retrieve_pmids,
)
from amr.pubmed.cache import (
    merge_with_existing,
    update_metadata_cache,
    get_metadata,
)


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
            rows.append({"pathogen": pathogen, "year": year, "count": cnt})
            done += 1
            if done % 50 == 0:
                print(f"Progress: {done}/{total_tasks} year-queries done")
    return pd.DataFrame(rows)


def count_pmids_per_year_for_antibiotics(
    antibiotics: Sequence[str],
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
        clause = prepare_antibiotic_pubmed_query_tiab(ab)
        for year in range(start_year, end_year + 1):
            cnt = esearch_count_for_year(clause, year, email=email)
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
    email: str = "your_email@example.com",
    batch_size: int = 5000,
    extra_clauses: Optional[Sequence[str]] = None,
) -> list[str]:
    """Retrieve PMIDs within a date range; splits queries that exceed PubMed limits."""
    query, count, webenv, query_key = search_pmids(
        antibiotic,
        pathogen,
        bacteria_synonym_df,
        extra_clauses=extra_clauses,
        min_date=min_date,
        max_date=max_date,
        email=email,
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
            email=email,
            batch_size=batch_size,
            extra_clauses=extra_clauses,
        )
        right_pmids = fetch_pmids_by_date_and_keywords(
            min_date=(mid_dt + timedelta(days=1)).strftime("%Y/%m/%d"),
            max_date=max_date,
            antibiotic=antibiotic,
            pathogen=pathogen,
            bacteria_synonym_df=bacteria_synonym_df,
            email=email,
            batch_size=batch_size,
            extra_clauses=extra_clauses,
        )
        return left_pmids + right_pmids
    return retrieve_pmids(query, webenv, query_key, count, batch_size=batch_size)


def fetch_pmids_intervals(
    pathogens: Sequence[str],
    antibiotics: Sequence[str],
    bacteria_synonym_df: pd.DataFrame,
    min_date: str,
    max_date: str,
    year_gap: int = 5,
    email: str = "your_email@example.com",
    parquet_path: str = "pmids_results.parquet",
    save_every: int = 50,
    extra_keywords: Optional[Sequence[str]] = None,
    full_fetch_threshold: int = 2000,
    summary_batch_size: int = 200,
    verbose: int = 1,
    progress_every: int = 200,
) -> pd.DataFrame:
    """Query PubMed in intervals for all pathogen/antibiotic combinations."""
    import os

    if os.path.exists(parquet_path):
        existing_df = pd.read_parquet(parquet_path)
        if "venue_histogram" not in existing_df.columns:
            existing_df["venue_histogram"] = None
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

    new_results: list[dict] = []
    metadata_cache: dict[str, dict] = {}

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
    task_counter = 0
    event_counter = 0
    progress_every = max(1, int(progress_every))
    event_stats = {"cached": 0, "zero": 0, "full_fetch_partitioned": 0, "new": 0}

    def _log_event(kind: str, message: str) -> None:
        nonlocal event_counter
        event_counter += 1
        event_stats[kind] = event_stats.get(kind, 0) + 1
        if verbose >= 2:
            print(message)
        elif verbose >= 1 and event_counter % progress_every == 0:
            print(
                f"Progress events {event_counter}: "
                f"cached={event_stats.get('cached',0)} "
                f"zero={event_stats.get('zero',0)} "
                f"full_fetch={event_stats.get('full_fetch_partitioned',0)} "
                f"new={event_stats.get('new',0)}"
            )

    for pathogen in pathogens:
        for antibiotic in antibiotics:
            task_counter += 1
            task_progress = (task_counter / max(total_tasks, 1)) * 100

            # Determine missing intervals for this pair
            interval_entries = []
            missing_entries = []
            for interval_start, interval_end in intervals:
                start_dt = datetime.strptime(interval_start, "%Y/%m/%d")
                end_dt = datetime.strptime(interval_end, "%Y/%m/%d")

                mask = (
                    (existing_df["pathogen"] == pathogen)
                    & (existing_df["antibiotic"] == antibiotic)
                    & (existing_df["start_date"] == interval_start)
                    & (existing_df["end_date"] == interval_end)
                )

                entry = {
                    "start": interval_start,
                    "end": interval_end,
                    "start_dt": start_dt,
                    "end_dt": end_dt,
                    "cached": False,
                    "count": 0,
                }

                if mask.any():
                    entry["cached"] = True
                    entry["count"] = int(existing_df.loc[mask, "count"].iloc[0])
                    _log_event(
                        "cached",
                        f"[{task_counter:>3}/{total_tasks}] "
                        f"({task_progress:5.1f}%) | "
                        f"{pathogen:<40} | {antibiotic:<20} | "
                        f"{interval_start} – {interval_end} | "
                        f"{entry['count']:5} PMIDs (cached)"
                    )
                else:
                    missing_entries.append(entry)
                interval_entries.append(entry)

            if not missing_entries:
                continue

            query, total_count, webenv, query_key = search_pmids(
                antibiotic,
                pathogen,
                bacteria_synonym_df,
                extra_clauses=extra_keywords,
                min_date=min_date,
                max_date=max_date,
                email=email,
            )

            if total_count == 0:
                for entry in missing_entries:
                    new_results.append(
                        {
                            "pathogen": pathogen,
                            "antibiotic": antibiotic,
                            "start_date": entry["start"],
                            "end_date": entry["end"],
                            "pmids": [],
                            "count": 0,
                            "venue_histogram": json.dumps({}),
                        }
                    )
                    _log_event(
                        "zero",
                        f"[{task_counter:>3}/{total_tasks}] "
                        f"({task_progress:5.1f}%) | "
                        f"{pathogen:<40} | {antibiotic:<20} | "
                        f"{entry['start']} – {entry['end']} | "
                        f"{0:5} PMIDs (new)"
                    )
                if len(new_results) >= save_every:
                    df_new = pd.DataFrame(new_results)
                    existing_df = merge_with_existing(df_new, parquet_path=parquet_path)
                    new_results = []
                continue

            if total_count <= full_fetch_threshold:
                pmids_all = retrieve_pmids(query, webenv, query_key, total_count)
                update_metadata_cache(pmids_all, metadata_cache, email=email, batch_size=summary_batch_size)

                for entry in missing_entries:
                    pmids_interval = []
                    venue_counter = Counter()
                    for pmid in pmids_all:
                        meta = get_metadata(pmid, metadata_cache)
                        pub_date = meta["date"]
                        if not pub_date:
                            continue
                        if not (overall_start_dt <= pub_date <= overall_end_dt):
                            continue
                        if entry["start_dt"] <= pub_date <= entry["end_dt"]:
                            pmids_interval.append(pmid)
                            venue_value = meta["venue"] or "Unknown"
                            venue_counter[venue_value] += 1
                    histogram_json = json.dumps(dict(venue_counter))
                    new_results.append(
                        {
                            "pathogen": pathogen,
                            "antibiotic": antibiotic,
                            "start_date": entry["start"],
                            "end_date": entry["end"],
                            "pmids": pmids_interval,
                            "count": len(pmids_interval),
                            "venue_histogram": histogram_json,
                        }
                    )
                    _log_event(
                        "full_fetch_partitioned",
                        f"[{task_counter:>3}/{total_tasks}] "
                        f"({task_progress:5.1f}%) | "
                        f"{pathogen:<40} | {antibiotic:<20} | "
                        f"{entry['start']} – {entry['end']} | "
                        f"{len(pmids_interval):5} PMIDs (full fetch partitioned)"
                    )
                    if len(new_results) >= save_every:
                        df_new = pd.DataFrame(new_results)
                        existing_df = merge_with_existing(df_new, parquet_path=parquet_path)
                        new_results = []
                continue

            for entry in missing_entries:
                pmids_interval = fetch_pmids_by_date_and_keywords(
                    min_date=entry["start"],
                    max_date=entry["end"],
                    antibiotic=antibiotic,
                    pathogen=pathogen,
                    bacteria_synonym_df=bacteria_synonym_df,
                    email=email,
                    extra_clauses=extra_keywords,
                )
                update_metadata_cache(pmids_interval, metadata_cache, email=email, batch_size=summary_batch_size)
                from collections import Counter as _Counter

                venue_counter = _Counter()
                filtered_pmids = []
                for pmid in pmids_interval:
                    meta = get_metadata(pmid, metadata_cache)
                    pub_date = meta["date"]
                    if pub_date and not (entry["start_dt"] <= pub_date <= entry["end_dt"]):
                        continue
                    filtered_pmids.append(pmid)
                    venue_value = meta["venue"] or "Unknown"
                    venue_counter[venue_value] += 1
                histogram_json = json.dumps(dict(venue_counter))
                new_results.append(
                    {
                        "pathogen": pathogen,
                        "antibiotic": antibiotic,
                        "start_date": entry["start"],
                        "end_date": entry["end"],
                        "pmids": filtered_pmids,
                        "count": len(filtered_pmids),
                        "venue_histogram": histogram_json,
                    }
                )
                _log_event(
                    "new",
                    f"[{task_counter:>3}/{total_tasks}] "
                    f"({task_progress:5.1f}%) | "
                    f"{pathogen:<40} | {antibiotic:<20} | "
                    f"{entry['start']} – {entry['end']} | "
                    f"{len(filtered_pmids):5} PMIDs (new)"
                )
                if len(new_results) >= save_every:
                    df_new = pd.DataFrame(new_results)
                    existing_df = merge_with_existing(df_new, parquet_path=parquet_path)
                    new_results = []

    if new_results:
        df_new = pd.DataFrame(new_results)
        existing_df = merge_with_existing(df_new, parquet_path=parquet_path)

    if verbose >= 1:
        print(
            "Collection summary: "
            f"events={event_counter}, "
            f"cached={event_stats.get('cached',0)}, "
            f"zero={event_stats.get('zero',0)}, "
            f"full_fetch={event_stats.get('full_fetch_partitioned',0)}, "
            f"new={event_stats.get('new',0)}"
        )

    return existing_df


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

    existing[entity_col] = existing[entity_col].astype(str)
    existing["year"] = pd.to_numeric(existing["year"], errors="coerce").astype("Int64")

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

    results: list[dict] = []
    metadata_cache: dict[str, dict] = {}

    import time
    from collections import deque

    recent_progress = deque()
    RECENT_WINDOW_SEC = 900

    try:
        for idx, (entity_value, window_start, window_end) in enumerate(tasks, start=1):
            clause = clause_builder(entity_value)
            window_rows = _count_pmids_per_year_via_windows(
                entity_value=entity_value,
                entity_col=entity_col,
                clause=clause,
                start_year=window_start,
                end_year=window_end,
                email=email,
                metadata_cache=metadata_cache,
                fetch_threshold=fetch_threshold,
                metadata_batch_size=metadata_batch_size,
            )
            results.extend(window_rows)

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

            if save_every and (len(results) >= save_every):
                batch_df = pd.DataFrame(results)
                combined = pd.concat([existing, batch_df], ignore_index=True)
                combined = combined.drop_duplicates(subset=[entity_col, "year"], keep="last")
                combined.to_parquet(output_path, index=False)
                existing = combined
                results = []
    except KeyboardInterrupt:
        print("Interrupted. Saving partial results…")
    finally:
        if results:
            batch_df = pd.DataFrame(results)
            combined = pd.concat([existing, batch_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=[entity_col, "year"], keep="last")
            try:
                combined.to_parquet(output_path, index=False)
            except Exception:
                combined.to_csv(output_path, index=False)
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
    fetch_threshold: int = 2000,
    metadata_batch_size: int = 200,
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
    *,
    start_year: int | None = None,
    end_year: int | None = None,
    email: str = "your_email@example.com",
    output_path: str = "antibiotic_tiab_counts.parquet",
    save_every: int = 200,
    progress_every: int = 50,
    fetch_threshold: int = 2000,
    metadata_batch_size: int = 200,
) -> pd.DataFrame:
    """Resumable yearly PMID counts per antibiotic (Title/Abstract; hyphen/slash variants).

    - Uses prepare_antibiotic_pubmed_query_tiab to handle hyphens/slashes via AND splits.
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
        clause_builder=prepare_antibiotic_pubmed_query_tiab,
        output_path=output_path,
        start_year=start_year,
        end_year=end_year,
        email=email,
        save_every=save_every,
        progress_every=progress_every,
        fetch_threshold=fetch_threshold,
        metadata_batch_size=metadata_batch_size,
    )
