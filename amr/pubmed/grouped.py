"""Grouped PubMed collection with resumable search and local pair attribution."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence
from uuid import uuid4

import pandas as pd

from amr.io import atomic_write_parquet
from amr.validation import clean_interval_results, normalize_entity_key
from .cache import get_metadata, update_metadata_cache
from .pmids import normalize_pmids
from .queries import (
    build_antibiotic_group_pathogen_query,
    build_pathogen_tiab_query,
    expand_antibiotic_synonyms,
)
from .search import fetch_pubmed_abstracts_batch, retrieve_pmids, search_term_clause
from .yearly import (
    load_pubmed_metadata_cache,
    metadata_dataframe_to_dict,
    metadata_dict_to_dataframe,
    save_pubmed_metadata_cache,
)


def split_antibiotic_groups(antibiotics: Sequence[str], group_size: int = 10) -> list[list[str]]:
    """Sort antibiotics deterministically, then split into groups of at most 10."""
    if not 1 <= int(group_size) <= 10:
        raise ValueError("antibiotic group_size must be between 1 and 10")
    ordered = sorted({str(value).strip() for value in antibiotics if str(value).strip()}, key=str.casefold)
    size = int(group_size)
    return [ordered[start : start + size] for start in range(0, len(ordered), size)]


def _db_connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=60)
    connection.execute("PRAGMA busy_timeout = 60000")
    return connection


def _initialize_cache(path: Path) -> None:
    connection = _db_connect(path)
    try:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        connection.execute(
            """CREATE TABLE IF NOT EXISTS search_nodes (
                cache_key TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                split_date TEXT,
                pmids_json TEXT
            )"""
        )
        connection.execute(
            """CREATE TABLE IF NOT EXISTS abstracts (
                pmid TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                abstract TEXT NOT NULL
            )"""
        )
        connection.execute(
            """CREATE TABLE IF NOT EXISTS collection_runs (
                scope_hash TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                status TEXT NOT NULL
            )"""
        )
        connection.commit()
    finally:
        connection.close()


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _node_key(base_query: str, start: str, end: str, namespace: str) -> str:
    return _hash_text(f"{namespace}\0{base_query}\0{start}\0{end}")


def _start_collection_run(cache_path: Path, scope_hash: str, *, force_refresh: bool) -> str:
    connection = _db_connect(cache_path)
    try:
        row = connection.execute(
            "SELECT run_id, status FROM collection_runs WHERE scope_hash = ?", (scope_hash,)
        ).fetchone()
        if row and (row[1] == "active" or not force_refresh):
            run_id = str(row[0])
        else:
            run_id = uuid4().hex
        connection.execute(
            """INSERT INTO collection_runs(scope_hash, run_id, status) VALUES (?, ?, 'active')
               ON CONFLICT(scope_hash) DO UPDATE SET run_id=excluded.run_id, status='active'""",
            (scope_hash, run_id),
        )
        connection.commit()
        return run_id
    finally:
        connection.close()


def _finish_collection_run(cache_path: Path, scope_hash: str, run_id: str) -> None:
    connection = _db_connect(cache_path)
    try:
        connection.execute(
            "UPDATE collection_runs SET status='complete' WHERE scope_hash=? AND run_id=?",
            (scope_hash, run_id),
        )
        connection.commit()
    finally:
        connection.close()


def _get_node(connection: sqlite3.Connection, key: str) -> tuple[str, str | None, list[str]] | None:
    row = connection.execute(
        "SELECT status, split_date, pmids_json FROM search_nodes WHERE cache_key = ?", (key,)
    ).fetchone()
    if row is None:
        return None
    status, split_date, pmids_json = row
    return status, split_date, json.loads(pmids_json) if pmids_json else []


def _save_node(
    connection: sqlite3.Connection,
    key: str,
    status: str,
    *,
    split_date: str | None = None,
    pmids: Sequence[str] = (),
) -> None:
    connection.execute(
        """INSERT INTO search_nodes(cache_key, status, split_date, pmids_json)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(cache_key) DO UPDATE SET
             status=excluded.status,
             split_date=excluded.split_date,
             pmids_json=excluded.pmids_json""",
        (key, status, split_date, json.dumps(list(pmids)) if pmids else "[]"),
    )
    connection.commit()


def _date_midpoint(start: str, end: str) -> str:
    start_dt = datetime.strptime(start, "%Y/%m/%d")
    end_dt = datetime.strptime(end, "%Y/%m/%d")
    midpoint = start_dt + (end_dt - start_dt) // 2
    return midpoint.strftime("%Y/%m/%d")


def _collect_query_tree(
    connection: sqlite3.Connection,
    base_query: str,
    start: str,
    end: str,
    *,
    email: str,
    retrieve_batch_size: int,
    split_threshold: int,
    verbose: bool,
    cache_namespace: str,
) -> list[str]:
    """Retrieve one query, recursively partitioning large date windows.

    Completed leaves and split markers are committed to SQLite as they finish, so
    restarting resumes from the saved query tree instead of repeating retrieval.
    """
    key = _node_key(base_query, start, end, cache_namespace)
    cached = _get_node(connection, key)
    if cached and cached[0] == "complete":
        return cached[2]
    if cached and cached[0] == "split":
        midpoint = cached[1]
        if not midpoint:
            raise RuntimeError("Grouped search cache has a split node without a split date.")
        right_start = (datetime.strptime(midpoint, "%Y/%m/%d") + timedelta(days=1)).strftime("%Y/%m/%d")
        return normalize_pmids(
            _collect_query_tree(
                connection, base_query, start, midpoint, email=email,
                retrieve_batch_size=retrieve_batch_size, split_threshold=split_threshold,
                verbose=verbose, cache_namespace=cache_namespace,
            )
            + _collect_query_tree(
                connection, base_query, right_start, end, email=email,
                retrieve_batch_size=retrieve_batch_size, split_threshold=split_threshold,
                verbose=verbose, cache_namespace=cache_namespace,
            )
        )

    query, count, webenv, query_key = search_term_clause(
        base_query, min_date=start, max_date=end, email=email, verbose=verbose
    )
    if count is None:
        raise RuntimeError(f"Grouped PubMed search failed for {start}..{end}.")
    if count <= split_threshold:
        pmids = (
            retrieve_pmids(query, webenv, query_key, count, batch_size=retrieve_batch_size, verbose=verbose)
            if count
            else []
        )
        pmids = normalize_pmids(pmids)
        _save_node(connection, key, "complete", pmids=pmids)
        return pmids

    if start == end:
        raise RuntimeError(
            f"A single-day grouped PubMed query returned {count} records; reduce the antibiotic group size."
        )
    midpoint = _date_midpoint(start, end)
    if midpoint >= end:
        midpoint = (datetime.strptime(end, "%Y/%m/%d") - timedelta(days=1)).strftime("%Y/%m/%d")
    _save_node(connection, key, "split", split_date=midpoint)
    if verbose:
        print(f"Grouped PubMed query has {count:,} records; splitting {start}..{end} at {midpoint}.")
    right_start = (datetime.strptime(midpoint, "%Y/%m/%d") + timedelta(days=1)).strftime("%Y/%m/%d")
    return normalize_pmids(
        _collect_query_tree(
            connection, base_query, start, midpoint, email=email,
            retrieve_batch_size=retrieve_batch_size, split_threshold=split_threshold,
            verbose=verbose, cache_namespace=cache_namespace,
        )
        + _collect_query_tree(
            connection, base_query, right_start, end, email=email,
            retrieve_batch_size=retrieve_batch_size, split_threshold=split_threshold,
            verbose=verbose, cache_namespace=cache_namespace,
        )
    )


def _normalize_for_match(text: str) -> str:
    text = str(text or "").casefold().replace("-", " ").replace("/", " ")
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _variant_matches(text: str, variant: str) -> bool:
    normalized_text = _normalize_for_match(text)
    normalized_variant = _normalize_for_match(variant)
    if not normalized_variant:
        return False
    if re.search(rf"(?<!\w){re.escape(normalized_variant)}(?!\w)", normalized_text):
        return True
    parts = [part for part in re.split(r"[-/]", variant) if part.strip()]
    return len(parts) > 1 and all(
        re.search(rf"(?<!\w){re.escape(_normalize_for_match(part))}(?!\w)", normalized_text)
        for part in parts
    )


def _matched_antibiotics(
    text: str,
    antibiotics: Sequence[str],
    variants_by_antibiotic: dict[str, list[str]],
) -> list[str]:
    matched = []
    for antibiotic in antibiotics:
        if any(_variant_matches(text, variant) for variant in variants_by_antibiotic[antibiotic]):
            matched.append(antibiotic)
    return matched


def _load_abstract_cache(path: str | Path | None, search_cache_path: Path) -> dict[str, tuple[str, str]]:
    cache: dict[str, tuple[str, str]] = {}
    if not path or not Path(path).exists():
        frame = None
    else:
        try:
            frame = pd.read_parquet(path)
        except Exception:
            frame = None
    if frame is not None and {"pmid", "title", "abstract"}.issubset(frame.columns):
        cache.update({
            str(row.pmid): (str(row.title or ""), str(row.abstract or ""))
            for row in frame[["pmid", "title", "abstract"]].itertuples(index=False)
        })
    connection = _db_connect(search_cache_path)
    try:
        cache.update({
            str(pmid): (str(title or ""), str(abstract or ""))
            for pmid, title, abstract in connection.execute("SELECT pmid, title, abstract FROM abstracts")
        })
    finally:
        connection.close()
    return cache


def _save_abstract_cache(cache: dict[str, tuple[str, str]], path: str | Path | None) -> None:
    if not path:
        return
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        [(pmid, title, abstract) for pmid, (title, abstract) in cache.items()],
        columns=["pmid", "title", "abstract"],
    )
    atomic_write_parquet(frame, cache_path, index=False)


def _fetch_missing_abstracts(
    pmids: Sequence[str],
    abstract_cache: dict[str, tuple[str, str]],
    *,
    email: str,
    batch_size: int,
    search_cache_path: Path,
) -> None:
    missing = [pmid for pmid in normalize_pmids(pmids) if pmid not in abstract_cache]
    for offset in range(0, len(missing), max(1, batch_size)):
        batch = missing[offset : offset + max(1, batch_size)]
        fetched = fetch_pubmed_abstracts_batch(batch, email=email)
        if set(fetched) != set(batch):
            raise RuntimeError("PubMed abstract response omitted requested records; rerun to resume.")
        abstract_cache.update(fetched)
        connection = _db_connect(search_cache_path)
        try:
            connection.executemany(
                """INSERT INTO abstracts(pmid, title, abstract) VALUES (?, ?, ?)
                   ON CONFLICT(pmid) DO UPDATE SET title=excluded.title, abstract=excluded.abstract""",
                [(pmid, title, abstract) for pmid, (title, abstract) in fetched.items()],
            )
            connection.commit()
        finally:
            connection.close()


def fetch_pmids_intervals_grouped(
    *,
    pathogens: Sequence[str],
    antibiotics: Sequence[str],
    bacteria_synonym_df: pd.DataFrame,
    min_date: str,
    max_date: str,
    antibiotic_synonym_df: pd.DataFrame | None = None,
    email: str,
    parquet_path: str | Path,
    metadata_cache_path: str | Path | None,
    abstract_cache_path: str | Path | None,
    group_search_cache_path: str | Path,
    extra_keywords: Sequence[str] | None = None,
    year_gap: int = 5,
    group_size: int = 10,
    full_fetch_threshold: int = 9500,
    retrieve_batch_size: int = 9000,
    abstract_batch_size: int = 200,
    metadata_batch_size: int = 1000,
    max_workers: int = 4,
    max_pending_futures: int | None = None,
    refresh_cached_intervals: bool = False,
    refresh_cached_pairs: Sequence[tuple[str, str]] | None = None,
    verbose: int = 1,
    progress_every: int = 100,
) -> pd.DataFrame:
    """Search antibiotics in groups, cache each search partition, then assign PMIDs locally."""
    if not email or email == "your_email@example.com":
        raise ValueError("A real PubMed email is required for grouped collection.")
    groups = split_antibiotic_groups(antibiotics, group_size)
    pathogens = sorted({str(value).strip() for value in pathogens if str(value).strip()}, key=str.casefold)
    parquet_path = Path(parquet_path)
    checkpoint_dir = Path(f"{parquet_path}.checkpoints")
    cached_frames = []
    if parquet_path.exists():
        cached_frames.append(pd.read_parquet(parquet_path))
    if checkpoint_dir.exists():
        cached_frames.extend(pd.read_parquet(path) for path in sorted(checkpoint_dir.glob("part-*.parquet")))
    if cached_frames:
        existing_df = clean_interval_results(pd.concat(cached_frames, ignore_index=True))
    else:
        existing_df = pd.DataFrame(
            columns=["pathogen", "antibiotic", "start_date", "end_date", "pmids", "count", "venue_histogram"]
        )

    overall_start = datetime.strptime(min_date, "%Y/%m/%d")
    overall_end = datetime.strptime(max_date, "%Y/%m/%d")
    intervals: list[tuple[str, str, datetime, datetime]] = []
    current = overall_start
    while current <= overall_end:
        end = datetime(min(current.year + int(year_gap) - 1, overall_end.year), 12, 31)
        if end > overall_end:
            end = overall_end
        intervals.append((current.strftime("%Y/%m/%d"), end.strftime("%Y/%m/%d"), current, end))
        current = datetime(end.year + 1, 1, 1)

    refresh_keys = {
        (normalize_entity_key(p), normalize_entity_key(a))
        for p, a in (refresh_cached_pairs or [])
    }
    expected_ranges = {f"{start}\0{end}" for start, end, _, _ in intervals}
    interval_labels = existing_df["start_date"].astype(str) + "\0" + existing_df["end_date"].astype(str)
    in_scope = interval_labels.isin(expected_ranges)
    coverage = existing_df.loc[in_scope, ["pathogen", "antibiotic"]].copy()
    coverage["pathogen_key"] = coverage["pathogen"].map(normalize_entity_key)
    coverage["antibiotic_key"] = coverage["antibiotic"].map(normalize_entity_key)
    coverage_counts = coverage.groupby(["pathogen_key", "antibiotic_key"], sort=False).size()
    complete_pairs = set(coverage_counts[coverage_counts.ge(len(intervals))].index.tolist())

    targets: dict[str, list[str]] = {}
    for pathogen in pathogens:
        target_antibiotics = []
        for group in groups:
            for antibiotic in group:
                pair_key = (normalize_entity_key(pathogen), normalize_entity_key(antibiotic))
                selected = not refresh_keys or pair_key in refresh_keys
                refresh_pair = refresh_cached_intervals and selected
                complete = pair_key in complete_pairs
                if refresh_pair or not complete:
                    target_antibiotics.append(antibiotic)
        if target_antibiotics:
            targets[pathogen] = target_antibiotics

    variants_by_antibiotic = {
        antibiotic: expand_antibiotic_synonyms(antibiotic, antibiotic_synonym_df)
        for pathogen_antibiotics in targets.values()
        for antibiotic in pathogen_antibiotics
    }
    pathogen_queries = {
        pathogen: build_pathogen_tiab_query(pathogen, bacteria_synonym_df)
        for pathogen in targets
    }
    tasks = []
    for pathogen, target_antibiotics in targets.items():
        for group in split_antibiotic_groups(target_antibiotics, group_size):
            query = build_antibiotic_group_pathogen_query(
                group,
                pathogen,
                bacteria_synonym_df,
                antibiotic_synonym_df,
                list(extra_keywords or []),
                pathogen_clause=pathogen_queries[pathogen],
                variants_by_antibiotic=variants_by_antibiotic,
            )
            tasks.append({"pathogen": pathogen, "antibiotics": group, "query": query})

    cache_path = Path(group_search_cache_path)
    _initialize_cache(cache_path)
    scope_hash = _hash_text(
        json.dumps(
            {
                "min_date": min_date,
                "max_date": max_date,
                "queries": [
                    [task["pathogen"], task["antibiotics"], task["query"]]
                    for task in tasks
                ],
            },
            sort_keys=True,
        )
    )
    run_id = _start_collection_run(
        cache_path, scope_hash, force_refresh=bool(refresh_cached_intervals)
    )

    workers = max(1, int(max_workers))
    max_pending = max(workers, int(max_pending_futures or workers * 2))
    abstract_cache = _load_abstract_cache(abstract_cache_path, cache_path)
    metadata_cache = {}
    if metadata_cache_path and Path(metadata_cache_path).exists():
        metadata_cache = metadata_dataframe_to_dict(load_pubmed_metadata_cache(metadata_cache_path))
    metadata_dirty = False
    pair_pmids: dict[tuple[str, str, str, str], set[str]] = {}
    pair_venues: dict[tuple[str, str, str, str], Counter[str]] = {}
    completed = 0
    total = len(tasks)

    def _run_task(task: dict[str, object]) -> dict[str, object]:
        connection = _db_connect(cache_path)
        try:
            pmids = _collect_query_tree(
                connection,
                str(task["query"]),
                min_date,
                max_date,
                email=email,
                retrieve_batch_size=retrieve_batch_size,
                split_threshold=full_fetch_threshold,
                verbose=verbose >= 2,
                cache_namespace=run_id,
            )
        finally:
            connection.close()
        return {**task, "pmids": pmids}

    def _consume(result: dict[str, object]) -> None:
        nonlocal metadata_dirty, completed
        pathogen = str(result["pathogen"])
        group = list(result["antibiotics"])
        pmids = normalize_pmids(result["pmids"])
        _fetch_missing_abstracts(
            pmids,
            abstract_cache,
            email=email,
            batch_size=abstract_batch_size,
            search_cache_path=cache_path,
        )
        before = len(metadata_cache)
        update_metadata_cache(pmids, metadata_cache, email=email, batch_size=metadata_batch_size, verbose=False)
        metadata_dirty = metadata_dirty or len(metadata_cache) > before

        for pmid in pmids:
            pub_date = get_metadata(pmid, metadata_cache)["date"]
            if not pub_date or not (overall_start <= pub_date <= overall_end):
                continue
            interval = next(
                ((start, end) for start, end, start_dt, end_dt in intervals if start_dt <= pub_date <= end_dt),
                None,
            )
            if interval is None:
                continue
            title, abstract = abstract_cache.get(pmid, ("", ""))
            text = f"{title}\n{abstract}"
            for antibiotic in _matched_antibiotics(text, group, variants_by_antibiotic):
                key = (pathogen, antibiotic, interval[0], interval[1])
                pair_set = pair_pmids.setdefault(key, set())
                if pmid not in pair_set:
                    pair_set.add(pmid)
                    venue = str(get_metadata(pmid, metadata_cache).get("venue") or "Unknown")
                    pair_venues.setdefault(key, Counter())[venue] += 1

        completed += 1
        if verbose >= 1 and (completed % max(1, progress_every) == 0 or completed == total):
            print(f"Grouped PubMed searches: {completed:,}/{total:,} ({pathogen}; {len(group)} antibiotics).")
        if metadata_dirty and metadata_cache_path and (completed % 25 == 0 or completed == total):
            save_pubmed_metadata_cache(metadata_dict_to_dataframe(metadata_cache), metadata_cache_path)
            metadata_dirty = False

    pending = set()
    task_iterator = iter(tasks)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="pubmed-group") as executor:
        exhausted = False
        while pending or not exhausted:
            while not exhausted and len(pending) < max_pending:
                try:
                    task = next(task_iterator)
                except StopIteration:
                    exhausted = True
                    break
                pending.add(executor.submit(_run_task, task))
            if not pending:
                continue
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                _consume(future.result())

    if abstract_cache_path:
        _save_abstract_cache(abstract_cache, abstract_cache_path)
    if metadata_dirty and metadata_cache_path:
        save_pubmed_metadata_cache(metadata_dict_to_dataframe(metadata_cache), metadata_cache_path)

    refreshed_pairs = {
        (normalize_entity_key(pathogen), normalize_entity_key(antibiotic))
        for pathogen, antibiotics_for_pathogen in targets.items()
        for antibiotic in antibiotics_for_pathogen
    }
    existing_pair_index = pd.MultiIndex.from_arrays(
        [
            existing_df["pathogen"].map(normalize_entity_key),
            existing_df["antibiotic"].map(normalize_entity_key),
        ]
    )
    within_refresh_window = (
        pd.to_datetime(existing_df["end_date"], errors="coerce").ge(overall_start)
        & pd.to_datetime(existing_df["start_date"], errors="coerce").le(overall_end)
    )
    pair_is_refreshed = pd.Series(existing_pair_index.isin(refreshed_pairs), index=existing_df.index)
    kept_df = existing_df.loc[~(pair_is_refreshed & within_refresh_window)]

    rows: list[dict[str, object]] = []
    for pathogen, antibiotics_for_pathogen in targets.items():
        for antibiotic in antibiotics_for_pathogen:
            for start, end, _, _ in intervals:
                pmids = sorted(pair_pmids.get((pathogen, antibiotic, start, end), set()), key=int)
                rows.append(
                    {
                        "pathogen": pathogen,
                        "antibiotic": antibiotic,
                        "start_date": start,
                        "end_date": end,
                        "pmids": pmids,
                        "count": len(pmids),
                        "venue_histogram": json.dumps(dict(pair_venues.get((pathogen, antibiotic, start, end), {}))),
                    }
                )
    refreshed_df = pd.DataFrame(rows)
    combined = clean_interval_results(pd.concat([kept_df, refreshed_df], ignore_index=True))
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(combined, parquet_path, index=False)
    if checkpoint_dir.exists():
        for checkpoint in checkpoint_dir.glob("part-*.parquet"):
            checkpoint.unlink()
        try:
            checkpoint_dir.rmdir()
        except OSError:
            pass
    _finish_collection_run(cache_path, scope_hash, run_id)
    return combined


__all__ = ["split_antibiotic_groups", "fetch_pmids_intervals_grouped"]
