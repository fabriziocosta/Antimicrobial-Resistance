"""Data collection and preprocessing helpers for PubMed AMR queries."""

from __future__ import annotations

import json
import os
import re
import time
from collections import Counter
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError

import pandas as pd
from Bio import Entrez

RETRIES = 3
BACKOFF_FACTOR = 5
RATE_LIMIT_DELAY = 0.34  # ~3 requests/sec per NCBI policy

MONTH_LOOKUP = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


# ---------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------
def prepare_antibiotic_pubmed_query(name: str) -> str:
    """Build a PubMed query clause for an antibiotic name (handles hyphens/slashes)."""
    parts = re.split(r"[-/]", name)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) > 1:
        and_part = " AND ".join(f'"{p}"[Abstract]' for p in parts)
        query = f'("{name}"[Abstract] OR ({and_part}))'
    else:
        query = f'"{name}"[Abstract]'
    return query


def prepare_antibiotic_pubmed_query_tiab(name: str) -> str:
    """Build a PubMed query clause for an antibiotic name in Title/Abstract (handles hyphens/slashes).

    Examples:
    - "co-trimoxazole" -> ('"co-trimoxazole"[Title/Abstract] OR ("co"[Title/Abstract] AND "trimoxazole"[Title/Abstract]))
    - "sulfamethoxazole/trimethoprim" -> ("sulfamethoxazole/trimethoprim"[Title/Abstract] OR ("sulfamethoxazole"[Title/Abstract] AND "trimethoprim"[Title/Abstract]))
    - otherwise -> '"term"[Title/Abstract]'
    """
    parts = re.split(r"[-/]", name)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) > 1:
        and_part = " AND ".join(f'"{p}"[Title/Abstract]' for p in parts)
        query = f'("{name}"[Title/Abstract] OR ({and_part}))'
    else:
        query = f'"{name}"[Title/Abstract]'
    return query


def expand_synonyms(pathogen: str, bacteria_synonym_df: pd.DataFrame) -> List[str]:
    """Return a list of synonyms for `pathogen` from the lookup table."""
    mask = bacteria_synonym_df["bacteria name"].str.lower() == pathogen.lower()
    vals = (
        bacteria_synonym_df.loc[mask, "bacteria synonym"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    return sorted({v for v in vals if v})


def build_pathogen_tiab_query(pathogen: str, bacteria_synonym_df: pd.DataFrame) -> str:
    """Construct a Title/Abstract-only clause OR-ing pathogen name with its synonyms."""
    variants = expand_synonyms(pathogen, bacteria_synonym_df)
    if not variants:
        variants = [pathogen]
    elif pathogen not in variants:
        variants = [pathogen] + variants
    clause = "(" + " OR ".join(f'"{v}"[Title/Abstract]' for v in variants) + ")"
    return clause


def esearch_count_for_year(term_clause: str, year: int, *, email: str) -> int:
    """Return PubMed count for `term_clause` limited to a single publication year."""
    Entrez.email = email
    date_clause = f'("{year}/01/01"[Date - Publication] : "{year}/12/31"[Date - Publication])'
    term = f"({term_clause}) AND {date_clause}"
    for i in range(RETRIES):
        try:
            time.sleep(RATE_LIMIT_DELAY)
            handle = Entrez.esearch(db="pubmed", term=term, retmax=0)
            try:
                record = Entrez.read(handle)
            except Exception as e:
                print(f"Failed to parse Entrez response: {e}")
                record = {}
            handle.close()
            return int(record.get("Count", 0))
        except (HTTPError, URLError) as e:
            delay = BACKOFF_FACTOR * (2**i)
            print(f"Network/HTTP error during Entrez search: {e}. Retrying in {delay}s...")
            time.sleep(delay)
        except Exception as e:  # pragma: no cover - defensive
            print(f"Unexpected error in esearch_count_for_year: {e}")
            break
    return 0


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
    if not pathogens:
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
    if not antibiotics:
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


def parse_pubdate(pubdate: str | None) -> Optional[datetime]:
    """Parse PubMed date strings into datetime objects."""
    if not pubdate:
        return None
    text = str(pubdate).strip()
    if not text:
        return None
    match = re.search(r"(\\d{4})", text)
    if not match:
        return None
    year = int(match.group(1))
    lowered = text.lower()
    month = 1
    for key, value in MONTH_LOOKUP.items():
        if key in lowered:
            month = value
            break
    try:
        return datetime(year, month, 1)
    except ValueError:
        return datetime(year, 1, 1)


def build_antibiotic_pathogen_query(
    antibiotic: str,
    pathogen: str,
    bacteria_synonym_df: pd.DataFrame,
    extra_clauses: Optional[Sequence[str]] = None,
    min_date: Optional[str] = None,
    max_date: Optional[str] = None,
) -> str:
    """Construct a PubMed query tying together antibiotic, pathogen, and optional filters."""
    pathogen_variants = expand_synonyms(pathogen, bacteria_synonym_df)
    if not pathogen_variants:
        pathogen_variants = [pathogen]
    elif pathogen not in pathogen_variants:
        pathogen_variants = [pathogen] + pathogen_variants

    pathogen_clause = "(" + " OR ".join(f'"{p}"[Abstract]' for p in pathogen_variants) + ")"
    antibiotic_clause = prepare_antibiotic_pubmed_query(antibiotic)

    term_parts = [f"({antibiotic_clause} AND {pathogen_clause})"]
    if extra_clauses:
        term_parts.extend(extra_clauses)
    if min_date and max_date:
        term_parts.append(f'("{min_date}"[Date - Publication] : "{max_date}"[Date - Publication])')
    return " AND ".join(term_parts)


# ---------------------------------------------------------------------
# PubMed retrieval
# ---------------------------------------------------------------------
def search_pmids(
    antibiotic: str,
    pathogen: str,
    bacteria_synonym_df: pd.DataFrame,
    extra_clauses: Optional[Sequence[str]],
    min_date: str,
    max_date: str,
    email: str,
) -> Tuple[str, int, Optional[str], Optional[str]]:
    """Search PubMed for PMIDs with retries and exponential backoff."""
    Entrez.email = email
    query = build_antibiotic_pathogen_query(
        antibiotic,
        pathogen,
        bacteria_synonym_df,
        extra_clauses=extra_clauses,
        min_date=min_date,
        max_date=max_date,
    )

    for i in range(RETRIES):
        try:
            time.sleep(RATE_LIMIT_DELAY)
            handle = Entrez.esearch(db="pubmed", term=query, usehistory="y", retmax=0)
            try:
                record = Entrez.read(handle)
            except Exception as e:
                print(f"Failed to parse Entrez response: {e}")
                record = {}
            handle.close()

            total_count = int(record.get("Count", 0))
            webenv = record.get("WebEnv")
            query_key = record.get("QueryKey")

            if not webenv or not query_key:
                print("Warning: missing WebEnv or QueryKey — skipping retrieval.")
                return query, 0, None, None

            return query, total_count, webenv, query_key

        except (HTTPError, URLError) as e:
            delay = BACKOFF_FACTOR * (2**i)
            print(f"Network/HTTP error during Entrez search: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
        except Exception as e:  # pragma: no cover - defensive
            print(f"Unexpected error during Entrez search: {e}")
            break

    print(f"Failed to connect to Entrez after {RETRIES} retries.")
    return query, 0, None, None


def retrieve_pmids(query: str, webenv: str, query_key: str, count: int, batch_size: int = 5000) -> List[str]:
    """Retrieve PMIDs in batches for a given search."""
    pmids: List[str] = []
    if not query or not webenv or not query_key or count == 0:
        return pmids

    for start in range(0, count, batch_size):
        for attempt in range(RETRIES):
            try:
                time.sleep(RATE_LIMIT_DELAY)
                handle = Entrez.esearch(
                    db="pubmed",
                    term=query,
                    usehistory="y",
                    webenv=webenv,
                    query_key=query_key,
                    retstart=start,
                    retmax=batch_size,
                )
                try:
                    record = Entrez.read(handle)
                except Exception as e:
                    print(f"Failed to parse Entrez record: {e}")
                    record = {}
                handle.close()

                pmids.extend(record.get("IdList", []))
                break
            except (HTTPError, URLError) as e:
                delay = BACKOFF_FACTOR * (2**attempt)
                print(f"Network error fetching PMIDs: {e}. Retrying in {delay}s...")
                time.sleep(delay)
            except Exception as e:  # pragma: no cover - defensive
                print(f"Unexpected error retrieving PMIDs: {e}")
                break

    return pmids


def fetch_pub_metadata(pmids: Sequence[str], email: str = "your_email@example.com", batch_size: int = 200):
    """Fetch publication dates and venues for a list of PMIDs."""
    Entrez.email = email
    date_map: Dict[str, datetime] = {}
    venue_map: Dict[str, str] = {}

    for start in range(0, len(pmids), batch_size):
        batch = pmids[start : start + batch_size]
        for attempt in range(RETRIES):
            try:
                time.sleep(RATE_LIMIT_DELAY)
                handle = Entrez.esummary(db="pubmed", id=",".join(batch))
                try:
                    records = Entrez.read(handle)
                except Exception as e:
                    print(f"Failed to parse Entrez summary: {e}")
                    records = {}
                handle.close()

                if isinstance(records, dict):
                    summaries = records.get("DocumentSummarySet", {}).get("DocumentSummary", [])
                else:
                    summaries = records

                for summary in summaries:
                    pmid_value = summary.get("Id")
                    if pmid_value is None:
                        continue
                    pmid = str(pmid_value)
                    pubdate = summary.get("PubDate") or summary.get("EPubDate") or ""
                    epub = summary.get("EPubDate")
                    venue = summary.get("FullJournalName") or summary.get("Source") or "Unknown"
                    parsed_date = parse_pubdate(pubdate)
                    if not parsed_date and epub:
                        parsed_date = parse_pubdate(epub)
                    if parsed_date:
                        date_map[pmid] = parsed_date
                    venue_map[pmid] = venue
                break
            except (HTTPError, URLError) as e:
                delay = BACKOFF_FACTOR * (2**attempt)
                print(f"HTTP/Network error in fetch_pub_metadata: {e}. Retrying in {delay}s...")
                time.sleep(delay)
            except Exception as e:  # pragma: no cover - defensive
                print(f"Unexpected error in fetch_pub_metadata: {e}")
                break

    return date_map, venue_map


def merge_with_existing(new_df: pd.DataFrame, parquet_path: str = "pmids_results.parquet") -> pd.DataFrame:
    """Merge new results into existing parquet (if present) safely."""
    if os.path.exists(parquet_path):
        old_df = pd.read_parquet(parquet_path)
        for df in (old_df, new_df):
            if "venue_histogram" not in df.columns:
                df["venue_histogram"] = json.dumps({})
        merged_df = pd.concat([old_df, new_df], ignore_index=True)
        merged_df = merged_df.drop_duplicates(
            subset=["pathogen", "antibiotic", "start_date", "end_date"],
            keep="last",
        )
    else:
        if "venue_histogram" not in new_df.columns:
            new_df["venue_histogram"] = json.dumps({})
        merged_df = new_df

    for attempt in range(3):
        try:
            merged_df.to_parquet(parquet_path, index=False)
            break
        except OSError as e:
            delay = 2**attempt
            print(f"File write failed ({e}), retrying in {delay}s...")
            time.sleep(delay)
    return merged_df


def fetch_pmids_by_date_and_keywords(
    min_date: str,
    max_date: str,
    antibiotic: str,
    pathogen: str,
    bacteria_synonym_df: pd.DataFrame,
    email: str = "your_email@example.com",
    batch_size: int = 5000,
    extra_clauses: Optional[Sequence[str]] = None,
) -> List[str]:
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


def update_metadata_cache(pmids: Sequence[str], metadata_cache: Dict[str, dict], email: str = "your_email@example.com", batch_size: int = 200):
    """Populate metadata cache with publication date and venue details."""
    missing = [pmid for pmid in pmids if pmid not in metadata_cache]
    if not missing:
        return
    date_map, venue_map = fetch_pub_metadata(missing, email=email, batch_size=batch_size)
    for pmid in missing:
        metadata_cache[pmid] = {
            "date": date_map.get(pmid),
            "venue": venue_map.get(pmid, "Unknown"),
        }


def get_metadata(pmid: str, metadata_cache: Dict[str, dict]) -> dict:
    """Convenience accessor for cached PMID metadata."""
    return metadata_cache.get(pmid, {"date": None, "venue": "Unknown"})


# ---------------------------------------------------------------------
# Venue helpers
# ---------------------------------------------------------------------
def add_venue_vectors(df: pd.DataFrame, histogram_col: str = "venue_histogram"):
    """Return a copy of df with a `venue_vector` column and the venue index map."""
    if histogram_col not in df.columns:
        raise KeyError(f"Column '{histogram_col}' not present in DataFrame")

    hist_series = df[histogram_col].fillna("{}")

    def _coerce_hist(value):
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return {}
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return {}
        return {}

    parsed = hist_series.apply(_coerce_hist)
    all_venues = sorted({venue for hist in parsed for venue in hist.keys()})
    venue_index = {venue: idx for idx, venue in enumerate(all_venues)}

    def _vectorize(hist):
        vector = [0] * len(all_venues)
        for venue, count in hist.items():
            idx = venue_index.get(venue)
            if idx is not None:
                vector[idx] = count
        return vector

    venue_vectors = parsed.apply(_vectorize)

    df_copy = df.copy()
    df_copy["venue_vector"] = venue_vectors
    return df_copy, venue_index


def plot_venue_vector(
    df_with_vecs: pd.DataFrame,
    venue_index: Dict[str, int],
    row: Optional[int] = None,
    ax=None,
    rotation: int = 0,
    title: Optional[str] = None,
    use_all_dates: bool = False,
    pathogen: Optional[str] = None,
    antibiotic: Optional[str] = None,
):
    """Bar plot of venue counts for a given pathogen/antibiotic row."""
    import matplotlib.pyplot as plt

    required_cols = {"venue_vector", "pathogen", "antibiotic"}
    missing = required_cols - set(df_with_vecs.columns)
    if missing:
        raise KeyError(f"DataFrame missing required columns: {missing}")

    if not isinstance(venue_index, dict):
        raise TypeError("venue_index must be a dictionary of venue->position.")

    if row is not None and (row < 0 or row >= len(df_with_vecs)):
        raise IndexError("Row index out of range for venue_vector plot.")

    if row is None:
        if pathogen is None or antibiotic is None:
            raise ValueError("Provide either `row` or both `pathogen` and `antibiotic`.")
        row_mask = (df_with_vecs["pathogen"] == pathogen) & (df_with_vecs["antibiotic"] == antibiotic)
        if not row_mask.any():
            raise ValueError("No rows match the specified pathogen/antibiotic combination.")
        current_row = df_with_vecs.loc[row_mask].iloc[0]
    else:
        current_row = df_with_vecs.iloc[row]
        pathogen = pathogen or current_row["pathogen"]
        antibiotic = antibiotic or current_row["antibiotic"]

    venues_sorted = [venue for venue, _ in sorted(venue_index.items(), key=lambda kv: kv[1])]

    def _coerce_vector(vector_like):
        if not isinstance(vector_like, (list, tuple)):
            raise TypeError("venue_vector entries must be list-like.")
        vec = list(vector_like)
        if len(vec) != len(venues_sorted):
            adjusted = [0] * len(venues_sorted)
            for venue, idx in venue_index.items():
                if idx < len(vec):
                    adjusted[idx] = vec[idx]
            vec = adjusted
        return vec

    if use_all_dates:
        mask = (df_with_vecs["pathogen"] == pathogen) & (df_with_vecs["antibiotic"] == antibiotic)
        grouped = df_with_vecs.loc[mask, "venue_vector"]
        aggregated = [0] * len(venues_sorted)
        for vector_like in grouped:
            try:
                vec = _coerce_vector(vector_like)
            except TypeError:
                continue
            for idx, value in enumerate(vec):
                aggregated[idx] += value
        vector = aggregated
    else:
        vector = _coerce_vector(current_row["venue_vector"])

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, max(4, len(venues_sorted) * 0.35)))
    else:
        fig = ax.figure

    positions = list(range(len(venues_sorted)))
    max_value = max(vector) if vector else 0
    offset = 0.02 * max_value if max_value else 0.1
    bars = ax.barh(positions, vector)
    for bar, value in zip(bars, vector):
        width = bar.get_width()
        y_pos = bar.get_y() + bar.get_height() / 2
        x_pos = width + offset if width else offset
        ax.text(x_pos, y_pos, f"{value}", va="center", ha="left", fontsize=8)
    ax.set_yticks(positions)
    ax.set_yticklabels(venues_sorted, rotation=rotation)
    ax.set_xlabel("Count")
    ax.set_ylabel("Venue")
    if title:
        ax.set_title(title)
    ax.margins(y=0.01)
    try:
        fig.tight_layout()
    except ValueError:
        pass
    return ax


# ---------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------
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
) -> pd.DataFrame:
    """Query PubMed in intervals for all pathogen/antibiotic combinations."""
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

    overall_start_dt = datetime.strptime(min_date, "%Y/%m/%d")
    overall_end_dt = datetime.strptime(max_date, "%Y/%m/%d")

    total_tasks = len(pathogens) * len(antibiotics)
    task_counter = 0
    new_results = []
    metadata_cache: Dict[str, dict] = {}

    for ai, antibiotic in enumerate(antibiotics, start=1):
        for pi, pathogen in enumerate(pathogens, start=1):
            task_counter += 1
            task_progress = (task_counter / total_tasks) * 100

            synonyms = expand_synonyms(pathogen, bacteria_synonym_df)
            synonyms_str = ", ".join(synonyms) if synonyms else "—"
            print(
                f"Processing antibiotic '{antibiotic}' | pathogen '{pathogen}' | synonyms: {synonyms_str}"
            )

            interval_entries = []
            missing_entries = []

            for year in range(overall_start_dt.year, overall_end_dt.year + 1, year_gap):
                interval_start = f"{year}/01/01"
                interval_end_year = min(year + year_gap - 1, overall_end_dt.year)
                interval_end = f"{interval_end_year}/12/31"
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
                    print(
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
                    print(
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
                    print(
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
                venue_counter = Counter()
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
                print(
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

    return existing_df


__all__ = [
    "RETRIES",
    "BACKOFF_FACTOR",
    "RATE_LIMIT_DELAY",
    "prepare_antibiotic_pubmed_query",
    "expand_synonyms",
    "parse_pubdate",
    "prepare_antibiotic_pubmed_query_tiab",
    "build_pathogen_tiab_query",
    "esearch_count_for_year",
    "count_pmids_per_year_for_pathogens",
    "count_pmids_per_year_for_antibiotics",
    "build_antibiotic_pathogen_query",
    "search_pmids",
    "retrieve_pmids",
    "fetch_pub_metadata",
    "merge_with_existing",
    "fetch_pmids_by_date_and_keywords",
    "update_metadata_cache",
    "get_metadata",
    "add_venue_vectors",
    "plot_venue_vector",
    "fetch_pmids_intervals",
]
