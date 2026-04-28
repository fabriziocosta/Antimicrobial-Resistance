"""PubMed Entrez search and summary utilities."""

from __future__ import annotations

import re
import socket
import time
from datetime import datetime, timedelta
from typing import Callable, Dict, Optional, Sequence, Tuple, List
from urllib.error import HTTPError, URLError

import pandas as pd
from Bio import Entrez

from amr.config import (
    EUTILS_API_KEY as API_KEY,
    EUTILS_RETRIES as RETRIES,
    EUTILS_BACKOFF_FACTOR as BACKOFF_FACTOR,
    EUTILS_RATE_LIMIT_DELAY as RATE_LIMIT_DELAY,
    EUTILS_TIMEOUT_SECONDS as TIMEOUT_SECONDS,
)
from .pmids import normalize_pmids
from .queries import build_antibiotic_pathogen_query


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
NETWORK_EXCEPTIONS = (HTTPError, URLError, TimeoutError, socket.timeout)


def _configure_entrez(email: str) -> None:
    Entrez.email = email
    if API_KEY:
        Entrez.api_key = API_KEY
    socket.setdefaulttimeout(TIMEOUT_SECONDS)


def _format_duration_hms(seconds: float | int) -> str:
    total_seconds = max(0, int(round(float(seconds))))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}h {minutes}m {secs}s"


def parse_pubdate(pubdate: str | None) -> Optional[datetime]:
    """Parse PubMed date strings into datetime objects."""
    if not pubdate:
        return None
    text = str(pubdate).strip()
    if not text:
        return None
    match = re.search(r"(\d{4})", text)
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


def esearch_count_for_year(term_clause: str, year: int, *, email: str) -> int | None:
    """Return PubMed count for a year, or ``None`` if the query did not complete."""
    _configure_entrez(email)
    date_clause = f'("{year}/01/01"[Date - Publication] : "{year}/12/31"[Date - Publication])'
    term = f"({term_clause}) AND {date_clause}"
    print(f"[esearch_count_for_year] querying year={year} with up to {RETRIES} attempts")
    for i in range(RETRIES):
        try:
            time.sleep(RATE_LIMIT_DELAY)
            handle = Entrez.esearch(db="pubmed", term=term, retmax=0)
            try:
                record = Entrez.read(handle)
            except Exception as e:
                if i == RETRIES - 1:
                    print(f"Failed to parse Entrez response after retries: {e}")
                    return None
                else:
                    delay = BACKOFF_FACTOR * (2**i)
                    print(f"Failed to parse Entrez response (attempt {i + 1}/{RETRIES}): {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                    continue
            handle.close()
            return int(record.get("Count", 0))
        except NETWORK_EXCEPTIONS as e:
            delay = BACKOFF_FACTOR * (2**i)
            print(f"Network/HTTP error during Entrez search: {e}. Retrying in {delay}s...")
            time.sleep(delay)
        except Exception as e:  # pragma: no cover - defensive
            print(f"Unexpected error in esearch_count_for_year: {e}")
            break
    return None


def search_term_clause(
    term_clause: str,
    *,
    min_date: str,
    max_date: str,
    email: str,
) -> Tuple[str, Optional[int], Optional[str], Optional[str]]:
    """Search PubMed for a term/date window.

    Returns ``count=None`` when the query failed, distinguishing failures from true
    zero-result searches.
    """
    _configure_entrez(email)
    query = f'({term_clause}) AND ("{min_date}"[Date - Publication] : "{max_date}"[Date - Publication])'
    print(f"[search_term_clause] starting query window {min_date}..{max_date}; max attempts={RETRIES}")

    for i in range(RETRIES):
        try:
            time.sleep(RATE_LIMIT_DELAY)
            handle = Entrez.esearch(db="pubmed", term=query, usehistory="y", retmax=0)
            try:
                record = Entrez.read(handle)
            except Exception as e:
                if i == RETRIES - 1:
                    print(f"Failed to parse Entrez response after retries: {e}")
                    return query, None, None, None
                else:
                    delay = BACKOFF_FACTOR * (2**i)
                    print(f"Failed to parse Entrez response (attempt {i + 1}/{RETRIES}): {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                    continue
            handle.close()

            total_count = int(record.get("Count", 0))
            webenv = record.get("WebEnv")
            query_key = record.get("QueryKey")
            if total_count == 0:
                return query, 0, None, None
            if not webenv or not query_key:
                print("Warning: missing WebEnv or QueryKey — treating search as failed.")
                return query, None, None, None
            return query, total_count, webenv, query_key
        except NETWORK_EXCEPTIONS as e:
            delay = BACKOFF_FACTOR * (2**i)
            print(f"Network/HTTP error during Entrez search: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
        except Exception as e:  # pragma: no cover - defensive
            print(f"Unexpected error during Entrez search: {e}")
            break

    print(f"Failed to connect to Entrez after {RETRIES} retries.")
    return query, None, None, None


def search_pmids(
    antibiotic: str,
    pathogen: str,
    bacteria_synonym_df: pd.DataFrame,
    extra_clauses: Optional[Sequence[str]],
    min_date: str,
    max_date: str,
    email: str,
    antibiotic_synonym_df: pd.DataFrame | None = None,
) -> Tuple[str, Optional[int], Optional[str], Optional[str]]:
    """Search PubMed for PMIDs with retries and exponential backoff.

    Returns ``count=None`` when the query failed, distinguishing failures from true
    zero-result searches.
    """
    _configure_entrez(email)
    query = build_antibiotic_pathogen_query(
        antibiotic,
        pathogen,
        bacteria_synonym_df,
        antibiotic_synonym_df=antibiotic_synonym_df,
        extra_clauses=extra_clauses,
        min_date=min_date,
        max_date=max_date,
    )
    print(f"[search_pmids] starting query window {min_date}..{max_date}; max attempts={RETRIES}")

    for i in range(RETRIES):
        try:
            time.sleep(RATE_LIMIT_DELAY)
            handle = Entrez.esearch(db="pubmed", term=query, usehistory="y", retmax=0)
            try:
                record = Entrez.read(handle)
            except Exception as e:
                if i == RETRIES - 1:
                    print(f"Failed to parse Entrez response after retries: {e}")
                    return query, None, None, None
                else:
                    delay = BACKOFF_FACTOR * (2**i)
                    print(f"Failed to parse Entrez response (attempt {i + 1}/{RETRIES}): {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                    continue
            handle.close()

            total_count = int(record.get("Count", 0))
            webenv = record.get("WebEnv")
            query_key = record.get("QueryKey")

            if total_count == 0:
                return query, 0, None, None
            if not webenv or not query_key:
                print("Warning: missing WebEnv or QueryKey — treating search as failed.")
                return query, None, None, None

            return query, total_count, webenv, query_key

        except NETWORK_EXCEPTIONS as e:
            delay = BACKOFF_FACTOR * (2**i)
            print(f"Network/HTTP error during Entrez search: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
        except Exception as e:  # pragma: no cover - defensive
            print(f"Unexpected error during Entrez search: {e}")
            break

    print(f"Failed to connect to Entrez after {RETRIES} retries.")
    return query, None, None, None


def retrieve_pmids(query: str, webenv: str, query_key: str, count: int, batch_size: int = 5000) -> List[str]:
    """Retrieve PMIDs in batches for a given search."""
    pmids: List[str] = []
    if not query or not webenv or not query_key or count == 0:
        return pmids

    total_batches = (count + batch_size - 1) // batch_size
    for batch_idx, start in enumerate(range(0, count, batch_size), start=1):
        print(f"[retrieve_pmids] batch {batch_idx}/{total_batches} (retstart={start})")
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
                    if attempt == RETRIES - 1:
                        print(f"Failed to parse Entrez record after retries: {e}")
                        record = {}
                    else:
                        delay = BACKOFF_FACTOR * (2**attempt)
                        print(f"Failed to parse Entrez record (attempt {attempt + 1}/{RETRIES}): {e}. Retrying in {delay}s...")
                        time.sleep(delay)
                        continue
                handle.close()

                pmids.extend(record.get("IdList", []))
                break
            except NETWORK_EXCEPTIONS as e:
                delay = BACKOFF_FACTOR * (2**attempt)
                print(f"Network error fetching PMIDs: {e}. Retrying in {delay}s...")
                time.sleep(delay)
            except Exception as e:  # pragma: no cover - defensive
                print(f"Unexpected error retrieving PMIDs: {e}")
                break

    return pmids


def fetch_pub_metadata(
    pmids: Sequence[str],
    email: str = "your_email@example.com",
    batch_size: int = 200,
    on_batch_complete: Callable[[int, int, list[str], Dict[str, datetime], Dict[str, str]], None] | None = None,
):
    """Fetch publication dates and venues for a list of PMIDs."""
    _configure_entrez(email)
    pmids = normalize_pmids(pmids)
    date_map: Dict[str, datetime] = {}
    venue_map: Dict[str, str] = {}

    total_batches = (len(pmids) + batch_size - 1) // batch_size if len(pmids) > 0 else 0
    if total_batches:
        print(f"[fetch_pub_metadata] fetching metadata for {len(pmids)} PMIDs in {total_batches} batch(es)")
    start_time = time.time()
    for batch_idx, start in enumerate(range(0, len(pmids), batch_size), start=1):
        batch = pmids[start : start + batch_size]
        completed_batches = batch_idx - 1
        if completed_batches > 0:
            elapsed = time.time() - start_time
            avg_batch_seconds = elapsed / completed_batches
            remaining_batches = total_batches - completed_batches
            eta_seconds = avg_batch_seconds * remaining_batches
            eta_text = _format_duration_hms(eta_seconds)
            est_finish = (datetime.now() + timedelta(seconds=eta_seconds)).strftime("%H:%M:%S")
        else:
            eta_text = "estimating"
            est_finish = "--:--:--"
        print(
            f"[fetch_pub_metadata] starting batch {batch_idx}/{total_batches} "
            f"({len(batch)} PMIDs) | ETA {eta_text} | est finish {est_finish}"
        )
        batch_date_map: Dict[str, datetime] = {}
        batch_venue_map: Dict[str, str] = {}
        batch_completed = False
        for attempt in range(RETRIES):
            try:
                time.sleep(RATE_LIMIT_DELAY)
                handle = Entrez.esummary(db="pubmed", id=",".join(batch))
                try:
                    records = Entrez.read(handle)
                except Exception as e:
                    if attempt == RETRIES - 1:
                        print(f"Failed to parse Entrez summary after retries: {e}")
                        records = {}
                    else:
                        delay = BACKOFF_FACTOR * (2**attempt)
                        print(
                            f"Failed to parse Entrez summary (batch {batch_idx}/{total_batches}, "
                            f"attempt {attempt + 1}/{RETRIES}): {e}. Retrying in {delay}s..."
                        )
                        time.sleep(delay)
                        continue
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
                        batch_date_map[pmid] = parsed_date
                    venue_map[pmid] = venue
                    batch_venue_map[pmid] = venue
                batch_completed = True
                break
            except NETWORK_EXCEPTIONS as e:
                delay = BACKOFF_FACTOR * (2**attempt)
                print(f"HTTP/Network error in fetch_pub_metadata: {e}. Retrying in {delay}s...")
                time.sleep(delay)
            except Exception as e:  # pragma: no cover - defensive
                print(f"Unexpected error in fetch_pub_metadata: {e}")
                break
        if batch_completed and on_batch_complete is not None:
            on_batch_complete(batch_idx, total_batches, batch, batch_date_map, batch_venue_map)

    return date_map, venue_map


__all__ = [
    "parse_pubdate",
    "esearch_count_for_year",
    "search_pmids",
    "retrieve_pmids",
    "fetch_pub_metadata",
]
