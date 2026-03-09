"""PubMed Entrez search and summary utilities."""

from __future__ import annotations

import re
import time
from datetime import datetime
from typing import Dict, Optional, Sequence, Tuple, List
from urllib.error import HTTPError, URLError

import pandas as pd
from Bio import Entrez

from ..config import (
    EUTILS_RETRIES as RETRIES,
    EUTILS_BACKOFF_FACTOR as BACKOFF_FACTOR,
    EUTILS_RATE_LIMIT_DELAY as RATE_LIMIT_DELAY,
)
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


__all__ = [
    "parse_pubdate",
    "esearch_count_for_year",
    "search_pmids",
    "retrieve_pmids",
    "fetch_pub_metadata",
]
