"""PubMed Entrez search and summary utilities."""

from __future__ import annotations

import re
import socket
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Callable, Dict, Optional, Sequence, Tuple, List
from urllib.error import HTTPError, URLError

import pandas as pd
from Bio import Entrez

from amr.config import (
    EUTILS_API_KEY as API_KEY,
    EUTILS_RETRIES as RETRIES,
    EUTILS_BACKOFF_FACTOR as BACKOFF_FACTOR,
    EUTILS_429_BACKOFF_SECONDS as RATE_LIMIT_BACKOFF_SECONDS,
    EUTILS_MAX_BACKOFF_SECONDS as MAX_BACKOFF_SECONDS,
    EUTILS_RATE_LIMIT_DELAY as RATE_LIMIT_DELAY,
    EUTILS_TIMEOUT_SECONDS as TIMEOUT_SECONDS,
)
from .pmids import normalize_pmids
from .queries import build_antibiotic_pathogen_query
from .rate_limit import RequestRateLimiter


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
ENTREZ_RATE_LIMITER = RequestRateLimiter(RATE_LIMIT_DELAY)


def _configure_entrez(email: str) -> None:
    Entrez.email = email
    if API_KEY:
        Entrez.api_key = API_KEY
    socket.setdefaulttimeout(TIMEOUT_SECONDS)


def _is_rate_limit_error(error: BaseException) -> bool:
    """Return whether an Entrez error indicates server-side throttling."""
    return isinstance(error, HTTPError) and error.code == 429


def _retry_after_seconds(error: BaseException) -> float | None:
    """Read a numeric Retry-After header when the server provides one."""
    if not _is_rate_limit_error(error):
        return None
    headers = getattr(error, "headers", None)
    if headers is None:
        return None
    value = headers.get("Retry-After")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


def _retry_delay(error: BaseException, attempt: int) -> float:
    """Return a bounded delay, with a longer global cooldown for HTTP 429."""
    ordinary_delay = BACKOFF_FACTOR * (2**attempt)
    if not _is_rate_limit_error(error):
        return ordinary_delay
    configured_delay = RATE_LIMIT_BACKOFF_SECONDS * (2**attempt)
    server_delay = _retry_after_seconds(error)
    return min(
        MAX_BACKOFF_SECONDS,
        max(ordinary_delay, configured_delay, server_delay or 0.0),
    )


def _wait_before_retry(error: BaseException, attempt: int) -> float:
    """Apply a retry delay and globally cool down after HTTP 429."""
    delay = _retry_delay(error, attempt)
    if _is_rate_limit_error(error):
        ENTREZ_RATE_LIMITER.penalize(delay)
    else:
        time.sleep(delay)
    return delay


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
            ENTREZ_RATE_LIMITER.wait()
            handle = Entrez.esearch(db="pubmed", term=term, retmax=0)
            try:
                record = Entrez.read(handle)
            except Exception as e:
                if i == RETRIES - 1:
                    print(f"Failed to parse Entrez response after retries: {e}")
                    return None
                else:
                    delay = _retry_delay(e, i)
                    print(f"Failed to parse Entrez response (attempt {i + 1}/{RETRIES}): {e}. Retrying in {delay}s...")
                    _wait_before_retry(e, i)
                    continue
            handle.close()
            return int(record.get("Count", 0))
        except NETWORK_EXCEPTIONS as e:
            delay = _retry_delay(e, i)
            print(f"Network/HTTP error during Entrez search: {e}. Retrying in {delay}s...")
            _wait_before_retry(e, i)
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
    verbose: bool = False,
) -> Tuple[str, Optional[int], Optional[str], Optional[str]]:
    """Search PubMed for a term/date window.

    Returns ``count=None`` when the query failed, distinguishing failures from true
    zero-result searches.
    """
    _configure_entrez(email)
    query = f'({term_clause}) AND ("{min_date}"[Date - Publication] : "{max_date}"[Date - Publication])'
    if verbose:
        print(f"[search_term_clause] starting query window {min_date}..{max_date}; max attempts={RETRIES}")

    for i in range(RETRIES):
        try:
            ENTREZ_RATE_LIMITER.wait()
            handle = Entrez.esearch(db="pubmed", term=query, usehistory="y", retmax=0)
            try:
                record = Entrez.read(handle)
            except Exception as e:
                if i == RETRIES - 1:
                    print(f"Failed to parse Entrez response after retries: {e}")
                    return query, None, None, None
                else:
                    delay = _retry_delay(e, i)
                    print(f"Failed to parse Entrez response (attempt {i + 1}/{RETRIES}): {e}. Retrying in {delay}s...")
                    _wait_before_retry(e, i)
                    continue
            handle.close()

            total_count = int(record.get("Count", 0))
            webenv = record.get("WebEnv")
            query_key = record.get("QueryKey")
            if total_count == 0:
                return query, 0, None, None
            if not webenv or not query_key:
                if i == RETRIES - 1:
                    print("Missing WebEnv or QueryKey after retries — treating search as failed.")
                    return query, None, None, None
                delay = BACKOFF_FACTOR * (2**i)
                print(
                    f"Missing WebEnv or QueryKey (attempt {i + 1}/{RETRIES}). "
                    f"Retrying in {delay}s..."
                )
                time.sleep(delay)
                continue
            return query, total_count, webenv, query_key
        except NETWORK_EXCEPTIONS as e:
            delay = _retry_delay(e, i)
            print(f"Network/HTTP error during Entrez search: {e}. Retrying in {delay} seconds...")
            _wait_before_retry(e, i)
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
    verbose: bool = True,
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
    if verbose:
        print(f"[search_pmids] starting query window {min_date}..{max_date}; max attempts={RETRIES}")

    for i in range(RETRIES):
        try:
            ENTREZ_RATE_LIMITER.wait()
            handle = Entrez.esearch(db="pubmed", term=query, usehistory="y", retmax=0)
            try:
                record = Entrez.read(handle)
            except Exception as e:
                if i == RETRIES - 1:
                    print(f"Failed to parse Entrez response after retries: {e}")
                    return query, None, None, None
                else:
                    delay = _retry_delay(e, i)
                    print(f"Failed to parse Entrez response (attempt {i + 1}/{RETRIES}): {e}. Retrying in {delay}s...")
                    _wait_before_retry(e, i)
                    continue
            handle.close()

            total_count = int(record.get("Count", 0))
            webenv = record.get("WebEnv")
            query_key = record.get("QueryKey")

            if total_count == 0:
                return query, 0, None, None
            if not webenv or not query_key:
                if i == RETRIES - 1:
                    print("Missing WebEnv or QueryKey after retries — treating search as failed.")
                    return query, None, None, None
                delay = BACKOFF_FACTOR * (2**i)
                print(
                    f"Missing WebEnv or QueryKey (attempt {i + 1}/{RETRIES}). "
                    f"Retrying in {delay}s..."
                )
                time.sleep(delay)
                continue

            return query, total_count, webenv, query_key

        except NETWORK_EXCEPTIONS as e:
            delay = _retry_delay(e, i)
            print(f"Network/HTTP error during Entrez search: {e}. Retrying in {delay} seconds...")
            _wait_before_retry(e, i)
        except Exception as e:  # pragma: no cover - defensive
            print(f"Unexpected error during Entrez search: {e}")
            break

    print(f"Failed to connect to Entrez after {RETRIES} retries.")
    return query, None, None, None


def retrieve_pmids(
    query: str,
    webenv: str,
    query_key: str,
    count: int,
    batch_size: int = 9000,
    *,
    verbose: bool = False,
) -> List[str]:
    """Retrieve every PMID in batches or fail without returning partial results."""
    pmids: List[str] = []
    if count == 0:
        return pmids
    if count < 0 or batch_size <= 0:
        raise ValueError("count must be non-negative and batch_size must be positive")
    if not query or not webenv or not query_key:
        raise ValueError("query, webenv, and query_key are required when count is positive")

    total_batches = (count + batch_size - 1) // batch_size
    for batch_idx, start in enumerate(range(0, count, batch_size), start=1):
        expected_batch_count = min(batch_size, count - start)
        if verbose:
            print(f"[retrieve_pmids] batch {batch_idx}/{total_batches} (retstart={start})")
        batch_pmids: List[str] | None = None
        last_error: Exception | None = None
        for attempt in range(RETRIES):
            handle = None
            try:
                ENTREZ_RATE_LIMITER.wait()
                handle = Entrez.esearch(
                    db="pubmed",
                    term=query,
                    usehistory="y",
                    webenv=webenv,
                    query_key=query_key,
                    retstart=start,
                    retmax=batch_size,
                )
                record = Entrez.read(handle)
                ids = normalize_pmids(record.get("IdList", []))
                if len(ids) != expected_batch_count:
                    raise ValueError(
                        f"PubMed returned {len(ids)} PMIDs for batch {batch_idx}/{total_batches}; "
                        f"expected {expected_batch_count}."
                    )
                batch_pmids = ids
                break
            except Exception as e:
                last_error = e
                if attempt < RETRIES - 1:
                    delay = _retry_delay(e, attempt) if isinstance(e, NETWORK_EXCEPTIONS) else BACKOFF_FACTOR * (2**attempt)
                    print(
                        f"PMID batch {batch_idx}/{total_batches} failed "
                        f"(attempt {attempt + 1}/{RETRIES}): {e}. Retrying in {delay}s..."
                    )
                    if isinstance(e, NETWORK_EXCEPTIONS):
                        _wait_before_retry(e, attempt)
                    else:
                        time.sleep(delay)
            finally:
                if handle is not None:
                    handle.close()
        if batch_pmids is None:
            raise RuntimeError(
                f"Failed to retrieve PMID batch {batch_idx}/{total_batches}; refusing to return "
                "a partial result."
            ) from last_error
        pmids.extend(batch_pmids)

    if len(pmids) != count or len(set(pmids)) != count:
        raise RuntimeError(
            f"PubMed PMID retrieval was incomplete or duplicated: expected {count}, received "
            f"{len(pmids)} unique {len(set(pmids))}."
        )
    return pmids


def fetch_pub_metadata(
    pmids: Sequence[str],
    email: str = "your_email@example.com",
    batch_size: int = 1000,
    on_batch_complete: Callable[[int, int, list[str], Dict[str, datetime], Dict[str, str]], None] | None = None,
    verbose: bool = True,
):
    """Fetch publication dates and venues for a list of PMIDs."""
    _configure_entrez(email)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    pmids = normalize_pmids(pmids)
    date_map: Dict[str, datetime] = {}
    venue_map: Dict[str, str] = {}

    total_batches = (len(pmids) + batch_size - 1) // batch_size if len(pmids) > 0 else 0
    if verbose and total_batches:
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
        if verbose:
            print(
                f"[fetch_pub_metadata] starting batch {batch_idx}/{total_batches} "
                f"({len(batch)} PMIDs) | ETA {eta_text} | est finish {est_finish}"
            )
        batch_date_map: Dict[str, datetime] = {}
        batch_venue_map: Dict[str, str] = {}
        batch_completed = False
        batch_result: tuple[Dict[str, datetime], Dict[str, str]] | None = None
        last_error: Exception | None = None
        for attempt in range(RETRIES):
            handle = None
            try:
                ENTREZ_RATE_LIMITER.wait()
                handle = Entrez.esummary(db="pubmed", id=",".join(batch))
                records = Entrez.read(handle)

                if isinstance(records, dict):
                    summaries = records.get("DocumentSummarySet", {}).get("DocumentSummary", [])
                else:
                    summaries = records

                this_date_map: Dict[str, datetime] = {}
                this_venue_map: Dict[str, str] = {}
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
                        this_date_map[pmid] = parsed_date
                    this_venue_map[pmid] = venue
                missing_pmids = set(batch) - set(this_venue_map)
                if missing_pmids:
                    raise ValueError(
                        f"PubMed metadata response omitted {len(missing_pmids)} requested PMID(s)."
                    )
                batch_result = (this_date_map, this_venue_map)
                batch_completed = True
                break
            except Exception as e:
                last_error = e
                if attempt < RETRIES - 1:
                    delay = _retry_delay(e, attempt) if isinstance(e, NETWORK_EXCEPTIONS) else BACKOFF_FACTOR * (2**attempt)
                    print(f"Metadata batch {batch_idx}/{total_batches} failed: {e}. Retrying in {delay}s...")
                    if isinstance(e, NETWORK_EXCEPTIONS):
                        _wait_before_retry(e, attempt)
                    else:
                        time.sleep(delay)
            finally:
                if handle is not None:
                    handle.close()
        if not batch_completed or batch_result is None:
            raise RuntimeError(
                f"Failed to fetch metadata batch {batch_idx}/{total_batches}; refusing to cache "
                "incomplete metadata."
            ) from last_error
        batch_date_map, batch_venue_map = batch_result
        date_map.update(batch_date_map)
        venue_map.update(batch_venue_map)
        if batch_completed and on_batch_complete is not None:
            on_batch_complete(batch_idx, total_batches, batch, batch_date_map, batch_venue_map)

    return date_map, venue_map


def fetch_pubmed_abstracts_batch(
    pmids: Sequence[str],
    *,
    email: str,
    verbose: bool = False,
) -> Dict[str, Tuple[str, str]]:
    """Fetch PubMed titles and abstracts for one bounded PMID batch.

    PubMed can omit a record from a multi-ID response. Retry omitted IDs in
    smaller requests so one unavailable record does not discard the rest of a
    large batch. An ID still omitted from its single-record response is returned
    with blank text, allowing callers to continue while treating it as lacking
    usable abstract evidence.
    """
    _configure_entrez(email)
    normalized = normalize_pmids(pmids)
    if not normalized:
        return {}
    for attempt in range(RETRIES):
        title_abstracts: Dict[str, Tuple[str, str]] = {}
        try:
            ENTREZ_RATE_LIMITER.wait()
            handle = Entrez.efetch(db="pubmed", id=normalized, retmode="xml")
            try:
                root = ET.parse(handle).getroot()
            finally:
                handle.close()
            records = root.findall(".//PubmedArticle") + root.findall(".//PubmedBookArticle")
            for article in records:
                pmid_el = article.find(".//PMID")
                if pmid_el is None or not pmid_el.text:
                    continue
                pmid = pmid_el.text.strip()
                title_el = article.find(".//ArticleTitle")
                if title_el is None:
                    title_el = article.find(".//BookTitle")
                title = " ".join(title_el.itertext()).strip() if title_el is not None else ""
                abstract_el = article.find(".//Abstract")
                abstract = "\n".join(
                    " ".join(section.itertext()).strip()
                    for section in abstract_el.findall("./AbstractText")
                    if abstract_el is not None and " ".join(section.itertext()).strip()
                ) if abstract_el is not None else ""
                title_abstracts[pmid] = (title, abstract)
            missing_pmids = set(normalized) - set(title_abstracts)
            if missing_pmids:
                if len(normalized) > 1:
                    missing_ids = sorted(missing_pmids)
                    retry_batch_size = max(1, len(missing_ids) // 2)
                    for retry_start in range(0, len(missing_ids), retry_batch_size):
                        missing_results = fetch_pubmed_abstracts_batch(
                            missing_ids[retry_start : retry_start + retry_batch_size],
                            email=email,
                            verbose=verbose,
                        )
                        title_abstracts.update(missing_results)
                elif attempt + 1 < RETRIES:
                    time.sleep(BACKOFF_FACTOR**attempt)
                    continue
                else:
                    missing_pmid = next(iter(missing_pmids))
                    print(
                        "Warning: PubMed EFetch omitted PMID "
                        f"{missing_pmid} from its single-record response; continuing with blank title/abstract."
                    )
                    title_abstracts[missing_pmid] = ("", "")
            return title_abstracts
        except NETWORK_EXCEPTIONS as exc:
            delay = _wait_before_retry(exc, attempt)
            if verbose:
                print(f"Abstract batch fetch failed (attempt {attempt + 1}/{RETRIES}); retrying in {delay:g}s: {exc}")
        except Exception as exc:
            if attempt == RETRIES - 1:
                raise RuntimeError(f"Failed to fetch abstracts for a batch of {len(normalized)} PMIDs.") from exc
            delay = BACKOFF_FACTOR * (2**attempt)
            time.sleep(delay)
    raise RuntimeError(f"Failed to fetch abstracts for a batch of {len(normalized)} PMIDs.")


__all__ = [
    "parse_pubdate",
    "esearch_count_for_year",
    "search_pmids",
    "retrieve_pmids",
    "fetch_pub_metadata",
    "fetch_pubmed_abstracts_batch",
]
