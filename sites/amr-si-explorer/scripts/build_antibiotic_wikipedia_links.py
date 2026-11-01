#!/usr/bin/env python3
"""Resolve English Wikipedia links for the antibiotics shown in the explorer.

The CSV contains one row for every explorer antibiotic. Exact title matches,
Wikipedia normalization, and redirects are preferred. Search candidates are
recorded for review when no exact page is found.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
PAIR_DATA = ROOT / "sites/amr-si-explorer/dist/pairs.json"
OUTPUT = ROOT / "sites/amr-si-explorer/data/antibiotic_wikipedia_links.csv"
OUTPUT_JSON = ROOT / "sites/amr-si-explorer/dist/antibiotic_wikipedia_links.json"
API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "AMRSIExplorer/1.0 (antibiotic Wikipedia link resolver)"
BATCH_SIZE = 25


def request_json(params: dict[str, str]) -> dict:
    query = urlencode(params)
    request = Request(f"{API_URL}?{query}", headers={"User-Agent": USER_AGENT})
    for attempt in range(7):
        try:
            with urlopen(request, timeout=30) as response:
                payload = json.load(response)
            if payload.get("error"):
                raise RuntimeError(f"Wikipedia API returned an error: {payload['error']}")
            return payload
        except (HTTPError, URLError, TimeoutError) as error:
            if attempt == 6:
                raise RuntimeError(f"Wikipedia API request failed: {error}") from error
            retry_after = error.headers.get("Retry-After") if isinstance(error, HTTPError) else None
            delay = float(retry_after) if retry_after and retry_after.isdigit() else min(5 * (2**attempt), 60)
            time.sleep(delay)
    raise AssertionError("Retry loop ended unexpectedly")


def title_candidates(name: str) -> list[str]:
    candidates = [name]
    spaced = name.replace("-", " ")
    candidates.extend([spaced, spaced.replace("/", " / "), spaced.replace("/", " ")])
    return list(dict.fromkeys(value.strip() for value in candidates if value.strip()))


def resolve_titles(names: list[str], rows: dict[str, dict[str, str]]) -> None:
    candidate_to_names: dict[str, list[str]] = {}
    for name in names:
        for candidate in title_candidates(name):
            candidate_to_names.setdefault(candidate, []).append(name)

    candidates = list(candidate_to_names)
    for start in range(0, len(candidates), BATCH_SIZE):
        batch = candidates[start : start + BATCH_SIZE]
        payload = request_json({
            "action": "query", "format": "json", "formatversion": "2",
            "redirects": "1", "maxlag": "5", "titles": "|".join(batch),
        })
        query = payload.get("query", {})
        title_map = {}
        for key in ("normalized", "redirects"):
            for mapping in query.get(key, []):
                title_map[mapping["from"].casefold()] = mapping["to"]
        pages = {
            page["title"].casefold(): page["title"]
            for page in query.get("pages", []) if "missing" not in page
        }
        for requested in batch:
            title = requested
            visited = set()
            while title.casefold() in title_map and title.casefold() not in visited:
                visited.add(title.casefold())
                title = title_map[title.casefold()]
            canonical = pages.get(title.casefold())
            if canonical:
                for source_name in candidate_to_names[requested]:
                    rows.setdefault(source_name, {
                        "wikipedia_title": canonical,
                        "wikipedia_url": "https://en.wikipedia.org/wiki/" + quote(canonical.replace(" ", "_"), safe="()_,-."),
                        "match_status": "found",
                        "match_type": "exact_title_or_redirect" if requested == source_name else "normalized_title_or_redirect",
                    })
        write_table(rows)
        print(f"Checked exact title candidates {min(start + len(batch), len(candidates)):,}/{len(candidates):,}.", flush=True)
        time.sleep(1.1)


def search_candidates(name: str) -> dict[str, str]:
    payload = request_json({
        "action": "query", "format": "json", "formatversion": "2",
        "list": "search", "srsearch": f'intitle:"{name}"', "srlimit": "5", "maxlag": "5",
    })
    results = payload.get("query", {}).get("search", [])
    if not results:
        return {"wikipedia_title": "", "wikipedia_url": "", "match_status": "not_found", "match_type": "no_exact_title_or_search_result"}
    page = results[0]
    title = page["title"]
    return {
        "wikipedia_title": title,
        "wikipedia_url": "https://en.wikipedia.org/wiki/" + quote(title.replace(" ", "_"), safe="()_,-."),
        "match_status": "review_search_result",
        "match_type": "wikipedia_title_search",
    }


def write_table(rows: dict[str, dict[str, str]]) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "antibiotic", "wikipedia_title", "wikipedia_url", "match_status", "match_type",
        ])
        writer.writeheader()
        for name in sorted(rows, key=str.casefold):
            writer.writerow({"antibiotic": name, **rows[name]})


def write_site_data(names: list[str], rows: dict[str, dict[str, str]]) -> None:
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": "sites/amr-si-explorer/data/antibiotic_wikipedia_links.csv",
        "count": len(names),
        "records": [{"antibiotic": name, **rows[name]} for name in names],
    }
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def main(force_refresh: bool = False) -> None:
    if not PAIR_DATA.exists():
        raise FileNotFoundError(f"Build the explorer pair data first: {PAIR_DATA}")
    records = json.loads(PAIR_DATA.read_text(encoding="utf-8"))["records"]
    names = sorted({str(record["antibiotic"]).strip() for record in records if record.get("antibiotic")}, key=str.casefold)
    if not names:
        raise ValueError("Explorer pair data contains no antibiotics.")

    rows: dict[str, dict[str, str]] = {}
    if OUTPUT.exists() and not force_refresh:
        previous = pd.read_csv(OUTPUT, keep_default_na=False)
        required = {"antibiotic", "wikipedia_title", "wikipedia_url", "match_status", "match_type"}
        if required.issubset(previous.columns):
            allowed = set(names)
            for row in previous.to_dict(orient="records"):
                name = str(row["antibiotic"])
                if name in allowed and row["match_status"] in {"found", "review_search_result", "not_found"}:
                    rows[name] = {column: str(row[column]) for column in required - {"antibiotic"}}

    pending = [name for name in names if name not in rows]
    print(f"Resolving {len(pending):,} antibiotics; reusing {len(rows):,} existing rows.", flush=True)
    resolve_titles(pending, rows)
    for index, name in enumerate(pending, start=1):
        if name in rows:
            print(f"Resolved {index:,}/{len(pending):,} pending antibiotics: {name}.", flush=True)
            continue
        rows[name] = search_candidates(name)
        write_table(rows)
        time.sleep(1.1)
        print(f"Completed {index:,}/{len(pending):,} pending antibiotics.", flush=True)

    write_table({name: rows[name] for name in names})
    write_site_data(names, rows)
    print(f"Wrote {OUTPUT.relative_to(ROOT)}: {len(rows):,} antibiotics, "
          f"{sum(row['match_status'] == 'found' for row in rows.values()):,} exact/normalized links, "
          f"{sum(row['match_status'] == 'review_search_result' for row in rows.values()):,} search candidates.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-refresh", action="store_true", help="Redo all Wikipedia lookups.")
    main(force_refresh=parser.parse_args().force_refresh)
