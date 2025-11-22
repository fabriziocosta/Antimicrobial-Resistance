"""LLM-assisted filtering of PubMed AMR abstracts."""

from __future__ import annotations

import json
import os
import time
from typing import Iterable
from xml.etree import ElementTree as ET

import pandas as pd
import requests

DEFAULT_SYSTEM_PROMPT = """
You are a careful biomedical evidence reviewer. Given a title and abstract, decide whether the authors explicitly state or conclude that the specified pathogen is showing clinically meaningful resistance to the specified antibiotic.

Rules:
- Answer "yes" only if the abstract clearly reports resistance, reduced susceptibility, treatment failure due to resistance, or emergence of resistant isolates for that pathogen/antibiotic pair.
- Answer "no" if the abstract does not make that claim, is about a different organism/drug, only mentions testing without resistance, or only provides background.
- Answer "uncertain" if the abstract is missing, ambiguous, or only indirectly related.
- Base the decision only on the provided text; do not infer beyond it.

Return compact JSON with keys: decision (yes/no/uncertain) and rationale (one short sentence).
""".strip()

PUBMED_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
RETRIES = 3
BACKOFF_FACTOR = 2


def fetch_pubmed_abstract(
    pmid: str,
    *,
    email: str | None = None,
    session: requests.Session | None = None,
) -> tuple[str, str]:
    """Fetch title and abstract text for a PMID via PubMed efetch with retries."""
    sess = session or requests
    params = {"db": "pubmed", "id": pmid, "retmode": "xml"}
    if email:
        params["email"] = email
    for attempt in range(RETRIES):
        try:
            resp = sess.get(PUBMED_EFETCH, params=params, timeout=30)
            resp.raise_for_status()
            root = ET.fromstring(resp.text)
            article = root.find(".//Article")
            if article is None:
                return "", ""
            title_el = article.find("./ArticleTitle")
            title = " ".join(title_el.itertext()).strip() if title_el is not None else ""
            abstract_el = article.find(".//Abstract")
            if abstract_el is None:
                return title, ""
            parts = [" ".join(p.itertext()).strip() for p in abstract_el.findall(".//AbstractText")]
            abstract = "\n".join([p for p in parts if p])
            return title, abstract
        except Exception as exc:
            if attempt == RETRIES - 1:
                print(f"PubMed fetch failed for PMID {pmid}: {exc}")
                return "", ""
            delay = BACKOFF_FACTOR**attempt
            time.sleep(delay)


def build_user_prompt(pmid: str, pathogen: str, antibiotic: str, title: str, abstract: str) -> str:
    """Create the LLM prompt for a single PMID."""
    return f"""
PMID: {pmid}
Target pathogen (exact match only): {pathogen}
Target antibiotic (exact match only): {antibiotic}
Title: {title}
Abstract:
{abstract or '[no abstract available]'}

Task: Decide if the authors explicitly claim that THIS pathogen shows significant resistance to THIS antibiotic (ignore other organisms/drugs mentioned).
Return JSON with keys decision (yes/no/uncertain) and rationale (short phrase).
""".strip()


def classify_abstract(
    llm,
    pmid: str,
    pathogen: str,
    antibiotic: str,
    title: str,
    abstract: str,
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    max_tokens: int = 256,
) -> dict:
    """Call the LLM to classify a single abstract."""
    prompt = build_user_prompt(pmid, pathogen, antibiotic, title, abstract)
    raw = llm.complete(prompt, system_prompt=system_prompt, max_tokens=max_tokens, temperature=0)
    try:
        data = json.loads(raw)
    except Exception:
        data = {"decision": "uncertain", "rationale": raw.strip()}
    return {
        "pmid": pmid,
        "decision": data.get("decision", "uncertain"),
        "rationale": data.get("rationale", ""),
    }


def prepare_pmids_to_review(pmid_df: pd.DataFrame, *, max_pmids: int | None = None) -> pd.DataFrame:
    """Flatten pmids_results.parquet structure to one row per PMID/pathogen/antibiotic."""
    long_df = pmid_df.explode("pmids").dropna(subset=["pmids"]).copy()
    long_df["pmid"] = long_df["pmids"].astype(str).str.extract(r"(\\d+)")[0]
    long_df = long_df.dropna(subset=["pmid"])
    long_df = long_df[["pmid", "pathogen", "antibiotic", "start_date", "end_date"]].drop_duplicates()
    if max_pmids:
        long_df = long_df.head(max_pmids)
    return long_df.reset_index(drop=True)


def run_llm_filter(
    long_df: pd.DataFrame,
    llm,
    *,
    email: str | None = None,
    rate_limit: float = 0.35,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    progress_every: int = 10,
    resume_from: pd.DataFrame | None = None,
    seen_pmids: set[str] | None = None,
    output_path: str | None = None,
) -> pd.DataFrame:
    """
    Process each PMID row and return a DataFrame with LLM decisions.

    - If resume_from or seen_pmids provided, processed PMIDs are skipped.
    - Keeps the “seen” index in-memory to avoid many filesystem writes; save once via output_path if desired.
    """
    seen = set(seen_pmids or [])
    existing = pd.DataFrame()
    if resume_from is not None and not resume_from.empty:
        existing = resume_from.copy()
        if "pmid" in existing.columns:
            seen.update(existing["pmid"].astype(str))

    results = [] if existing.empty else existing.to_dict("records")
    abstract_cache: dict[str, tuple[str, str]] = {}
    session = requests.Session()

    for idx, row in long_df.iterrows():
        pmid = str(row.pmid)
        if pmid in seen:
            continue
        if pmid not in abstract_cache:
            title, abstract = fetch_pubmed_abstract(pmid, email=email, session=session)
            abstract_cache[pmid] = (title, abstract)
            time.sleep(rate_limit)
        else:
            title, abstract = abstract_cache[pmid]

        if not abstract:
            # Skip PMIDs without abstracts to avoid ambiguous judgments
            continue

        verdict = classify_abstract(
            llm,
            pmid,
            row.pathogen,
            row.antibiotic,
            title,
            abstract,
            system_prompt=system_prompt,
        )
        verdict.update(
            {
                "pathogen": row.pathogen,
                "antibiotic": row.antibiotic,
                "title": title,
                "abstract": abstract,
                "start_date": row.start_date,
                "end_date": row.end_date,
            }
        )
        results.append(verdict)
        seen.add(pmid)

        if progress_every and (len(results) % progress_every == 0):
            print(f"Processed {len(seen)}/{len(long_df)}")

    df_out = pd.DataFrame(results)
    if output_path:
        df_out.to_parquet(output_path, index=False)
    session.close()
    return df_out


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "PUBMED_EFETCH",
    "fetch_pubmed_abstract",
    "build_user_prompt",
    "classify_abstract",
    "prepare_pmids_to_review",
    "run_llm_filter",
]
