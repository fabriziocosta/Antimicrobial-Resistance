"""LLM-assisted filtering of PubMed AMR abstracts."""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Iterable
from xml.etree import ElementTree as ET

import pandas as pd
import requests

from amr.config import EUTILS_API_KEY
from amr.io import atomic_write_parquet

DEFAULT_SYSTEM_PROMPT = """
You are a careful biomedical evidence reviewer. Given a title and abstract, decide whether the authors explicitly state or conclude that the specified pathogen is showing clinically meaningful resistance to the specified antibiotic.

Rules:
- Answer "yes" only if the abstract clearly reports resistance, reduced susceptibility, treatment failure due to resistance, or emergence of resistant isolates for that pathogen/antibiotic pair.
- Answer "no" if the abstract does not make that claim, is about a different organism/drug, only mentions testing without resistance, or only provides background.
- Answer "uncertain" if the abstract is missing, ambiguous, or only indirectly related.
- Base the decision only on the provided text; do not infer beyond it.

Return compact JSON with keys: decision (yes/no/uncertain) and rationale (one short sentence).
""".strip()

UNIFIED_AMR_INTEREST_SYSTEM_PROMPT = """
You are classifying scientific abstracts about antimicrobial resistance (AMR).

Task:
Given a target pathogen, a target antibiotic or antibiotic class, a title, and an abstract,
classify the abstract into exactly one primary category describing why the paper contributes
to scientific interest for the target pathogen-antibiotic pair.

Core rule:
Do not classify by keyword co-occurrence. First determine whether the abstract clearly links
the target pathogen to the target antibiotic/class. Only assign a substantive category if the
target pair relation is explicitly supported by the title or abstract.

If the target pathogen-antibiotic relation is unclear, ambiguous, incidental, or unsupported,
discard it by setting:
- decision = "no"
- primary_category = "NONE"
- target_pair_relation_supported = false

Attribution rules:
A pathogen is linked to resistance, susceptibility, testing, treatment, or AMR relevance for an
antimicrobial agent only when the text clearly asserts that relation.

Supported links include:
- direct modification: "methicillin-resistant Staphylococcus aureus", "meropenem-resistant Klebsiella pneumoniae";
- explicit assertion: "E. coli isolates were resistant to gentamicin", "Klebsiella pneumoniae showed reduced susceptibility to meropenem";
- group attribution: "gentamicin-resistant coliforms, including Enterobacter cloacae, Klebsiella pneumoniae and Serratia marcescens".

Do not transfer resistance, susceptibility, testing, treatment, or AMR relevance from one pathogen
to another merely because they occur nearby in the same sentence.

Examples:
- "Methicillin-resistant Staphylococcus aureus and E. coli were detected" supports methicillin resistance for Staphylococcus aureus, not for E. coli.
- "Resistance to gentamicin was common in Enterococcus faecalis, while E. coli was also detected" does not support gentamicin resistance in E. coli.
- "Gentamicin-resistant coliforms, including Enterobacter cloacae, Klebsiella pneumoniae and Serratia marcescens" supports gentamicin resistance for all three listed species.
- "Carbapenem-resistant Klebsiella pneumoniae and Pseudomonas aeruginosa were isolated" is ambiguous unless the grammar clearly makes "carbapenem-resistant" apply to both species. If ambiguous for the target pathogen, discard.

Categories:

1. ENVIRONMENTAL_ANIMAL_NON_HUMAN
Use when the main setting is outside direct human clinical care, including animals, livestock,
companion animals, wildlife, aquaculture, veterinary medicine, food production, farms,
slaughterhouses, wastewater, sewage, rivers, soil, water, environmental sampling, non-human
reservoirs, or One Health transmission. Use this category when the main contribution is about
non-human reservoirs, environmental spread, animal carriage, food-chain transmission, or
cross-sector ecology.

2. CLINICAL_OUTBREAKS_REPORTS
Use when the abstract mainly concerns human clinical cases, patient cohorts, infections, case
reports, case series, hospital events, clinical burden, treatment experience, patient outcomes,
or outbreak investigations. This includes bloodstream infection, urinary tract infection,
pneumonia, sepsis, wound infection, intensive care unit outbreaks, nosocomial spread, mortality,
morbidity, and clinical risk factors. Do not use this category merely because isolates came from
a hospital laboratory if the main contribution is routine resistance monitoring. Use SURVEILLANCE instead.

3. SURVEILLANCE
Use when the abstract mainly reports resistance rates, susceptibility patterns, prevalence,
incidence, frequency, epidemiological trends, geographic spread, temporal changes, or routine
monitoring of the target pathogen-antibiotic pair. This includes antibiogram-style studies,
multicentre surveillance, national or regional monitoring, and reports of percentages or
frequencies of resistant isolates. Do not use this category when the main contribution is
explaining the genetic, molecular, or biological basis of resistance. Use
MECHANISMS_OF_RESISTANCE_GENETICS_LAB_STUDIES instead.

4. MECHANISMS_OF_RESISTANCE_GENETICS_LAB_STUDIES
Use when the abstract mainly studies how resistance occurs, how it is encoded, how it spreads
biologically, or how it behaves under laboratory investigation. This includes resistance genes,
mutations, plasmids, integrons, transposons, horizontal gene transfer, beta-lactamases,
carbapenemases, ESBLs, KPC, NDM, OXA, CTX-M, MCR, VAN, efflux pumps, porin loss, target
modification, enzymatic inactivation, biofilm-associated resistance, adaptive resistance,
whole-genome sequencing, molecular epidemiology, MIC shifts, gene expression, fitness cost, or
laboratory resistance evolution. Prefer this category over SURVEILLANCE when the abstract both
reports resistance rates and explains the mechanism of resistance.

5. OTHER
Use when the abstract is AMR-relevant but does not fit any of the categories above, including
cases where the target pathogen-antibiotic relation demonstrates resistance but does not clearly
belong in the other four categories. Use OTHER for diagnostics, policy, stewardship, guidelines,
prescribing behaviour, antibiotic consumption, reviews, modelling, economic burden, education,
or broad AMR discussions.

Decision labels:
- yes: the abstract is AMR-relevant for the target pathogen-antibiotic pair, and the target pair relation is explicitly supported.
- no: the abstract is not AMR-relevant for the target pair, or the target pair relation is ambiguous, incidental, unsupported, or based only on co-occurrence.
- uncertain: abstract missing, ambiguous, or insufficient to judge confidently.

Internal reasoning steps:
1. Identify pathogens mentioned.
2. Identify antibiotics/classes mentioned.
3. Extract explicit pathogen-antibiotic relations.
4. Decide whether the target pathogen is clearly linked to the target antibiotic/class.
5. If the target pair relation is unclear or ambiguous, discard it.
6. Choose the primary category based only on evidence explicitly linked to the target pair.
7. If attribution is ambiguous, do not infer the relation. Set decision = "no", primary_category = "NONE", and target_pair_relation_supported = false.

Output only valid compact JSON using this schema:
{
  "decision": "yes | no | uncertain",
  "primary_category": "ENVIRONMENTAL_ANIMAL_NON_HUMAN | CLINICAL_OUTBREAKS_REPORTS | SURVEILLANCE | MECHANISMS_OF_RESISTANCE_GENETICS_LAB_STUDIES | OTHER | NONE",
  "rationale": "Brief explanation in 1-3 sentences. Mention the specific title/abstract evidence and explain whether the target pathogen-antibiotic relation is supported.",
  "evidence_phrases": ["short phrase copied exactly from the title or abstract"],
  "target_pair_relation_supported": true,
  "is_amr_relevant": true,
}

Consistency rules:
- If target_pair_relation_supported is false, decision must be "no".
- If the target pathogen and target antibiotic/class are mentioned separately but not clearly linked, decision must be "no" and primary_category must be "NONE".
- If resistance, susceptibility, testing, or treatment is attributed to a different pathogen or a different antibiotic than the target pair, decision must be "no" and primary_category must be "NONE".
- If attribution is grammatically or semantically ambiguous, decision must be "no" and primary_category must be "NONE".
- evidence_phrases must only include phrases that support the classification.
- Do not infer missing relations.
- When in doubt, discard the target pair relation.
""".strip()

PUBMED_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
RETRIES = 3
BACKOFF_FACTOR = 2
LLM_RETRIES = 3
LLM_BACKOFF_FACTOR = 2
LLM_CLASSIFICATION_MAX_TOKENS = 768
AMR_CATEGORY_COLUMNS = [
    "amr_primary_category",
    "amr_evidence_phrases",
    "amr_target_pair_relation_supported",
    "amr_is_relevant",
]
AMR_PRIMARY_CATEGORIES = {
    "ENVIRONMENTAL_ANIMAL_NON_HUMAN",
    "CLINICAL_OUTBREAKS_REPORTS",
    "SURVEILLANCE",
    "MECHANISMS_OF_RESISTANCE_GENETICS_LAB_STUDIES",
    "OTHER",
    "NONE",
}
def _log(log_path: str | None, message: str) -> None:
    """Write log to stdout and, if provided, append to a file."""
    import datetime as _dt
    stamped = f"[{_dt.datetime.now().isoformat(timespec='seconds')}] {message}"
    print(stamped)
    if log_path:
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(stamped + "\n")


def _format_hms(seconds: float | int) -> str:
    total_seconds = max(0, int(round(float(seconds))))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}h {minutes}m {secs}s"


def fetch_pubmed_abstract(
    pmid: str,
    *,
    email: str | None = None,
    session: requests.Session | None = None,
) -> tuple[str, str, str]:
    """Fetch title and abstract text for a PMID via PubMed efetch with retries.

    Returns (title, abstract, status), where status is one of:
    - "ok": fetch succeeded (abstract may be empty if truly missing)
    - "no_abstract": fetch succeeded but no abstract present
    - "network_error": fetch failed after retries (caller should retry later)
    """
    sess = session or requests
    params = {"db": "pubmed", "id": pmid, "retmode": "xml"}
    if email:
        params["email"] = email
    if EUTILS_API_KEY:
        params["api_key"] = EUTILS_API_KEY
    for attempt in range(RETRIES):
        try:
            resp = sess.get(PUBMED_EFETCH, params=params, timeout=30)
            resp.raise_for_status()
            root = ET.fromstring(resp.text)
            article = root.find(".//Article")
            if article is None:
                return "", "", "no_abstract"
            title_el = article.find("./ArticleTitle")
            title = " ".join(title_el.itertext()).strip() if title_el is not None else ""
            abstract_el = article.find(".//Abstract")
            if abstract_el is None:
                return title, "", "no_abstract"
            parts = [" ".join(p.itertext()).strip() for p in abstract_el.findall(".//AbstractText")]
            abstract = "\n".join([p for p in parts if p])
            return title, abstract, "ok"
        except Exception as exc:
            if attempt == RETRIES - 1:
                print(f"PubMed fetch failed for PMID {pmid}: {exc}")
                return "", "", "network_error"
            delay = BACKOFF_FACTOR**attempt
            time.sleep(delay)


def fetch_pubmed_abstracts_batch(
    pmids: list[str],
    *,
    email: str | None = None,
    session: requests.Session | None = None,
) -> dict[str, tuple[str, str, str]]:
    """Fetch title/abstract for multiple PMIDs in one efetch request.

    Returns mapping ``pmid -> (title, abstract, status)``, where status is:
    ``ok``, ``no_abstract``, or ``network_error``.
    """
    pmids = [str(p).strip() for p in pmids if str(p).strip()]
    if not pmids:
        return {}
    sess = session or requests
    params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}
    if email:
        params["email"] = email
    if EUTILS_API_KEY:
        params["api_key"] = EUTILS_API_KEY

    for attempt in range(RETRIES):
        try:
            resp = sess.get(PUBMED_EFETCH, params=params, timeout=60)
            resp.raise_for_status()
            root = ET.fromstring(resp.text)

            out = {pmid: ("", "", "no_abstract") for pmid in pmids}
            for article in root.findall(".//PubmedArticle"):
                pmid_el = article.find(".//MedlineCitation/PMID")
                if pmid_el is None or not pmid_el.text:
                    continue
                pmid = str(pmid_el.text).strip()
                if not pmid:
                    continue
                article_el = article.find(".//Article")
                if article_el is None:
                    out[pmid] = ("", "", "no_abstract")
                    continue
                title_el = article_el.find("./ArticleTitle")
                title = " ".join(title_el.itertext()).strip() if title_el is not None else ""
                abstract_el = article_el.find(".//Abstract")
                if abstract_el is None:
                    out[pmid] = (title, "", "no_abstract")
                    continue
                parts = [" ".join(p.itertext()).strip() for p in abstract_el.findall(".//AbstractText")]
                abstract = "\n".join([p for p in parts if p])
                if abstract:
                    out[pmid] = (title, abstract, "ok")
                else:
                    out[pmid] = (title, "", "no_abstract")
            return out
        except Exception as exc:
            if attempt == RETRIES - 1:
                print(f"PubMed batch fetch failed for {len(pmids)} PMIDs: {exc}")
                return {pmid: ("", "", "network_error") for pmid in pmids}
            delay = BACKOFF_FACTOR**attempt
            time.sleep(delay)
    return {pmid: ("", "", "network_error") for pmid in pmids}


def build_user_prompt(pmid: str, pathogen: str, antibiotic: str, title: str, abstract: str) -> str:
    """Create the LLM prompt for a single PMID."""
    return f"""
PMID: {pmid}
target_pathogen:
{pathogen}

target_antibiotic:
{antibiotic}

title:
{title}

abstract:
{abstract or '[no abstract available]'}

Task:
Classify only this target pathogen-antibiotic pair. Ignore other organisms or drugs unless they
determine whether the target pair relation is explicitly supported. Return only the requested JSON.
""".strip()


def _extract_json_object(raw: str) -> dict:
    try:
        data = json.loads(raw)
    except Exception:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {"decision": "uncertain", "rationale": raw.strip()}
        try:
            data = json.loads(raw[start : end + 1])
        except Exception:
            return {"decision": "uncertain", "rationale": raw.strip()}
    return data if isinstance(data, dict) else {"decision": "uncertain", "rationale": raw.strip()}


def _normalize_primary_category(value: object, target_pair_relation_supported: bool = False) -> str:
    text = str(value or "").strip().upper()
    if not target_pair_relation_supported:
        return "NONE"
    if text in AMR_PRIMARY_CATEGORIES:
        return text
    return "NONE"


def _normalize_bool(value: object, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"true", "yes", "1"}:
        return True
    if text in {"false", "no", "0"}:
        return False
    return default


def _normalize_evidence_phrases(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    phrases = []
    for item in value:
        text = str(item or "").strip()
        if text:
            phrases.append(text)
    return phrases


def _derive_decision(data: dict, target_pair_relation_supported: bool, is_amr_relevant: bool) -> str:
    decision = _normalize_unified_decision(data.get("decision"))
    if decision != "uncertain" or "decision" in data:
        return decision
    if is_amr_relevant and target_pair_relation_supported:
        return "yes"
    if not is_amr_relevant or not target_pair_relation_supported:
        return "no"
    return "uncertain"


def _normalize_classification(data: dict, pmid: str) -> dict:
    target_pair_relation_supported = _normalize_bool(data.get("target_pair_relation_supported"), default=False)
    is_amr_relevant = _normalize_bool(data.get("is_amr_relevant"), default=False)
    primary_category = _normalize_primary_category(data.get("primary_category"), target_pair_relation_supported)
    decision = _derive_decision(data, target_pair_relation_supported, is_amr_relevant)
    if not target_pair_relation_supported and decision == "yes":
        decision = "no"
    if decision == "no" or not is_amr_relevant:
        primary_category = "NONE"
    return {
        "pmid": pmid,
        "decision": decision,
        "rationale": str(data.get("rationale", "") or "").strip(),
        "primary_category": primary_category,
        "evidence_phrases": _normalize_evidence_phrases(data.get("evidence_phrases")),
        "target_pair_relation_supported": target_pair_relation_supported,
        "is_amr_relevant": is_amr_relevant,
    }


def _default_no_abstract_classification(pmid: str) -> dict:
    return {
        "pmid": pmid,
        "decision": "uncertain",
        "rationale": "no abstract",
        "primary_category": "NONE",
        "evidence_phrases": [],
        "target_pair_relation_supported": False,
        "is_amr_relevant": False,
    }


def classify_abstract(
    llm,
    pmid: str,
    pathogen: str,
    antibiotic: str,
    title: str,
    abstract: str,
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    max_tokens: int = LLM_CLASSIFICATION_MAX_TOKENS,
    log_path: str | None = None,
    retries: int = LLM_RETRIES,
    backoff: int = LLM_BACKOFF_FACTOR,
) -> dict:
    """Call the LLM to classify a single abstract with retries/backoff."""
    prompt = build_user_prompt(pmid, pathogen, antibiotic, title, abstract)
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            raw = llm.complete(prompt, system_prompt=system_prompt, max_tokens=max_tokens, temperature=0)
            return _normalize_classification(_extract_json_object(raw), pmid)
        except Exception as exc:  # pragma: no cover - network/API handling
            last_error = exc
            if attempt == retries - 1:
                _log(log_path, f"LLM failed for PMID {pmid}: {exc}")
                break
            delay = backoff**attempt
            _log(log_path, f"LLM error for PMID {pmid}: {exc}. Retrying in {delay}s...")
            time.sleep(delay)
    data = {"decision": "uncertain", "rationale": f"LLM failure: {last_error}"}
    return _normalize_classification(data, pmid)


def prepare_pmids_to_review(
    pmid_df: pd.DataFrame,
    *,
    exclude_pmids: set[str] | None = None,
    max_pmids: int | None = None,
) -> pd.DataFrame:
    """Flatten pmids_results.parquet structure to one row per PMID/pathogen/antibiotic."""
    long_df = pmid_df.explode("pmids").dropna(subset=["pmids"]).copy()
    long_df["pmid"] = long_df["pmids"].astype(str).str.extract("(\\d+)")[0]
    long_df = long_df.dropna(subset=["pmid"])
    long_df = long_df[["pmid", "pathogen", "antibiotic", "start_date", "end_date"]].drop_duplicates()
    if exclude_pmids:
        long_df = long_df[~long_df["pmid"].astype(str).isin(exclude_pmids)]
    if max_pmids:
        long_df = long_df.head(max_pmids)
    return long_df.reset_index(drop=True)


def prepare_yearly_candidate_pmids(
    yearly_df: pd.DataFrame,
    *,
    exclude_pmids: set[str] | None = None,
    max_pmids: int | None = None,
) -> pd.DataFrame:
    """Flatten yearly pair rows to one row per PMID/pathogen/antibiotic/year."""
    required_cols = {"pathogen", "antibiotic", "year", "pmids"}
    missing = required_cols - set(yearly_df.columns)
    if missing:
        raise KeyError(f"yearly_df missing required columns: {sorted(missing)}")

    work = yearly_df.copy()
    work = work.explode("pmids").dropna(subset=["pmids"]).copy()
    work["pmid"] = work["pmids"].astype(str).str.extract(r"(\d+)")[0]
    work = work.dropna(subset=["pmid"])
    work["year"] = pd.to_numeric(work["year"], errors="coerce").astype("Int64")
    work = work.dropna(subset=["year"])
    work["year"] = work["year"].astype(int)
    work = work[["pmid", "pathogen", "antibiotic", "year"]].drop_duplicates()
    if exclude_pmids:
        work = work[~work["pmid"].astype(str).isin(exclude_pmids)]
    if max_pmids:
        work = work.head(max_pmids)
    return work.sort_values(["year", "pathogen", "antibiotic", "pmid"]).reset_index(drop=True)


def _normalize_unified_decision(value: object) -> str:
    text = str(value or "").strip().lower()
    if text.startswith("yes"):
        return "yes"
    if text.startswith("no"):
        return "no"
    if text.startswith("uncertain"):
        return "uncertain"
    return "uncertain"


def run_unified_llm_filter(
    candidate_df: pd.DataFrame,
    llm,
    *,
    email: str | None = None,
    rate_limit: float = 0.35,
    system_prompt: str = UNIFIED_AMR_INTEREST_SYSTEM_PROMPT,
    output_path: str | None = None,
    resume_from: pd.DataFrame | None = None,
    save_every: int = 50,
    progress_every: int = 25,
    max_llm_calls: int | None = None,
    dry_run: bool = False,
    verbose: int = 1,
    log_path: str | None = None,
    abstract_cache_path: str | None = "pmid_abstracts.parquet",
    abstract_cache_save_every: int | None = 200,
    auto_resume_from_output: bool = True,
    llm_max_workers: int = 1,
    llm_batch_size: int = 8,
    failed_retry_path: str | None = "yearly_amr_unified_failures.parquet",
    max_failure_retries: int | None = 5,
    abstract_fetch_batch_size: int = 200,
    prefetch_abstracts: bool = True,
    category_backfill_decisions: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Run the unified AMR + scientific-interest LLM screen on yearly candidates."""
    required_cols = {"pmid", "pathogen", "antibiotic", "year"}
    missing = required_cols - set(candidate_df.columns)
    if missing:
        raise KeyError(f"candidate_df missing required columns: {sorted(missing)}")
    category_cols = AMR_CATEGORY_COLUMNS

    def _append_log(message: str) -> None:
        if log_path:
            _log(log_path, message)

    def _progress(message: str) -> None:
        if verbose >= 1:
            print(message)
        _append_log(message)

    existing = pd.DataFrame()
    key_cols = ["pmid", "pathogen", "antibiotic", "year"]
    seen_keys: set[tuple[str, str, str, int]] = set()
    category_backfill_set = (
        {str(value).strip().lower() for value in category_backfill_decisions}
        if category_backfill_decisions is not None
        else None
    )
    if auto_resume_from_output and resume_from is None and output_path and os.path.exists(output_path):
        try:
            resume_from = pd.read_parquet(output_path)
        except Exception as exc:
            _append_log(f"Failed loading existing unified output from {output_path}: {exc}")
    if resume_from is not None and not resume_from.empty:
        existing = resume_from.copy()
        if set(key_cols).issubset(existing.columns):
            for col in category_cols:
                if col not in existing.columns:
                    existing[col] = pd.NA
            complete_mask = existing[category_cols].notna().all(axis=1)
            if "amr_evidence_phrases" in existing.columns:
                complete_mask &= existing["amr_evidence_phrases"].apply(
                    lambda v: isinstance(v, (list, tuple)) or hasattr(v, "tolist")
                )
            if "unified_rationale" in existing.columns:
                llm_failure_mask = existing["unified_rationale"].astype(str).str.startswith("LLM failure:")
                complete_mask &= ~llm_failure_mask
            complete_existing = existing.loc[complete_mask].copy()
            incomplete_existing = existing.loc[~complete_mask].copy()
            if category_backfill_set is not None and not incomplete_existing.empty:
                incomplete_decisions = incomplete_existing["unified_decision"].map(_normalize_unified_decision)
                defer_mask = ~incomplete_decisions.isin(category_backfill_set)
                if defer_mask.any():
                    deferred_existing = incomplete_existing.loc[defer_mask].copy()
                    complete_existing = pd.concat([complete_existing, deferred_existing], ignore_index=True)
                    incomplete_existing = incomplete_existing.loc[~defer_mask].copy()
            if not incomplete_existing.empty:
                _append_log(
                    f"Requeueing {len(incomplete_existing)} existing unified rows missing AMR category fields "
                    "or from prior LLM failures."
                )
            existing = complete_existing
            for row in existing[key_cols].itertuples(index=False):
                seen_keys.add((str(row.pmid), str(row.pathogen), str(row.antibiotic), int(row.year)))

    # Persistent failures for flaky network-fetch rows
    failure_cols = ["pmid", "pathogen", "antibiotic", "year", "fail_count", "last_error", "last_attempt_ts"]
    failures_df = pd.DataFrame(columns=failure_cols)
    if failed_retry_path and os.path.exists(failed_retry_path):
        try:
            failures_df = pd.read_parquet(failed_retry_path)
        except Exception as exc:
            _append_log(f"Failed loading retry state from {failed_retry_path}: {exc}")
            failures_df = pd.DataFrame(columns=failure_cols)
    for col in failure_cols:
        if col not in failures_df.columns:
            failures_df[col] = pd.NA
    failure_map: dict[tuple[str, str, str, int], dict[str, object]] = {}
    for row in failures_df[failure_cols].itertuples(index=False):
        k = (str(row.pmid), str(row.pathogen), str(row.antibiotic), int(row.year))
        failure_map[k] = {
            "fail_count": int(pd.to_numeric(row.fail_count, errors="coerce") if pd.notna(row.fail_count) else 0),
            "last_error": str(row.last_error) if pd.notna(row.last_error) else "",
            "last_attempt_ts": str(row.last_attempt_ts) if pd.notna(row.last_attempt_ts) else "",
        }

    def _candidate_key(row: object) -> tuple[str, str, str, int]:
        return (str(row.pmid), str(row.pathogen), str(row.antibiotic), int(row.year))

    def _has_exhausted_failure_retries(key: tuple[str, str, str, int]) -> bool:
        if max_failure_retries is None:
            return False
        existing_failure = failure_map.get(key)
        return bool(existing_failure and int(existing_failure.get("fail_count", 0)) >= int(max_failure_retries))

    progress_seen_keys = set(seen_keys)
    skipped_existing_rows = 0
    skipped_retry_exhausted_rows = 0
    pending_candidate_rows = 0
    for row in candidate_df.itertuples(index=False):
        key = _candidate_key(row)
        if key in progress_seen_keys:
            skipped_existing_rows += 1
            continue
        if _has_exhausted_failure_retries(key):
            skipped_retry_exhausted_rows += 1
            progress_seen_keys.add(key)
            continue
        pending_candidate_rows += 1
        progress_seen_keys.add(key)

    abstract_cache: dict[str, tuple[str, str]] = {}
    cache_dirty = False
    session = requests.Session()
    if abstract_cache_path and os.path.exists(abstract_cache_path):
        try:
            cache_df = pd.read_parquet(abstract_cache_path)
            if {"pmid", "title", "abstract"}.issubset(cache_df.columns):
                for row in cache_df[["pmid", "title", "abstract"]].itertuples(index=False):
                    abstract_cache[str(row.pmid)] = (row.title or "", row.abstract or "")
        except Exception as exc:
            _append_log(f"Failed loading abstract cache from {abstract_cache_path}: {exc}")

    failed_prefetch_pmids: set[str] = set()
    if prefetch_abstracts and not dry_run:
        candidate_pmids = []
        for row in candidate_df.itertuples(index=False):
            key = _candidate_key(row)
            if key not in seen_keys and not _has_exhausted_failure_retries(key):
                candidate_pmids.append(str(row.pmid))
        candidate_pmids = pd.Index(candidate_pmids).dropna().unique().tolist()
        missing_pmids = [pmid for pmid in candidate_pmids if pmid not in abstract_cache]
        if missing_pmids:
            batch_n = max(1, int(abstract_fetch_batch_size))
            total_batches = (len(missing_pmids) + batch_n - 1) // batch_n
            for b_idx, start in enumerate(range(0, len(missing_pmids), batch_n), start=1):
                batch_pmids = missing_pmids[start : start + batch_n]
                batch_map = fetch_pubmed_abstracts_batch(batch_pmids, email=email, session=session)
                for pmid, (title, abstract, status) in batch_map.items():
                    if status == "network_error":
                        failed_prefetch_pmids.add(str(pmid))
                        continue
                    abstract_cache[str(pmid)] = (title or "", abstract or "")
                    cache_dirty = True
                if verbose >= 1 and (b_idx == 1 or b_idx % 10 == 0 or b_idx == total_batches):
                    print(f"Prefetched abstracts batch {b_idx}/{total_batches} ({len(batch_pmids)} PMIDs).")
                time.sleep(rate_limit)
            if abstract_cache_path and cache_dirty:
                _append_log(f"Saving prefetched abstract cache ({len(abstract_cache)} PMIDs).")
                # checkpoint helper declared below; call deferred via flag

    def _checkpoint_results() -> None:
        if not output_path:
            return
        tmp_df = pd.concat([existing, pd.DataFrame(results)], ignore_index=True)
        if not tmp_df.empty:
            tmp_df = tmp_df.drop_duplicates(subset=key_cols, keep="last")
        atomic_write_parquet(tmp_df, output_path, index=False)

    def _checkpoint_cache() -> None:
        nonlocal cache_dirty
        if not abstract_cache_path or not abstract_cache:
            return
        cache_df = pd.DataFrame([{"pmid": p, "title": t, "abstract": a} for p, (t, a) in abstract_cache.items()])
        atomic_write_parquet(cache_df, abstract_cache_path, index=False)
        cache_dirty = False

    def _checkpoint_failures() -> None:
        if not failed_retry_path:
            return
        rows = []
        for (pmid, pathogen, antibiotic, year), meta in failure_map.items():
            rows.append(
                {
                    "pmid": pmid,
                    "pathogen": pathogen,
                    "antibiotic": antibiotic,
                    "year": int(year),
                    "fail_count": int(meta.get("fail_count", 0)),
                    "last_error": str(meta.get("last_error", "")),
                    "last_attempt_ts": str(meta.get("last_attempt_ts", "")),
                }
            )
        atomic_write_parquet(pd.DataFrame(rows, columns=failure_cols), failed_retry_path, index=False)

    def _classification_output_fields(verdict: dict) -> dict:
        return {
            "amr_primary_category": verdict["primary_category"],
            "amr_evidence_phrases": verdict["evidence_phrases"],
            "amr_target_pair_relation_supported": verdict["target_pair_relation_supported"],
            "amr_is_relevant": verdict["is_amr_relevant"],
        }

    def _default_no_abstract_output_fields(pmid: str) -> dict:
        return _classification_output_fields(_default_no_abstract_classification(pmid))

    def _record_result(item: dict) -> None:
        results.append(item)
        seen_keys.add((str(item["pmid"]), str(item["pathogen"]), str(item["antibiotic"]), int(item["year"])))
        # clear retry state once row is successfully decided/stored
        failure_map.pop((str(item["pmid"]), str(item["pathogen"]), str(item["antibiotic"]), int(item["year"])), None)

        if output_path and save_every and len(results) % save_every == 0:
            _checkpoint_results()
        if abstract_cache_path and abstract_cache_save_every and cache_dirty and len(results) % int(max(1, abstract_cache_save_every)) == 0:
            _checkpoint_cache()
        if failed_retry_path and save_every and len(results) % save_every == 0:
            _checkpoint_failures()
        if verbose >= 1 and progress_every and len(results) % progress_every == 0:
            batch_df = pd.DataFrame(results)
            yes_n = int((batch_df["unified_decision"] == "yes").sum())
            no_n = int((batch_df["unified_decision"] == "no").sum())
            unc_n = int((batch_df["unified_decision"] == "uncertain").sum())
            completed = len(results)
            if completed > 0 and total > 0:
                elapsed_s = time.time() - run_started_at
                avg_s_per_item = elapsed_s / completed
                remaining_items = max(0, total - completed)
                eta_s = avg_s_per_item * remaining_items
                eta_text = _format_hms(eta_s)
                est_finish = (datetime.now() + timedelta(seconds=eta_s)).strftime("%H:%M:%S")
            else:
                eta_text = "estimating"
                est_finish = "--:--:--"
            _progress(
                f"Processed {completed:>6}/{total:<6} yes:{yes_n:>6} no:{no_n:>6} uncertain:{unc_n:>6} "
                f"| ETA {eta_text} | est finish {est_finish}"
            )

    def _run_llm_batch(batch: list[dict]) -> list[dict]:
        if not batch:
            return []
        if dry_run:
            return [
                {
                    "pmid": item["pmid"],
                    "pathogen": item["pathogen"],
                    "antibiotic": item["antibiotic"],
                    "year": item["year"],
                    "unified_decision": "uncertain",
                    "unified_raw_decision": "dry_run",
                    "unified_rationale": "dry_run",
                    "title": item["title"] or "",
                    "abstract": item["abstract"] or "",
                    **_default_no_abstract_output_fields(item["pmid"]),
                }
                for item in batch
            ]

        def _classify(item: dict) -> dict:
            verdict = classify_abstract(
                llm,
                item["pmid"],
                item["pathogen"],
                item["antibiotic"],
                item["title"],
                item["abstract"],
                system_prompt=system_prompt,
                max_tokens=LLM_CLASSIFICATION_MAX_TOKENS,
                log_path=log_path,
            )
            raw_decision = str(verdict.get("decision", "uncertain"))
            return {
                "pmid": item["pmid"],
                "pathogen": item["pathogen"],
                "antibiotic": item["antibiotic"],
                "year": item["year"],
                "unified_decision": _normalize_unified_decision(raw_decision),
                "unified_raw_decision": raw_decision,
                "unified_rationale": str(verdict.get("rationale", "") or "").strip(),
                "title": item["title"] or "",
                "abstract": item["abstract"] or "",
                **_classification_output_fields(verdict),
            }

        if int(max(1, llm_max_workers)) == 1:
            return [_classify(item) for item in batch]

        out: list[dict] = []
        with ThreadPoolExecutor(max_workers=int(max(1, llm_max_workers))) as pool:
            futures = [pool.submit(_classify, item) for item in batch]
            for fut in as_completed(futures):
                out.append(fut.result())
        return out

    results: list[dict] = []
    total = pending_candidate_rows
    llm_calls_used = 0
    pending_batch: list[dict] = []
    run_started_at = time.time()
    if verbose >= 1:
        _progress(
            f"[unified_filter] starting run for {total} pending candidate rows "
            f"({len(candidate_df)} input rows; skipped {skipped_existing_rows} already complete"
            f"{', skipped ' + str(skipped_retry_exhausted_rows) + ' retry-exhausted' if skipped_retry_exhausted_rows else ''})"
        )

    for row in candidate_df.itertuples(index=False):
        key = _candidate_key(row)
        if key in seen_keys:
            continue

        if _has_exhausted_failure_retries(key):
            continue

        pmid = str(row.pmid)
        pathogen = str(row.pathogen)
        antibiotic = str(row.antibiotic)
        year = int(row.year)

        if dry_run:
            _record_result(
                {
                    "pmid": pmid,
                    "pathogen": pathogen,
                    "antibiotic": antibiotic,
                    "year": year,
                    "unified_decision": "uncertain",
                    "unified_raw_decision": "dry_run",
                    "unified_rationale": "dry_run",
                    "title": "",
                    "abstract": "",
                    **_default_no_abstract_output_fields(pmid),
                }
            )
            continue

        if pmid in failed_prefetch_pmids:
            now_ts = pd.Timestamp.utcnow().isoformat()
            prev = failure_map.get(key, {"fail_count": 0, "last_error": "", "last_attempt_ts": ""})
            failure_map[key] = {
                "fail_count": int(prev.get("fail_count", 0)) + 1,
                "last_error": "network_error_prefetch",
                "last_attempt_ts": now_ts,
            }
            if failed_retry_path:
                _checkpoint_failures()
            continue
        elif pmid in abstract_cache:
            title, abstract = abstract_cache[pmid]
        else:
            title, abstract, status = fetch_pubmed_abstract(pmid, email=email, session=session)
            if status == "network_error":
                now_ts = pd.Timestamp.utcnow().isoformat()
                prev = failure_map.get(key, {"fail_count": 0, "last_error": "", "last_attempt_ts": ""})
                failure_map[key] = {
                    "fail_count": int(prev.get("fail_count", 0)) + 1,
                    "last_error": "network_error",
                    "last_attempt_ts": now_ts,
                }
                _append_log(f"Skipping PMID {pmid} due to network error (attempt {failure_map[key]['fail_count']}).")
                if failed_retry_path:
                    _checkpoint_failures()
                time.sleep(rate_limit)
                continue
            abstract_cache[pmid] = (title, abstract)
            cache_dirty = True
            time.sleep(rate_limit)

        if not abstract:
            no_abstract = _default_no_abstract_classification(pmid)
            _record_result(
                {
                    "pmid": pmid,
                    "pathogen": pathogen,
                    "antibiotic": antibiotic,
                    "year": year,
                    "unified_decision": no_abstract["decision"],
                    "unified_raw_decision": "no_abstract",
                    "unified_rationale": no_abstract["rationale"],
                    "title": title or "",
                    "abstract": abstract or "",
                    **_classification_output_fields(no_abstract),
                }
            )
            continue

        if max_llm_calls is not None and llm_calls_used >= max_llm_calls:
            break

        pending_batch.append(
            {
                "pmid": pmid,
                "pathogen": pathogen,
                "antibiotic": antibiotic,
                "year": year,
                "title": title,
                "abstract": abstract,
            }
        )
        if max_llm_calls is not None and (llm_calls_used + len(pending_batch)) >= max_llm_calls:
            should_flush = True
        else:
            should_flush = len(pending_batch) >= int(max(1, llm_batch_size))
        if should_flush:
            for item in _run_llm_batch(pending_batch):
                _record_result(item)
                llm_calls_used += 1
            pending_batch = []
            time.sleep(rate_limit)

    # Flush remaining LLM items
    if pending_batch and (max_llm_calls is None or llm_calls_used < max_llm_calls):
        if max_llm_calls is not None:
            remaining = int(max_llm_calls - llm_calls_used)
            pending_batch = pending_batch[: max(0, remaining)]
        for item in _run_llm_batch(pending_batch):
            _record_result(item)
            llm_calls_used += 1

    out_df = pd.concat([existing, pd.DataFrame(results)], ignore_index=True)
    if not out_df.empty:
        out_df = out_df.drop_duplicates(subset=key_cols, keep="last").reset_index(drop=True)
    if output_path:
        atomic_write_parquet(out_df, output_path, index=False)
    if abstract_cache_path and abstract_cache:
        _checkpoint_cache()
    if failed_retry_path:
        _checkpoint_failures()
    session.close()
    return out_df


def run_llm_filter(
    long_df: pd.DataFrame,
    llm,
    *,
    email: str | None = None,
    rate_limit: float = 0.35,
    save_every: int = 50,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    progress_every: int = 10,
    resume_from: pd.DataFrame | None = None,
    seen_pmids: set | None = None,
    output_path: str | None = None,
    log_path: str | None = None,
    max_llm_calls: int | None = None,
    dry_run: bool = False,
    verbose: int = 1,
    abstract_cache_path: str | None = "pmid_abstracts.parquet",
    abstract_cache_save_every: int | None = 200,
) -> pd.DataFrame:
    """
    Process each PMID row and return a DataFrame with LLM decisions.

    - If resume_from or seen_pmids provided, processed PMIDs are skipped.
    - Keeps the “seen” index in-memory to avoid many filesystem writes; save once via output_path if desired.
    - max_llm_calls limits LLM invocations per run; dry_run skips LLM calls.
    - verbose controls stdout logging: 0 = silent, 1 = progress_every summaries, 2 = full output.
    """
    def _stamp(message: str) -> str:
        import datetime as _dt

        return f"[{_dt.datetime.now().isoformat(timespec='seconds')}] {message}"

    def _append_log(message: str) -> None:
        if log_path:
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(_stamp(message) + "\n")

    def _log_verbose(message: str) -> None:
        if verbose >= 2:
            _log(log_path, message)
        else:
            _append_log(message)

    def _progress(message: str) -> None:
        if verbose >= 1:
            print(message)
        _append_log(message)

    # Use combination keys (pmid, pathogen, antibiotic) for exclusion
    seen_keys: set[tuple[str, str, str]] = set()
    seen_pmids_only: set[str] = set()
    if seen_pmids:
        for item in seen_pmids:
            if isinstance(item, tuple) and len(item) >= 3:
                pmid_i, pathogen_i, antibiotic_i = item[0], item[1], item[2]
                seen_keys.add((str(pmid_i), str(pathogen_i), str(antibiotic_i)))
            else:
                seen_pmids_only.add(str(item))
    # Persistent abstract cache across runs
    abstract_cache: dict[str, tuple[str, str]] = {}
    cache_dirty = False
    # Load on-disk cache if available
    if abstract_cache_path and os.path.exists(abstract_cache_path):
        try:
            _df_cache = pd.read_parquet(abstract_cache_path)
            if {"pmid", "title", "abstract"}.issubset(_df_cache.columns):
                for r in _df_cache[["pmid", "title", "abstract"]].itertuples(index=False):
                    abstract_cache[str(r.pmid)] = (r.title or "", r.abstract or "")
                _log_verbose(f"Loaded abstract cache: {len(abstract_cache)} PMIDs from {abstract_cache_path}")
        except Exception as e:
            _log_verbose(f"Failed to load abstract cache from {abstract_cache_path}: {e}")

    existing = pd.DataFrame()
    if resume_from is not None and not resume_from.empty:
        existing = resume_from.copy()
        if {"pmid", "pathogen", "antibiotic"}.issubset(existing.columns):
            for r in existing[["pmid", "pathogen", "antibiotic"]].astype(str).itertuples(index=False):
                seen_keys.add((r.pmid, r.pathogen, r.antibiotic))
        elif "pmid" in existing.columns:
            seen_pmids_only.update(existing["pmid"].astype(str))
        # Seed abstract cache from existing decisions if available
        try:
            if {"pmid", "title", "abstract"}.issubset(existing.columns):
                for r in existing[["pmid", "title", "abstract"]].itertuples(index=False):
                    if pd.notna(r.pmid):
                        pmid_key = str(r.pmid)
                        title0 = (r.title or "") if not pd.isna(r.title) else ""
                        abstr0 = (r.abstract or "") if not pd.isna(r.abstract) else ""
                        if pmid_key and pmid_key not in abstract_cache:
                            abstract_cache[pmid_key] = (title0, abstr0)
        except Exception:
            pass

    results = []  # only new rows collected this run
    session = requests.Session()
    total = len(long_df)
    processed = len(existing)
    positives = (existing["decision"].str.lower() == "yes").sum() if not existing.empty else 0
    llm_calls_used = 0

    if total == 0:
        _log_verbose("No PMIDs to process.")
        return existing

    for run_idx, row in enumerate(long_df.itertuples(index=False), start=1):
        pmid = str(row.pmid)
        key = (pmid, str(row.pathogen), str(row.antibiotic))
        if key in seen_keys or pmid in seen_pmids_only:
            continue
        if pmid not in abstract_cache:
            title, abstract, status = fetch_pubmed_abstract(pmid, email=email, session=session)
            if status == "network_error":
                _log_verbose(f"[{run_idx}/{total}] PMID {pmid} fetch skipped due to network error; will retry later.")
                # Do not mark as seen or processed; leave for a future run
                time.sleep(rate_limit)
                continue
            abstract_cache[pmid] = (title, abstract)
            cache_dirty = True
            time.sleep(rate_limit)
        else:
            title, abstract = abstract_cache[pmid]

        if not abstract:
            # Skip PMIDs without abstracts to avoid ambiguous judgments
            _log_verbose(f"[{run_idx}/{total}] PMID {pmid} skipped (no abstract).")
            verdict = {
                "pmid": pmid,
                "decision": "uncertain",
                "rationale": "no abstract",
            }
            title = title or ""
            abstract = abstract or ""
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
            seen_keys.add(key)
            processed += 1
            continue

        _log_verbose(
            f"[{run_idx}/{total}] PMID {pmid} | {row.pathogen} vs {row.antibiotic} | interval {row.start_date}–{row.end_date}",
        )

        if dry_run:
            verdict = {
                "pmid": pmid,
                "decision": "uncertain",
                "rationale": "dry_run",
            }
        else:
            if max_llm_calls is not None and llm_calls_used >= max_llm_calls:
                _log_verbose(f"Reached max_llm_calls={max_llm_calls}, stopping.")
                break
            verdict = classify_abstract(
                llm,
                pmid,
                row.pathogen,
                row.antibiotic,
                title,
                abstract,
                system_prompt=system_prompt,
                log_path=log_path,
            )
            llm_calls_used += 1
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
        seen_keys.add(key)
        processed += 1
        decision = str(verdict.get("decision", "")).lower()
        if decision == "yes":
            positives += 1

        _log_verbose(f"    decision={decision} | rationale={verdict.get('rationale','')[:120]}")

        # Periodic checkpoints: decisions and abstract cache
        if output_path and save_every and (len(results) % save_every == 0):
            df_tmp = pd.concat([existing, pd.DataFrame(results)], ignore_index=True)
            # Deduplicate on combination-level keys to allow multiple pathogen/antibiotic per PMID
            key_cols = ["pmid", "pathogen", "antibiotic", "start_date", "end_date"]
            keep_cols = [c for c in key_cols if c in df_tmp.columns]
            df_tmp = df_tmp.drop_duplicates(subset=keep_cols, keep="last")
            atomic_write_parquet(df_tmp, output_path, index=False)
            _log_verbose(f"Checkpoint saved {len(df_tmp)} rows to {output_path}")
        # Save abstract cache periodically, independent of decision checkpoints
        if abstract_cache_path and cache_dirty and abstract_cache_save_every and (len(results) % max(1, abstract_cache_save_every) == 0):
            try:
                df_cache = pd.DataFrame(
                    [{"pmid": k, "title": v[0], "abstract": v[1]} for k, v in abstract_cache.items()]
                )
                # Keep last occurrence per PMID (dict already unique keys)
                atomic_write_parquet(df_cache, abstract_cache_path, index=False)
                _log_verbose(f"Checkpoint saved abstract cache with {len(df_cache)} PMIDs to {abstract_cache_path}")
                cache_dirty = False
            except Exception as e:
                _log_verbose(f"Failed to save abstract cache to {abstract_cache_path}: {e}")

        if progress_every and (len(results) % progress_every == 0):
            # compute interim summary on current in-memory results
            df_now = pd.DataFrame(results)
            no_count = (df_now["decision"].str.lower() == "no").sum() if not df_now.empty else 0
            yes_count = (df_now["decision"].str.lower() == "yes").sum() if not df_now.empty else 0
            unc_count = (df_now["decision"].str.lower() == "uncertain").sum() if not df_now.empty else 0
            # cumulative counts (existing + current batch)
            df_cum = pd.concat([existing, df_now], ignore_index=True)
            cum_no = (df_cum["decision"].str.lower() == "no").sum() if not df_cum.empty else 0
            cum_yes = (df_cum["decision"].str.lower() == "yes").sum() if not df_cum.empty else 0
            cum_unc = (df_cum["decision"].str.lower() == "uncertain").sum() if not df_cum.empty else 0

            run_processed = len(results)
            cumulative_processed = len(df_cum)
            # Pretty, fixed-width progress line so columns align across updates
            _progress(
                (
                    f"Processed {run_processed:>3}/{total:<3}  "
                    f"no:{no_count:>7}  yes:{yes_count:>7}  uncertain:{unc_count:>7}  |  "
                    f"cumulative {cumulative_processed:>7}  "
                    f"no:{cum_no:>7}  yes:{cum_yes:>7}  uncertain:{cum_unc:>7}"
                )
            )

    df_out = pd.DataFrame(results)
    if output_path:
        df_all = pd.concat([existing, df_out], ignore_index=True)
        key_cols = ["pmid", "pathogen", "antibiotic", "start_date", "end_date"]
        keep_cols = [c for c in key_cols if c in df_all.columns]
        df_all = df_all.drop_duplicates(subset=keep_cols, keep="last")
        atomic_write_parquet(df_all, output_path, index=False)
        df_out = df_all  # return deduped full set for convenience
    # Final save of abstract cache at end of run
    if abstract_cache_path and abstract_cache:
        try:
            df_cache = pd.DataFrame(
                [{"pmid": k, "title": v[0], "abstract": v[1]} for k, v in abstract_cache.items()]
            )
            atomic_write_parquet(df_cache, abstract_cache_path, index=False)
            _log_verbose(f"Saved abstract cache with {len(df_cache)} PMIDs to {abstract_cache_path}")
        except Exception as e:
            _log_verbose(f"Failed to save abstract cache to {abstract_cache_path}: {e}")
    # Batch-level summary
    yes_count = (df_out["decision"].str.lower() == "yes").sum() if not df_out.empty else 0
    no_count = (df_out["decision"].str.lower() == "no").sum() if not df_out.empty else 0
    unc_count = (df_out["decision"].str.lower() == "uncertain").sum() if not df_out.empty else 0
    _log_verbose(
        f"Finished: processed {processed} / {total}; positives {positives} ({(positives/max(processed,1)):.3f} fraction).",
    )
    _log_verbose(f"Batch summary: yes={yes_count}, no={no_count}, uncertain={unc_count}")
    session.close()
    return df_out


def build_yearly_screened_pairs(
    yearly_df: pd.DataFrame,
    unified_df: pd.DataFrame,
    *,
    include_decisions: Iterable[str] = ("yes", "uncertain"),
    output_path: str | None = None,
    merge_with_existing_output: bool = True,
) -> pd.DataFrame:
    """Filter yearly PMIDs by unified decisions and reaggregate yearly pair counts."""
    required_yearly = {"pathogen", "antibiotic", "year", "pmids"}
    missing_yearly = required_yearly - set(yearly_df.columns)
    if missing_yearly:
        raise KeyError(f"yearly_df missing required columns: {sorted(missing_yearly)}")

    required_screen = {"pmid", "pathogen", "antibiotic", "year", "unified_decision"}
    missing_screen = required_screen - set(unified_df.columns)
    if missing_screen:
        raise KeyError(f"unified_df missing required columns: {sorted(missing_screen)}")

    include_set = {str(v).strip().lower() for v in include_decisions}
    if not include_set:
        raise ValueError("include_decisions cannot be empty.")

    work = yearly_df.copy()
    work = work.explode("pmids").dropna(subset=["pmids"]).copy()
    work["pmid"] = work["pmids"].astype(str).str.extract(r"(\d+)")[0]
    work = work.dropna(subset=["pmid"])
    work["year"] = pd.to_numeric(work["year"], errors="coerce").astype("Int64")
    work = work.dropna(subset=["year"])
    work["year"] = work["year"].astype(int)

    keep_keys = (
        unified_df.loc[
            unified_df["unified_decision"].astype(str).str.lower().isin(include_set),
            ["pmid", "pathogen", "antibiotic", "year"],
        ]
        .drop_duplicates()
        .astype({"pmid": str, "pathogen": str, "antibiotic": str, "year": int})
    )
    if keep_keys.empty:
        out = pd.DataFrame(columns=["pathogen", "antibiotic", "year", "count", "pmids"])
        if output_path:
            atomic_write_parquet(out, output_path, index=False)
        return out

    filtered = work.merge(keep_keys, on=["pmid", "pathogen", "antibiotic", "year"], how="inner")
    filtered = filtered.drop_duplicates(subset=["pmid", "pathogen", "antibiotic", "year"])
    out = (
        filtered.groupby(["pathogen", "antibiotic", "year"], as_index=False)
        .agg(
            count=("pmid", "nunique"),
            pmids=("pmid", lambda values: sorted(pd.unique(values))),
        )
        .sort_values(["pathogen", "antibiotic", "year"])
        .reset_index(drop=True)
    )
    if output_path and merge_with_existing_output and os.path.exists(output_path):
        existing_out = pd.read_parquet(output_path)
        existing_out = existing_out[["pathogen", "antibiotic", "year", "count", "pmids"]].copy()
        existing_out["year"] = pd.to_numeric(existing_out["year"], errors="coerce").astype("Int64")
        existing_out = existing_out.dropna(subset=["year"]).copy()
        existing_out["year"] = existing_out["year"].astype(int)
        existing_out["pmids"] = existing_out["pmids"].apply(lambda v: v if isinstance(v, list) else [])
        out["pmids"] = out["pmids"].apply(lambda v: v if isinstance(v, list) else [])
        combined = pd.concat([existing_out, out], ignore_index=True)
        out = (
            combined.groupby(["pathogen", "antibiotic", "year"], as_index=False)
            .agg(
                pmids=("pmids", lambda rows: sorted({str(p) for row in rows for p in row})),
                count=("count", "max"),
            )
            .sort_values(["pathogen", "antibiotic", "year"])
            .reset_index(drop=True)
        )
        out["count"] = out["pmids"].apply(len)
    if output_path:
        atomic_write_parquet(out, output_path, index=False)
    return out


def build_positive_dataset(
    pmid_df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    *,
    output_path: str | None = None,
) -> pd.DataFrame:
    """Build a positive-only dataset at the original interval grain.

    This preserves the combination-level review grain of ``filtered_df``:
    a PMID is retained only for the specific reviewed
    ``(pmid, pathogen, antibiotic, start_date, end_date)`` rows marked ``yes``.
    """
    required_pmid_cols = {"pathogen", "antibiotic", "start_date", "end_date", "pmids"}
    missing_pmid = required_pmid_cols - set(pmid_df.columns)
    if missing_pmid:
        raise KeyError(f"pmid_df missing required columns: {missing_pmid}")

    required_filtered_cols = {"pmid", "pathogen", "antibiotic", "decision"}
    missing_filtered = required_filtered_cols - set(filtered_df.columns)
    if missing_filtered:
        raise KeyError(f"filtered_df missing required columns: {missing_filtered}")

    work = pmid_df.copy()
    work = work.explode("pmids").dropna(subset=["pmids"]).copy()
    work["pmid"] = work["pmids"].astype(str).str.extract(r"(\d+)")[0]
    work = work.dropna(subset=["pmid"])

    merge_cols = ["pmid", "pathogen", "antibiotic"]
    optional_cols = ["start_date", "end_date"]
    merge_cols.extend([col for col in optional_cols if col in filtered_df.columns and col in work.columns])

    positive_keys = (
        filtered_df.loc[filtered_df["decision"].astype(str).str.lower() == "yes", merge_cols]
        .astype({col: str for col in merge_cols})
        .drop_duplicates()
    )
    if positive_keys.empty:
        out = pd.DataFrame(
            columns=["pathogen", "antibiotic", "start_date", "end_date", "pmids", "count", "venue_histogram"]
        )
        if output_path:
            atomic_write_parquet(out, output_path, index=False)
        return out

    work = work.astype({col: str for col in merge_cols})
    positive_rows = work.merge(positive_keys, on=merge_cols, how="inner")
    positive_rows = positive_rows.drop_duplicates(subset=merge_cols)

    def _unique_pmids(series: pd.Series) -> list[str]:
        return list(dict.fromkeys(series.astype(str).tolist()))

    out = (
        positive_rows.groupby(["pathogen", "antibiotic", "start_date", "end_date"], as_index=False)
        .agg(
            pmids=("pmid", _unique_pmids),
            count=("pmid", "nunique"),
            venue_histogram=("venue_histogram", "first"),
        )
    )

    if output_path:
        atomic_write_parquet(out, output_path, index=False)
    return out


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "UNIFIED_AMR_INTEREST_SYSTEM_PROMPT",
    "LLM_CLASSIFICATION_MAX_TOKENS",
    "AMR_CATEGORY_COLUMNS",
    "PUBMED_EFETCH",
    "fetch_pubmed_abstract",
    "fetch_pubmed_abstracts_batch",
    "build_user_prompt",
    "classify_abstract",
    "prepare_pmids_to_review",
    "prepare_yearly_candidate_pmids",
    "run_unified_llm_filter",
    "build_yearly_screened_pairs",
    "run_llm_filter",
    "build_positive_dataset",
]
