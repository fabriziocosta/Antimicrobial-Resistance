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

UNIFIED_AMR_INTEREST_SYSTEM_PROMPT = """
You are screening PubMed abstracts for pathogen-antibiotic antimicrobial resistance evidence.

A study should be included only when BOTH are true:
1) The abstract explicitly reports that the specified pathogen shows resistance / reduced susceptibility / treatment failure due to resistance for the specified antibiotic.
2) The study is scientifically relevant for real-world AMR interest (clinical, surveillance, epidemiological, environmental, food, veterinary, or naturally occurring strain contexts), and not primarily a laboratory model-system experiment where AMR is incidental.

Decision labels:
- yes: both conditions are clearly satisfied.
- no: at least one condition is clearly not satisfied.
- uncertain: abstract missing, ambiguous, or insufficient to judge confidently.

Return compact JSON with keys: decision (yes/no/uncertain) and rationale (one short sentence).
""".strip()

PUBMED_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
RETRIES = 3
BACKOFF_FACTOR = 2
LLM_RETRIES = 3
LLM_BACKOFF_FACTOR = 2


def _log(log_path: str | None, message: str) -> None:
    """Write log to stdout and, if provided, append to a file."""
    import datetime as _dt
    stamped = f"[{_dt.datetime.now().isoformat(timespec='seconds')}] {message}"
    print(stamped)
    if log_path:
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(stamped + "\n")


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
            try:
                data = json.loads(raw)
            except Exception:
                # attempt to locate JSON substring if the model wrapped output
                start = raw.find("{")
                end = raw.rfind("}")
                if start != -1 and end != -1 and end > start:
                    fragment = raw[start : end + 1]
                    try:
                        data = json.loads(fragment)
                    except Exception:
                        data = {"decision": "uncertain", "rationale": raw.strip()}
                else:
                    data = {"decision": "uncertain", "rationale": raw.strip()}
            return {
                "pmid": pmid,
                "decision": data.get("decision", "uncertain"),
                "rationale": data.get("rationale", ""),
            }
        except Exception as exc:  # pragma: no cover - network/API handling
            last_error = exc
            if attempt == retries - 1:
                _log(log_path, f"LLM failed for PMID {pmid}: {exc}")
                break
            delay = backoff**attempt
            _log(log_path, f"LLM error for PMID {pmid}: {exc}. Retrying in {delay}s...")
            time.sleep(delay)
    return {"pmid": pmid, "decision": "uncertain", "rationale": f"LLM failure: {last_error}"}


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
) -> pd.DataFrame:
    """Run the unified AMR + scientific-interest LLM screen on yearly candidates."""
    required_cols = {"pmid", "pathogen", "antibiotic", "year"}
    missing = required_cols - set(candidate_df.columns)
    if missing:
        raise KeyError(f"candidate_df missing required columns: {sorted(missing)}")

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
    if resume_from is not None and not resume_from.empty:
        existing = resume_from.copy()
        if set(key_cols).issubset(existing.columns):
            for row in existing[key_cols].itertuples(index=False):
                seen_keys.add((str(row.pmid), str(row.pathogen), str(row.antibiotic), int(row.year)))

    abstract_cache: dict[str, tuple[str, str]] = {}
    cache_dirty = False
    if abstract_cache_path and os.path.exists(abstract_cache_path):
        try:
            cache_df = pd.read_parquet(abstract_cache_path)
            if {"pmid", "title", "abstract"}.issubset(cache_df.columns):
                for row in cache_df[["pmid", "title", "abstract"]].itertuples(index=False):
                    abstract_cache[str(row.pmid)] = (row.title or "", row.abstract or "")
        except Exception as exc:
            _append_log(f"Failed loading abstract cache from {abstract_cache_path}: {exc}")

    results: list[dict] = []
    total = len(candidate_df)
    llm_calls_used = 0
    session = requests.Session()
    for idx, row in enumerate(candidate_df.itertuples(index=False), start=1):
        key = (str(row.pmid), str(row.pathogen), str(row.antibiotic), int(row.year))
        if key in seen_keys:
            continue
        pmid = str(row.pmid)
        pathogen = str(row.pathogen)
        antibiotic = str(row.antibiotic)
        year = int(row.year)

        if pmid in abstract_cache:
            title, abstract = abstract_cache[pmid]
        else:
            title, abstract, status = fetch_pubmed_abstract(pmid, email=email, session=session)
            if status == "network_error":
                _append_log(f"Skipping PMID {pmid} due to network error.")
                time.sleep(rate_limit)
                continue
            abstract_cache[pmid] = (title, abstract)
            cache_dirty = True
            time.sleep(rate_limit)

        if not abstract:
            decision = "uncertain"
            rationale = "no abstract"
            raw_decision = "no_abstract"
        elif dry_run:
            decision = "uncertain"
            rationale = "dry_run"
            raw_decision = "dry_run"
        else:
            if max_llm_calls is not None and llm_calls_used >= max_llm_calls:
                break
            verdict = classify_abstract(
                llm,
                pmid,
                pathogen,
                antibiotic,
                title,
                abstract,
                system_prompt=system_prompt,
                max_tokens=256,
                log_path=log_path,
            )
            raw_decision = str(verdict.get("decision", "uncertain"))
            decision = _normalize_unified_decision(raw_decision)
            rationale = str(verdict.get("rationale", "") or "").strip()
            llm_calls_used += 1
            time.sleep(rate_limit)

        results.append(
            {
                "pmid": pmid,
                "pathogen": pathogen,
                "antibiotic": antibiotic,
                "year": year,
                "unified_decision": decision,
                "unified_raw_decision": raw_decision,
                "unified_rationale": rationale,
                "title": title or "",
                "abstract": abstract or "",
            }
        )
        seen_keys.add(key)

        if output_path and save_every and len(results) % save_every == 0:
            tmp_df = pd.concat([existing, pd.DataFrame(results)], ignore_index=True)
            tmp_df = tmp_df.drop_duplicates(subset=key_cols, keep="last")
            tmp_df.to_parquet(output_path, index=False)

        if (
            abstract_cache_path
            and abstract_cache_save_every
            and cache_dirty
            and len(results) % int(abstract_cache_save_every) == 0
        ):
            cache_df = pd.DataFrame(
                [{"pmid": p, "title": t, "abstract": a} for p, (t, a) in abstract_cache.items()]
            )
            cache_df.to_parquet(abstract_cache_path, index=False)
            cache_dirty = False

        if verbose >= 1 and progress_every and len(results) % progress_every == 0:
            batch_df = pd.DataFrame(results)
            yes_n = int((batch_df["unified_decision"] == "yes").sum())
            no_n = int((batch_df["unified_decision"] == "no").sum())
            unc_n = int((batch_df["unified_decision"] == "uncertain").sum())
            _progress(
                f"Processed {len(results):>6}/{total:<6} yes:{yes_n:>6} no:{no_n:>6} uncertain:{unc_n:>6}"
            )

    out_df = pd.concat([existing, pd.DataFrame(results)], ignore_index=True)
    if not out_df.empty:
        out_df = out_df.drop_duplicates(subset=key_cols, keep="last").reset_index(drop=True)
    if output_path:
        out_df.to_parquet(output_path, index=False)
    if abstract_cache_path and abstract_cache:
        cache_df = pd.DataFrame(
            [{"pmid": p, "title": t, "abstract": a} for p, (t, a) in abstract_cache.items()]
        )
        cache_df.to_parquet(abstract_cache_path, index=False)
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
            df_tmp.to_parquet(output_path, index=False)
            _log_verbose(f"Checkpoint saved {len(df_tmp)} rows to {output_path}")
        # Save abstract cache periodically, independent of decision checkpoints
        if abstract_cache_path and cache_dirty and abstract_cache_save_every and (len(results) % max(1, abstract_cache_save_every) == 0):
            try:
                df_cache = pd.DataFrame(
                    [{"pmid": k, "title": v[0], "abstract": v[1]} for k, v in abstract_cache.items()]
                )
                # Keep last occurrence per PMID (dict already unique keys)
                df_cache.to_parquet(abstract_cache_path, index=False)
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
        df_all.to_parquet(output_path, index=False)
        df_out = df_all  # return deduped full set for convenience
    # Final save of abstract cache at end of run
    if abstract_cache_path and abstract_cache:
        try:
            df_cache = pd.DataFrame(
                [{"pmid": k, "title": v[0], "abstract": v[1]} for k, v in abstract_cache.items()]
            )
            df_cache.to_parquet(abstract_cache_path, index=False)
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
            out.to_parquet(output_path, index=False)
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
    if output_path:
        out.to_parquet(output_path, index=False)
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
            out.to_parquet(output_path, index=False)
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
        out.to_parquet(output_path, index=False)
    return out


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "UNIFIED_AMR_INTEREST_SYSTEM_PROMPT",
    "PUBMED_EFETCH",
    "fetch_pubmed_abstract",
    "build_user_prompt",
    "classify_abstract",
    "prepare_pmids_to_review",
    "prepare_yearly_candidate_pmids",
    "run_unified_llm_filter",
    "build_yearly_screened_pairs",
    "run_llm_filter",
    "build_positive_dataset",
]
