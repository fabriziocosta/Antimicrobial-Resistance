"""System 1 (Jev) AMR screening through OpenRouter's Decisions API."""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
import requests

from amr.config import OPENROUTER_API_KEY
from amr.io import atomic_write_parquet
from amr.screening.core import (
    _merge_checkpoint_rows,
    fetch_pubmed_abstracts_batch,
)

OPENROUTER_DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
JEV_DEFAULT_MODEL = "typesafe/jev-1.13"
SYSTEM1_QUESTION_VERSION = "amr-pair-support-noul-v1"


class DecisionClient(Protocol):
    def decide(self, state: str, *, question: str) -> dict[str, Any]: ...


class OpenRouterJevClient:
    """Small client for Jev's typed Decisions API (not chat completions)."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = JEV_DEFAULT_MODEL,
        timeout: float = 90.0,
        retries: int = 3,
        backoff: float = 2.0,
        session: requests.Session | None = None,
    ) -> None:
        self.api_key = api_key or OPENROUTER_API_KEY
        if not self.api_key:
            raise ValueError("Set OPENROUTER_API_KEY before running Jev predictions.")
        self.model = model
        self.timeout = timeout
        self.retries = max(1, int(retries))
        self.backoff = max(0.0, float(backoff))
        self.session = session or requests.Session()

    def decide(self, state: str, *, question: str) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "state": state,
            "questions": {
                "amr_pair_supported": {
                    "type": "noul",
                    "instructions": question,
                }
            },
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                response = self.session.post(
                    OPENROUTER_DECISIONS_URL,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict):
                    raise ValueError("OpenRouter response must be a JSON object.")
                return data
            except Exception as exc:
                last_error = exc
                if attempt + 1 < self.retries:
                    time.sleep(self.backoff**attempt)
        raise RuntimeError(f"OpenRouter Jev request failed: {last_error}") from last_error


def _build_state(
    pmid: str,
    pathogen: str,
    antibiotic: str,
    title: str,
    abstract: str,
) -> str:
    return (
        f"Target pathogen: {pathogen}\n"
        f"Target antibiotic: {antibiotic}\n"
        f"PMID: {pmid}\n"
        f"Title: {title}\n"
        f"Abstract: {abstract}"
    )


def _extract_probability(response: dict[str, Any]) -> float:
    """Validate the probability from the single configured Jev noul answer."""
    answers = response.get("answers")
    answer = answers.get("amr_pair_supported") if isinstance(answers, dict) else None
    if not isinstance(answer, dict) or answer.get("type") != "noul":
        raise ValueError("OpenRouter response is missing the AMR noul answer.")
    probability = answer.get("probability")
    if isinstance(probability, bool) or not isinstance(probability, (int, float)):
        raise ValueError("Jev returned a non-numeric AMR probability.")
    value = float(probability)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"Jev probability must be between 0 and 1; got {value}.")
    return value


def _load_abstract_cache(path: str | Path | None) -> dict[str, tuple[str, str]]:
    if not path or not Path(path).exists():
        return {}
    try:
        cache_df = pd.read_parquet(path)
    except Exception:
        return {}
    if not {"pmid", "title", "abstract"}.issubset(cache_df.columns):
        return {}
    return {
        str(row.pmid): (str(row.title or ""), str(row.abstract or ""))
        for row in cache_df[["pmid", "title", "abstract"]].itertuples(index=False)
    }


def run_system1_filter(
    candidate_df: pd.DataFrame,
    client: DecisionClient,
    *,
    model: str = JEV_DEFAULT_MODEL,
    run_id: str | None = None,
    output_path: str | Path | None = None,
    resume_from: pd.DataFrame | None = None,
    abstract_cache_path: str | Path | None = "pmid_abstracts.parquet",
    email: str | None = None,
    save_every: int = 100,
    progress_every: int = 100,
    rate_limit: float = 0.0,
    max_calls: int | None = None,
    question_version: str = SYSTEM1_QUESTION_VERSION,
) -> pd.DataFrame:
    """Score candidate PMID/pathogen/antibiotic/year rows and checkpoint each run.

    Successful results are reused on resume. Failed API rows are retried. A run ID
    scopes resumability; use a new ID to preserve a fresh prediction snapshot.
    """
    required = {"pmid", "pathogen", "antibiotic", "year"}
    missing = required - set(candidate_df.columns)
    if missing:
        raise KeyError(f"candidate_df missing required columns: {sorted(missing)}")
    if not 0 <= float(rate_limit):
        raise ValueError("rate_limit must be non-negative.")

    run_id = str(run_id or f"jev-{uuid.uuid4().hex[:12]}")
    key_cols = ["pmid", "pathogen", "antibiotic", "year", "system1_run_id"]
    existing = resume_from.copy() if resume_from is not None else pd.DataFrame()
    if output_path and Path(output_path).exists():
        try:
            existing = pd.read_parquet(output_path)
        except Exception:
            pass
    if not existing.empty:
        missing_meta = set(key_cols) - set(existing.columns)
        if missing_meta:
            existing = pd.DataFrame()
        else:
            current_run = existing.loc[existing["system1_run_id"].astype(str) == run_id]
            if not current_run.empty:
                models = set(current_run["system1_model"].dropna().astype(str))
                versions = set(current_run["system1_question_version"].dropna().astype(str))
                if models and models != {str(model)}:
                    raise ValueError(
                        f"System 1 run_id {run_id!r} already belongs to model(s) {sorted(models)}; "
                        "choose a new run_id for a new model run."
                    )
                if versions and versions != {str(question_version)}:
                    raise ValueError(
                        f"System 1 run_id {run_id!r} already belongs to question version(s) "
                        f"{sorted(versions)}; choose a new run_id for a new question version."
                    )
            completed = current_run.loc[
                current_run["system1_status"].astype(str).isin({"ok", "no_abstract"})
            ]
            done_keys = {
                (str(r.pmid), str(r.pathogen), str(r.antibiotic), int(r.year))
                for r in completed[key_cols[:-1]].itertuples(index=False)
            }
    if existing.empty:
        done_keys = set()

    candidates = candidate_df.copy()
    candidates["pmid"] = candidates["pmid"].astype(str)
    candidates["year"] = pd.to_numeric(candidates["year"], errors="coerce")
    candidates = candidates.dropna(subset=["year"])
    candidates["year"] = candidates["year"].astype(int)
    candidates = candidates.drop_duplicates(subset=["pmid", "pathogen", "antibiotic", "year"])
    pending = []
    for row in candidates.itertuples(index=False):
        key = (str(row.pmid), str(row.pathogen), str(row.antibiotic), int(row.year))
        if key not in done_keys:
            pending.append(row)
    if max_calls is not None:
        pending = pending[: max(0, int(max_calls))]

    abstract_cache = _load_abstract_cache(abstract_cache_path)
    missing_pmids = sorted({str(row.pmid) for row in pending if str(row.pmid) not in abstract_cache})
    abstract_updates: dict[str, tuple[str, str]] = {}
    if missing_pmids:
        fetched = fetch_pubmed_abstracts_batch(missing_pmids, email=email)
        for pmid, (title, abstract, status) in fetched.items():
            if status != "network_error":
                abstract_cache[str(pmid)] = (title or "", abstract or "")
                abstract_updates[str(pmid)] = (title or "", abstract or "")
        if abstract_cache_path and abstract_updates:
            cache_df = pd.DataFrame(
                [{"pmid": pmid, "title": title, "abstract": abstract} for pmid, (title, abstract) in abstract_cache.items()]
            )
            atomic_write_parquet(cache_df, abstract_cache_path, index=False)

    question = (
        "Is the following statement true based only on the provided title and abstract? "
        "The paper provides substantive antimicrobial-resistance evidence or context for the "
        "specified target pathogen and target antibiotic (or an explicitly applicable antibiotic class). "
        "Count resistance, susceptibility testing or patterns, AMR mechanisms, treatment, surveillance, "
        "stewardship, or policy as relevant. Do not infer support from unrelated co-mentions. "
        "The target pathogen and antibiotic must be linked by the paper's evidence or study context."
    )
    rows: list[dict[str, Any]] = []
    attempted = 0
    for row in pending:
        pmid = str(row.pmid)
        pathogen = str(row.pathogen)
        antibiotic = str(row.antibiotic)
        year = int(row.year)
        title, abstract = abstract_cache.get(pmid, ("", ""))
        now = datetime.now(timezone.utc).isoformat()
        result: dict[str, Any] = {
            "pmid": pmid,
            "pathogen": pathogen,
            "antibiotic": antibiotic,
            "year": year,
            "system1_run_id": run_id,
            "system1_model": model,
            "system1_question_version": question_version,
            "system1_p_yes": pd.NA,
            "system1_status": "error",
            "system1_error": "",
            "system1_served_model": "",
            "system1_request_id": "",
            "system1_provider": "",
            "system1_updated_at": now,
        }
        if pmid not in abstract_cache:
            result["system1_status"] = "abstract_error"
            result["system1_error"] = "PubMed abstract fetch failed."
        elif not abstract.strip():
            result["system1_status"] = "no_abstract"
        else:
            try:
                response = client.decide(
                    _build_state(pmid, pathogen, antibiotic, title, abstract),
                    question=question,
                )
                result["system1_p_yes"] = _extract_probability(response)
                result["system1_status"] = "ok"
                result["system1_served_model"] = str(response.get("model") or model)
                result["system1_request_id"] = str(response.get("id") or "")
                result["system1_provider"] = str(response.get("provider") or "")
                result["system1_error"] = ""
                attempted += 1
            except Exception as exc:
                result["system1_error"] = f"{type(exc).__name__}: {exc}"
                attempted += 1
        rows.append(result)
        if output_path and len(rows) % max(1, int(save_every)) == 0:
            _merge_checkpoint_rows(output_path, pd.DataFrame(rows), key_cols=key_cols, fallback=existing)
            rows = []
        if progress_every > 0 and (attempted > 0 and attempted % progress_every == 0):
            print(f"System 1 predictions attempted: {attempted:,}")
        if rate_limit:
            time.sleep(rate_limit)
    if output_path and rows:
        _merge_checkpoint_rows(output_path, pd.DataFrame(rows), key_cols=key_cols, fallback=existing)
    if output_path and not pending and not Path(output_path).exists():
        columns = key_cols + [
            "system1_model", "system1_question_version", "system1_p_yes", "system1_status",
            "system1_error", "system1_served_model", "system1_request_id", "system1_provider",
            "system1_updated_at",
        ]
        atomic_write_parquet(pd.DataFrame(columns=columns), output_path, index=False)

    if output_path and Path(output_path).exists():
        stored = pd.read_parquet(output_path)
        if "system1_run_id" in stored.columns:
            return stored.loc[stored["system1_run_id"].astype(str) == run_id].reset_index(drop=True)
    if rows:
        return pd.DataFrame(rows)
    if not existing.empty and "system1_run_id" in existing.columns:
        return existing.loc[existing["system1_run_id"].astype(str) == run_id].reset_index(drop=True)
    return pd.DataFrame()


__all__ = [
    "JEV_DEFAULT_MODEL",
    "OPENROUTER_DECISIONS_URL",
    "SYSTEM1_QUESTION_VERSION",
    "OpenRouterJevClient",
    "run_system1_filter",
]
