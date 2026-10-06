"""System 1 (Jev) AMR screening through OpenRouter's Decisions API."""

from __future__ import annotations

import time
import uuid
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, local
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
OPENROUTER_CREDITS_URL = "https://openrouter.ai/api/v1/credits"
OPENROUTER_KEY_URL = "https://openrouter.ai/api/v1/key"
JEV_DEFAULT_MODEL = "typesafe/jev-1.13"
SYSTEM1_QUESTION_VERSION = "amr-pair-support-noul-v1"
SYSTEM1_AMR_SUPPORT_QUESTION = (
    "Is the following statement true based only on the provided title and abstract? "
    "The paper provides substantive antimicrobial-resistance evidence or context for the "
    "specified target pathogen and target antibiotic (or an explicitly applicable antibiotic class). "
    "Count resistance, susceptibility testing or patterns, AMR mechanisms, treatment, surveillance, "
    "stewardship, or policy as relevant. Do not infer support from unrelated co-mentions. "
    "The target pathogen and antibiotic must be linked by the paper's evidence or study context."
)


def _nullable_text(value: object) -> str:
    """Return empty text for null cache fields, including pandas.NA."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    return f"{minutes}m {seconds:02d}s"


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
            raise ValueError(
                "OpenRouter credentials were not found. Set OPENROUTER_API_KEY or store a "
                "generic-password item in macOS Keychain (service: OpenRouter by default)."
            )
        self.model = model
        self.timeout = timeout
        self.retries = max(1, int(retries))
        self.backoff = max(0.0, float(backoff))
        self.session = session or requests.Session()

    def credit_status(self) -> tuple[str, float | None]:
        """Return a printable OpenRouter balance/limit, without interrupting scoring.

        The account credits endpoint requires a management key. If this key cannot
        access it, the key-status endpoint may still expose its remaining limit.
        """
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            response = self.session.get(OPENROUTER_CREDITS_URL, headers=headers, timeout=5)
            if response.ok:
                data = response.json().get("data", {})
                total = data.get("total_credits")
                used = data.get("total_usage")
                if isinstance(total, (int, float)) and isinstance(used, (int, float)):
                    remaining = max(0.0, float(total - used))
                    return f"account credits remaining ${remaining:.4f}", remaining
            credits_status = response.status_code
        except Exception:
            credits_status = None

        try:
            response = self.session.get(OPENROUTER_KEY_URL, headers=headers, timeout=5)
            if response.ok:
                data = response.json().get("data", {})
                remaining = data.get("limit_remaining")
                if isinstance(remaining, (int, float)):
                    remaining = max(0.0, float(remaining))
                    return f"API key limit remaining ${remaining:.4f}", remaining
            key_status = response.status_code
        except Exception:
            key_status = None

        if credits_status == 403:
            return "remaining credit unavailable (account credits require a management key)", None
        if key_status is not None:
            return f"remaining credit unavailable (OpenRouter HTTP {key_status})", None
        return "remaining credit unavailable (OpenRouter could not be reached)", None

    def for_worker(self) -> "OpenRouterJevClient":
        """Create a worker client with its own HTTP session for concurrent calls."""
        return OpenRouterJevClient(
            api_key=self.api_key,
            model=self.model,
            timeout=self.timeout,
            retries=self.retries,
            backoff=self.backoff,
        )

    def decide(self, state: str, *, question: str) -> dict[str, Any]:
        return self.decide_many(state, questions={"amr_pair_supported": question})

    def decide_many(self, state: str, *, questions: dict[str, str]) -> dict[str, Any]:
        """Ask multiple typed yes/no/uncertain questions about one state."""
        if not questions:
            raise ValueError("At least one question is required.")
        payload = {
            "model": self.model,
            "state": state,
            "questions": {
                question_id: {
                    "type": "noul",
                    "instructions": instructions,
                }
                for question_id, instructions in questions.items()
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


def build_system1_state(
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


def _extract_probability(response: dict[str, Any], answer_id: str = "amr_pair_supported") -> float:
    """Validate the probability from the single configured Jev noul answer."""
    answers = response.get("answers")
    answer = answers.get(answer_id) if isinstance(answers, dict) else None
    if not isinstance(answer, dict) or answer.get("type") != "noul":
        raise ValueError("OpenRouter response is missing the AMR noul answer.")
    # The Decisions API returns a noul answer as {"type": "noul", "noul": p_true}.
    probability = answer.get("noul")
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
        str(row.pmid): (_nullable_text(row.title), _nullable_text(row.abstract))
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
    max_workers: int = 1,
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
    if int(max_workers) < 1:
        raise ValueError("max_workers must be at least 1.")
    max_workers = int(max_workers)

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
    candidate_keys = {
        (str(row.pmid), str(row.pathogen), str(row.antibiotic), int(row.year))
        for row in candidates.itertuples(index=False)
    }
    already_done_keys = done_keys & candidate_keys

    prior_status_by_key: dict[tuple[str, str, str, int], str] = {}
    if not existing.empty and {"system1_run_id", "system1_status", *key_cols[:-1]}.issubset(existing.columns):
        current_run = existing.loc[existing["system1_run_id"].astype(str) == run_id]
        if not current_run.empty:
            current_run = current_run.drop_duplicates(subset=key_cols, keep="last")
            for row in current_run.itertuples(index=False):
                key = (str(row.pmid), str(row.pathogen), str(row.antibiotic), int(row.year))
                if key in candidate_keys:
                    prior_status_by_key[key] = str(row.system1_status)

    pending = []
    for row in candidates.itertuples(index=False):
        key = (str(row.pmid), str(row.pathogen), str(row.antibiotic), int(row.year))
        if key not in done_keys:
            pending.append(row)
    pending_before_cap = len(pending)
    capped_rows = 0
    if max_calls is not None:
        pending = pending[: max(0, int(max_calls))]
        capped_rows = pending_before_cap - len(pending)

    print(
        f"System 1 run {run_id}: {len(candidates):,} candidate rows; "
        f"resuming {len(already_done_keys):,} completed; "
        f"{len(pending):,} pending this session; up to {max_workers} concurrent requests."
    )
    previous_failures = sum(
        prior_status_by_key.get(key) in {"error", "abstract_error"}
        for key in candidate_keys - already_done_keys
    )
    if previous_failures:
        print(f"Retrying {previous_failures:,} previously failed rows.")
    if capped_rows:
        print(f"max_calls deferred {capped_rows:,} pending rows for a later run.")

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

    rows: list[dict[str, Any]] = []
    attempted = 0
    session_processed = 0
    session_successes = 0
    session_request_failures = 0
    session_abstract_failures = 0
    session_no_abstract = 0
    progressed_keys = set(already_done_keys)
    status_by_key = dict(prior_status_by_key)
    run_started_at = time.monotonic()
    credit_status = getattr(client, "credit_status", None)
    if callable(credit_status):
        latest_credit_status, latest_credit_amount = credit_status()
    else:
        latest_credit_status, latest_credit_amount = "balance lookup not supported by client", None
    starting_credit_amount = latest_credit_amount
    last_credit_amount = latest_credit_amount
    observed_cost_per_call: float | None = None

    def _refresh_credit_info() -> None:
        nonlocal latest_credit_status, latest_credit_amount, last_credit_amount
        nonlocal observed_cost_per_call
        if not callable(credit_status):
            return
        latest_credit_status, latest_credit_amount = credit_status()
        if (
            latest_credit_amount is not None
            and last_credit_amount is not None
            and starting_credit_amount is not None
            and latest_credit_amount < last_credit_amount
            and attempted > 0
        ):
            # OpenRouter's balance can remain unchanged across several polls, then
            # catch up in a lump. Update the cost rate only when the reported
            # balance moves so a stale balance is never mistaken for zero spend.
            total_observed_spend = max(0.0, starting_credit_amount - latest_credit_amount)
            if total_observed_spend > 0:
                observed_cost_per_call = total_observed_spend / attempted
        if latest_credit_amount is not None:
            last_credit_amount = latest_credit_amount

    def _progress_line() -> str:
        counts = Counter(status_by_key.get(key, "pending") for key in progressed_keys)
        progressed = len(progressed_keys)
        total = len(candidates)
        elapsed = time.monotonic() - run_started_at
        rate = session_processed / elapsed if elapsed > 0 else 0.0
        remaining = max(0, len(pending) - session_processed)
        eta = remaining / rate if rate > 0 else 0.0
        percent = (100.0 * progressed / total) if total else 100.0
        unresolved = max(0, total - counts["ok"] - counts["no_abstract"])
        estimated_end_amount = None
        if latest_credit_amount is not None and observed_cost_per_call is not None:
            if unresolved == 0:
                estimated_end_amount = latest_credit_amount
            else:
                estimated_end_amount = max(
                    0.0, latest_credit_amount - observed_cost_per_call * unresolved
                )
        estimate_text = (
            f"estimated balance after {unresolved:,} unresolved candidates "
            f"${estimated_end_amount:.4f}"
            if estimated_end_amount is not None
            else "end balance estimate waiting for OpenRouter to report spend"
        )
        return (
            f"{progressed:,}/{total:,} candidates ({percent:.1f}%) "
            f"| this session {session_processed:,}/{len(pending):,} "
            f"| successes {counts['ok']:,} | request/parse failures {counts['error']:,} "
            f"| abstract failures {counts['abstract_error']:,} | no abstract {counts['no_abstract']:,} "
            f"| API attempts this session {attempted:,} "
            f"({session_successes:,} succeeded, {session_request_failures:,} failed) "
            f"| elapsed {_format_duration(elapsed)} | ETA this session {_format_duration(eta)} "
            f"| {latest_credit_status} | {estimate_text}"
        )

    worker_local = local()
    rate_limit_lock = Lock()
    next_request_at = 0.0

    def _worker_client() -> DecisionClient:
        if not hasattr(worker_local, "client"):
            make_worker = getattr(client, "for_worker", None)
            worker_local.client = make_worker() if callable(make_worker) else client
        return worker_local.client

    def _score_candidate(row: Any) -> tuple[dict[str, Any], bool]:
        nonlocal next_request_at
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
                if rate_limit:
                    with rate_limit_lock:
                        now_monotonic = time.monotonic()
                        scheduled_at = max(now_monotonic, next_request_at)
                        next_request_at = scheduled_at + rate_limit
                    if scheduled_at > now_monotonic:
                        time.sleep(scheduled_at - now_monotonic)
                response = _worker_client().decide(
                    build_system1_state(pmid, pathogen, antibiotic, title, abstract),
                    question=SYSTEM1_AMR_SUPPORT_QUESTION,
                )
                result["system1_p_yes"] = _extract_probability(response)
                result["system1_status"] = "ok"
                result["system1_served_model"] = str(response.get("model") or model)
                result["system1_request_id"] = str(response.get("id") or "")
                result["system1_provider"] = str(response.get("provider") or "")
                result["system1_error"] = ""
            except Exception as exc:
                result["system1_error"] = f"{type(exc).__name__}: {exc}"
        return result, result["system1_status"] in {"ok", "error"} and bool(abstract.strip())

    def _result_stream():
        if max_workers == 1:
            for pending_row in pending:
                yield _score_candidate(pending_row)
            return
        row_iter = iter(pending)
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="system1") as executor:
            futures: set[Future] = set()
            max_in_flight = max_workers * 2
            while True:
                while len(futures) < max_in_flight:
                    try:
                        pending_row = next(row_iter)
                    except StopIteration:
                        break
                    futures.add(executor.submit(_score_candidate, pending_row))
                if not futures:
                    break
                completed, futures = wait(futures, return_when=FIRST_COMPLETED)
                for future in completed:
                    yield future.result()

    for result, was_api_attempt in _result_stream():
        attempted += int(was_api_attempt)
        if result["system1_status"] == "ok":
            session_successes += 1
        elif was_api_attempt:
            session_request_failures += 1
        rows.append(result)
        pmid = str(result["pmid"])
        pathogen = str(result["pathogen"])
        antibiotic = str(result["antibiotic"])
        year = int(result["year"])
        key = (pmid, pathogen, antibiotic, year)
        session_processed += 1
        progressed_keys.add(key)
        status_by_key[key] = str(result["system1_status"])
        if result["system1_status"] == "abstract_error":
            session_abstract_failures += 1
        elif result["system1_status"] == "no_abstract":
            session_no_abstract += 1
        if output_path and len(rows) >= max(1, int(save_every)):
            _merge_checkpoint_rows(output_path, pd.DataFrame(rows), key_cols=key_cols, fallback=existing)
            rows = []
        if progress_every > 0 and session_processed % progress_every == 0:
            # Make every reported progress point durable so a restart resumes from it.
            if output_path and rows:
                _merge_checkpoint_rows(output_path, pd.DataFrame(rows), key_cols=key_cols, fallback=existing)
                rows = []
            _refresh_credit_info()
            print(_progress_line())
    if output_path and rows:
        _merge_checkpoint_rows(output_path, pd.DataFrame(rows), key_cols=key_cols, fallback=existing)
    _refresh_credit_info()
    print(f"System 1 final: {_progress_line()}")
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
    "OPENROUTER_CREDITS_URL",
    "OPENROUTER_KEY_URL",
    "SYSTEM1_AMR_SUPPORT_QUESTION",
    "SYSTEM1_QUESTION_VERSION",
    "OpenRouterJevClient",
    "build_system1_state",
    "run_system1_filter",
]
