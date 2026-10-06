"""Concept discovery data and forecasting helpers for scientific interest."""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from amr.io import atomic_write_parquet


SEED_CONCEPTS: list[dict[str, Any]] = [
    {"concept_id": "surveillance", "name": "Surveillance", "parent_id": None,
     "definition": "Systematic monitoring or measurement of antimicrobial resistance burden, prevalence, incidence, or trends.",
     "inclusion_criteria": ["resistance-rate monitoring", "antibiograms", "population-level susceptibility trends"],
     "exclusion_criteria": ["single-patient case reports without monitoring"], "version": 1},
    {"concept_id": "outbreak_evidence", "name": "Outbreak evidence", "parent_id": None,
     "definition": "Evidence describing a cluster, outbreak, or spread event involving antimicrobial-resistant organisms.",
     "inclusion_criteria": ["outbreak investigations", "transmission clusters"], "exclusion_criteria": ["routine surveillance without a cluster"], "version": 1},
    {"concept_id": "treatment_failure", "name": "Treatment failure", "parent_id": None,
     "definition": "Clinical evidence that infection treatment is ineffective, complicated, or fails in a way connected to resistance.",
     "inclusion_criteria": ["failure or relapse during antimicrobial therapy", "resistance-associated poor outcomes"], "exclusion_criteria": ["in vitro resistance without treatment outcomes"], "version": 1},
    {"concept_id": "mechanism_discovery", "name": "Mechanism discovery", "parent_id": None,
     "definition": "Evidence identifying or characterising a biological mechanism that causes or contributes to antimicrobial resistance.",
     "inclusion_criteria": ["resistance mechanisms", "gene transfer", "mutations or expression mechanisms"], "exclusion_criteria": ["resistance prevalence without mechanistic analysis"], "version": 1},
    {"concept_id": "genomic_evidence", "name": "Genomic evidence", "parent_id": None,
     "definition": "Use of genome sequencing or genomic analysis to detect, track, or explain antimicrobial resistance.",
     "inclusion_criteria": ["whole-genome sequencing", "genomic epidemiology", "resistome analysis"], "exclusion_criteria": ["non-genomic molecular tests alone"], "version": 1},
    {"concept_id": "diagnostics", "name": "Diagnostics", "parent_id": None,
     "definition": "Development or evaluation of methods that detect resistance or guide susceptibility testing.",
     "inclusion_criteria": ["diagnostic accuracy", "rapid susceptibility testing", "resistance detection assays"], "exclusion_criteria": ["routine susceptibility results without diagnostic evaluation"], "version": 1},
    {"concept_id": "hospital_transmission", "name": "Hospital transmission", "parent_id": None,
     "definition": "Evidence about transmission or spread of resistant organisms in healthcare settings.",
     "inclusion_criteria": ["nosocomial transmission", "hospital contact or infection-control spread"], "exclusion_criteria": ["hospital prevalence without transmission evidence"], "version": 1},
    {"concept_id": "agricultural_use", "name": "Agricultural use", "parent_id": None,
     "definition": "Antimicrobial use or resistance in livestock, aquaculture, food production, or agricultural environments.",
     "inclusion_criteria": ["veterinary antimicrobial use", "farm or food-chain resistance"], "exclusion_criteria": ["human clinical use without an agricultural link"], "version": 1},
    {"concept_id": "policy_guideline_change", "name": "Policy or guideline change", "parent_id": None,
     "definition": "Changes to formal policy, clinical guidelines, stewardship recommendations, or prescribing rules related to resistance.",
     "inclusion_criteria": ["guideline revisions", "policy actions", "stewardship recommendations"], "exclusion_criteria": ["discussion of policy without a change or recommendation"], "version": 1},
]

CONCEPT_SYSTEM_PROMPT = (
    "You classify scientific abstracts for a reproducible antimicrobial-resistance study. "
    "Use only evidence in the supplied title and abstract. Return valid JSON. "
    "Do not infer a concept from the pathogen, antibiotic, year, or any outcome."
)
CLASSIFICATION_PROMPT_VERSION = hashlib.sha256(
    (CONCEPT_SYSTEM_PROMPT + "|score-json-v1|0=absent;1=explicit;fraction=ambiguous").encode("utf-8")
).hexdigest()[:16]
JEV_CLASSIFICATION_PROMPT_VERSION = hashlib.sha256(
    b"jev-noul-concept-presence-title-abstract-v1"
).hexdigest()[:16]


def _normalise_pmid(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text or None


def concept_catalog_hash(catalog: Sequence[Mapping[str, Any]]) -> str:
    """Return a stable short hash for a concept catalog and its definitions."""
    payload = json.dumps(list(catalog), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def concept_definition_hash(concept: Mapping[str, Any]) -> str:
    """Hash one concept independently so catalog additions do not invalidate it."""
    payload = json.dumps(dict(concept), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def pareto_concept_frontier(
    population_summary: pd.DataFrame,
    *,
    horizon: int = 0,
    excluded_concepts: Iterable[str] = (),
) -> pd.DataFrame:
    """Return non-dominated concepts on generality and specialist strength.

    Larger average gain (generality) and larger 90th-percentile gain
    (specialist strength) are both treated as better. ``horizon=0`` selects
    the equal-weight aggregate across the evaluated forecast horizons.
    Excluded concepts remain on the returned frontier, with their expansion
    eligibility marked false; they can still dominate other concepts.
    """
    required = {"model", "horizon", "generality", "specialist_strength"}
    if required - set(population_summary.columns):
        raise KeyError(f"population_summary is missing columns: {sorted(required - set(population_summary.columns))}")
    work = population_summary.loc[
        pd.to_numeric(population_summary["horizon"], errors="coerce").eq(int(horizon))
        & population_summary["model"].astype(str).str.startswith("concept:")
    ].copy()
    work["concept_id"] = work["model"].astype(str).str.removeprefix("concept:")
    excluded = {str(value) for value in excluded_concepts}
    for column in ("generality", "specialist_strength"):
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna(subset=["generality", "specialist_strength"]).drop_duplicates("concept_id")
    if work.empty:
        return work.assign(on_pareto_front=pd.Series(dtype=bool), eligible_for_expansion=pd.Series(dtype=bool))
    points = work[["generality", "specialist_strength"]].to_numpy(dtype=float)
    pareto = []
    for index, point in enumerate(points):
        dominates = (
            (points[:, 0] >= point[0]) & (points[:, 1] >= point[1])
            & ((points[:, 0] > point[0]) | (points[:, 1] > point[1]))
        )
        pareto.append(not bool(dominates.any()))
    work["on_pareto_front"] = pareto
    work["eligible_for_expansion"] = ~work["concept_id"].isin(excluded)
    return work.sort_values(["on_pareto_front", "generality", "specialist_strength"],
                            ascending=[False, False, False]).reset_index(drop=True)


def make_pair_split(
    pairs: Iterable[tuple[object, object]], *, heldout_fraction: float = 0.2, seed: int = 42
) -> pd.DataFrame:
    """Create a deterministic pair-level development/confirmation split."""
    if not 0 < heldout_fraction < 1:
        raise ValueError("heldout_fraction must be between 0 and 1.")
    unique = sorted({(str(pathogen), str(antibiotic)) for pathogen, antibiotic in pairs})
    if len(unique) < 2:
        raise ValueError("At least two unique pathogen-antibiotic pairs are required to split.")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique))
    heldout_n = min(len(unique) - 1, max(1, int(round(len(unique) * heldout_fraction))))
    heldout = set(order[:heldout_n].tolist())
    return pd.DataFrame([
        {"pathogen": pair[0], "antibiotic": pair[1],
         "split": "confirmation" if index in heldout else "development"}
        for index, pair in enumerate(unique)
    ])


def build_concept_corpus(
    screened_df: pd.DataFrame,
    abstract_df: pd.DataFrame,
    si_df: pd.DataFrame,
    *,
    heldout_fraction: float = 0.2,
    split_seed: int = 42,
    pathogen_priority_df: pd.DataFrame | None = None,
    num_pathogens: int | None = None,
    min_pathogenic_index: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join screened pair-year PMID membership to abstracts and observed SI.

    The returned corpus has one row per pair-year-PMID association. Articles are
    classified once by PMID; a PMID associated with several pairs contributes to
    each pair's yearly prevalence signal.
    """
    required_screen = {"pathogen", "antibiotic", "year", "pmids"}
    required_abstract = {"pmid", "title", "abstract"}
    required_si = {"pathogen", "antibiotic", "year", "SI"}
    for frame, required, name in (
        (screened_df, required_screen, "screened_df"),
        (abstract_df, required_abstract, "abstract_df"),
        (si_df, required_si, "si_df"),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise KeyError(f"{name} is missing required columns: {sorted(missing)}")

    if num_pathogens is not None and int(num_pathogens) < 1:
        raise ValueError("num_pathogens must be a positive integer or None.")
    si = si_df[["pathogen", "antibiotic", "year", "SI"]].copy()
    if num_pathogens is not None:
        if pathogen_priority_df is None:
            raise ValueError("pathogen_priority_df is required when num_pathogens is set.")
        required_priority = {"CanonicalSpecies", "pathogenic_index"}
        missing_priority = required_priority - set(pathogen_priority_df.columns)
        if missing_priority:
            raise KeyError(f"pathogen_priority_df is missing columns: {sorted(missing_priority)}")
        priority = pathogen_priority_df[["CanonicalSpecies", "pathogenic_index"]].copy()
        priority["CanonicalSpecies"] = priority["CanonicalSpecies"].astype(str).str.strip()
        priority["pathogenic_index"] = pd.to_numeric(priority["pathogenic_index"], errors="coerce")
        priority = priority.dropna(subset=["CanonicalSpecies", "pathogenic_index"])
        if min_pathogenic_index is not None:
            priority = priority.loc[priority["pathogenic_index"] > float(min_pathogenic_index)]
        priority = priority.sort_values(["pathogenic_index", "CanonicalSpecies"], ascending=[False, True])
        present = set(si["pathogen"].dropna().astype(str).str.strip())
        if "pathogen" in screened_df:
            present &= set(screened_df["pathogen"].dropna().astype(str).str.strip())
        priority = priority.drop_duplicates("CanonicalSpecies")
        selected = priority.loc[priority["CanonicalSpecies"].isin(present), "CanonicalSpecies"].head(int(num_pathogens))
        si["pathogen"] = si["pathogen"].astype(str).str.strip()
        si = si.loc[si["pathogen"].isin(set(selected))].copy()
    si["year"] = pd.to_numeric(si["year"], errors="coerce")
    si = si.dropna(subset=["pathogen", "antibiotic", "year", "SI"])
    si["year"] = si["year"].astype(int)
    si = si.drop_duplicates(["pathogen", "antibiotic", "year"], keep="last")

    member = screened_df[["pathogen", "antibiotic", "year", "pmids"]].copy()
    member["year"] = pd.to_numeric(member["year"], errors="coerce")
    member = member.dropna(subset=["pathogen", "antibiotic", "year"])
    member["year"] = member["year"].astype(int)
    member = member.merge(si, on=["pathogen", "antibiotic", "year"], how="inner", validate="many_to_one")
    member = member.explode("pmids").rename(columns={"pmids": "pmid"})
    member["pmid"] = member["pmid"].map(_normalise_pmid)
    member = member.dropna(subset=["pmid"])
    member = member.drop_duplicates(["pathogen", "antibiotic", "year", "pmid"])

    abstracts = abstract_df[["pmid", "title", "abstract"]].copy()
    abstracts["pmid"] = abstracts["pmid"].map(_normalise_pmid)
    abstracts = abstracts.dropna(subset=["pmid"]).drop_duplicates("pmid", keep="last")
    abstracts["title"] = abstracts["title"].fillna("").astype(str)
    abstracts["abstract"] = abstracts["abstract"].fillna("").astype(str)
    abstracts["abstract_available"] = abstracts["abstract"].str.strip().ne("")
    corpus = member.merge(abstracts, on="pmid", how="left", validate="many_to_one")
    corpus["title"] = corpus["title"].fillna("")
    corpus["abstract"] = corpus["abstract"].fillna("")
    corpus["abstract_available"] = corpus["abstract_available"].fillna(False).astype(bool)
    split = make_pair_split(
        corpus[["pathogen", "antibiotic"]].drop_duplicates().itertuples(index=False, name=None),
        heldout_fraction=heldout_fraction,
        seed=split_seed,
    )
    corpus = corpus.merge(split, on=["pathogen", "antibiotic"], how="left", validate="many_to_one")
    return corpus.sort_values(["pathogen", "antibiotic", "year", "pmid"]).reset_index(drop=True), split


def estimate_classification_cost(
    abstract_count: int,
    *,
    estimated_input_tokens_per_call: int,
    estimated_output_tokens_per_call: int,
    input_usd_per_million_tokens: float,
    output_usd_per_million_tokens: float,
) -> dict[str, float | int]:
    """Estimate one multi-label LLM request per unique abstract."""
    calls = max(0, int(abstract_count))
    input_tokens = calls * max(0, int(estimated_input_tokens_per_call))
    output_tokens = calls * max(0, int(estimated_output_tokens_per_call))
    estimated_usd = (
        input_tokens * float(input_usd_per_million_tokens)
        + output_tokens * float(output_usd_per_million_tokens)
    ) / 1_000_000
    return {"abstracts": calls, "estimated_calls": calls, "input_tokens": input_tokens,
            "output_tokens": output_tokens, "estimated_usd": estimated_usd}


def _extract_json_object(raw: str) -> dict[str, Any]:
    text = str(raw or "").strip()
    if not text:
        raise ValueError("Model returned an empty response; no JSON object was available to parse.")
    try:
        result = json.loads(text)
    except json.JSONDecodeError as exc:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            preview = text[:300].replace("\n", " ")
            raise ValueError(
                f"Model response did not contain a JSON object. Response preview: {preview!r}"
            ) from exc
        try:
            result = json.loads(match.group(0))
        except json.JSONDecodeError as nested_exc:
            raise ValueError(
                f"Model response contained malformed JSON. Response preview: {text[:300]!r}"
            ) from nested_exc
    if not isinstance(result, dict):
        raise ValueError("Model response must be a JSON object.")
    return result


def build_classification_prompt(title: str, abstract: str, catalog: Sequence[Mapping[str, Any]]) -> str:
    concepts = [{key: concept.get(key) for key in (
        "concept_id", "name", "definition", "inclusion_criteria", "exclusion_criteria"
    )} for concept in catalog]
    payload = {"title": title, "abstract": abstract, "concepts": concepts}
    return (
        "For each concept, estimate whether the abstract contains evidence for it. "
        "Return JSON with a `scores` object mapping every concept_id to a number from 0 to 1. "
        "Use 0 when evidence is absent, 1 when explicit, and intermediate values only for genuine ambiguity.\n"
        + json.dumps(payload, ensure_ascii=False)
    )


def _normalise_score_response(data: Mapping[str, Any], concept_ids: Sequence[str]) -> dict[str, float]:
    raw_scores = data.get("scores", data)
    if not isinstance(raw_scores, Mapping):
        raise ValueError("Classifier response must contain a mapping named 'scores'.")
    result: dict[str, float] = {}
    for concept_id in concept_ids:
        value = raw_scores.get(concept_id, 0.0)
        try:
            score = float(value)
        except (TypeError, ValueError):
            score = 0.0
        if not np.isfinite(score):
            score = 0.0
        result[concept_id] = min(1.0, max(0.0, score))
    return result


def classify_concept_abstracts(
    abstracts: pd.DataFrame,
    catalog: Sequence[Mapping[str, Any]],
    *,
    llm: Any,
    model_name: str,
    cache_path: str | Path,
    max_calls: int | None = None,
) -> pd.DataFrame:
    """Classify unique abstracts once per catalog version, resuming from parquet."""
    required = {"pmid", "title", "abstract"}
    if required - set(abstracts.columns):
        raise KeyError(f"abstracts is missing columns: {sorted(required - set(abstracts.columns))}")
    if not catalog:
        raise ValueError("At least one concept is required.")
    catalog_version = concept_catalog_hash(catalog)
    concept_ids = [str(c["concept_id"]) for c in catalog]
    work = abstracts[["pmid", "title", "abstract"]].copy()
    work["pmid"] = work["pmid"].map(_normalise_pmid)
    work = work.dropna(subset=["pmid"]).drop_duplicates("pmid", keep="last")
    work["title"] = work["title"].fillna("").astype(str)
    work["abstract"] = work["abstract"].fillna("").astype(str)
    work = work.loc[work["abstract"].str.strip().ne("")]

    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["pmid", "concept_id", "probability", "catalog_version", "model_name",
               "prompt_version", "status", "error"]
    all_cached = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=columns)
    if not set(columns).issubset(all_cached.columns):
        all_cached = pd.DataFrame(columns=columns)
    cached_current = all_cached.loc[
        all_cached["catalog_version"].astype(str).eq(catalog_version)
        & all_cached["model_name"].astype(str).eq(model_name)
        & all_cached["prompt_version"].astype(str).eq(CLASSIFICATION_PROMPT_VERSION)
    ]
    completed = set(cached_current.loc[cached_current["status"].eq("ok"), "pmid"].astype(str))
    pending = work.loc[~work["pmid"].isin(completed)]
    if max_calls is not None:
        pending = pending.head(max(0, int(max_calls)))
    new_rows: list[dict[str, Any]] = []
    for row in pending.itertuples(index=False):
        try:
            raw = llm.complete(
                build_classification_prompt(row.title, row.abstract, catalog),
                system_prompt=CONCEPT_SYSTEM_PROMPT,
                model=model_name,
            )
            scores = _normalise_score_response(_extract_json_object(raw), concept_ids)
            new_rows.extend({"pmid": row.pmid, "concept_id": concept_id, "probability": score,
                             "catalog_version": catalog_version, "model_name": model_name,
                             "prompt_version": CLASSIFICATION_PROMPT_VERSION,
                             "status": "ok", "error": ""}
                            for concept_id, score in scores.items())
        except Exception as exc:
            new_rows.extend({"pmid": row.pmid, "concept_id": concept_id, "probability": np.nan,
                             "catalog_version": catalog_version, "model_name": model_name,
                             "prompt_version": CLASSIFICATION_PROMPT_VERSION,
                             "status": "error", "error": str(exc)}
                            for concept_id in concept_ids)
        if new_rows and len(new_rows) % 100 == 0:
            combined = pd.concat([all_cached, pd.DataFrame(new_rows)], ignore_index=True)
            atomic_write_parquet(combined, path, index=False)
    combined = pd.concat([all_cached, pd.DataFrame(new_rows)], ignore_index=True)
    if not combined.empty:
        combined = combined.drop_duplicates(
            ["pmid", "concept_id", "catalog_version", "model_name", "prompt_version"], keep="last"
        )
    atomic_write_parquet(combined, path, index=False)
    return combined.loc[
        combined["catalog_version"].astype(str).eq(catalog_version)
        & combined["model_name"].astype(str).eq(model_name)
        & combined["prompt_version"].astype(str).eq(CLASSIFICATION_PROMPT_VERSION)
    ].reset_index(drop=True)


def classify_concept_abstracts_jev(
    abstracts: pd.DataFrame,
    catalog: Sequence[Mapping[str, Any]],
    *,
    jev_client: Any,
    model_name: str,
    cache_path: str | Path,
    max_calls: int | None = None,
    max_workers: int = 1,
    checkpoint_abstracts: int = 1000,
) -> pd.DataFrame:
    """Score concept presence with Jev's typed NOUL endpoint, resumably.

    One request contains the abstract and one binary-confidence question per
    concept. No pair identifiers or scientific-interest outcomes are passed.
    """
    required = {"pmid", "title", "abstract"}
    if required - set(abstracts.columns):
        raise KeyError(f"abstracts is missing columns: {sorted(required - set(abstracts.columns))}")
    if not catalog:
        raise ValueError("At least one concept is required.")
    catalog_version = concept_catalog_hash(catalog)
    concepts = {str(c["concept_id"]): c for c in catalog}
    if len(concepts) != len(catalog):
        raise ValueError("Concept catalog contains duplicate concept_id values.")
    concept_versions = {concept_id: concept_definition_hash(concept)
                        for concept_id, concept in concepts.items()}
    work = abstracts[["pmid", "title", "abstract"]].copy()
    work["pmid"] = work["pmid"].map(_normalise_pmid)
    work = work.dropna(subset=["pmid"]).drop_duplicates("pmid", keep="last")
    work["title"] = work["title"].fillna("").astype(str)
    work["abstract"] = work["abstract"].fillna("").astype(str)
    work = work.loc[work["abstract"].str.strip().ne("")]
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["pmid", "concept_id", "concept_version", "probability", "catalog_version",
               "model_name", "prompt_version", "status", "error"]
    legacy_columns = [column for column in columns if column != "concept_version"]
    cached = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=columns)
    if not set(legacy_columns).issubset(cached.columns):
        cached = pd.DataFrame(columns=columns)
    if "concept_version" not in cached.columns:
        cached["concept_version"] = ""

    # Backfill pre-concept-version cache entries when their catalog snapshot is
    # available. Unknown legacy versions remain usable only through their exact
    # catalog version and are never mistaken for a revised definition.
    catalog_versions: dict[tuple[str, str], str] = {}
    catalog_files = list((path.parent / "catalog_versions").glob("concept_catalog_*.json"))
    current_catalog_file = path.parent / "concept_catalog.json"
    if current_catalog_file.exists():
        catalog_files.append(current_catalog_file)
    for catalog_file in catalog_files:
        try:
            saved_catalog = json.loads(catalog_file.read_text())
            saved_version = concept_catalog_hash(saved_catalog)
            catalog_versions.update({
                (saved_version, str(saved_concept["concept_id"])): concept_definition_hash(saved_concept)
                for saved_concept in saved_catalog
            })
        except (OSError, ValueError, KeyError, TypeError):
            continue
    cached["concept_version"] = cached["concept_version"].fillna("").astype(str)
    needs_version = cached["concept_version"].eq("")
    if needs_version.any():
        cached.loc[needs_version, "concept_version"] = [
            catalog_versions.get((str(row.catalog_version), str(row.concept_id)), "")
            for row in cached.loc[needs_version, ["catalog_version", "concept_id"]].itertuples(index=False)
        ]

    prompt_mask = (
        cached["model_name"].astype(str).eq(model_name)
        & cached["prompt_version"].astype(str).eq(JEV_CLASSIFICATION_PROMPT_VERSION)
        & cached["status"].eq("ok")
    )
    successful_cache = cached.loc[prompt_mask]
    cache_lookup: dict[tuple[str, str, str], pd.Series] = {}
    for cache_row in successful_cache.itertuples(index=False):
        version = str(cache_row.concept_version)
        if version:
            cache_lookup[(str(cache_row.pmid), str(cache_row.concept_id), version)] = pd.Series(cache_row._asdict())

    current_successes = set(zip(
        cached.loc[prompt_mask & cached["catalog_version"].astype(str).eq(catalog_version), "pmid"].astype(str),
        cached.loc[prompt_mask & cached["catalog_version"].astype(str).eq(catalog_version), "concept_id"].astype(str),
        cached.loc[prompt_mask & cached["catalog_version"].astype(str).eq(catalog_version), "concept_version"].astype(str),
    ))
    pending_rows: list[tuple[Any, list[str]]] = []
    reused_rows: list[dict[str, Any]] = []
    for row in work.itertuples(index=False):
        missing_concepts: list[str] = []
        for concept_id, concept_version in concept_versions.items():
            key = (str(row.pmid), concept_id, concept_version)
            cached_row = cache_lookup.get(key)
            if cached_row is None:
                missing_concepts.append(concept_id)
            elif key not in current_successes:
                record = cached_row.to_dict()
                record["catalog_version"] = catalog_version
                reused_rows.append(record)
                current_successes.add(key)
        if missing_concepts:
            pending_rows.append((row, missing_concepts))
    total_pending_rows = len(pending_rows)
    if max_calls is not None:
        pending_rows = pending_rows[:max(0, int(max_calls))]
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1.")
    if checkpoint_abstracts < 1:
        raise ValueError("checkpoint_abstracts must be at least 1.")
    worker_local = threading.local()

    def score_one(row: Any, missing_concepts: list[str]) -> list[dict[str, Any]]:
        state = f"Title: {row.title.strip()}\nAbstract: {row.abstract.strip()}"
        questions = {
            concept_id: (
                "Based only on this title and abstract, how likely is it that the paper contains "
                f"explicit evidence of this concept: {concept.get('definition', concept.get('name', concept_id))} "
                f"Inclusion examples: {concept.get('inclusion_criteria', [])}. "
                f"Exclude: {concept.get('exclusion_criteria', [])}. Score presence, not importance or future impact."
            )
            for concept_id in missing_concepts
            for concept in [concepts[concept_id]]
        }
        try:
            if not hasattr(worker_local, "client"):
                make_worker = getattr(jev_client, "for_worker", None)
                worker_local.client = make_worker() if callable(make_worker) else jev_client
            response = worker_local.client.decide_many(state, questions=questions)
            answers = response.get("answers", {})
            rows = []
            for concept_id in missing_concepts:
                answer = answers.get(concept_id, {}) if isinstance(answers, Mapping) else {}
                probability = answer.get("noul") if isinstance(answer, Mapping) else None
                if probability is None:
                    rows.append({"pmid": row.pmid, "concept_id": concept_id,
                                 "concept_version": concept_versions[concept_id], "probability": np.nan,
                                 "catalog_version": catalog_version, "model_name": model_name,
                                 "prompt_version": JEV_CLASSIFICATION_PROMPT_VERSION,
                                 "status": "error", "error": f"Jev response omitted a NOUL score for {concept_id}."})
                    continue
                try:
                    score = float(probability)
                    if not np.isfinite(score) or not 0 <= score <= 1:
                        raise ValueError(f"Invalid NOUL score: {probability}")
                    rows.append({"pmid": row.pmid, "concept_id": concept_id,
                                 "concept_version": concept_versions[concept_id], "probability": score,
                                 "catalog_version": catalog_version, "model_name": model_name,
                                 "prompt_version": JEV_CLASSIFICATION_PROMPT_VERSION,
                                 "status": "ok", "error": ""})
                except (TypeError, ValueError) as exc:
                    rows.append({"pmid": row.pmid, "concept_id": concept_id,
                                 "concept_version": concept_versions[concept_id], "probability": np.nan,
                                 "catalog_version": catalog_version, "model_name": model_name,
                                 "prompt_version": JEV_CLASSIFICATION_PROMPT_VERSION,
                                 "status": "error", "error": str(exc)})
            return rows
        except Exception as exc:
            return [{"pmid": row.pmid, "concept_id": concept_id,
                             "concept_version": concept_versions[concept_id], "probability": np.nan,
                             "catalog_version": catalog_version, "model_name": model_name,
                             "prompt_version": JEV_CLASSIFICATION_PROMPT_VERSION,
                             "status": "error", "error": str(exc)} for concept_id in missing_concepts]

    if reused_rows:
        cached = pd.concat([cached, pd.DataFrame(reused_rows)], ignore_index=True)
        cached = cached.drop_duplicates(
            ["pmid", "concept_id", "catalog_version", "model_name", "prompt_version"], keep="last")
        atomic_write_parquet(cached, path, index=False)
    batches = (pending_rows[i:i + checkpoint_abstracts]
               for i in range(0, len(pending_rows), checkpoint_abstracts))
    completed_count = len(work) - total_pending_rows
    session_started = time.monotonic()
    session_attempts = 0
    session_successes = 0
    session_failures = 0
    get_credit_status = getattr(jev_client, "credit_status", None)

    def read_credit() -> tuple[str, float | None]:
        if not callable(get_credit_status):
            return "remaining credit unavailable (client has no balance lookup)", None
        try:
            label, amount = get_credit_status()
            return str(label), float(amount) if amount is not None else None
        except Exception as exc:
            return f"remaining credit unavailable ({type(exc).__name__})", None

    latest_credit_status, latest_credit_amount = read_credit()
    starting_credit_amount = latest_credit_amount
    last_credit_amount = latest_credit_amount
    observed_cost_per_call: float | None = None

    def refresh_credit() -> None:
        nonlocal latest_credit_status, latest_credit_amount, last_credit_amount
        nonlocal observed_cost_per_call
        latest_credit_status, latest_credit_amount = read_credit()
        if (
            latest_credit_amount is not None and last_credit_amount is not None
            and starting_credit_amount is not None
            and latest_credit_amount < last_credit_amount and session_attempts > 0
        ):
            total_observed_spend = max(0.0, starting_credit_amount - latest_credit_amount)
            if total_observed_spend > 0:
                observed_cost_per_call = total_observed_spend / session_attempts
        if latest_credit_amount is not None:
            last_credit_amount = latest_credit_amount

    def format_duration(seconds: float) -> str:
        total = max(0, int(seconds))
        hours, remainder = divmod(total, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours}h {minutes:02d}m {seconds:02d}s"
        return f"{minutes}m {seconds:02d}s"

    def progress_line(latest_batch_errors: int) -> str:
        elapsed = max(0.0, time.monotonic() - session_started)
        rate = session_attempts / elapsed if elapsed > 0 else 0.0
        remaining_this_session = max(0, len(pending_rows) - session_attempts)
        eta = remaining_this_session / rate if rate > 0 else 0.0
        unresolved = max(0, len(pending_rows) - session_successes)
        if latest_credit_amount is not None and observed_cost_per_call is not None:
            estimated_end_amount = max(0.0, latest_credit_amount - observed_cost_per_call * unresolved)
            estimate_text = f"estimated balance after {unresolved:,} unresolved requests ${estimated_end_amount:.4f}"
        else:
            estimate_text = "end balance estimate waiting for OpenRouter to report spend"
        percent = 100.0 * completed_count / len(work) if len(work) else 100.0
        return (
            f"{completed_count:,}/{len(work):,} abstracts ({percent:.1f}%) "
            f"| requests this session {session_attempts:,}/{len(pending_rows):,} "
            f"({session_successes:,} succeeded, {session_failures:,} failed) "
            f"| {latest_batch_errors:,} errors in latest checkpoint "
            f"| elapsed {format_duration(elapsed)} | ETA this session {format_duration(eta)} "
            f"| {latest_credit_status} | {estimate_text}"
        )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for batch in batches:
            batch_rows: list[dict[str, Any]] = []
            futures = [pool.submit(score_one, row, missing) for row, missing in batch]
            for future in as_completed(futures):
                batch_rows.extend(future.result())
            cached = pd.concat([cached, pd.DataFrame(batch_rows)], ignore_index=True)
            cached = cached.drop_duplicates(
                ["pmid", "concept_id", "catalog_version", "model_name", "prompt_version"], keep="last")
            atomic_write_parquet(cached, path, index=False)
            completed_count += len(batch)
            failed_pmids = {str(row["pmid"]) for row in batch_rows if row["status"] == "error"}
            session_attempts += len(batch)
            session_failures += len(failed_pmids)
            session_successes += len(batch) - len(failed_pmids)
            refresh_credit()
            print("Jev concept classification: " + progress_line(len(failed_pmids)))
    if not cached.empty:
        cached = cached.drop_duplicates(
            ["pmid", "concept_id", "catalog_version", "model_name", "prompt_version"], keep="last")
    atomic_write_parquet(cached, path, index=False)
    return cached.loc[
        cached["catalog_version"].astype(str).eq(catalog_version)
        & cached["model_name"].astype(str).eq(model_name)
        & cached["prompt_version"].astype(str).eq(JEV_CLASSIFICATION_PROMPT_VERSION)
    ].reset_index(drop=True)


def build_yearly_concept_signals(
    corpus: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    catalog_version: str,
    model_name: str | None = None,
) -> pd.DataFrame:
    """Aggregate abstract scores into pair-year concept prevalence signals."""
    required_corpus = {"pathogen", "antibiotic", "year", "pmid", "abstract_available"}
    required_scores = {"pmid", "concept_id", "probability", "catalog_version", "status"}
    if required_corpus - set(corpus.columns) or required_scores - set(scores.columns):
        raise KeyError("Corpus or score table is missing required concept-signal columns.")
    mask = scores["catalog_version"].astype(str).eq(catalog_version) & scores["status"].eq("ok")
    if model_name is not None:
        if "model_name" not in scores:
            raise KeyError("model_name filtering was requested but scores have no model_name column.")
        mask &= scores["model_name"].astype(str).eq(model_name)
    valid_scores = scores.loc[mask, ["pmid", "concept_id", "probability"]].copy()
    valid_scores["pmid"] = valid_scores["pmid"].map(_normalise_pmid)
    valid_scores["probability"] = pd.to_numeric(valid_scores["probability"], errors="coerce").clip(0, 1)
    work = corpus.loc[corpus["abstract_available"].astype(bool)].copy()
    work["pmid"] = work["pmid"].map(_normalise_pmid)
    joined = work.merge(valid_scores, on="pmid", how="inner", validate="many_to_many")
    grouped = joined.groupby(["pathogen", "antibiotic", "year", "concept_id"], dropna=False)
    out = grouped.agg(score_sum=("probability", "sum"), classified_articles=("pmid", "nunique")).reset_index()
    denominator = (work.groupby(["pathogen", "antibiotic", "year"])["pmid"].nunique()
                   .rename("abstract_articles").reset_index())
    out = out.merge(denominator, on=["pathogen", "antibiotic", "year"], how="left", validate="many_to_one")
    out["concept_prevalence"] = out["score_sum"] / out["abstract_articles"].replace(0, np.nan)
    out["classification_coverage"] = out["classified_articles"] / out["abstract_articles"].replace(0, np.nan)
    return out


def propose_concept_children(
    parent: Mapping[str, Any],
    abstracts: Sequence[Mapping[str, str]],
    *,
    llm: Any,
    model_name: str,
    num_children: int | None = None,
) -> dict[str, Any]:
    """Ask an LLM for child hypotheses from abstracts only; caller reviews before adoption."""
    if num_children is not None and int(num_children) < 1:
        raise ValueError("num_children must be at least 1.")
    child_count_instruction = (
        f"Propose exactly {int(num_children)} recurrent, distinguishable child concepts. "
        if num_children is not None else "Propose up to five recurrent, distinguishable child concepts. "
    )
    prompt = {
        "parent": {key: parent.get(key) for key in (
            "concept_id", "name", "definition", "inclusion_criteria", "exclusion_criteria"
        )},
        "abstract_examples": list(abstracts),
        "task": (
            f"{child_count_instruction}Do not use or infer outcomes. "
            "For each return name, definition, parent_id, scientific_rationale, inclusion_criteria, "
            "exclusion_criteria, representative_evidence, closest_competing_concept, difference, "
            "expected_temporal_relationship, testable_hypothesis, and rejection_conditions. "
            "Return JSON as {children: [...]} ."
        ),
    }
    raw = llm.complete(
        json.dumps(prompt, ensure_ascii=False), system_prompt=CONCEPT_SYSTEM_PROMPT,
        model=model_name, response_format={"type": "json_object"},
    )
    result = _extract_json_object(raw)
    if not isinstance(result.get("children", []), list):
        raise ValueError("Child-concept response must contain a list named 'children'.")
    return result


def _make_training_rows(
    si: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    lag_length: int,
    horizon: int,
    max_origin_year: int,
    concept_ids: Sequence[str],
) -> pd.DataFrame:
    y_lookup = {(str(r.pathogen), str(r.antibiotic), int(r.year)): float(r.SI)
                for r in si.itertuples(index=False) if pd.notna(r.SI)}
    x_lookup = {(str(r.pathogen), str(r.antibiotic), int(r.year), str(r.concept_id)):
                float(r.concept_prevalence) if pd.notna(r.concept_prevalence) else 0.0
                for r in signals.itertuples(index=False)}
    pairs = sorted({(p, a) for p, a, _ in y_lookup})
    rows: list[dict[str, Any]] = []
    for pathogen, antibiotic in pairs:
        years = sorted(y for p, a, y in y_lookup if p == pathogen and a == antibiotic)
        for origin in years:
            target_year = origin + horizon
            if target_year > max_origin_year + horizon or (pathogen, antibiotic, target_year) not in y_lookup:
                continue
            features: dict[str, float] = {}
            valid = True
            for lag in range(1, lag_length + 1):
                y_value = y_lookup.get((pathogen, antibiotic, origin - lag))
                if y_value is None:
                    valid = False
                    break
                features[f"y_lag_{lag}"] = y_value
                for concept_id in concept_ids:
                    features[f"{concept_id}_lag_{lag}"] = x_lookup.get(
                        (pathogen, antibiotic, origin - lag, concept_id), 0.0
                    )
            if not valid:
                continue
            rows.append({"pathogen": pathogen, "antibiotic": antibiotic, "origin_year": origin,
                         "target_year": target_year, "target": y_lookup[(pathogen, antibiotic, target_year)],
                         **features})
    return pd.DataFrame(rows)


def _fit_panel_slopes(train: pd.DataFrame, feature_cols: Sequence[str]) -> np.ndarray | None:
    if len(train) <= len(feature_cols) or train.empty:
        return None
    train = train.reset_index(drop=True)
    x = train[list(feature_cols)].to_numpy(dtype=float)
    y = train["target"].to_numpy(dtype=float)
    groups = train["pathogen"].astype(str) + "\x1f" + train["antibiotic"].astype(str)
    xw = x.copy()
    yw = y.copy()
    group_frame = pd.DataFrame({"group": groups})
    for indices in group_frame.groupby("group", sort=False).indices.values():
        idx = np.asarray(indices, dtype=int)
        xw[idx] -= x[idx].mean(axis=0)
        yw[idx] -= y[idx].mean()
    if not np.isfinite(xw).all() or not np.isfinite(yw).all():
        return None
    return np.linalg.lstsq(xw, yw, rcond=None)[0]


def rolling_origin_concept_forecasts(
    si_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    *,
    concept_ids: Sequence[str] = (),
    model_sets: Mapping[str, Sequence[str]] | None = None,
    development_pairs: set[tuple[str, str]] | None = None,
    fit_start_year: int = 2000,
    fit_end_year: int = 2023,
    lag_length: int = 3,
    horizons: Sequence[int] = (1, 2, 3),
    min_prior_observations: int = 10,
    progress_every: int = 0,
) -> pd.DataFrame:
    """Generate direct, expanding-window, fixed-effect panel forecasts.

    Common lag slopes are learned on development pairs. A pair intercept is
    estimated from that pair's observations available at each forecast origin.
    Models with or without concepts use the same target rows.
    """
    required_si = {"pathogen", "antibiotic", "year", "SI"}
    if required_si - set(si_df.columns):
        raise KeyError(f"si_df is missing columns: {sorted(required_si - set(si_df.columns))}")
    si = si_df[list(required_si)].copy()
    si["year"] = pd.to_numeric(si["year"], errors="coerce")
    si["SI"] = pd.to_numeric(si["SI"], errors="coerce")
    si = si.dropna(subset=["pathogen", "antibiotic", "year", "SI"])
    si["year"] = si["year"].astype(int)
    si = si.drop_duplicates(["pathogen", "antibiotic", "year"], keep="last")
    dev_set = None if development_pairs is None else {(str(p), str(a)) for p, a in development_pairs}
    result: list[dict[str, Any]] = []
    if model_sets is None:
        configured_models: dict[str, Sequence[str]] = {
            "baseline": (),
            **{f"concept:{concept_id}": (concept_id,) for concept_id in concept_ids},
        }
    else:
        configured_models = {"baseline": (), **{name: tuple(ids) for name, ids in model_sets.items()}}
    progress_interval = max(0, int(progress_every))
    origins = range(fit_start_year, fit_end_year + 1)
    total_steps = len(horizons) * len(configured_models) * len(origins)
    completed_steps = 0
    progress_started = time.monotonic()

    def report_progress(steps: int = 1) -> None:
        nonlocal completed_steps
        completed_steps += steps
        if not progress_interval or (completed_steps % progress_interval and completed_steps != total_steps):
            return
        elapsed = max(0.0, time.monotonic() - progress_started)
        rate = completed_steps / elapsed if elapsed > 0 else 0.0
        remaining = max(0, total_steps - completed_steps)
        eta = remaining / rate if rate > 0 else 0.0
        total_seconds = max(0, int(elapsed))
        hours, rem = divmod(total_seconds, 3600)
        minutes, seconds = divmod(rem, 60)
        elapsed_text = f"{hours}h {minutes:02d}m {seconds:02d}s" if hours else f"{minutes}m {seconds:02d}s"
        total_seconds = max(0, int(eta))
        hours, rem = divmod(total_seconds, 3600)
        minutes, seconds = divmod(rem, 60)
        eta_text = f"{hours}h {minutes:02d}m {seconds:02d}s" if hours else f"{minutes}m {seconds:02d}s"
        percent = 100.0 * completed_steps / total_steps if total_steps else 100.0
        print(f"Forecast progress: {completed_steps:,}/{total_steps:,} model-origin steps ({percent:.1f}%) "
              f"| elapsed {elapsed_text} | ETA {eta_text}")

    si_years_by_pair: dict[tuple[str, str], np.ndarray] = {}
    for (pathogen, antibiotic), pair_si in si.groupby(["pathogen", "antibiotic"], sort=False):
        si_years_by_pair[(str(pathogen), str(antibiotic))] = np.sort(pair_si["year"].to_numpy(dtype=int))
    for horizon in horizons:
        for model_name, selected_ids in configured_models.items():
            rows = _make_training_rows(si, signals_df, lag_length=lag_length, horizon=int(horizon),
                                       max_origin_year=fit_end_year, concept_ids=selected_ids)
            if rows.empty:
                report_progress(len(origins))
                continue
            rows_by_pair = {
                (str(pathogen), str(antibiotic)): pair_rows
                for (pathogen, antibiotic), pair_rows in rows.groupby(["pathogen", "antibiotic"], sort=False)
            }
            y_cols = [f"y_lag_{lag}" for lag in range(1, lag_length + 1)]
            feature_cols = y_cols + [f"{c}_lag_{lag}" for c in selected_ids for lag in range(1, lag_length + 1)]
            for origin in origins:
                try:
                    train = rows.loc[(rows["origin_year"] + int(horizon)) <= origin].copy()
                    if dev_set is not None:
                        train_pairs = list(zip(train["pathogen"].astype(str), train["antibiotic"].astype(str)))
                        train = train.loc[[pair in dev_set for pair in train_pairs]]
                    slopes = _fit_panel_slopes(train, feature_cols)
                    if slopes is None:
                        continue
                    candidate = rows.loc[rows["origin_year"].eq(origin)].copy()
                    for pair_values, pair_candidates in candidate.groupby(["pathogen", "antibiotic"], sort=False):
                        pair = (str(pair_values[0]), str(pair_values[1]))
                        observed_years = si_years_by_pair.get(pair, np.asarray([], dtype=int))
                        prior_observation_count = int(np.searchsorted(observed_years, origin, side="right"))
                        if prior_observation_count < min_prior_observations:
                            continue
                        pair_rows = rows_by_pair[pair]
                        prior = pair_rows.loc[(pair_rows["origin_year"] + int(horizon)) <= origin]
                        if prior.empty:
                            continue
                        xprior = prior[list(feature_cols)].to_numpy(dtype=float)
                        yprior = prior["target"].to_numpy(dtype=float)
                        alpha = float(np.mean(yprior - xprior @ slopes))
                        xrows = pair_candidates[list(feature_cols)].to_numpy(dtype=float)
                        predictions = alpha + xrows @ slopes
                        for row, prediction in zip(pair_candidates.itertuples(index=False), predictions):
                            result.append({"pathogen": pair[0], "antibiotic": pair[1], "model": model_name,
                                           "horizon": int(horizon), "origin_year": int(origin),
                                           "target_year": int(row.target_year), "observed": float(row.target),
                                           "predicted": float(prediction),
                                           "squared_error": (float(row.target) - float(prediction)) ** 2})
                finally:
                    report_progress()
    return pd.DataFrame(result)


def summarise_concept_forecasts(forecasts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-pair and population summaries of baseline-relative MSE gains."""
    if forecasts.empty:
        empty = pd.DataFrame()
        return empty, empty
    comparisons = {
        str(model): "baseline"
        for model in forecasts["model"].dropna().unique()
        if str(model) != "baseline"
    }
    return summarise_incremental_forecast_gains(forecasts, comparisons)


def summarise_incremental_forecast_gains(
    forecasts: pd.DataFrame,
    comparisons: Mapping[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare each named model with its reference on matched forecast origins.

    ``comparisons`` maps an evaluated model name to its reference model, enabling
    conditional gains and parent-versus-children replacement comparisons.
    """
    keys = ["pathogen", "antibiotic", "horizon", "origin_year", "target_year"]
    pair_results: list[pd.DataFrame] = []
    for candidate, reference in comparisons.items():
        left = forecasts.loc[forecasts["model"].eq(candidate), keys + ["squared_error"]]
        right = forecasts.loc[forecasts["model"].eq(reference), keys + ["squared_error"]]
        merged = left.merge(right, on=keys, how="inner", suffixes=("_candidate", "_reference"))
        if merged.empty:
            continue
        merged["model"] = candidate
        merged["reference_model"] = reference
        merged["forecast_gain"] = merged["squared_error_reference"] - merged["squared_error_candidate"]
        pair_results.append(merged)
    if not pair_results:
        empty = pd.DataFrame()
        return empty, empty
    paired = pd.concat(pair_results, ignore_index=True)
    pair_summary = paired.groupby(["model", "reference_model", "pathogen", "antibiotic", "horizon"],
                                  as_index=False).agg(
        candidate_mse=("squared_error_candidate", "mean"),
        reference_mse=("squared_error_reference", "mean"),
        forecast_gain=("forecast_gain", "mean"),
        forecasts=("forecast_gain", "size"),
    )
    by_horizon = pair_summary.groupby(["model", "reference_model", "horizon"], as_index=False).agg(
        generality=("forecast_gain", "mean"),
        specialist_strength=("forecast_gain", lambda values: float(values.quantile(0.9))),
        pairs=("forecast_gain", "count"),
    )
    pair_overall = pair_summary.groupby(
        ["model", "reference_model", "pathogen", "antibiotic"], as_index=False
    )["forecast_gain"].mean()
    overall = pair_overall.groupby(["model", "reference_model"], as_index=False).agg(
        generality=("forecast_gain", "mean"),
        specialist_strength=("forecast_gain", lambda values: float(values.quantile(0.9))),
        pairs=("forecast_gain", "count"),
    )
    overall["horizon"] = 0  # 0 denotes the equal-weight mean over horizons 1, 2, and 3.
    population = pd.concat([by_horizon, overall[by_horizon.columns]], ignore_index=True)
    return pair_summary, population


def compare_parent_children(
    parent_forecasts: pd.DataFrame,
    children_forecasts: pd.DataFrame,
    *,
    parent_id: str,
) -> pd.DataFrame:
    """Compare parent-only and child-replacement forecasts on matched origins."""
    keys = ["pathogen", "antibiotic", "horizon", "origin_year", "target_year"]
    parent_mask = parent_forecasts["model"].isin({f"concept:{parent_id}", f"parent:{parent_id}"})
    parent = parent_forecasts.loc[parent_mask, keys + ["squared_error"]]
    children = children_forecasts.loc[
        children_forecasts["model"].astype(str).str.startswith("children:"), keys + ["squared_error"]
    ]
    if parent.empty or children.empty:
        return pd.DataFrame(columns=keys + ["parent_mse", "children_mse", "split_gain"])
    child_mean = children.groupby(keys, as_index=False)["squared_error"].mean()
    merged = parent.merge(child_mean, on=keys, how="inner", suffixes=("_parent", "_children"))
    merged = merged.rename(columns={"squared_error_parent": "parent_mse", "squared_error_children": "children_mse"})
    merged["split_gain"] = merged["parent_mse"] - merged["children_mse"]
    return merged


__all__ = [
    "SEED_CONCEPTS", "CONCEPT_SYSTEM_PROMPT", "CLASSIFICATION_PROMPT_VERSION", "concept_catalog_hash",
    "concept_definition_hash", "pareto_concept_frontier", "make_pair_split",
    "build_concept_corpus", "estimate_classification_cost", "build_classification_prompt",
    "classify_concept_abstracts", "classify_concept_abstracts_jev", "build_yearly_concept_signals", "propose_concept_children",
    "rolling_origin_concept_forecasts", "summarise_concept_forecasts", "compare_parent_children",
    "summarise_incremental_forecast_gains",
]
