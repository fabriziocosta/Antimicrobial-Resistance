"""Orchestration functions shared by the inspectable SI concept notebooks.

Notebook files expose experiment parameters, call these functions, and display
returned tables or figures. Model APIs are imported lazily by the stage that uses them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from amr.pipeline_config import cfg, cfg_path, load_pipeline_config
from amr.pipelines.concepts import (
    build_si_concept_paths,
    corpus_preflight,
    load_concept_corpus,
)
from amr.si.concepts import (
    SEED_CONCEPTS,
    JEV_CLASSIFICATION_PROMPT_VERSION,
    concept_catalog_hash,
    build_yearly_concept_signals,
    classify_concept_abstracts_jev,
    rolling_origin_concept_forecasts,
    summarise_incremental_forecast_gains,
)
from amr.si.concept_causal import estimate_pair_concept_causal_rankings


def _project(project_root: str | Path):
    root = Path(project_root).resolve()
    config = load_pipeline_config(project_root=root)
    return root, config, build_si_concept_paths(root)


def _load_pair_split(paths: Mapping[str, Path]) -> pd.DataFrame:
    split = pd.read_csv(paths["pair_split"])
    split[["pathogen", "antibiotic"]] = split[["pathogen", "antibiotic"]].astype(str)
    return split


def prepare_concept_corpus(
    project_root: str | Path,
    *,
    num_pathogens: int | None = 10,
    heldout_fraction: float = 0.20,
    split_seed: int = 42,
    min_pathogenic_index: float | None = None,
    screened_path: str | Path | None = None,
    abstracts_path: str | Path | None = None,
    si_path: str | Path | None = None,
    pathogen_priority_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build and persist the pair-year corpus using the same inputs as notebook 10a."""
    root, config, paths = _project(project_root)
    screened_path = screened_path or cfg_path(config, "paths.yearly_screened", root)
    abstracts_path = abstracts_path or cfg_path(config, "paths.abstract_cache", root)
    si_path = si_path or root / "outputs" / "parquets" / "yearly_si.parquet"
    pathogen_priority_path = pathogen_priority_path or cfg_path(config, "paths.pathogen_priority", root)
    if min_pathogenic_index is None:
        min_pathogenic_index = float(cfg(config, "collection.min_pathogenic_index", 0))
    corpus, pair_split = load_concept_corpus(
        screened_path=screened_path,
        abstracts_path=abstracts_path,
        si_path=si_path,
        heldout_fraction=heldout_fraction,
        split_seed=split_seed,
        pathogen_priority_path=pathogen_priority_path,
        num_pathogens=num_pathogens,
        min_pathogenic_index=min_pathogenic_index,
    )
    paths["directory"].mkdir(parents=True, exist_ok=True)
    unique_abstracts = corpus[["pmid", "title", "abstract", "abstract_available"]].drop_duplicates("pmid")
    corpus.drop(columns=["title", "abstract"]).to_parquet(paths["corpus"], index=False)
    unique_abstracts.to_parquet(paths["abstracts"], index=False)
    pair_split.to_csv(paths["pair_split"], index=False)
    preflight = corpus_preflight(corpus)
    checks = {
        "duplicate_pair_year_pmid_links": int(corpus[["pathogen", "antibiotic", "year", "pmid"]].duplicated().sum()),
        "split_labels": sorted(pair_split["split"].dropna().unique().tolist()),
        "selected_pathogens": int(corpus["pathogen"].nunique()),
        "pathogen_limit": num_pathogens,
        "all_antibiotics_retained": True,
        "preflight": preflight,
        "coverage": corpus.groupby("split").agg(
            pairs=("pathogen", "nunique"), pair_year_article_links=("pmid", "size"),
            unique_pmids=("pmid", "nunique"), abstract_coverage=("abstract_available", "mean"),
        ),
        "artifacts": paths,
    }
    if checks["duplicate_pair_year_pmid_links"]:
        raise ValueError("Corpus contains duplicate pathogen-antibiotic-year-PMID associations.")
    if set(checks["split_labels"]) != {"development", "confirmation"}:
        raise ValueError("The pair split must contain development and confirmation pairs.")
    return checks


def load_concept_catalog_context(project_root: str | Path) -> dict[str, Any]:
    """Load the persisted corpus, abstract table, catalog, and paths for notebook 10b."""
    root, _, paths = _project(project_root)
    corpus = pd.read_parquet(paths["corpus"])
    abstract_table = pd.read_parquet(paths["abstracts"])
    if paths["catalog"].exists():
        catalog = json.loads(paths["catalog"].read_text())
    else:
        catalog = [dict(concept) for concept in SEED_CONCEPTS]
    return {
        "corpus": corpus,
        "abstract_table": abstract_table,
        "catalog": catalog,
        "catalog_version": concept_catalog_hash(catalog),
        "paths": paths,
        "project_root": root,
    }


def build_development_proposal_examples(
    project_root: str | Path,
    *,
    parent_id: str,
    positive_threshold: float = 0.5,
    max_examples: int = 30,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Select cached positive examples exclusive to development pairs."""
    context = load_concept_catalog_context(project_root)
    corpus, abstract_table = context["corpus"], context["abstract_table"]
    paths = context["paths"]
    development_pmids = set(corpus.loc[corpus["split"].eq("development"), "pmid"].astype(str))
    confirmation_pmids = set(corpus.loc[corpus["split"].eq("confirmation"), "pmid"].astype(str))
    exclusive = development_pmids - confirmation_pmids
    examples_df = pd.DataFrame(columns=["pmid", "title", "abstract"])
    if paths["scores"].exists():
        scores = pd.read_parquet(paths["scores"])
        jev_model = str(cfg(load_pipeline_config(project_root=context["project_root"]), "system1.model", "typesafe/jev-1.13"))
        selected = scores.loc[
            scores["concept_id"].astype(str).eq(str(parent_id))
            & scores["catalog_version"].astype(str).eq(context["catalog_version"])
            & scores["model_name"].astype(str).eq(jev_model)
            & scores["prompt_version"].astype(str).eq(JEV_CLASSIFICATION_PROMPT_VERSION)
            & scores["status"].eq("ok")
            & scores["pmid"].astype(str).isin(exclusive)
        ].drop_duplicates("pmid", keep="last")
        selected = selected.loc[pd.to_numeric(selected["probability"], errors="coerce").ge(positive_threshold)]
        examples_df = (
            selected[["pmid"]]
            .merge(abstract_table, on="pmid", how="inner", validate="one_to_one")
            .loc[lambda frame: frame["abstract_available"].astype(bool)]
            .sample(frac=1, random_state=random_seed)
            .head(max_examples)
        )
    examples = examples_df[["title", "abstract"]].to_dict("records")
    if set(examples_df["pmid"].astype(str)) & confirmation_pmids:
        raise AssertionError("Proposal examples include confirmation-linked PMIDs.")
    return {**context, "examples_df": examples_df, "examples": examples, "development_only_pmids": exclusive}


def request_child_concepts(
    parent: Mapping[str, Any],
    examples: Sequence[Mapping[str, str]],
    *,
    model_name: str,
    temperature: float = 0,
    max_tokens: int = 4000,
    num_children: int = 3,
) -> dict[str, Any]:
    """Ask the configured LLM for child hypotheses; save/review remains a notebook step."""
    from amr.llm import OpenAIConfig, OpenAILanguageModel
    from amr.si.concepts import propose_concept_children

    llm = OpenAILanguageModel(OpenAIConfig(model=model_name, temperature=temperature, max_tokens=max_tokens))
    return propose_concept_children(parent, examples, llm=llm, model_name=model_name, num_children=num_children)



def save_child_proposal(project_root: str | Path, parent_id: str, proposal: Mapping[str, Any]) -> Path:
    """Persist an LLM child proposal for review without changing the active catalog."""
    _, _, paths = _project(project_root)
    paths["directory"].mkdir(parents=True, exist_ok=True)
    proposal_path = paths["directory"] / f"proposals_{parent_id}.json"
    proposal_path.write_text(json.dumps(proposal, indent=2, ensure_ascii=False))
    return proposal_path


def save_reviewed_concept_catalog(project_root: str | Path, catalog: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Save a reviewed catalog and its immutable hash-named snapshot."""
    _, _, paths = _project(project_root)
    catalog = list(catalog)
    version = concept_catalog_hash(catalog)
    paths["catalog_versions"].mkdir(parents=True, exist_ok=True)
    snapshot = paths["catalog_versions"] / f"concept_catalog_{version}.json"
    snapshot.write_text(json.dumps(catalog, indent=2, ensure_ascii=False))
    paths["catalog"].write_text(json.dumps(catalog, indent=2, ensure_ascii=False))
    return {"catalog_version": version, "catalog_path": paths["catalog"], "snapshot_path": snapshot}


def load_classification_context(project_root: str | Path) -> dict[str, Any]:
    """Load classifier inputs and a safe prompt preview for notebook 10c."""
    context = load_concept_catalog_context(project_root)
    abstract_table = context["abstract_table"]
    abstracts = abstract_table.loc[
        abstract_table["abstract_available"].astype(bool), ["pmid", "title", "abstract"]
    ].drop_duplicates("pmid")
    abstracts = abstracts.loc[abstracts["abstract"].fillna("").astype(str).str.strip().ne("")]
    return {**context, "abstracts": abstracts}


def run_cached_concept_classification(
    project_root: str | Path,
    *,
    max_calls: int | None = None,
    max_workers: int = 128,
    run_classification: bool = True,
) -> dict[str, Any]:
    """Run or inspect resumable Jev classification for the current catalog."""
    from amr.screening.system1 import OpenRouterJevClient

    context = load_classification_context(project_root)
    config = load_pipeline_config(project_root=context["project_root"])
    model_name = str(cfg(config, "system1.model", "typesafe/jev-1.13"))
    paths = context["paths"]
    if run_classification:
        client = OpenRouterJevClient(model=model_name)
        scores = classify_concept_abstracts_jev(
            context["abstracts"], context["catalog"], jev_client=client, model_name=model_name,
            cache_path=paths["scores"], max_calls=max_calls, max_workers=max_workers,
        )
    elif paths["scores"].exists():
        scores = pd.read_parquet(paths["scores"])
    else:
        scores = pd.DataFrame()
    if scores.empty:
        status = pd.DataFrame(columns=["catalog_version", "status", "labels"])
    else:
        current = scores.loc[
            scores["catalog_version"].astype(str).eq(context["catalog_version"])
            & scores["model_name"].astype(str).eq(model_name)
        ]
        status = current.groupby(["catalog_version", "status"]).size().rename("labels").reset_index()
    preview = context["abstracts"].iloc[0].to_dict() if len(context["abstracts"]) else None
    return {**context, "model_name": model_name, "scores": scores, "status_summary": status, "preview": preview}


def evaluate_development_concepts(
    project_root: str | Path,
    *,
    current_population: Sequence[str],
    candidate_ids: Sequence[str],
    parent_child_replacements: Mapping[str, Sequence[str]],
    fit_start_year: int = 2000,
    fit_end_year: int = 2023,
    lag_length: int = 3,
    horizons: Sequence[int] = (1, 2, 3),
    min_prior_observations: int = 10,
    progress_every: int = 10,
) -> dict[str, Any]:
    """Build yearly signals and rolling-origin development forecasts for notebook 10d."""
    root, config, paths = _project(project_root)
    corpus = pd.read_parquet(paths["corpus"])
    si_df = pd.read_parquet(root / "outputs" / "parquets" / "yearly_si.parquet")
    catalog = json.loads(paths["catalog"].read_text())
    version = concept_catalog_hash(catalog)
    model_name = str(cfg(config, "system1.model", "typesafe/jev-1.13"))
    scores = pd.read_parquet(paths["scores"])
    valid = scores.loc[
        scores["catalog_version"].astype(str).eq(version)
        & scores["model_name"].astype(str).eq(model_name)
        & scores["status"].eq("ok")
    ]
    expected_pmids = set(corpus.loc[corpus["abstract_available"], "pmid"].astype(str))
    missing_pmids = expected_pmids - set(valid["pmid"].astype(str))
    if missing_pmids:
        raise RuntimeError(f"Catalog scores are incomplete for {len(missing_pmids):,} abstracts. Resume notebook 10c first.")
    signals = build_yearly_concept_signals(corpus, scores, catalog_version=version, model_name=model_name)
    signals.to_parquet(paths["signals"], index=False)
    concept_ids = [str(concept["concept_id"]) for concept in catalog]
    current_population = list(map(str, current_population))
    candidate_ids = list(map(str, candidate_ids))
    model_sets = {f"concept:{concept_id}": [concept_id] for concept_id in candidate_ids}
    comparisons = {f"concept:{concept_id}": "baseline" for concept_id in candidate_ids}
    if current_population:
        model_sets["population"] = list(current_population)
        comparisons["population"] = "baseline"
        for candidate in candidate_ids:
            if candidate not in current_population:
                name = f"population_plus:{candidate}"
                model_sets[name] = [*current_population, candidate]
                comparisons[name] = "population"
    for parent_id, child_ids in parent_child_replacements.items():
        background = [concept_id for concept_id in current_population if concept_id != parent_id]
        model_sets[f"parent:{parent_id}"] = [*background, parent_id]
        model_sets[f"children:{parent_id}"] = [*background, *map(str, child_ids)]
        comparisons[f"parent:{parent_id}"] = "baseline"
        comparisons[f"children:{parent_id}"] = f"parent:{parent_id}"
    split = _load_pair_split(paths)
    development_pairs = set(map(tuple, split.loc[split["split"].eq("development"), ["pathogen", "antibiotic"]].to_numpy()))
    forecasts = rolling_origin_concept_forecasts(
        si_df, signals, concept_ids=concept_ids, model_sets=model_sets,
        development_pairs=development_pairs, fit_start_year=fit_start_year, fit_end_year=fit_end_year,
        lag_length=lag_length, horizons=horizons, min_prior_observations=min_prior_observations,
        progress_every=progress_every,
    )
    forecasts = forecasts.merge(split, on=["pathogen", "antibiotic"], how="left", validate="many_to_one")
    forecasts = forecasts.loc[forecasts["split"].eq("development")].drop(columns="split")
    forecasts.to_parquet(paths["forecasts"], index=False)
    pair_summary, population_summary = summarise_incremental_forecast_gains(forecasts, comparisons)
    pair_summary.to_parquet(paths["pair_summary"], index=False)
    population_summary.to_parquet(paths["population_summary"], index=False)
    return {
        "signals": signals, "forecasts": forecasts, "pair_summary": pair_summary,
        "population_summary": population_summary, "concept_ids": concept_ids,
        "paths": paths,
    }


def confirm_frozen_concepts(
    project_root: str | Path,
    *,
    expected_catalog_version: str,
    enabled: bool,
    frozen_population: Sequence[str] = (),
    parent_child_replacements: Mapping[str, Sequence[str]] | None = None,
    fit_start_year: int = 2000,
    fit_end_year: int = 2023,
    lag_length: int = 3,
    horizons: Sequence[int] = (1, 2, 3),
    min_prior_observations: int = 10,
    min_attribution_origins: int = 5,
) -> dict[str, Any]:
    """Evaluate a frozen catalog/population once on confirmation pairs."""
    import matplotlib.pyplot as plt

    root, config, paths = _project(project_root)
    corpus = pd.read_parquet(paths["corpus"])
    si_df = pd.read_parquet(root / "outputs" / "parquets" / "yearly_si.parquet")
    catalog = json.loads(paths["catalog"].read_text())
    version = concept_catalog_hash(catalog)
    if not enabled or str(expected_catalog_version) != version:
        raise RuntimeError("Freeze the catalog, set its expected hash, and explicitly enable confirmation.")
    model_name = str(cfg(config, "system1.model", "typesafe/jev-1.13"))
    scores = pd.read_parquet(paths["scores"])
    valid = scores.loc[
        scores["catalog_version"].astype(str).eq(version)
        & scores["model_name"].astype(str).eq(model_name)
        & scores["status"].eq("ok")
    ]
    expected_pmids = set(corpus.loc[corpus["abstract_available"], "pmid"].astype(str))
    if expected_pmids - set(valid["pmid"].astype(str)):
        raise RuntimeError("Frozen catalog scores are incomplete; finish notebook 10c first.")
    signals = build_yearly_concept_signals(corpus, scores, catalog_version=version, model_name=model_name)
    signals.to_parquet(paths["signals"], index=False)
    concept_ids = [str(concept["concept_id"]) for concept in catalog]
    frozen_population = list(map(str, frozen_population))
    parent_child_replacements = parent_child_replacements or {}
    if parent_child_replacements and not frozen_population:
        raise ValueError("Set FROZEN_POPULATION before requesting nested replacement estimates.")
    if any(str(parent) not in frozen_population for parent in parent_child_replacements):
        raise ValueError("Every nested replacement parent must be in FROZEN_POPULATION.")
    model_sets = {f"concept:{concept_id}": [concept_id] for concept_id in concept_ids}
    comparisons = {name: "baseline" for name in model_sets}
    if frozen_population:
        model_sets["frozen_population"] = frozen_population
        comparisons["frozen_population"] = "baseline"
        for concept_id in frozen_population:
            full_name = f"conditional_with:{concept_id}"
            without_name = f"conditional_without:{concept_id}"
            model_sets[full_name] = list(frozen_population)
            model_sets[without_name] = [item for item in frozen_population if item != concept_id]
            comparisons[full_name] = without_name
        for parent_id, child_ids in parent_child_replacements.items():
            parent_id = str(parent_id)
            child_ids = list(map(str, child_ids))
            background = [item for item in frozen_population if item != parent_id]
            parent_name = f"nested_parent:{parent_id}"
            children_name = f"nested_children:{parent_id}"
            model_sets[parent_name] = list(frozen_population)
            model_sets[children_name] = [*background, *child_ids]
            comparisons[children_name] = parent_name
            for child_id in child_ids:
                single_name = f"nested_single_child:{parent_id}:{child_id}"
                model_sets[single_name] = [*background, child_id]
                comparisons[single_name] = parent_name
    split = _load_pair_split(paths)
    development_pairs = set(map(tuple, split.loc[split["split"].eq("development"), ["pathogen", "antibiotic"]].to_numpy()))
    confirmation_pairs = set(map(tuple, split.loc[split["split"].eq("confirmation"), ["pathogen", "antibiotic"]].to_numpy()))
    forecasts = rolling_origin_concept_forecasts(
        si_df, signals, concept_ids=concept_ids, model_sets=model_sets,
        development_pairs=development_pairs, fit_start_year=fit_start_year, fit_end_year=fit_end_year,
        lag_length=lag_length, horizons=horizons, min_prior_observations=min_prior_observations,
    )
    pairs = list(zip(forecasts["pathogen"].astype(str), forecasts["antibiotic"].astype(str)))
    forecasts = forecasts.loc[[pair in confirmation_pairs for pair in pairs]].copy()
    forecasts.to_parquet(paths["confirmation_forecasts"], index=False)
    pair_summary, population_summary = summarise_incremental_forecast_gains(forecasts, comparisons)
    pair_summary.to_parquet(paths["confirmation_summary"], index=False)
    causal_horizons, causal_ranking = estimate_pair_concept_causal_rankings(
        si_df, signals, corpus, catalog, split,
        fit_start_year=fit_start_year, fit_end_year=fit_end_year,
        horizons=horizons, lag_length=lag_length,
        min_prior_observations=min_prior_observations,
        min_pair_origins=min_attribution_origins,
    )
    causal_horizons.to_parquet(paths["pair_concept_causal_horizons"], index=False)
    causal_ranking.to_parquet(paths["pair_concept_causal_ranking"], index=False)
    horizon_report = pair_summary.groupby(["model", "horizon"], as_index=False).agg(
        generality=("forecast_gain", "mean"),
        specialist_strength=("forecast_gain", lambda values: values.quantile(0.9)),
        confirmation_pairs=("forecast_gain", "count"),
    ) if not pair_summary.empty else pd.DataFrame()
    paths["figures"].mkdir(parents=True, exist_ok=True)
    figure_path = paths["figures"] / "confirmation_generality_by_horizon.png"
    fig, ax = plt.subplots(figsize=(11, 5))
    if not horizon_report.empty:
        for model, group in horizon_report.groupby("model"):
            ax.plot(group["horizon"], group["generality"], marker="o", label=model)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(xticks=list(horizons), xlabel="Forecast horizon (years)", ylabel="Held-out MSE gain vs reference",
           title=f"Frozen catalog confirmation — {version}")
    if not horizon_report.empty:
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    return {
        "forecasts": forecasts, "pair_summary": pair_summary,
        "population_summary": population_summary, "horizon_report": horizon_report,
        "pair_concept_causal_horizons": causal_horizons,
        "pair_concept_causal_ranking": causal_ranking,
        "figure": fig, "figure_path": figure_path, "catalog_version": version,
        "paths": paths,
    }
