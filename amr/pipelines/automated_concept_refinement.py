"""Orchestration for the automated SI concept refinement notebook (10f)."""
from __future__ import annotations

import hashlib
import json
import re
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

from amr.pipeline_config import cfg, cfg_path, load_pipeline_config
from amr.pipelines.concepts import build_si_concept_paths, corpus_preflight, load_concept_corpus
from amr.screening.system1 import OpenRouterJevClient
from amr.si.concepts import (
    SEED_CONCEPTS, concept_catalog_hash, concept_definition_hash, JEV_CLASSIFICATION_PROMPT_VERSION,
    pareto_concept_frontier, classify_concept_abstracts_jev, build_yearly_concept_signals,
    rolling_origin_concept_forecasts, summarise_incremental_forecast_gains, propose_concept_children,
)



def _save_concept_phylogeny(catalog, survivor_ids, output_path: Path) -> Path:
    """Render a compact-label refinement forest and save a full node index beside it."""
    import matplotlib.pyplot as plt

    concepts = {str(concept["concept_id"]): concept for concept in catalog}
    children = {concept_id: [] for concept_id in concepts}
    roots = []
    for concept_id, concept in concepts.items():
        parent_id = concept.get("parent_id")
        parent_id = str(parent_id) if parent_id not in (None, "") else None
        if parent_id in concepts:
            children[parent_id].append(concept_id)
        else:
            roots.append(concept_id)
    name_key = lambda concept_id: (str(concepts[concept_id].get("name", concept_id)).casefold(), concept_id)
    for child_ids in children.values():
        child_ids.sort(key=name_key)
    roots.sort(key=name_key)

    survivor_ids = {str(concept_id) for concept_id in survivor_ids if str(concept_id) in concepts}
    surviving_paths = set(survivor_ids)
    for survivor_id in survivor_ids:
        parent_id = concepts[survivor_id].get("parent_id")
        while parent_id not in (None, "") and str(parent_id) in concepts:
            parent_id = str(parent_id)
            if parent_id in surviving_paths:
                break
            surviving_paths.add(parent_id)
            parent_id = concepts[parent_id].get("parent_id")

    node_labels = {concept_id: f"C{index:03d}" for index, concept_id in enumerate(sorted(concepts), start=1)}
    positions = {}
    next_leaf_y = 0
    max_depth = 0
    active = set()
    visited = set()

    def place(concept_id, depth):
        nonlocal next_leaf_y, max_depth
        if concept_id in active:
            raise ValueError(f"Cycle detected in concept hierarchy at {concept_id!r}.")
        active.add(concept_id)
        visited.add(concept_id)
        max_depth = max(max_depth, depth)
        if not children[concept_id]:
            y = next_leaf_y
            name_lines = max(1, len(textwrap.wrap(str(concepts[concept_id].get("name", concept_id)), width=28)))
            next_leaf_y += max(1.25, name_lines * 0.62)
        else:
            child_ys = [place(child_id, depth + 1) for child_id in children[concept_id]]
            y = sum(child_ys) / len(child_ys)
        positions[concept_id] = (depth, y)
        active.remove(concept_id)
        return y

    for root_id in roots:
        place(root_id, 0)
        next_leaf_y += 0.8
    if visited != set(concepts):
        remaining = sorted(set(concepts) - visited)
        raise ValueError(f"Concept hierarchy contains a cycle or disconnected cycle: {remaining[:10]}.")

    figure_height = max(6.0, 0.34 * max(next_leaf_y, 1) + 1.5)
    figure_width = max(16.0, 4.7 * max_depth + 9.0)
    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    for parent_id, child_ids in children.items():
        parent_x, parent_y = positions[parent_id]
        for child_id in child_ids:
            child_x, child_y = positions[child_id]
            highlighted = child_id in surviving_paths and parent_id in surviving_paths
            color = "#1d4ed8" if highlighted else "#cbd5e1"
            width = 3.2 if highlighted else 1.15
            mid_x = (parent_x + child_x) / 2
            ax.plot([parent_x + 0.08, mid_x], [parent_y, parent_y], color=color, linewidth=width, zorder=1)
            ax.plot([mid_x, mid_x], [parent_y, child_y], color=color, linewidth=width, zorder=1)
            ax.plot([mid_x, child_x - 0.08], [child_y, child_y], color=color, linewidth=width, zorder=1)

    for concept_id, (x, y) in positions.items():
        is_survivor = concept_id in survivor_ids
        is_path = concept_id in surviving_paths
        is_dead_end = not children[concept_id] and not is_survivor
        color = "#15803d" if is_survivor else "#2563eb" if is_path else "#dc6b19" if is_dead_end else "#64748b"
        size = 110 if is_survivor else 75 if is_path else 58
        ax.scatter([x], [y], s=size, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        name = str(concepts[concept_id].get("name", concept_id))
        wrapped_name = "\n".join(textwrap.wrap(name, width=28) or [name])
        suffix = "  ★" if is_survivor else "  ×" if is_dead_end else ""
        ax.annotate(f"{node_labels[concept_id]}  {wrapped_name}{suffix}", (x, y),
                    xytext=(9, 0), textcoords="offset points", va="center",
                    fontsize=8, linespacing=1.15, fontweight="bold" if is_survivor else "normal",
                    color="#14532d" if is_survivor else "#1f2937", zorder=4)

    ax.set_title("Concept refinement phylogeny — blue paths lead to final Pareto survivors", loc="left", pad=16)
    ax.set_xlabel("Refinement depth from root")
    ax.set_xticks(range(max_depth + 1), ["Root" if depth == 0 else str(depth) for depth in range(max_depth + 1)])
    ax.set_ylim(next_leaf_y - 0.5, -0.5)
    ax.set_yticks([])
    ax.set_xlim(-0.25, max_depth + 1.8)
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.8)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    parent_ids = {
        concept_id: str(concepts[concept_id].get("parent_id") or "")
        for concept_id in concepts
    }
    node_index = pd.DataFrame([
        {
            "node_label": node_labels[concept_id],
            "concept_id": concept_id,
            "name": str(concepts[concept_id].get("name", concept_id)),
            "definition": str(concepts[concept_id].get("definition", "")),
            "parent_id": parent_ids[concept_id],
            "depth": int(positions[concept_id][0]),
            "is_survivor": concept_id in survivor_ids,
            "on_path_to_survivor": concept_id in surviving_paths,
            "is_dead_end": not children[concept_id] and concept_id not in survivor_ids,
        }
        for concept_id in sorted(concepts, key=lambda item: node_labels[item])
    ])
    node_index.to_csv(output_path.with_name(f"{output_path.stem}_nodes.csv"), index=False)
    return output_path

def run_automated_concept_refinement(
    project_root: str | Path,
    parameters: Mapping[str, Any],
    *,
    plot_callback: Callable[[Path], None] | None = None,
) -> dict[str, Any]:
    """Run notebook 10f from exposed parameters, preserving all resumable artifacts."""
    project_root = Path(project_root).resolve()
    PIPELINE_CONFIG = load_pipeline_config(project_root=project_root)
    CONCEPT_PATHS = build_si_concept_paths(project_root)
    YEARLY_SCREENED_PATH = parameters["YEARLY_SCREENED_PATH"]
    ABSTRACT_CACHE_PATH = parameters["ABSTRACT_CACHE_PATH"]
    SI_PATH = parameters["SI_PATH"]
    PATHOGEN_PRIORITY_PATH = parameters["PATHOGEN_PRIORITY_PATH"]
    MIN_PATHOGENIC_INDEX = parameters["MIN_PATHOGENIC_INDEX"]
    NUM_PATHOGENS = parameters["NUM_PATHOGENS"]
    HELDOUT_FRACTION = parameters["HELDOUT_FRACTION"]
    SPLIT_SEED = parameters["SPLIT_SEED"]
    RESET_CATALOG_TO_SEEDS = parameters["RESET_CATALOG_TO_SEEDS"]
    JEV_MODEL = parameters["JEV_MODEL"]
    MAX_WORKERS = parameters["MAX_WORKERS"]
    MAX_CLASSIFICATION_CALLS = parameters["MAX_CLASSIFICATION_CALLS"]
    CHILD_GENERATION_MODEL = parameters["CHILD_GENERATION_MODEL"]
    LLM_TEMPERATURE = parameters["LLM_TEMPERATURE"]
    LLM_MAX_TOKENS = parameters["LLM_MAX_TOKENS"]
    MAX_EXAMPLES_PER_PARENT = parameters["MAX_EXAMPLES_PER_PARENT"]
    PROPOSAL_EXAMPLE_SEED = parameters["PROPOSAL_EXAMPLE_SEED"]
    POSITIVE_EXAMPLE_THRESHOLD = parameters["POSITIVE_EXAMPLE_THRESHOLD"]
    FIT_START_YEAR = parameters["FIT_START_YEAR"]
    FORECAST_ORIGIN_END_YEAR = parameters["FORECAST_ORIGIN_END_YEAR"]
    LAG_LENGTH = parameters["LAG_LENGTH"]
    FORECAST_HORIZONS = parameters["FORECAST_HORIZONS"]
    MIN_PRIOR_OBSERVATIONS = parameters["MIN_PRIOR_OBSERVATIONS"]
    FORECAST_PROGRESS_EVERY = parameters["FORECAST_PROGRESS_EVERY"]
    FRONTIER_DISPLAY_ROWS = parameters["FRONTIER_DISPLAY_ROWS"]
    N_ITER = parameters["N_ITER"]
    K_PARENTS = parameters["K_PARENTS"]
    NUM_CHILDREN = parameters["NUM_CHILDREN"]
    RUN_AUTOMATED_CYCLE = parameters["RUN_AUTOMATED_CYCLE"]

    if not 0 < HELDOUT_FRACTION < 1:
        raise ValueError("HELDOUT_FRACTION must be strictly between 0 and 1.")
    if NUM_PATHOGENS is not None and (not isinstance(NUM_PATHOGENS, int) or NUM_PATHOGENS < 1):
        raise ValueError("NUM_PATHOGENS must be a positive integer or None.")
    integer_parameters = {
        "N_ITER": N_ITER, "K_PARENTS": K_PARENTS, "NUM_CHILDREN": NUM_CHILDREN,
        "LAG_LENGTH": LAG_LENGTH, "MIN_PRIOR_OBSERVATIONS": MIN_PRIOR_OBSERVATIONS,
        "MAX_EXAMPLES_PER_PARENT": MAX_EXAMPLES_PER_PARENT, "MAX_WORKERS": MAX_WORKERS,
    }
    if any(not isinstance(value, int) or isinstance(value, bool) for value in integer_parameters.values()):
        raise ValueError(f"These parameters must be integers: {integer_parameters}")
    if any(value < 1 for value in integer_parameters.values()):
        raise ValueError(f"These parameters must be positive: {integer_parameters}")
    if not FORECAST_HORIZONS or any(not isinstance(h, int) or isinstance(h, bool) or h < 1 for h in FORECAST_HORIZONS):
        raise ValueError("FORECAST_HORIZONS must contain positive integers.")
    if len(set(FORECAST_HORIZONS)) != len(FORECAST_HORIZONS):
        raise ValueError("FORECAST_HORIZONS must not contain duplicates.")
    if FIT_START_YEAR > FORECAST_ORIGIN_END_YEAR:
        raise ValueError("FIT_START_YEAR must be no later than FORECAST_ORIGIN_END_YEAR.")
    if not 0 <= POSITIVE_EXAMPLE_THRESHOLD <= 1:
        raise ValueError("POSITIVE_EXAMPLE_THRESHOLD must be between 0 and 1.")
    if MAX_CLASSIFICATION_CALLS is not None and (
        not isinstance(MAX_CLASSIFICATION_CALLS, int) or isinstance(MAX_CLASSIFICATION_CALLS, bool)
        or MAX_CLASSIFICATION_CALLS < 0
    ):
        raise ValueError("MAX_CLASSIFICATION_CALLS must be a nonnegative integer or None.")

    corpus, pair_split = load_concept_corpus(
        screened_path=YEARLY_SCREENED_PATH,
        abstracts_path=ABSTRACT_CACHE_PATH,
        si_path=SI_PATH,
        heldout_fraction=HELDOUT_FRACTION,
        split_seed=SPLIT_SEED,
        pathogen_priority_path=PATHOGEN_PRIORITY_PATH,
        num_pathogens=NUM_PATHOGENS,
        min_pathogenic_index=MIN_PATHOGENIC_INDEX,
    )
    CONCEPT_PATHS["directory"].mkdir(parents=True, exist_ok=True)
    abstract_table = corpus[["pmid", "title", "abstract", "abstract_available"]].drop_duplicates("pmid")
    corpus.drop(columns=["title", "abstract"]).to_parquet(CONCEPT_PATHS["corpus"], index=False)
    abstract_table.to_parquet(CONCEPT_PATHS["abstracts"], index=False)
    pair_split.to_csv(CONCEPT_PATHS["pair_split"], index=False)
    preflight = corpus_preflight(corpus)

    CONCEPT_PATHS["catalog_versions"].mkdir(parents=True, exist_ok=True)
    reference_catalog = [dict(concept) for concept in SEED_CONCEPTS]
    REFERENCE_CATALOG_VERSION = concept_catalog_hash(reference_catalog)
    reference_path = CONCEPT_PATHS["catalog_versions"] / f"concept_catalog_{REFERENCE_CATALOG_VERSION}.json"
    if not reference_path.exists():
        reference_path.write_text(json.dumps(reference_catalog, indent=2, ensure_ascii=False))

    if RESET_CATALOG_TO_SEEDS or not CONCEPT_PATHS["catalog"].exists():
        catalog = [dict(concept) for concept in reference_catalog]
        CONCEPT_PATHS["catalog"].write_text(json.dumps(catalog, indent=2, ensure_ascii=False))
    else:
        catalog = json.loads(CONCEPT_PATHS["catalog"].read_text())

    # Keep an immutable snapshot of this run's starting definition; `catalog` below
    # is the mutable working copy that receives children in each iteration.
    STARTING_CATALOG_VERSION = concept_catalog_hash(catalog)
    starting_catalog_path = CONCEPT_PATHS["catalog_versions"] / f"concept_catalog_{STARTING_CATALOG_VERSION}.json"
    if not starting_catalog_path.exists():
        starting_catalog_path.write_text(json.dumps(catalog, indent=2, ensure_ascii=False))

    si_df = pd.read_parquet(SI_PATH)
    split = pair_split.copy()
    split[["pathogen", "antibiotic"]] = split[["pathogen", "antibiotic"]].astype(str)
    development_pairs = set(map(tuple, split.loc[split["split"].eq("development"), ["pathogen", "antibiotic"]].to_numpy()))
    development_pmids = set(corpus.loc[corpus["split"].eq("development"), "pmid"].astype(str))
    confirmation_pmids = set(corpus.loc[corpus["split"].eq("confirmation"), "pmid"].astype(str))
    development_only_pmids = development_pmids - confirmation_pmids
    abstract_table["pmid"] = abstract_table["pmid"].astype(str)
    corpus["pmid"] = corpus["pmid"].astype(str)
    abstracts = abstract_table.loc[
        abstract_table["abstract_available"].astype(bool), ["pmid", "title", "abstract"]
    ].drop_duplicates("pmid")
    abstracts = abstracts.loc[abstracts["abstract"].fillna("").astype(str).str.strip().ne("")]
    if abstracts.empty:
        raise RuntimeError("The selected corpus has no nonblank abstracts to classify; concept signals cannot be evaluated.")

    # Estimate actual pending Jev requests from the concept-definition cache. Jev
    # receives one request per abstract with one or more missing concept questions.
    abstracts["pmid"] = abstracts["pmid"].map(
        lambda value: str(value).strip()[:-2]
        if str(value).strip().endswith(".0") and str(value).strip()[:-2].isdigit()
        else str(value).strip()
    )
    definition_versions = {
        str(concept["concept_id"]): concept_definition_hash(concept)
        for concept in catalog
    }
    cached_by_concept = {concept_id: set() for concept_id in definition_versions}
    if CONCEPT_PATHS["scores"].exists():
        cached_scores = pd.read_parquet(CONCEPT_PATHS["scores"])
        required_cache_columns = {
            "pmid", "concept_id", "concept_version", "model_name", "prompt_version", "status",
        }
        if required_cache_columns.issubset(cached_scores.columns):
            reusable = cached_scores.loc[
                cached_scores["model_name"].astype(str).eq(JEV_MODEL)
                & cached_scores["prompt_version"].astype(str).eq(JEV_CLASSIFICATION_PROMPT_VERSION)
                & cached_scores["status"].eq("ok")
            ]
            for concept_id, concept_version in definition_versions.items():
                concept_rows = reusable.loc[
                    reusable["concept_id"].astype(str).eq(concept_id)
                    & reusable["concept_version"].astype(str).eq(concept_version), "pmid"
                ]
                cached_by_concept[concept_id] = set(concept_rows.astype(str))
    pending_jev_pmids = {
        pmid for pmid in abstracts["pmid"].astype(str)
        if any(pmid not in cached_by_concept[concept_id] for concept_id in definition_versions)
    }
    del cached_by_concept
    CATALOG_VERSION = concept_catalog_hash(catalog)
    CORPUS_KEY = hashlib.sha256(
        ("|".join(sorted(corpus["pmid"].astype(str).unique())) + "|" +
         "|".join(f"{p}\t{a}\t{s}" for p, a, s in sorted(map(tuple, split[["pathogen", "antibiotic", "split"]].to_numpy())))).encode("utf-8")
    ).hexdigest()[:12]
    horizon_tag = "-".join(map(str, FORECAST_HORIZONS))
    run_configuration = {
        "corpus_key": CORPUS_KEY,
        "catalog_version": CATALOG_VERSION,
        "jev_model": JEV_MODEL,
        "child_generation_model": CHILD_GENERATION_MODEL,
        "llm_temperature": LLM_TEMPERATURE,
        "llm_max_tokens": LLM_MAX_TOKENS,
        "max_examples_per_parent": MAX_EXAMPLES_PER_PARENT,
        "positive_example_threshold": POSITIVE_EXAMPLE_THRESHOLD,
        "proposal_example_seed": PROPOSAL_EXAMPLE_SEED,
        "fit_start_year": FIT_START_YEAR,
        "forecast_origin_end_year": FORECAST_ORIGIN_END_YEAR,
        "lag_length": LAG_LENGTH,
        "forecast_horizons": list(FORECAST_HORIZONS),
        "min_prior_observations": MIN_PRIOR_OBSERVATIONS,
        "n_iter": N_ITER,
        "k_parents": K_PARENTS,
        "num_children": NUM_CHILDREN,
    }
    RUN_KEY = hashlib.sha256(json.dumps(run_configuration, sort_keys=True).encode("utf-8")).hexdigest()[:8]
    RUN_DIR = CONCEPT_PATHS["automated_refinement"] / (
        f"corpus_{CORPUS_KEY}_catalog_{CATALOG_VERSION}_n{N_ITER}_k{K_PARENTS}_c{NUM_CHILDREN}"
        f"_fit{FIT_START_YEAR}-{FORECAST_ORIGIN_END_YEAR}_l{LAG_LENGTH}_h{horizon_tag}_{RUN_KEY}"
    )
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    assert corpus[["pathogen", "antibiotic", "year", "pmid"]].duplicated().sum() == 0
    assert set(pair_split["split"]) == {"development", "confirmation"}
    print({
        "selected_pathogens": int(corpus["pathogen"].nunique()),
        "pathogen_limit": NUM_PATHOGENS,
        "all_antibiotics_retained": True,
        "pair_year_pmid_associations": preflight["pair_year_pmid_associations"],
        "unique_pmids": preflight["unique_pmids"],
        "unique_nonblank_abstracts_to_classify": preflight["unique_abstracts_to_classify"],
        "abstract_coverage": preflight["abstract_coverage"],
        "full_corpus_jev_request_upper_bound_if_uncached": preflight["unique_abstracts_to_classify"],
        "estimated_pending_jev_requests_after_cache": len(pending_jev_pmids),
        "abstracts_with_all_current_concept_labels_cached": len(abstracts) - len(pending_jev_pmids),
        "development_pairs": len(development_pairs),
        "confirmation_pairs": int(split["split"].eq("confirmation").sum()),
        "starting_catalog_version": CATALOG_VERSION,
        "starting_catalog_reference": str(starting_catalog_path),
        "seed_reference_version": REFERENCE_CATALOG_VERSION,
        "starting_concepts": len(catalog),
        "run_directory": str(RUN_DIR),
    })

    if not RUN_AUTOMATED_CYCLE:
        print("Automated refinement is disabled. Set RUN_AUTOMATED_CYCLE=True to run the cycle.")
        return {"status": "disabled"}
    else:
        from amr.llm import OpenAIConfig, OpenAILanguageModel

        def progress(message):
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {message}", flush=True)

        llm = OpenAILanguageModel(OpenAIConfig(model=CHILD_GENERATION_MODEL, temperature=LLM_TEMPERATURE, max_tokens=LLM_MAX_TOKENS))
        jev = OpenRouterJevClient(model=JEV_MODEL)
        fit_start = FIT_START_YEAR
        fit_end = FORECAST_ORIGIN_END_YEAR
        iteration_plot_paths = []
        # When continuing from a saved working catalog, do not expand parents that
        # already have descendants; select their children for any further refinement.
        expanded_parent_ids = {
            str(concept["parent_id"])
            for concept in catalog
            if concept.get("parent_id") not in (None, "")
        }
        evaluated_concept_ids = [str(concept["concept_id"]) for concept in catalog]
        all_population_summaries = []
        all_pair_summaries = []
        round_log = []

        def save_catalog_snapshot(current_catalog):
            version = concept_catalog_hash(current_catalog)
            CONCEPT_PATHS["catalog_versions"].mkdir(parents=True, exist_ok=True)
            snapshot = CONCEPT_PATHS["catalog_versions"] / f"concept_catalog_{version}.json"
            snapshot.write_text(json.dumps(current_catalog, indent=2, ensure_ascii=False))
            CONCEPT_PATHS["catalog"].write_text(json.dumps(current_catalog, indent=2, ensure_ascii=False))
            return version

        def classify_and_signal(current_catalog, stage_label):
            version = concept_catalog_hash(current_catalog)
            progress(
                f"{stage_label}: checking Jev cache/classification for "
                f"{len(abstracts):,} nonblank abstracts × {len(current_catalog)} concepts."
            )
            scores = classify_concept_abstracts_jev(
                abstracts, current_catalog, jev_client=jev, model_name=JEV_MODEL,
                cache_path=CONCEPT_PATHS["scores"], max_calls=MAX_CLASSIFICATION_CALLS, max_workers=MAX_WORKERS,
            )
            # Never turn unclassified abstracts into implicit zero-prevalence evidence.
            nonblank_pmids = set(
                abstracts.loc[abstracts["abstract"].fillna("").astype(str).str.strip().ne(""), "pmid"].astype(str)
            )
            expected_labels = {
                (pmid, str(concept["concept_id"]))
                for pmid in nonblank_pmids for concept in current_catalog
            }
            successful = scores.loc[
                scores["catalog_version"].astype(str).eq(version)
                & scores["model_name"].astype(str).eq(JEV_MODEL)
                & scores["status"].eq("ok"), ["pmid", "concept_id"]
            ]
            scored_labels = set(zip(successful["pmid"].astype(str), successful["concept_id"].astype(str)))
            missing_labels = expected_labels - scored_labels
            if missing_labels:
                raise RuntimeError(
                    f"Classification is incomplete for {len(missing_labels):,} abstract/concept labels "
                    f"({len(nonblank_pmids):,} nonblank abstracts × {len(current_catalog)} concepts expected). "
                    "No forecasts were run with partial labels. Rerun this cell to resume from the cache, "
                    "or set MAX_CLASSIFICATION_CALLS=None to finish all pending labels."
                )
            progress(f"{stage_label}: all required Jev labels are available; building yearly concept signals.")
            signals = build_yearly_concept_signals(
                corpus, scores, catalog_version=version, model_name=JEV_MODEL,
            )
            signals.to_parquet(CONCEPT_PATHS["signals"], index=False)
            progress(f"{stage_label}: signals ready ({len(signals):,} pair-year/concept rows).")
            return scores, signals, version

        def evaluate_concepts(concept_ids, label, signals, survivor_population=None, children_by_parent=None):
            concept_ids = list(dict.fromkeys(map(str, concept_ids)))
            survivor_population = list(dict.fromkeys(map(str, survivor_population or [])))
            children_by_parent = children_by_parent or {}
            model_sets = {f"concept:{cid}": [cid] for cid in concept_ids}
            comparisons = {f"concept:{cid}": "baseline" for cid in concept_ids}

            # Conditional contribution: compare the selected population with and without
            # each member. This asks whether a concept adds information beyond its peers.
            if survivor_population:
                for concept_id in survivor_population:
                    full_name = f"conditional_with:{label}:{concept_id}"
                    without_name = f"conditional_without:{label}:{concept_id}"
                    model_sets[full_name] = list(survivor_population)
                    model_sets[without_name] = [cid for cid in survivor_population if cid != concept_id]
                    comparisons[full_name] = without_name

                # Nested hierarchy: hold the other selected concepts fixed while replacing
                # each parent with its child set; also report each child replacement alone.
                for parent_id, child_ids in children_by_parent.items():
                    child_ids = list(dict.fromkeys(map(str, child_ids)))
                    background = [cid for cid in survivor_population if cid != parent_id]
                    parent_name = f"nested_parent:{label}:{parent_id}"
                    children_name = f"nested_children:{label}:{parent_id}"
                    model_sets[parent_name] = list(survivor_population)
                    model_sets[children_name] = [*background, *child_ids]
                    comparisons[children_name] = parent_name
                    for child_id in child_ids:
                        single_name = f"nested_single_child:{label}:{parent_id}:{child_id}"
                        model_sets[single_name] = [*background, child_id]
                        comparisons[single_name] = parent_name

            model_concepts = list(dict.fromkeys(
                concept_ids
                + survivor_population
                + [child for children in children_by_parent.values() for child in children]
            ))
            forecasts = rolling_origin_concept_forecasts(
                si_df, signals, concept_ids=model_concepts, model_sets=model_sets,
                development_pairs=development_pairs, fit_start_year=fit_start, fit_end_year=fit_end,
                lag_length=LAG_LENGTH, horizons=FORECAST_HORIZONS, min_prior_observations=MIN_PRIOR_OBSERVATIONS,
                progress_every=FORECAST_PROGRESS_EVERY,
            )
            if forecasts.empty:
                return forecasts, pd.DataFrame(), pd.DataFrame()
            forecasts = forecasts.merge(split, on=["pathogen", "antibiotic"], how="left", validate="many_to_one")
            forecasts = forecasts.loc[forecasts["split"].eq("development")].drop(columns="split")
            pair_summary, population_summary = summarise_incremental_forecast_gains(forecasts, comparisons)
            safe_label = re.sub(r"[^A-Za-z0-9_-]+", "_", label)
            forecasts.to_parquet(RUN_DIR / f"{safe_label}_forecasts.parquet", index=False)
            pair_summary.to_parquet(RUN_DIR / f"{safe_label}_pair_summary.parquet", index=False)
            population_summary.to_parquet(RUN_DIR / f"{safe_label}_population_summary.parquet", index=False)
            return forecasts, pair_summary, population_summary

        def plot_iteration_progress(
            iteration, population_summary, selected_ids, new_child_ids=(),
            replacement_summary=None, concept_names=None,
        ):
            """Plot average and specialist MSE gains for the concepts considered so far."""
            import matplotlib.pyplot as plt

            concept_names = concept_names or {}
            fig, ax = plt.subplots(figsize=(10, 7))
            standalone = population_summary.loc[
                pd.to_numeric(population_summary["horizon"], errors="coerce").eq(0)
                & population_summary["model"].astype(str).str.startswith("concept:")
            ].copy()
            if not standalone.empty:
                standalone["concept_id"] = standalone["model"].astype(str).str.removeprefix("concept:")
                standalone = standalone.drop_duplicates("concept_id", keep="last")
                selected_set = set(map(str, selected_ids))
                child_set = set(map(str, new_child_ids))
                groups = [
                    ("Other evaluated concepts", standalone.loc[~standalone["concept_id"].isin(selected_set | child_set)], "#9ca3af", "o"),
                    ("Selected parents", standalone.loc[standalone["concept_id"].isin(selected_set)], "#2563eb", "s"),
                    ("New children", standalone.loc[standalone["concept_id"].isin(child_set)], "#16a34a", "o"),
                ]
                for label, rows, color, marker_style in groups:
                    if rows.empty:
                        continue
                    ax.scatter(
                        rows["generality"], rows["specialist_strength"],
                        label=label, color=color, marker=marker_style, s=75, alpha=0.85,
                    )
                    if label != "Other evaluated concepts":
                        for row in rows.itertuples(index=False):
                            concept_id = str(row.concept_id)
                            display_name = concept_names.get(concept_id, concept_id)
                            ax.annotate(display_name, (row.generality, row.specialist_strength),
                                        xytext=(5, 4), textcoords="offset points", fontsize=8)

            if replacement_summary is not None and not replacement_summary.empty:
                nested = replacement_summary.loc[
                    pd.to_numeric(replacement_summary["horizon"], errors="coerce").eq(0)
                    & replacement_summary["model"].astype(str).str.startswith("nested_children:")
                ]
                if not nested.empty:
                    ax.scatter(
                        nested["generality"], nested["specialist_strength"],
                        label="Parent replaced by child set", color="#9333ea", marker="D", s=90,
                    )
                    for row in nested.itertuples(index=False):
                        parent_id = str(row.model).rsplit(":", 1)[-1]
                        parent_name = concept_names.get(parent_id, parent_id)
                        ax.annotate(f"children of {parent_name}",
                                    (row.generality, row.specialist_strength),
                                    xytext=(5, -10), textcoords="offset points", fontsize=8)

            ax.axhline(0, color="#555555", linewidth=0.8, linestyle="--")
            ax.axvline(0, color="#555555", linewidth=0.8, linestyle="--")
            ax.set_xlabel("Generality: mean pair-level MSE gain")
            ax.set_ylabel("Specialist strength: 90th-percentile pair-level MSE gain")
            ax.set_title(f"Iteration {iteration}: development concept gains")
            ax.grid(alpha=0.2)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(loc="best", frameon=True)
            fig.tight_layout()
            plot_path = RUN_DIR / f"iteration_{iteration:02d}_progress.png"
            fig.savefig(plot_path, dpi=160, bbox_inches="tight")
            iteration_plot_paths.append(plot_path)
            if plot_callback is not None:
                plot_callback(plot_path)
            plt.close(fig)
            progress(f"Iteration {iteration}: progress plot saved to {plot_path}.")

        # A. Classify and evaluate the starting catalog; concept-specific cache hits are reused.
        save_catalog_snapshot(catalog)
        progress(f"Starting catalog evaluation: {len(evaluated_concept_ids)} concepts; catalog {CATALOG_VERSION}.")
        scores, signals, current_catalog_version = classify_and_signal(catalog, "Initial catalog")
        progress("Initial catalog: beginning rolling-origin forecast evaluation.")
        _, initial_pair_summary, initial_population_summary = evaluate_concepts(
            evaluated_concept_ids, "initial_catalog", signals,
        )
        if initial_population_summary.empty:
            raise RuntimeError("No development forecast summaries were produced; inspect SI coverage and fit years.")
        all_pair_summaries.append(initial_pair_summary)
        all_population_summaries.append(initial_population_summary)
        progress(
            f"Initial catalog evaluation complete: {len(initial_pair_summary):,} pair/horizon rows; "
            f"{len(initial_population_summary):,} population summary rows."
        )

        # B. Expand the strongest unexpanded concepts from the development Pareto frontier.
        for iteration in range(1, N_ITER + 1):
            progress(f"Iteration {iteration}/{N_ITER}: combining current concept results and selecting parents.")
            combined_population = pd.concat(all_population_summaries, ignore_index=True)
            frontier = pareto_concept_frontier(
                combined_population, horizon=0, excluded_concepts=expanded_parent_ids,
            )
            frontier_path = RUN_DIR / f"iteration_{iteration:02d}_pareto_frontier.csv"
            frontier.to_csv(frontier_path, index=False)
            eligible_frontier = frontier.loc[frontier["on_pareto_front"] & frontier["eligible_for_expansion"]].sort_values(
                ["generality", "specialist_strength"], ascending=[False, False],
            )
            survivor_ids = frontier.loc[frontier["on_pareto_front"], "concept_id"].astype(str).tolist()
            selected_parents = eligible_frontier.head(K_PARENTS)
            parent_ids = selected_parents["concept_id"].astype(str).tolist()
            frontier_names = {str(concept["concept_id"]): str(concept.get("name", concept["concept_id"])) for concept in catalog}
            selected_parent_records = [
                {
                    "concept_id": str(row["concept_id"]),
                    "name": frontier_names.get(str(row["concept_id"]), str(row["concept_id"])),
                    "generality": float(row["generality"]),
                    "specialist_strength": float(row["specialist_strength"]),
                }
                for _, row in selected_parents.iterrows()
            ]
            progress(
                f"Iteration {iteration}/{N_ITER}: {int(frontier['on_pareto_front'].sum())} concepts on the Pareto frontier; "
                f"{len(eligible_frontier)} are eligible for expansion."
            )
            if not parent_ids:
                progress(f"Iteration {iteration}/{N_ITER}: no unexpanded Pareto concepts remain; stopping early.")
                current_names = {str(concept["concept_id"]): str(concept.get("name", concept["concept_id"])) for concept in catalog}
                plot_iteration_progress(iteration, combined_population, [], concept_names=current_names)
                round_log.append({
                    "iteration": iteration,
                    "selected_parents": [],
                    "parents": [],
                    "children": [],
                    "status": "no_eligible_parents",
                })
                (RUN_DIR / "iteration_history.json").write_text(json.dumps(round_log, indent=2, ensure_ascii=False))
                break
            progress(f"Iteration {iteration}/{N_ITER}: selected {len(parent_ids)} parent(s) for expansion:")
            for rank, (_, row) in enumerate(selected_parents.iterrows(), start=1):
                concept_id = str(row["concept_id"])
                progress(
                    f"  Parent {rank}/{len(parent_ids)}: {frontier_names.get(concept_id, concept_id)} [{concept_id}] "
                    f"| average gain={row['generality']:.6g} "
                    f"| q90 specialist gain={row['specialist_strength']:.6g}"
                )

            current_by_id = {str(concept["concept_id"]): concept for concept in catalog}
            existing_ids = set(current_by_id)
            existing_definitions = {
                re.sub(r"\s+", " ", str(concept.get("definition", "")).strip().lower())
                for concept in catalog
            }
            new_children = []
            parent_results = []
            for parent_rank, parent_id in enumerate(parent_ids, start=1):
                parent = current_by_id[parent_id]
                progress(
                    f"Iteration {iteration}/{N_ITER}: parent {parent_rank}/{len(parent_ids)} "
                    f"{parent.get('name', parent_id)} [{parent_id}] — preparing development-only examples."
                )
                parent_scores = scores.loc[
                    scores["catalog_version"].astype(str).eq(current_catalog_version)
                    & scores["model_name"].astype(str).eq(JEV_MODEL)
                    & scores["status"].eq("ok")
                    & scores["concept_id"].astype(str).eq(parent_id)
                    & scores["pmid"].astype(str).isin(development_only_pmids)
                    & pd.to_numeric(scores["probability"], errors="coerce").ge(POSITIVE_EXAMPLE_THRESHOLD)
                ].drop_duplicates("pmid", keep="last")
                examples_df = (
                    parent_scores[["pmid"]].merge(abstract_table, on="pmid", how="inner", validate="one_to_one")
                    .loc[lambda frame: frame["abstract_available"].astype(bool)]
                    .sample(frac=1, random_state=PROPOSAL_EXAMPLE_SEED + iteration - 1)
                    .head(MAX_EXAMPLES_PER_PARENT)
                )
                if examples_df.empty:
                    progress(f"Iteration {iteration}: {parent_id} has no positive development-only examples; skipped.")
                    expanded_parent_ids.add(parent_id)
                    parent_results.append({"parent_id": parent_id, "children": [], "status": "no_examples"})
                    continue
                proposal_path = RUN_DIR / f"iteration_{iteration:02d}_proposal_{parent_id}.json"
                if proposal_path.exists():
                    progress(f"Iteration {iteration}: loading saved proposal for {parent_id} from {proposal_path.name}.")
                    proposal = json.loads(proposal_path.read_text())
                else:
                    progress(f"Iteration {iteration}: requesting up to {NUM_CHILDREN} child hypotheses for {parent_id}.")
                    try:
                        proposal = propose_concept_children(
                            parent, examples_df[["title", "abstract"]].to_dict("records"),
                            llm=llm, model_name=CHILD_GENERATION_MODEL, num_children=NUM_CHILDREN,
                        )
                    except (RuntimeError, ValueError) as exc:
                        error_path = RUN_DIR / f"iteration_{iteration:02d}_proposal_{parent_id}.error.json"
                        error_path.write_text(json.dumps({
                            "parent_id": parent_id,
                            "model": CHILD_GENERATION_MODEL,
                            "error": str(exc),
                        }, indent=2, ensure_ascii=False))
                        progress(
                            f"Iteration {iteration}: proposal for {parent_id} failed ({exc}). "
                            f"Recorded at {error_path.name}; continuing with other parents. "
                            "A later run can retry this parent."
                        )
                        expanded_parent_ids.add(parent_id)
                        parent_results.append({
                            "parent_id": parent_id, "children": [], "status": "proposal_error",
                            "error": str(exc), "error_file": str(error_path),
                        })
                        continue
                    proposal_path.write_text(json.dumps(proposal, indent=2, ensure_ascii=False))
                    progress(f"Iteration {iteration}: proposal saved for {parent_id}.")
                proposed_count = len(proposal.get("children", [])) if isinstance(proposal.get("children", []), list) else 0
                progress(f"Iteration {iteration}: proposal for {parent_id} contains {proposed_count} child candidate(s).")
                accepted_ids = []
                for child in proposal.get("children", [])[:NUM_CHILDREN]:
                    name = str(child.get("name", "")).strip()
                    definition = str(child.get("definition", "")).strip()
                    if not name or not definition:
                        continue
                    normalized_definition = re.sub(r"\s+", " ", definition.lower())
                    if normalized_definition in existing_definitions:
                        continue
                    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "child"
                    child_id = f"{parent_id}_{slug}"
                    suffix = 2
                    while child_id in existing_ids:
                        child_id = f"{parent_id}_{slug}_{suffix}"
                        suffix += 1
                    existing_ids.add(child_id)
                    existing_definitions.add(normalized_definition)
                    review_fields = (
                        "scientific_rationale",
                        "inclusion_criteria",
                        "exclusion_criteria",
                        "representative_evidence",
                        "closest_competing_concept",
                        "difference",
                        "expected_temporal_relationship",
                        "testable_hypothesis",
                        "rejection_conditions",
                    )
                    child_record = {
                        "concept_id": child_id,
                        "name": name,
                        "parent_id": parent_id,
                        "definition": definition,
                        "inclusion_criteria": child.get("inclusion_criteria", []),
                        "exclusion_criteria": child.get("exclusion_criteria", []),
                        "version": iteration,
                        "generation_model": CHILD_GENERATION_MODEL,
                        "refinement_iteration": iteration,
                    }
                    new_children.append(child_record)
                    accepted_ids.append(child_id)
                    print(f"\nIteration {iteration} — accepted child {child_id}")
                    for field in ("name", "definition", "parent_id", *review_fields):
                        value = child_record.get(field) if field in {"name", "definition", "parent_id"} else child.get(field)
                        if value is None or value == "":
                            value = "(not provided by the proposal)"
                        print(f"{field}: {json.dumps(value, ensure_ascii=False, indent=2)}")
                progress(
                    f"Iteration {iteration}: accepted {len(accepted_ids)} new child/children for {parent_id} "
                    f"({NUM_CHILDREN} requested)."
                )
                expanded_parent_ids.add(parent_id)
                parent_results.append({"parent_id": parent_id, "children": accepted_ids, "status": "expanded"})

            if not new_children:
                progress(f"Iteration {iteration}/{N_ITER}: no new child concepts to classify; stopping this search.")
                current_names = {str(concept["concept_id"]): str(concept.get("name", concept["concept_id"])) for concept in catalog}
                plot_iteration_progress(
                    iteration, combined_population, parent_ids, concept_names=current_names,
                )
                round_log.append({
                    "iteration": iteration,
                    "selected_parents": selected_parent_records,
                    "parents": parent_results,
                    "children": [],
                    "status": "no_new_children",
                })
                (RUN_DIR / "iteration_history.json").write_text(json.dumps(round_log, indent=2, ensure_ascii=False))
                progress(f"Iteration {iteration}: no valid children were produced; continuing to the next Pareto parents if rounds remain.")
                continue
            catalog.extend(new_children)
            evaluated_concept_ids.extend(child["concept_id"] for child in new_children)
            current_catalog_version = save_catalog_snapshot(catalog)
            progress(f"Iteration {iteration}/{N_ITER}: saved {len(new_children)} new children; active catalog {current_catalog_version}.")

            # Existing concept labels are reused; only new child definitions require Jev questions.
            scores, signals, current_catalog_version = classify_and_signal(catalog, f"Iteration {iteration} child classification")
            progress(f"Iteration {iteration}/{N_ITER}: beginning forecast evaluation for {len(new_children)} new child concepts.")
            child_ids = [child["concept_id"] for child in new_children]
            children_by_parent = {
                result["parent_id"]: result["children"]
                for result in parent_results if result.get("children")
            }
            _, child_pair_summary, child_population_summary = evaluate_concepts(
                child_ids, f"iteration_{iteration:02d}_children", signals,
                survivor_population=survivor_ids, children_by_parent=children_by_parent,
            )
            if child_population_summary.empty:
                status = "children_not_evaluable"
                progress(f"Iteration {iteration}/{N_ITER}: no child development forecast summaries were available.")
            else:
                all_pair_summaries.append(child_pair_summary)
                all_population_summaries.append(child_population_summary)
                status = "evaluated"
                progress(
                    f"Iteration {iteration}/{N_ITER}: evaluation complete — "
                    f"{len(child_pair_summary):,} pair/horizon rows and "
                    f"{len(child_population_summary):,} population summary rows."
                )
            current_names = {str(concept["concept_id"]): str(concept.get("name", concept["concept_id"])) for concept in catalog}
            plot_iteration_progress(
                iteration,
                pd.concat(all_population_summaries, ignore_index=True),
                parent_ids,
                new_child_ids=child_ids if status == "evaluated" else (),
                replacement_summary=child_population_summary,
                concept_names=current_names,
            )
            round_log.append({
                "iteration": iteration,
                "catalog_version": current_catalog_version,
                "selected_parents": selected_parent_records,
                "parents": parent_results,
                "children": child_ids,
                "status": status,
            })
            for parent_result in parent_results:
                child_text = ", ".join(parent_result["children"]) if parent_result["children"] else "no accepted children"
                progress(
                    f"Iteration {iteration} result: {parent_result['parent_id']} "
                    f"({parent_result['status']}) → {child_text}"
                )
            (RUN_DIR / "iteration_history.json").write_text(json.dumps(round_log, indent=2, ensure_ascii=False))
            progress(f"Iteration {iteration}/{N_ITER}: round status = {status}; history saved.")
            if status != "evaluated":
                progress(f"Iteration {iteration}: child models were not evaluable; continuing to the next search round if available.")
                continue

        if all_population_summaries:
            population_results = pd.concat(all_population_summaries, ignore_index=True)
            pair_results = pd.concat(all_pair_summaries, ignore_index=True)
            population_results.to_parquet(RUN_DIR / "all_development_population_summaries.parquet", index=False)
            pair_results.to_parquet(RUN_DIR / "all_development_pair_summaries.parquet", index=False)
            final_frontier = pareto_concept_frontier(
                population_results, horizon=0, excluded_concepts=expanded_parent_ids,
            )
            final_frontier.to_csv(RUN_DIR / "final_unexpanded_pareto_frontier.csv", index=False)
            all_concept_frontier = pareto_concept_frontier(population_results, horizon=0)
            all_concept_frontier.to_csv(RUN_DIR / "final_pareto_survivors.csv", index=False)
            surviving_concept_ids = set(
                all_concept_frontier.loc[all_concept_frontier["on_pareto_front"], "concept_id"].astype(str)
            )
            phylogeny_path = _save_concept_phylogeny(
                catalog, surviving_concept_ids, RUN_DIR / "final_concept_phylogeny.png",
            )
            survivor_names = [
                str(concept.get("name", concept["concept_id"]))
                for concept in catalog if str(concept["concept_id"]) in surviving_concept_ids
            ]
            print("Final concept phylogeny: roots at left; × marks terminal dead ends; thick blue paths end at final Pareto survivors.")
            print(f"Full node names, IDs, definitions, and parent links: {phylogeny_path.with_name(f'{phylogeny_path.stem}_nodes.csv')}")
            print("Final Pareto survivors:", ", ".join(survivor_names) if survivor_names else "none")
            progress("Final Pareto frontier (with expansion status):")
        else:
            surviving_concept_ids = set()
            phylogeny_path = None
        progress(f"Automated development cycle complete. Catalog {current_catalog_version}; outputs: {RUN_DIR}")
        return {
            "run_directory": RUN_DIR,
            "catalog_version": current_catalog_version,
            "catalog": catalog,
            "round_log": round_log,
            "final_frontier": final_frontier if all_population_summaries else pd.DataFrame(),
            "population_results": population_results if all_population_summaries else pd.DataFrame(),
            "pair_results": pair_results if all_pair_summaries else pd.DataFrame(),
            "iteration_plot_paths": iteration_plot_paths,
            "phylogeny_path": phylogeny_path,
            "phylogeny_nodes_path": phylogeny_path.with_name(f"{phylogeny_path.stem}_nodes.csv") if phylogeny_path else None,
            "surviving_concept_ids": sorted(surviving_concept_ids),
            "preflight": preflight,
        }
