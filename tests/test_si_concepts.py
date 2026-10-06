import json

import numpy as np
import pandas as pd

from amr.si.concepts import (
    build_concept_corpus,
    build_yearly_concept_signals,
    classify_concept_abstracts,
    compare_parent_children,
    concept_catalog_hash,
    make_pair_split,
    rolling_origin_concept_forecasts,
    summarise_incremental_forecast_gains,
)


class FakeLLM:
    def __init__(self):
        self.calls = 0

    def complete(self, prompt, **kwargs):
        self.calls += 1
        assert "scientific interest" not in prompt.lower()
        return json.dumps({"scores": {"surveillance": 0.75}})


def test_pair_split_is_reproducible_and_pair_disjoint():
    pairs = [(f"Pathogen {index}", "Drug") for index in range(10)]
    first = make_pair_split(pairs, heldout_fraction=0.2, seed=42)
    second = make_pair_split(pairs, heldout_fraction=0.2, seed=42)
    pd.testing.assert_frame_equal(first, second)
    assert first["split"].value_counts().to_dict() == {"development": 8, "confirmation": 2}


def test_corpus_joins_once_per_pair_year_pmid_and_preserves_shared_paper():
    screened = pd.DataFrame([
        {"pathogen": "P1", "antibiotic": "A", "year": 2000, "pmids": ["10", "10", "11"]},
        {"pathogen": "P2", "antibiotic": "A", "year": 2000, "pmids": ["10"]},
    ])
    abstracts = pd.DataFrame([
        {"pmid": "10", "title": "shared", "abstract": "evidence"},
        {"pmid": "11", "title": "blank", "abstract": " "},
    ])
    si = pd.DataFrame([
        {"pathogen": "P1", "antibiotic": "A", "year": 2000, "SI": 0.1},
        {"pathogen": "P2", "antibiotic": "A", "year": 2000, "SI": 0.2},
    ])
    corpus, split = build_concept_corpus(screened, abstracts, si)
    assert len(corpus) == 3
    assert not corpus[["pathogen", "antibiotic", "year", "pmid"]].duplicated().any()
    assert corpus.loc[corpus["pmid"].eq("10"), "abstract_available"].all()
    assert not corpus.loc[corpus["pmid"].eq("11"), "abstract_available"].any()
    assert len(split) == 2


def test_classification_cache_resumes_and_bounds_probabilities(tmp_path):
    abstracts = pd.DataFrame([
        {"pmid": "10", "title": "title", "abstract": "text"},
        {"pmid": "10", "title": "duplicate", "abstract": "duplicate"},
        {"pmid": "11", "title": "missing", "abstract": ""},
    ])
    catalog = [{"concept_id": "surveillance", "name": "Surveillance", "definition": "monitoring"}]
    llm = FakeLLM()
    path = tmp_path / "scores.parquet"
    first = classify_concept_abstracts(abstracts, catalog, llm=llm, model_name="fake", cache_path=path)
    second = classify_concept_abstracts(abstracts, catalog, llm=llm, model_name="fake", cache_path=path)
    assert llm.calls == 1
    assert first.loc[first["status"].eq("ok"), "probability"].item() == 0.75
    assert len(second.loc[second["status"].eq("ok")]) == 1
    assert second["catalog_version"].eq(concept_catalog_hash(catalog)).all()
    revised_catalog = [{"concept_id": "mechanism", "name": "Mechanism", "definition": "biological explanation"}]
    classify_concept_abstracts(abstracts, revised_catalog, llm=llm, model_name="fake", cache_path=path)
    classify_concept_abstracts(abstracts, catalog, llm=llm, model_name="fake", cache_path=path)
    assert llm.calls == 2
    cached = pd.read_parquet(path)
    assert set(cached["catalog_version"]) == {concept_catalog_hash(catalog), concept_catalog_hash(revised_catalog)}


def test_yearly_signal_uses_classifiable_unique_articles_as_denominator():
    corpus = pd.DataFrame([
        {"pathogen": "P", "antibiotic": "A", "year": 2000, "pmid": "1", "abstract_available": True},
        {"pathogen": "P", "antibiotic": "A", "year": 2000, "pmid": "2", "abstract_available": True},
        {"pathogen": "P", "antibiotic": "A", "year": 2000, "pmid": "3", "abstract_available": False},
    ])
    scores = pd.DataFrame([
        {"pmid": "1", "concept_id": "c", "probability": 1.0, "catalog_version": "v", "status": "ok"},
        {"pmid": "2", "concept_id": "c", "probability": 0.0, "catalog_version": "v", "status": "ok"},
    ])
    signals = build_yearly_concept_signals(corpus, scores, catalog_version="v")
    assert signals.loc[0, "concept_prevalence"] == 0.5
    assert signals.loc[0, "abstract_articles"] == 2
    assert signals.loc[0, "classification_coverage"] == 1.0


def test_rolling_origin_forecasts_compare_direct_horizons_and_conditional_models():
    si_rows = []
    signal_rows = []
    for pair_index in range(6):
        for year in range(2000, 2021):
            signal = ((year - 2000 + pair_index) % 5) / 4
            si_value = 0.4 + 0.2 * np.sin((year - 2000) / 3) + 0.1 * signal
            si_rows.append({"pathogen": f"P{pair_index}", "antibiotic": "Drug", "year": year, "SI": si_value})
            signal_rows.append({"pathogen": f"P{pair_index}", "antibiotic": "Drug", "year": year,
                                "concept_id": "c", "concept_prevalence": signal})
    si = pd.DataFrame(si_rows)
    signals = pd.DataFrame(signal_rows)
    dev_pairs = {(f"P{index}", "Drug") for index in range(4)}
    models = {"population": ["c"], "population_plus:c": ["c"]}
    # Distinct model names allow the same signal to be wired as candidate/refinement in notebooks.
    forecasts = rolling_origin_concept_forecasts(
        si, signals, concept_ids=["c"], model_sets=models, development_pairs=dev_pairs,
        fit_start_year=2012, fit_end_year=2017, lag_length=3, horizons=(1, 2, 3),
        min_prior_observations=10,
    )
    assert set(forecasts["horizon"]) == {1, 2, 3}
    assert (forecasts["target_year"] == forecasts["origin_year"] + forecasts["horizon"]).all()
    assert forecasts["model"].nunique() == 3  # baseline plus both configured concept models
    assert forecasts["squared_error"].ge(0).all()
    pair_summary, population = summarise_incremental_forecast_gains(
        forecasts, {"population": "baseline", "population_plus:c": "population"}
    )
    assert {"generality", "specialist_strength"}.issubset(population.columns)
    assert (pair_summary["horizon"].isin([1, 2, 3])).all()


def test_parent_child_replacement_uses_matched_origins():
    base = pd.DataFrame([
        {"pathogen": "P", "antibiotic": "A", "horizon": h, "origin_year": 2000,
         "target_year": 2000 + h, "model": "parent:p", "squared_error": 2.0}
        for h in (1, 2, 3)
    ])
    children = base.copy()
    children["model"] = "children:p"
    children["squared_error"] = 1.0
    compared = compare_parent_children(
        pd.concat([base, children], ignore_index=True),
        pd.concat([base, children], ignore_index=True),
        parent_id="p",
    )
    assert len(compared) == 3
    assert compared["split_gain"].eq(1.0).all()
