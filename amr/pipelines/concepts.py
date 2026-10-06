"""Notebook-facing paths and loaders for SI concept discovery."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from amr.si.concepts import build_concept_corpus


def build_si_concept_paths(project_root: str | Path) -> dict[str, Path]:
    """Return the isolated artifact paths used by the concept notebooks."""
    root = Path(project_root)
    base = root / "outputs" / "concept_search"
    return {
        "directory": base,
        "pair_split": base / "pair_split.csv",
        "corpus": base / "pair_year_abstract_corpus.parquet",
        "abstracts": base / "unique_abstracts.parquet",
        "catalog": base / "concept_catalog.json",
        "catalog_versions": base / "catalog_versions",
        "automated_refinement": base / "automated_refinement",
        "scores": base / "concept_scores.parquet",
        "signals": base / "yearly_concept_signals.parquet",
        "forecasts": base / "development_forecasts.parquet",
        "pair_summary": base / "pair_forecast_summary.parquet",
        "population_summary": base / "population_forecast_summary.parquet",
        "confirmation_forecasts": base / "confirmation_forecasts.parquet",
        "confirmation_summary": base / "confirmation_summary.parquet",
        "refinement_proposals": base / "refinement_proposals",
        "figures": base / "figures",
    }


def load_concept_corpus(
    *, screened_path: str | Path, abstracts_path: str | Path, si_path: str | Path,
    heldout_fraction: float = 0.2, split_seed: int = 42,
    pathogen_priority_path: str | Path | None = None, num_pathogens: int | None = None,
    min_pathogenic_index: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load canonical SI artifacts and assemble the concept corpus."""
    for path in (screened_path, abstracts_path, si_path):
        if not Path(path).exists():
            raise FileNotFoundError(path)
    priority = None
    if num_pathogens is not None:
        if pathogen_priority_path is None or not Path(pathogen_priority_path).exists():
            raise FileNotFoundError(pathogen_priority_path or "pathogen_priority_path required for restricted runs")
        priority = pd.read_csv(pathogen_priority_path)
    return build_concept_corpus(
        pd.read_parquet(screened_path), pd.read_parquet(abstracts_path), pd.read_parquet(si_path),
        heldout_fraction=heldout_fraction, split_seed=split_seed,
        pathogen_priority_df=priority, num_pathogens=num_pathogens,
        min_pathogenic_index=min_pathogenic_index,
    )


def corpus_preflight(corpus: pd.DataFrame) -> dict[str, Any]:
    """Summarise classification volume and abstract coverage before API calls."""
    available = corpus.loc[corpus["abstract_available"].astype(bool)]
    unique_abstracts = available[["pmid", "title", "abstract"]].drop_duplicates("pmid")
    total_associations = int(corpus[["pathogen", "antibiotic", "year", "pmid"]].drop_duplicates().shape[0])
    return {
        "pair_year_pmid_associations": total_associations,
        "unique_pmids": int(corpus["pmid"].nunique()),
        "unique_abstracts_to_classify": int(unique_abstracts["pmid"].nunique()),
        "abstract_available_associations": int(available.shape[0]),
        "abstract_coverage": float(available.shape[0] / total_associations) if total_associations else 0.0,
        "development_pairs": int(corpus.loc[corpus["split"].eq("development"), ["pathogen", "antibiotic"]].drop_duplicates().shape[0]),
        "confirmation_pairs": int(corpus.loc[corpus["split"].eq("confirmation"), ["pathogen", "antibiotic"]].drop_duplicates().shape[0]),
    }


__all__ = ["build_si_concept_paths", "load_concept_corpus", "corpus_preflight"]
