# Pathogen Atlas — Antimicrobial Resistance (Refactored Workflow)

This project now uses a module-first workflow with two active notebooks:

1. `01_collect_yearly_data.ipynb`
2. `02_model_scientific_interest.ipynb`

All earlier exploratory notebooks are considered legacy and should remain archived.

## Active Workflow

### Notebook 1: `01_collect_yearly_data.ipynb`

Builds the yearly AMR dataset end-to-end:

- Collects pathogen-antibiotic PMIDs through PubMed interval queries.
- Converts pair-PMID associations to yearly counts from publication dates.
- Collects yearly background publication counts for pathogens and antibiotics.
- Runs one unified strict LLM pass (`AMR evidence` + `scientific-interest relevance`).
- Exports the canonical screened yearly dataset.

Primary APIs:

- `amr.pipelines.yearly_collection.collect_yearly_amr_dataset(...)`
- `amr.pipelines.yearly_collection.collect_yearly_background_counts(...)`
- `amr.filtering.prepare_yearly_candidate_pmids(...)`
- `amr.filtering.run_unified_llm_filter(...)`
- `amr.filtering.build_yearly_screened_pairs(...)`

### Notebook 2: `02_model_scientific_interest.ipynb`

Performs SI modeling and forecasting:

- Builds yearly SI with pooled background-adjusted baseline.
- Forecasts future SI using yearly LOESS count modeling.
- Ranks pairs for reliable future high scientific interest.
- Produces focused plots for shortlisted pairs.

Primary APIs:

- `amr.pipelines.scientific_interest.build_yearly_si_dataset(...)`
- `amr.pipelines.scientific_interest.rank_future_high_interest_pairs(...)`

## Canonical Data Contracts

- `yearly_amr_pairs.parquet`
  - Columns: `pathogen`, `antibiotic`, `year`, `count`, `pmids`
- `yearly_amr_screened_pairs.parquet`
  - Columns: `pathogen`, `antibiotic`, `year`, `count`, `pmids`
  - Built from unified decisions, default include set: `yes` + `uncertain`
- `yearly_si.parquet`
  - Includes `bg_pathogen_n`, `bg_antibiotic_n`, `expected_log_count_global`, `SI`
- `yearly_si_ranked_future.parquet`
  - Per-pair forecast and rank-shift outputs for future SI

## Installation

Python 3.10+ is required.

```bash
pip install \
  pandas numpy requests biopython pyarrow \
  matplotlib seaborn scikit-learn jupyter
```

Optional model backends:

```bash
pip install openai
```

## Runtime Requirements

- PubMed access: use a real contact email with Entrez calls.
- LLM backend: configure OpenAI or Ollama credentials.
- Existing parquet caches are reusable and supported.

## Migration Notes

- The legacy second-pass screen in `amr.refined_filtering` is deprecated.
- Use `run_unified_llm_filter` as the canonical screening path.
- Legacy notebooks are kept for reproducibility only and should not be extended.
