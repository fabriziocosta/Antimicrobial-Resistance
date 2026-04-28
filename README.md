# Pathogen Atlas — Antimicrobial Resistance (Refactored Workflow)

This project now uses a module-first workflow with three active notebooks:

1. `notebooks/00_prepare_inputs.ipynb`
2. `notebooks/01_collect_yearly_data.ipynb`
3. `notebooks/02_model_scientific_interest.ipynb`

All earlier exploratory notebooks are considered legacy and should remain archived.

## Documentation

Detailed project documents live in `docs/`:

- [Data collection and filtering process](docs/data_collection_and_filtering.md)
- [Scientific interest modeling notes](docs/scientific_interest.md)
- [SI driver classification analysis](docs/si_driver_classification_analysis.md)

## Repository Layout

- `assets/`: static lookup inputs used by the notebooks and PubMed query builders.
- `notebooks/`: active notebook workflow entrypoints.
- `amr/pubmed/`: PubMed query construction, Entrez retrieval, metadata, and yearly PMID transforms.
- `amr/screening/`: LLM-assisted abstract screening and screened-pair dataset builders.
- `amr/si/`: scientific-interest modeling, forecasting, ranking, driver summaries, and plots.
- `amr/pipelines/`: notebook-facing orchestration functions.

## Active Workflow

Static lookup inputs live in `assets/`; generated parquet outputs stay at the project root by default.

### Notebook 0: `notebooks/00_prepare_inputs.ipynb`

Prepares static input assets. It downloads/validates the pathogen priority list and builds antibiotic lookup CSVs from `assets/antibiotic_list3.xlsx`:

- `assets/antibiotic_synonym_lookup.csv`
- `assets/antibiotic_coarsening_lookup.csv`

### Notebook 1: `notebooks/01_collect_yearly_data.ipynb`

Builds the yearly AMR dataset end-to-end:

- Collects pathogen-antibiotic PMIDs through PubMed interval queries.
- Converts pair-PMID associations to yearly counts from publication dates.
- Collects yearly background publication counts for pathogens and antibiotics.
- Runs one unified strict LLM pass (`AMR evidence` + `scientific-interest relevance`).
- Exports the canonical screened yearly dataset.
- Supports partitioned runs via `SLICE_START_YEAR` / `SLICE_END_YEAR` and auto-merges with existing parquet outputs.

Primary APIs:

- `amr.pipelines.collection.collect_yearly_amr_dataset(...)`
- `amr.pipelines.collection.collect_yearly_background_counts(...)`
- `amr.screening.prepare_yearly_candidate_pmids(...)`
- `amr.screening.run_unified_llm_filter(...)`
- `amr.screening.build_yearly_screened_pairs(...)`

### Notebook 2: `notebooks/02_model_scientific_interest.ipynb`

Performs SI modeling and forecasting:

- Builds yearly SI with pooled background-adjusted baseline.
- Forecasts future SI using yearly LOESS count modeling.
- Ranks pairs for reliable future high scientific interest.
- Produces focused plots for shortlisted pairs.

Primary APIs:

- `amr.pipelines.si.build_yearly_si_dataset(...)`
- `amr.pipelines.si.rank_future_high_interest_pairs(...)`

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
pip install -e ".[dev,notebooks]"
```

Optional model backends:

```bash
pip install -e ".[llm]"
```

Optional SI smoothing backend:

```bash
pip install -e ".[loess]"
```

## Runtime Requirements

- PubMed access: use a real contact email with Entrez calls.
- LLM backend: configure OpenAI or Ollama credentials.
- Existing parquet caches are reusable and supported.

## Migration Notes

- The legacy second-pass screen in `amr.refined_filtering` is deprecated.
- Use `run_unified_llm_filter` as the canonical screening path.
- Legacy notebooks are kept for reproducibility only and should not be extended.

## Incremental Update Pattern

The refactored pipeline is designed for yearly (or multi-year slice) updates:

1. Run notebook 1 on a slice (for example `2027-2027` or `2025-2027`).
2. Outputs merge into canonical stores:
   - `yearly_amr_pairs.parquet`
   - `yearly_amr_unified_decisions.parquet`
   - `yearly_amr_screened_pairs.parquet`
3. Re-run notebook 2 with your chosen modeling window (for example `FIT_START_YEAR=1970`, `FIT_END_YEAR=2024`).
