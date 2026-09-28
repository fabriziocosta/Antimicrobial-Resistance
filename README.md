# Pathogen Atlas — Antimicrobial Resistance (Refactored Workflow)

This project uses a module-first workflow with twelve active
notebooks:

0. `notebooks/00_configure_pipeline.ipynb`
1. `notebooks/01_prepare_inputs.ipynb`
2. `notebooks/02_estimate_llm_screening_cost.ipynb`
3. `notebooks/03_query_examples.ipynb`
4. `notebooks/04_collect_yearly_data.ipynb`
5. `notebooks/04a_prepare_candidate_abstracts.ipynb`
6. `notebooks/04b_predict_candidate_pairs.ipynb`
7. `notebooks/04c_build_screened_dataset.ipynb`
8. `notebooks/05_model_scientific_interest.ipynb`
9. `notebooks/06_glass_interactive_si_shift.ipynb`
10. `notebooks/07_glass_dedicated_pair_plots.ipynb`
11. `notebooks/08_inspect_pair_pmids.ipynb`

All earlier exploratory notebooks are considered legacy and should remain archived.

## Documentation

Detailed project documents live in `docs/`:

- [Data collection and filtering process](docs/data_collection_and_filtering.md)
- [Scientific interest modeling notes](docs/scientific_interest.md)
- [SI driver classification analysis](docs/si_driver_classification_analysis.md)
- [Collaborator communications log](collaborator_communications/communications.md)

## Repository Layout

- `assets/`: static lookup inputs used by the notebooks and PubMed query builders.
- `notebooks/`: active notebook workflow entrypoints.
- `outputs/parquets/`: generated parquet tables and caches, versioned with Git LFS
  because they are expensive to rebuild.
- `outputs/pipeline_config.yaml`: versioned configuration snapshot used to reproduce
  the current notebook outputs.
- `outputs/figures/`: generated plots, kept local and excluded from Git because they
  can be reconstructed from the versioned parquet outputs.
- `amr/pubmed/`: PubMed query construction, Entrez retrieval, metadata, and yearly PMID transforms.
- `amr/screening/`: LLM-assisted abstract screening and screened-pair dataset builders.
- `amr/si/`: scientific-interest modeling, forecasting, ranking, driver summaries, and plots.
- `amr/pipelines/`: notebook-facing orchestration functions.

## Active Workflow

Static lookup inputs live in `assets/`; generated parquet outputs are written under `outputs/parquets/`.

### Notebook 00: `notebooks/00_configure_pipeline.ipynb`

Writes the shared YAML configuration used by the remaining notebooks.

### Notebook 01: `notebooks/01_prepare_inputs.ipynb`

Prepares static input assets. It downloads/validates the pathogen priority list and builds antibiotic lookup CSVs from `assets/antibiotic_list3.xlsx`:

- `assets/antibiotic_synonym_lookup.csv`
- `assets/antibiotic_coarsening_lookup.csv`

### Notebook 02: `notebooks/02_estimate_llm_screening_cost.ipynb`

Estimates LLM screening costs before collection. It can project from an existing pilot candidate set to the configured pathogen count, while excluding rows already present in the decisions parquet.

### Notebook 03: `notebooks/03_query_examples.ipynb`

Prints example PubMed queries, cache preflight summaries, and prompt templates before collection.

### Notebook 04: `notebooks/04_collect_yearly_data.ipynb`

Collects yearly candidates and background marginals:

- Collects pathogen-antibiotic PMIDs through PubMed interval queries.
- Converts pair-PMID associations to yearly counts from publication dates.
- Collects yearly background publication counts for pathogens and antibiotics.
- Supports partitioned collection runs via `SLICE_START_YEAR` / `SLICE_END_YEAR`.

Primary APIs:

- `amr.pipelines.collection.collect_yearly_amr_dataset(...)`
- `amr.pipelines.collection.collect_yearly_background_counts(...)`

Run the screening stages in this order after collection:

1. `04a_prepare_candidate_abstracts.ipynb` fills `pmid_abstracts.parquet` for
   every unique candidate PMID. Existing cache rows are reused, no-abstract
   results are retained, and failed fetches are retried on later runs.
2. `02_estimate_llm_screening_cost.ipynb` estimates both LLM and Jev costs.
   It stops with missing PMID examples if abstract cache coverage is incomplete.
3. `04b_predict_candidate_pairs.ipynb` runs the active LLM or System 1 source
   for the configured year slice. The prediction stores remain separate and
   resume independently.
4. `04c_build_screened_dataset.ipynb` applies the configured source and, for
   System 1, run ID and probability threshold. It rebuilds the screened parquet
   offline from saved predictions.
5. `05_model_scientific_interest.ipynb` reads the screened dataset using its
   existing input contract.

Choose the active prediction source once in notebook 00 by setting `PREDICTION_SOURCE` to `"llm"` or `"system1"`, then run the config cell. It writes `screening.active_source` to `outputs/pipeline_config.yaml`; notebooks 04b and 04c use that same selection. Notebook 02 continues to estimate both approaches. For System 1, set `OPENROUTER_API_KEY`; adjust `screening.system1_probability_threshold` in notebook 00 and rerun the config cell, then run notebook 04c to rebuild from saved Jev probabilities without repeating API calls. Change `system1.run_id` to create a distinct prediction snapshot; results are stored at the configured `paths.system1_decisions` path. Jev uses OpenRouter's typed Decisions API and `typesafe/jev-1.13` by default. System 1 driver-category classification is not included yet.

### Notebook 05: `notebooks/05_model_scientific_interest.ipynb`

Performs SI modeling and forecasting:

- Builds yearly SI with pooled background-adjusted baseline.
- Forecasts future SI using yearly LOESS count modeling.
- Ranks pairs for reliable future high scientific interest.
- Produces focused plots for shortlisted pairs.
- Saves all-year driver-composition pie charts for the selected pairs in one PDF grid.

Primary APIs:

- `amr.pipelines.si.build_yearly_si_dataset(...)`
- `amr.pipelines.si.rank_future_high_interest_pairs(...)`

### Notebook 06: `notebooks/06_glass_interactive_si_shift.ipynb`

Builds interactive GLASS comparisons, future-SI shift plots, and their raw plotted
tables from the screened and modeled outputs.

### Notebook 07: `notebooks/07_glass_dedicated_pair_plots.ipynb`

Produces dedicated longitudinal plots for selected GLASS pathogen–antibiotic pairs.

### Notebook 08: `notebooks/08_inspect_pair_pmids.ipynb`

Exports and inspects the PMIDs, titles, abstracts, and screening decisions for a
configured pathogen–antibiotic pair.

## Canonical Data Contracts

- `outputs/parquets/candidate_amr_pair_year_counts.parquet`
  - Columns: `pathogen`, `antibiotic`, `year`, `count`, `pmids`
- `outputs/parquets/yearly_amr_screened_pairs.parquet`
  - Columns: `pathogen`, `antibiotic`, `year`, `count`, `pmids`
  - Rebuilt from the active LLM or System 1 prediction source and its configured selection rule
- `outputs/parquets/yearly_amr_system1_decisions.parquet`
  - Append-preserving System 1 prediction runs with `system1_p_yes`, status, model, and run metadata
- `outputs/parquets/yearly_si.parquet`
  - Includes `bg_pathogen_n`, `bg_antibiotic_n`, `expected_log_count_global`, `SI`
- `outputs/parquets/yearly_si_ranked_future.parquet`
  - Per-pair forecast and rank-shift outputs for future SI, including `total_publications`

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

The computed parquet cache uses [Git LFS](https://git-lfs.com/). After cloning,
install Git LFS and retrieve the cached datasets:

```bash
git lfs install
git lfs pull
```

### Re-running notebooks 05–08 from a clone

After installing the project and running `git lfs pull`, notebooks 05–08 can use the
versioned configuration, static assets, and parquet caches directly:

```bash
pip install -e ".[dev,notebooks,loess]"
git lfs pull
```

Run notebooks 05, 06, 07, and 08 in order. Notebook 05 rebuilds the SI and ranking
parquets from the screened AMR, background-count, and unified-decision caches.
Notebooks 06 and 07 rebuild the GLASS comparisons and figures; notebook 08 rebuilds
the selected pair-PMID Excel export. Generated figures and exports remain local.

Run notebook 00 first only when intentionally changing the tracked configuration
snapshot.

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

1. Update shared settings in notebook 00 if required.
2. Run notebook 04 on a slice (for example `2027-2027` or `2025-2027`), then run
   `04a → 02 → 04b → 04c` in order.
3. Collection and prediction outputs resume into their canonical stores:
   - `outputs/parquets/candidate_amr_pair_year_counts.parquet`
   - `outputs/parquets/candidate_amr_pair_pmid_links.parquet`
   - `outputs/parquets/yearly_amr_unified_decisions.parquet`
   - `outputs/parquets/yearly_amr_system1_decisions.parquet`
   - `outputs/parquets/yearly_amr_screened_pairs.parquet` (rebuilt from the
     selected source, run ID, and threshold)
4. Re-run notebook 05 with the chosen modeling window.
5. Re-run notebooks 06–08 as needed to refresh comparisons, figures, and PMID
   inspection exports.
