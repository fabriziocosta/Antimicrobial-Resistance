# Pathogen Atlas — Antimicrobial Resistance (Refactored Workflow)

This project uses a module-first workflow with nine sequentially numbered active
notebooks:

1. `notebooks/00_configure_pipeline.ipynb`
2. `notebooks/01_prepare_inputs.ipynb`
3. `notebooks/02_query_examples.ipynb`
4. `notebooks/03_collect_yearly_data.ipynb`
5. `notebooks/04_estimate_llm_screening_cost.ipynb`
6. `notebooks/05_model_scientific_interest.ipynb`
7. `notebooks/06_glass_interactive_si_shift.ipynb`
8. `notebooks/07_glass_dedicated_pair_plots.ipynb`
9. `notebooks/08_inspect_pair_pmids.ipynb`

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

### Notebook 02: `notebooks/02_query_examples.ipynb`

Prints example PubMed queries, cache preflight summaries, and prompt templates before collection.

### Notebook 03: `notebooks/03_collect_yearly_data.ipynb`

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

### Notebook 04: `notebooks/04_estimate_llm_screening_cost.ipynb`

Estimates LLM screening costs from the candidate rows and cached abstracts produced
by notebook 03. It does not call the LLM.

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
  - Built from unified decisions using the configured accepted decision set
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
2. Run notebook 03 on a slice (for example `2027-2027` or `2025-2027`).
3. Outputs merge into canonical stores:
   - `outputs/parquets/candidate_amr_pair_year_counts.parquet`
   - `outputs/parquets/candidate_amr_pair_pmid_links.parquet`
   - `outputs/parquets/yearly_amr_unified_decisions.parquet`
   - `outputs/parquets/yearly_amr_screened_pairs.parquet`
4. Re-run notebook 05 with the chosen modeling window.
5. Re-run notebooks 06–08 as needed to refresh comparisons, figures, and PMID
   inspection exports.
