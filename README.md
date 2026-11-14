<div align="center">
  <img src="sites/amr-si-explorer/dist/amr-si-explorer-logo.png?v=1.0.21" alt="AMR SI Explorer logo: bacterium and resistance shield over a globe" width="150">
</div>

# AMR SI Explorer

*Scientific interest in antimicrobial resistance*

**How is research attention to antimicrobial resistance changing across pathogen–antibiotic pairs?**

This project collects and screens PubMed records, estimates pair-level scientific interest (SI), and explores projected changes in an interactive atlas. Its central idea is to compare AMR publication activity for a pathogen–antibiotic pair with the activity expected from the general research attention to that pathogen and antibiotic.

<div align="center">
  <a href="https://fabriziocosta.github.io/Antimicrobial-Resistance/">Open the AMR SI Explorer →</a>
</div>

## What SI means

For each pathogen–antibiotic pair and year, the workflow counts papers that report resistance for that exact pair. A pooled model estimates the expected pair-level log-count from the general publication activity for the pathogen and antibiotic, together with pooled pathogen and antibiotic effects. SI is the difference between the observed and expected log-counts:

```text
SI = log1p(observed AMR publication count) − expected log-count
```

Positive SI means the pair receives more AMR research attention than the baseline expects; negative SI means less. The Explorer shows forecast SI and its annualized gradient to help readers find pairs whose research attention is high or changing.

**SI measures publication attention. It is not a clinical resistance rate, a measure of disease burden, or a prediction of future pathogenic threat.** PubMed searches find candidate papers; only papers passing the resistance-specific screen contribute to the SI counts. Screen provenance is marked as validated or provisional in the published data.

## Explore

The [AMR SI Explorer](https://fabriziocosta.github.io/Antimicrobial-Resistance/) lets you:

- Search and filter pathogen–antibiotic pairs, including GLASS matches.
- Compare projected future SI with its annualized gradient, and adjust the quadrant thresholds.
- Select a pair to view its yearly SI and publication counts, plus its linked PubMed records.
- Explore a collapsed map of antibiotic profiles weighted by future SI.
- Download the currently filtered pairs as CSV.

The site is a static browser application. It uses precomputed data committed in [`sites/amr-si-explorer/dist/`](sites/amr-si-explorer/dist/) and does not fit models in the browser. Changes to the Explorer payload on `main` trigger the [GitHub Pages deployment workflow](.github/workflows/deploy-amr-si-explorer.yml). See the [Explorer guide](sites/amr-si-explorer/README.md) to rebuild its data.

## Analysis workflow

The notebooks orchestrate a modular Python pipeline. The main sequence is:

| Stage | Notebook | Purpose |
| --- | --- | --- |
| Configure | [`00_configure_pipeline.ipynb`](notebooks/00_configure_pipeline.ipynb) | Set the shared run configuration. |
| Prepare inputs | [`01_prepare_inputs.ipynb`](notebooks/01_prepare_inputs.ipynb) | Build canonical pathogen and antibiotic lookup tables. |
| Collect | [`04_collect_yearly_data.ipynb`](notebooks/04_collect_yearly_data.ipynb) | Retrieve PubMed candidates and background publication counts. |
| Prepare and screen | [`04a_prepare_candidate_abstracts.ipynb`](notebooks/04a_prepare_candidate_abstracts.ipynb), [`04b_predict_candidate_pairs.ipynb`](notebooks/04b_predict_candidate_pairs.ipynb), [`04c_build_screened_dataset.ipynb`](notebooks/04c_build_screened_dataset.ipynb) | Cache abstracts, screen exact pairs for reported resistance, and assemble the yearly dataset. Notebook [`02_estimate_llm_screening_cost.ipynb`](notebooks/02_estimate_llm_screening_cost.ipynb) helps plan screening before a run. |
| Model and rank | [`05_model_scientific_interest.ipynb`](notebooks/05_model_scientific_interest.ipynb) | Estimate yearly SI, forecast publication counts, and rank pairs. |
| Inspect | [`06_glass_interactive_si_shift.ipynb`](notebooks/06_glass_interactive_si_shift.ipynb), [`07_glass_dedicated_pair_plots.ipynb`](notebooks/07_glass_dedicated_pair_plots.ipynb), [`08_inspect_pair_pmids.ipynb`](notebooks/08_inspect_pair_pmids.ipynb) | Compare with GLASS records, plot selected pairs, and inspect their papers. |

Additional notebooks support SI concept discovery, antibiotic reference links, and exploratory analyses. See [the SI methods](docs/scientific_interest.md), [data collection and screening](docs/data_collection_and_filtering.md), and [specialized analysis guides](docs/specialized/).

## Install

Python 3.10 or newer is required. To install the notebook and modeling dependencies and retrieve the versioned parquet data:

```bash
python -m pip install -e ".[dev,notebooks,loess]"
git lfs install
git lfs pull
```

The optional `llm` extra installs the OpenAI client for workflows that use it:

```bash
python -m pip install -e ".[llm]"
```

PubMed collection requires an Entrez contact email. Screening providers may require credentials; keep credentials in environment variables or the local system keychain and out of committed configuration.

## Repository map

- [`amr/`](amr/) — PubMed collection, resistance screening, SI modeling, forecasting, and pipeline APIs.
- [`notebooks/`](notebooks/) — pipeline entrypoints and exploratory analyses.
- [`assets/`](assets/) — pathogen and antibiotic lists, synonyms, and other static inputs.
- [`outputs/parquets/`](outputs/parquets/) — versioned intermediate and analysis tables stored with Git LFS.
- [`docs/`](docs/) — methods, data handling, and research notes.
- [`sites/amr-si-explorer/`](sites/amr-si-explorer/) — static Explorer and data builders.
- [`archive/`](archive/) — legacy code and notebooks retained for historical reference.

## Citation

Citation metadata is maintained in [`CITATION.cff`](CITATION.cff). Please cite the project when reusing its code or results.
