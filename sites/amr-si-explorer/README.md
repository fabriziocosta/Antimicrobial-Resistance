# AMR SI Explorer

A static, browser-local Plotly viewer built from the filtered pair dataset used by Notebook 06. It supports text and category filters, editable quadrant thresholds, point selection, detailed hover records, and filtered CSV download.

The canonical public explorer is [AMR SI Explorer on GitHub Pages](https://fabriziocosta.github.io/Antimicrobial-Resistance/). Its static files are committed under `dist/` and deployed by `.github/workflows/deploy-amr-si-explorer.yml` whenever this directory changes on `main`.

For each selected pair, the page also renders a collapsible single-pair SI and publication-count plot between the pair information and the PMID list. Clicking a year in that plot opens the matching year’s PubMed IDs. Precomputed annual SI and publication-count data are stored in `dist/pair_timeseries.json`; pair-year PMID lists are stored in `dist/pair_pmids_by_year.json`. The browser does not fit models or need access to repository data.
Selecting a pair adds a **Pathogen profile** link beside its details. The link opens the matching species in the Pathogen Atlas explorer.
The selected antibiotic also links to its English Wikipedia article when available. The association table is built by `notebooks/11_wikipedia_antibiotic_references.ipynb`, which resolves every antibiotic in `dist/pairs.json` and writes `data/antibiotic_wikipedia_links.csv` plus the browser payload `dist/antibiotic_wikipedia_links.json`. Re-run the notebook after changing the explorer antibiotic set; pushing the updated `dist/` to `main` deploys it to GitHub Pages.

To regenerate that precomputed table from the repository's current yearly SI data and plotted pair list, run from the repository root:

```sh
python sites/amr-si-explorer/scripts/build_pair_timeseries.py
```

This uses the same `fit_predict_yearly_si_loess` model called by `plot_single_pair_si`. GitHub Pages serves the generated JSON directly. Rebuild the JSON after changing the yearly SI source data or model settings, then push the changed `dist/` files to `main`.

To regenerate the pair-year PMID table from the checked-in screened-pair data, Jev decisions, and PubMed date metadata, run from the repository root:

```sh
python sites/amr-si-explorer/scripts/build_pair_pmids_by_year.py
```

The builder checks that each pair’s 2025 PMID list still matches `dist/pairs.json`. Compact PMID records include the PMID, its publication date, and Jev `system1_p_yes` confidence when available. In the explorer, sort a selected year’s PMID list by newest date or highest confidence. The source currently contains annual PMID lists from 2000 through 2026. Rebuild the JSON after updating the screened-pair, System 1 decision, or PubMed metadata parquet, then push the changed `dist/` files to `main` to deploy them.
