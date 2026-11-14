# AMR SI Explorer

A static, browser-local Plotly viewer for resistance-focused SI pairs. It supports pathogen, antibiotic, and GLASS filters, editable quadrant thresholds, point selection, detailed hover records, and filtered CSV download. Click the circled information buttons for definitions of SI, axes, rankings, and the plots. Pair counts and PMID lists are rebuilt from the resistance-only screen. Provisional data can be published for exploratory review, with an explicit uncalibrated status banner; final data require the human-reviewed precision target.

The explorer version is tracked in `dist/version.json` and shown in the header; increment it with each release. Header author links follow Pathogen Atlas V2. The canonical public explorer is [AMR SI Explorer on GitHub Pages](https://fabriziocosta.github.io/Antimicrobial-Resistance/). Its static files are committed under `dist/` and deployed by `.github/workflows/deploy-amr-si-explorer.yml` whenever this directory changes on `main`.

The explorer also includes a collapsed, explorer-wide Future-SI-weighted Atlas profile map, with one point per antibiotic, matching the notebook’s IMSTE view. Its coordinates and hover diagnostics are stored in `dist/future_si_profiles.json`; rerun `notebooks/exploratory_antibiotic_literature_similarity.ipynb` to regenerate that payload after changing forecast or IMSTE settings.

For each selected pair, the page also renders a collapsible single-pair SI and publication-count plot between the pair information and the PMID list. Clicking a year in that plot opens the matching year’s PubMed IDs. Precomputed annual SI and publication-count data are stored in `dist/pair_timeseries.json`; pair-year PMID lists are stored in `dist/pair_pmids_by_year.json`. The browser does not fit models or need access to repository data.
Selecting a pair adds a **Pathogen profile** link beside its details. The link opens the matching species in the Pathogen Atlas explorer.
The selected antibiotic also links to its English Wikipedia article when available. The association table is built by `notebooks/11_wikipedia_antibiotic_references.ipynb`, which resolves every antibiotic in `dist/pairs.json` and writes `data/antibiotic_wikipedia_links.csv` plus the browser payload `dist/antibiotic_wikipedia_links.json`. Re-run the notebook after changing the explorer antibiotic set; pushing the updated `dist/` to `main` deploys it to GitHub Pages.

After notebook 04c and the SI notebooks have rebuilt the screened counts and current ranking, regenerate `dist/pairs.json` from the ranking and the resistance-only 2025 PMID counts. The configured status may be `provisional_uncalibrated` for explicitly exploratory publication or `validated` after human calibration:

```sh
python sites/amr-si-explorer/scripts/build_pairs_json.py
```

Then regenerate the precomputed yearly plot table from Notebook 06's yearly SI data and plotted pair list:

```sh
python sites/amr-si-explorer/scripts/build_pair_timeseries.py
```

This uses the same `fit_predict_yearly_si_loess` model and fit settings as the Notebook 06 explorer summary. The builder checks that each pair's future SI in the yearly plot matches the summary value for the same year. GitHub Pages serves the generated JSON directly. Rebuild the JSON after changing the yearly SI source data or model settings, then push the changed `dist/` files to `main`.

Regenerate the pair-year PMID table from the screened-pair data, matching Jev run, and PubMed date metadata by running from the repository root:

```sh
python sites/amr-si-explorer/scripts/build_pair_pmids_by_year.py
```

The pair-bundle builder and pair-year PMID builder require matching validated or explicitly provisional resistance-screen provenance. The PMID builder checks that the 2025 arrays match the newly generated `dist/pairs.json`, so both outputs use the same resistance-positive screen. Compact PMID records include the PMID, its publication date, and Jev `system1_p_yes` confidence when available. In the explorer, sort a selected year’s PMID list by newest date or highest confidence. Rebuild the JSON after updating the screened-pair, System 1 decision, or PubMed metadata parquet. Pushing changed `dist/` files to `main` triggers the documented GitHub Pages deployment workflow.
