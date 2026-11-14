# Reproducing the antibiotic literature similarity and AST comparison

This guide accompanies [`exploratory_antibiotic_literature_similarity.ipynb`](exploratory_antibiotic_literature_similarity.ipynb), whose analysis pipeline is implemented in [`exploratory_antibiotic_literature_similarity.py`](exploratory_antibiotic_literature_similarity.py). The notebook contains parameters, result tables, and visualizations; the module handles source loading and calculations. Antibiotic names on the future-SI map are placed in a right-side label column with zoom-aware leader lines. This guide records the inputs, saved NCBI snapshot, cohort rules, and commands used for the notebook output as of **2026-11-14**. It does not turn the analyses into estimates of antibiotic efficacy or resistance.

## What the notebook compares

The notebook has two analyses. First, it builds antibiotic-by-bacterium presence sets from screened AMR PubMed records and shows Jaccard overlap and a classical-MDS projection. A missing pair means no screened record in this AMR dataset, not a negative susceptibility result.

The later comparison constructs an Atlas V2 feature profile for each antibiotic using either (a) projected future scientific-interest scores (SI) as species weights or (b) clinical susceptibility fractions as weights. The two profiles are compared in their shared **73-dimensional (73D)** Atlas feature space with cosine distances, then each set of profiles is embedded separately with IMSTE. A pairwise scatter plots the future-SI distance against the AST distance for every unordered antibiotic pair; hover shows the pair and exact values, and the full table is displayed below it. Because the embeddings are fitted independently, their 2D orientations and absolute coordinates are arbitrary; compare the 73D cosine distances, not point-to-point distances across the two panels.

## Exact local inputs and release provenance

The notebook locates the AMR repository from the working directory (or a parent directory containing `outputs/parquets/yearly_amr_screened_pairs.parquet`). `ATLAS_V2_ROOT`, if set, selects the Atlas checkout; otherwise the notebook uses the absolute default shown below.

| Input | Exact inspected path | Use |
|---|---|---|
| AMR screened-pair table | [`../outputs/parquets/yearly_amr_screened_pairs.parquet`](../outputs/parquets/yearly_amr_screened_pairs.parquet) (absolute: `/Users/f.costa/Documents/Projects/Repositories/ACTIVE/Antimicrobial-Resistance/outputs/parquets/yearly_amr_screened_pairs.parquet`) | Literature-footprint/Jaccard analysis |
| AMR forecast table | [`../outputs/parquets/yearly_si_ranked_future.parquet`](../outputs/parquets/yearly_si_ranked_future.parquet) (absolute: `/Users/f.costa/Documents/Projects/Repositories/ACTIVE/Antimicrobial-Resistance/outputs/parquets/yearly_si_ranked_future.parquet`) | Future-SI weights; successful forecasts at the selected 2025 → 2030 horizon |
| AMR screening provenance | [`../outputs/parquets/yearly_amr_screened_pairs.provenance.json`](../outputs/parquets/yearly_amr_screened_pairs.provenance.json) | Marks this screening run `provisional_uncalibrated`; its holdout precision is null |
| Atlas V2 feature matrix | `/Users/f.costa/Documents/Projects/Repositories/ACTIVE/Pathogen Atlas/Pathogen Atlas/Pathogen Atlas Data Acquisition/version2/data/04_tables/release/species_feature_matrix.parquet` | `feature_<question_id>` values, keyed by canonical species |
| Atlas V2 answer summary | `/Users/f.costa/Documents/Projects/Repositories/ACTIVE/Pathogen Atlas/Pathogen Atlas/Pathogen Atlas Data Acquisition/version2/data/04_tables/release/species_feature_summary.parquet` | Feature names/categories and answer coverage |
| Atlas V2 IMSTE configuration | `/Users/f.costa/Documents/Projects/Repositories/ACTIVE/Pathogen Atlas/Pathogen Atlas/Pathogen Atlas Data Acquisition/version2/configs/notebook_parameters.yaml` (`phase5_plot_creation`) | Embedding parameters |
| Atlas V2 release metadata | `/Users/f.costa/Documents/Projects/Repositories/ACTIVE/Pathogen Atlas/Pathogen Atlas/Pathogen Atlas Data Acquisition/version2/data/04_tables/release/table_creation_metadata.json` | Release run ID `jev-human-pathogen-evidence-v1`; matrix contains 1,703 species |

The inspected Atlas checkout is package `pathogen-atlas-v2` version `2.0.10` at commit `caaccd9`. The AMR screened-pair provenance file identifies run `jev-explicit-pair-resistance-v1-20261109`, with question version `explicit-pair-resistance-noul-v1` and threshold calibration status `provisional_uncalibrated`.

## NCBI AST source and preserved snapshot

AST records come from the [NCBI Pathogen Detection AST Browser](https://www.ncbi.nlm.nih.gov/pathogens/ast/) and its [field/method documentation](https://www.ncbi.nlm.nih.gov/pathogens/docs/ast/). The notebook retrieves records from `https://www.ncbi.nlm.nih.gov/pathogens/pathogens-srv/` (`action=retrieve`, `collection=ast`) with the initial filter `epi_type:clinical AND host:"Homo sapiens"`. Its NCBI manifest records the retrieval time as **2026-11-14T11:24:42.586253+00:00 UTC** and 149,327 records in that initial query. This is the retrieval time and size of the saved snapshot; it is not the time of a later notebook execution that reused it.

The preserved filtered row snapshot is at:

`/Users/f.costa/Documents/Codex/2026-11-14/antibiotic-similarity-exploration/outputs/clinical_ast_comparison/ncbi_pathogen_detection_clinical_ast_atlas_forecast_extract.parquet`

The paired CSV is in the same directory. These are filtered source rows (before isolate-level conflict/deduplication), not the full 149,327-row NCBI response. The accompanying manifest is `/Users/f.costa/Documents/Codex/2026-11-14/antibiotic-similarity-exploration/outputs/clinical_ast_comparison/manifest.json`; it records snapshot counts, filters, conflicts, and the retrieval timestamp. The filtered Parquet SHA-256 is `aeb5e145cafbc9dec5d9d316f0d33e4558718dfe411e838224597218db1d3f65`. In this saved snapshot the filtered extract has 68,586 rows and 130 unique species–drug cells, covering 16 species and 44 antibiotics. Identical calls are collapsed during analysis; 21 isolate–drug groups (48 rows) with conflicting S/I/R calls are excluded.

### Filters and comparison cohort

After retrieval (or loading the snapshot), the notebook requires all of the following:

1. Clinical isolation type, human host (`Homo sapiens`), CLSI standard, and an explicit susceptible/intermediate/resistant phenotype.
2. An exact species match to the Atlas V2 matrix after lowercasing and whitespace normalization, and an exact species–antibiotic match to a successful forecast after antibiotic-name normalization (including the `rifampicin`/`rifampin` alias).
3. One interpretable call per unique target accession (falling back to BioSample accession), species, and drug. Repeated identical calls collapse; all calls for a conflicting isolate–species–drug group are excluded. Counts and fractions use S+I+R only; no unobserved species–drug pair is imputed.
4. For the matched comparison, at least **20 interpretable isolates in each species–drug cell**, then at least **two supported species and 30 isolates total per drug**. The future-SI and AST profiles for a given drug use the same qualifying species–drug cells. AST weights are normalized susceptible fractions; a zero-susceptible cell has zero weight, and a drug with no positive susceptible mass cannot form an AST centroid.

The notebook selected 167 successful forecast pairs (22 species, 49 antibiotics) at 2025 → 2030; 130 had at least one matching clinical CLSI S/I/R observation. The final support rule yielded **20 antibiotics, 78 shared species–drug cells, and 190 antibiotic pairs** for the distance comparison. Across the retained cells, isolate counts ranged from 24 to 3,178.

NCBI describes these as submitter-provided phenotypic results and cautions that submitted methods/field relationships are not centrally verified. No separate reuse license is attached to the returned phenotype rows. Cite NCBI, retain accession/source attribution, and check the original submitter or study terms before redistributing the extract; do not describe this filtered subset as population-representative surveillance.

## Feature preparation, weights, and embedding

Atlas feature columns are converted to numeric. Missing values are filled with each feature’s mean across the full 1,703-species Atlas matrix, constant features are removed, and `StandardScaler` is fitted across that full species matrix. This reproduces the Atlas feature coordinate system and leaves 73 nonconstant dimensions. Per-antibiotic weighted averages are then formed from those standardized species vectors; the averages themselves are not L2-normalized before cosine comparison.

- **Future-SI profile:** within each antibiotic’s qualifying shared species, compute `exp(future_predicted_SI - max(SI))` and normalize to sum to one. This stable softmax is equivalent to normalized `exp(SI)` and gives larger future-SI values exponentially more weight.
- **Clinical-AST profile:** use the species-level susceptible fraction `S/(S+I+R)` as its weight, normalized within the same antibiotic and same supported species set. Intermediate and resistant counts remain separately available in the summary but do not add to this weight.

Both maps use cosine distance in 73D, then independently fit IMSTE with `metric="cosine"` and `gravity_mode=None`. The Atlas `phase5_plot_creation` settings are 64 MSTs, 1,000 epochs, `famst`, up to 15 MST neighbors, 5 inter-component edges, negative ratio 5, repulsion lambda 0.5, batch size 32,768, epsilon 0.0001, logistic margin 1, temperature 0.5, 10 representatives per component, automatic device selection, and seed 42. The notebook overrides the configured learning rate 0.05 with **0.001** for these 49- and 20-point inputs because 0.05 collapsed this smaller embedding in the notebook’s inspection. That override is explicit in the notebook; the resulting 2D map remains a visualization, not the distance metric.

The Plotly comparison is a 1,500 × 760 side-by-side figure with the SI-weighted and AST-weighted map. Both panels use the same small marker size and the same distinct color for each antibiotic, plus hover values for shared species, isolate counts, minimum cell support, AST effective sample size and bootstrap centroid radius, and a right-side label column with the same vertical name spacing in both panels, joined to each marker by a thin leader line. The static PNG is rendered at 1,800 × 950 with scale 1.25. Hover and zoom are available in the self-contained HTML.

## Outputs and key readout

All AST-comparison files go to `AST_DATA_DIR`. The notebook uses `AST_COMPARISON_DATA_DIR` when set; otherwise it creates/uses the AMR checkout’s ignored directory `/Users/f.costa/Documents/Projects/Repositories/ACTIVE/Antimicrobial-Resistance/outputs/clinical_ast_comparison/`. For the preserved run, the directory is `/Users/f.costa/Documents/Codex/2026-11-14/antibiotic-similarity-exploration/outputs/clinical_ast_comparison/`.

The notebook writes these files in that directory:

- `ncbi_pathogen_detection_clinical_ast_atlas_forecast_extract.parquet` and `.csv` — filtered source-row snapshot.
- `species_antibiotic_clinical_ast_summary.csv` — deduplicated S/I/R isolate counts and fractions by species–drug.
- `clinical_ast_coverage_by_antibiotic.csv` and `forecast_ast_pair_coverage.csv` — coverage and per-forecast-pair AST availability.
- `comparison_pair_support.csv` and `comparison_species_drug_weights.csv` — final comparison support and each method’s within-drug weights.
- `si_vs_ast_pairwise_cosine_distances.csv` — 73D cosine distances for the 190 drug pairs.
- `manifest.json` — source, snapshot/run metadata, and filters.
- `clinical_ast_vs_future_si_imste.html` and `clinical_ast_vs_future_si_imste.png` — interactive and static comparison maps.

For the preserved cohort, pairwise 73D cosine distances have Spearman correlation **0.890** and Pearson correlation **0.871** between the two weighting schemes. This is concordance of distance structure, not evidence that SI and susceptibility are interchangeable or that a 2D coordinate aligns between separately fitted maps. AST support and weights are highly uneven, individual drugs can be dominated by a few species, the data are a submitter-selected subset, Atlas features describe evidence-derived question answers rather than susceptibility, and the AMR screen is uncalibrated. Treat the results as exploratory.

## Environment and rerun

The AMR project declares base dependencies in [`../pyproject.toml`](../pyproject.toml) and a `notebooks` extra (`jupyter`, `kaleido`, `plotly`). The separate Atlas project declares its `viz` extra in `/Users/f.costa/Documents/Projects/Repositories/ACTIVE/Pathogen Atlas/Pathogen Atlas/Pathogen Atlas Data Acquisition/version2/pyproject.toml`; it supplies `imste` pinned to commit `0a34b643e0a33f201d2bae36c32297c5c397202e`, Plotly, scikit-learn, and the Atlas visualization dependencies. Install both extras in the same Python environment used by Jupyter:

```sh
cd "/Users/f.costa/Documents/Projects/Repositories/ACTIVE/Antimicrobial-Resistance"
python -m pip install -e '.[notebooks]'
python -m pip install -e '/Users/f.costa/Documents/Projects/Repositories/ACTIVE/Pathogen Atlas/Pathogen Atlas/Pathogen Atlas Data Acquisition/version2[viz]'
export ATLAS_V2_ROOT="/Users/f.costa/Documents/Projects/Repositories/ACTIVE/Pathogen Atlas/Pathogen Atlas/Pathogen Atlas Data Acquisition/version2"
export AST_COMPARISON_DATA_DIR="/Users/f.costa/Documents/Codex/2026-11-14/antibiotic-similarity-exploration/outputs/clinical_ast_comparison"
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/exploratory_antibiotic_literature_similarity.ipynb
```

`PROJECT_ROOT` is discovered from the current directory or its parents. If `ATLAS_V2_ROOT` is unset, the notebook falls back to the absolute Atlas checkout path above. If `AST_COMPARISON_DATA_DIR` is unset, AST outputs go under the AMR repository’s ignored `outputs/clinical_ast_comparison/`. If that directory has no `ncbi_pathogen_detection_clinical_ast_atlas_forecast_extract.parquet`, the notebook downloads the current NCBI clinical-human result set; if the parquet exists, it loads that saved snapshot and does not refresh the endpoint.

### Refresh without replacing the saved NCBI snapshot

Keep the snapshot directory above untouched. To retrieve a newer NCBI result, point `AST_COMPARISON_DATA_DIR` to a **new, timestamped directory** (so no source parquet exists there) and execute the notebook with the same command. The notebook will fetch current clinical-human rows and write the new filtered snapshot and derived outputs only into that new directory. Save the old and new `manifest.json` files together for comparison. The notebook records `last_notebook_run_utc`, but a brand-new directory does not receive a `retrieved_utc` key from the current code; record the retrieval start/end time in UTC alongside that new manifest. The preserved snapshot manifest already contains its exact `retrieved_utc` timestamp, and rerunning with its parquet present loads rather than silently replaces it.
