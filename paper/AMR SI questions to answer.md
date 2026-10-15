# Questions from Michiel Vos to Resolve

These questions and unresolved points are extracted from the Word comments authored by **Vos, Michiel** in *AMR SI first draft.docx*. The answers below are updated against the current repository configuration, code, saved parquet outputs, and manuscript draft. Saved outputs are not all consistent with the active configuration; those discrepancies are marked explicitly and must be resolved before manuscript counts are treated as final.

## Important provenance issue

The active configuration has collection slice 2000–2026 and selects `system1` (Jev `typesafe/jev-1.13`, run ID `jev-1.13-amr-support-v1`) at probability threshold 0.75. It separately sets 2000–2023 for the validation/ranking fit, 2000–2025 for pair eligibility and displayed curves, and 2025–2030 for the forecast. The saved LLM decision table is an earlier/different screen and must not be described as the active source. The System 1 decision cache contains 5,607 accepted rows outside the current candidate cache, but a row-level join shows that the saved screened table exactly matches the current candidates that pass the configured threshold (see Question 1).

## 1 What are all the steps and counts in Figure 1?

**Answer so far.** The checked-in configuration and current saved tables support this description. A row-level join reconciles the saved screened table to the current candidates; the full prediction cache includes additional rows outside those candidates:

1. The active collection configuration takes all 915 pathogen names with `pathogenic_index > 0` from the 1,279-row ranked file, plus 245 canonical antibiotics. The saved candidate association output covers 2000–2026 and has 514,470 unique pathogen–antibiotic–PMID links, 128,007 unique PMIDs, 21,435 observed pairs, 789 pathogen labels, and 217 antibiotic labels. These are observed query results, not the configured vocabulary sizes.
2. The yearly candidate table contains 97,575 pair–year rows over 21,435 pairs. Its `count` values sum to 514,470 pair–PMID associations. A PMID can appear under multiple target pairs, so none of these association counts is a count of unique abstracts.
3. The saved unified LLM decisions table has 186,724 rows with `strict` 54,757, `yes` 78,741, `no` 28,593, and `uncertain` 24,633. It is not the active screening source in the current config. The active source is the System 1 decision table, with 529,091 rows: 524,511 `ok`, 4,565 `no_abstract`, and 15 `error` for the configured Jev run ID. Across the whole prediction table, 312,325 rows are `ok` with `p_yes >= 0.75`; 5,607 of those accepted rows do not occur in the current candidate cache (14,621 prediction rows total are outside that cache). This is consistent with the System 1 store’s append-preserving/resume behavior: it reuses matching rows for the current candidate input but does not delete old rows that no longer match it. The 5,607 have no matching pair–PMID under another year either, so this is not a year-assignment discrepancy. They most likely came from an earlier or different candidate snapshot; the saved prediction table has no candidate-snapshot ID to identify the exact change that left them outside the current cache.
4. The saved screened table has 64,721 pair–year rows, 15,539 pairs, 719 pathogen labels, and 213 antibiotic labels over 2000–2026; the annual `count` values sum to 306,718 pair–PMID–year instances (287,995 within 2000–2025). Joining the System 1 predictions to the current candidate table on `(PMID, pathogen, antibiotic, year)` leaves exactly 306,718 rows at `status == ok` and `p_yes >= 0.75`, and these exactly match the PMIDs in the saved screened output. Thus, 306,718 is the accepted count for the current candidate set; 312,325 is the unjoined prediction-table total.
5. The saved SI table contains 15,806 pair–year rows for 644 pairs, 58 pathogens, and 96 antibiotics. The saved future-ranking output has 644 pairs; the `yearly_si_ranked_future_interesting.parquet` shortlist has 127 pairs, all marked `strict_filter` (no fallback rows). These replace the older 187-eligible/26-shortlist figures in the preliminary answer. Rerun downstream analyses after any future collection refresh or screening change.

**Configuration gates behind these counts.**

- **Retrieval:** `collection.slice_start_year = 2000`, `slice_end_year = 2026`; all 915 pathogen names with positive pathogenicity index; all 245 antibiotics; synonym-expanded Title/Abstract clauses for both entities; and the configured AMR/susceptibility OR clause shown in Question 4.
- **Screening:** `screening.active_source = system1`, Jev run `jev-1.13-amr-support-v1`, and `system1_probability_threshold = 0.75`. Only successful predictions matched to current candidate rows can pass the threshold into the screened output; use 306,718 for this saved candidate snapshot, not the unjoined 312,325 prediction-table rows.
- **Eligible SI pairs:** `si.publication_filter_end_year = 2025` and `si.min_tot_amr_publications = 50` require at least 50 screened pair publications summed over 2000–2025. The pipeline uses this total-count rule; `si.min_avg_amr_publications_per_year = 1.0` is an alternative/legacy setting and is not simultaneously applied. For the SI model, `si.min_amr_pair_publication_count = 3` makes annual pair counts below 3 unavailable; `si.min_num_years_with_valid_counts = 10` requires at least 10 usable years. Pair-years without either pathogen or antibiotic background counts are also dropped before pooled-model fitting.
- **Forecast shortlist (separate from SI-pair eligibility):** ranking uses fit years 2000–2023, curves through 2025, and a 2030 horizon. The strict selection requires predicted counts of at least 10 in 2025 and 2030, at least 15 observed AMR papers in 2025, SI growth of at least 0.01, at least one post-fit validation year, and mean absolute post-fit SI percentage error no greater than 100%. The saved shortlist has two validation years per pair; the configured fallback can add up to 10 pairs only if strict selection needs it, but none were used here.

**Configuration consistency.** Notebook 00 and the checked-in `outputs/pipeline_config.yaml` now both set `collection.refresh_cached_intervals: true`. Running notebook 00 writes this shared configuration for downstream notebooks. The setting takes effect only when collection is run; it does not by itself rebuild screening, SI, or plots. After running collection with a refresh, rerun screening and SI/plot notebooks to update the saved outputs.

The draft’s “all approximately 24 million PubMed abstracts” most likely refers to the searchable PubMed corpus against which the keyword queries were run, not to records downloaded into this project. The collection code sends targeted queries to PubMed; it does not download and locally filter every PubMed record. As of 15 October 2026, NCBI’s database directory describes PubMed as containing **more than 41 million citations**, while the PubMed homepage rounds this to **more than 40 million** ([NCBI database directory](https://www.ncbi.nlm.nih.gov/search/), [PubMed homepage](https://pubmed.ncbi.nlm.nih.gov/)). These are rounded citation counts, not exact abstract counts. The repository does not establish the historical 24-million figure or the date it describes, and “PubMed records” may be more accurate than “abstracts” because not every indexed record has an abstract. Figure 1 should show the corpus as the search universe, then distinguish the queried candidate results and later screened outputs as pair–PMID associations or pair–PMID instances, never call those counts “abstracts”. The checked-in ranked file has 1,279 rows; applying the configured positive pathogenicity-index filter gives 915 pathogens.

**Still needed.** The current candidate-level accepted count is reconciled: 306,718 pair–PMID–year rows match both the thresholded predictions and saved screened table. Record the final run date and rerun downstream outputs after any refreshed PubMed collection so all counts share the same snapshot. Figure 1 should show raw retrieval associations and deduplicated review units separately if both are useful, with each unit named precisely.

## 2 What is the correct analysis period: 2000–2023 or 2000–2025?

**Answer so far.** The active configuration uses different windows for different purposes:

- PubMed collection currently covers 2000–2026.
- The saved screened dataset spans 2000–2026 and matches the current candidate-level System 1 thresholded predictions.
- Pair eligibility is filtered by at least 50 accepted publications from 2000–2025, with yearly counts below 3 unavailable to the SI fit and at least 10 valid years required for pair-level forecasting.
- The pooled SI table contains years 2000–2026, but yearly counts below 3 are unavailable to the SI model.
- The auxiliary validation/ranking fit uses 2000–2023.
- The displayed historical curve is fitted through 2025, with 2025 as the reference year and 2030 as the five-year forecast horizon.
- The 2024 and 2025 observations provide the two post-fit validation years in the saved 127-pair interesting shortlist. Although 2026 is present in collection and yearly SI outputs, it is not part of the 2025-reference-year ranking output.

The notebooks do not record a separate execution timestamp for the collection run. `collection_run_min_date` and `collection_run_max_date` are query bounds, not the date on which the run was performed. A final run date should be added from the data-generation record before submission.

**Draft mismatch.** The current Methods draft says pairs were retained at an average of at least one paper per year over 2000–2023. The checked-in configuration instead sets a minimum total of 50 screened papers over 2000–2025, plus the yearly-count and valid-year constraints above. Replace the draft criterion with the final run’s exact rule.

**Proposed manuscript wording.** “We retrieved records published from 2000 through 2026. Pair eligibility and displayed historical trajectories used data through 2025. We fitted the validation/ranking model through 2023, evaluated it against 2024–2025, and extrapolated the 2025 trajectory to 2030.” Add the final collection/rebuild date from a run record; the `collection_run_min_date` and `collection_run_max_date` fields are PubMed query bounds, not execution timestamps.

## 3 What model was used for the LLM screening, and where should the prompt and model details be provided?

**Answer so far.** The current configuration selects System 1 rather than the LLM for the saved screened dataset: Jev `typesafe/jev-1.13`, run ID `jev-1.13-amr-support-v1`, question version `amr-pair-support-noul-v1`, probability threshold 0.75, with up to 128 workers. The persisted System 1 table records served model `typesafe/jev-1.13-20260917` and provider `TypeSafe` for successful rows. Its question asks whether the title and abstract provide substantive AMR evidence or context for the specified pathogen–antibiotic pair, including applicable class context, without inferring support from unrelated co-mentions. The separate OpenAI LLM path is configured as `gpt-5-nano`, `reasoning_effort="minimal"`, compact JSON output, two workers, and one unified screening pass, but its persisted decision table is not the active source in this snapshot. The LLM user payload is deterministic JSON containing:

```json
{"target_pathogen":"...","target_antibiotic":"...","title":"...","abstract":"...","pmid":"..."}
```

The current production LLM prompt is defined in `amr/screening/core.py` and displayed by `notebooks/03_query_examples.ipynb`. It requires `strict`, `yes`, `no`, or `uncertain`, a primary AMR category, and compact JSON with `decision`, `primary_category`, `target_pair_relation_supported`, and `is_amr_relevant`. It disallows unsupported co-mention transfer but allows explicit pair statements, shared study-panel context, and explicit antibiotic-class attribution. The saved System 1 table records question version/model metadata; report the exact Jev question version and served model from that run, not the OpenAI prompt as though it produced the active screened dataset.

Those five primary categories (environmental/animal/non-human, clinical/outbreaks/reports, surveillance, mechanisms of resistance/genetics/laboratory studies, and other, plus `NONE` for unsupported pairs) belong to the LLM workflow. The active System 1 screen does not emit these categories; any category analysis based on the saved LLM table should be described as a separate annotation layer.

The LLM decision parquet retains rationale and evidence columns even though the active LLM configuration requests compact output. Treat those columns as persisted schema/history, not proof that the compact prompt generated them. The final manuscript should document the System 1 question/model/version if System 1 remains the final source; if LLM screening is selected instead, document its exact prompt/cache key and decision inclusion rule. Include the exact active instrument in Supplementary Material or a versioned repository file.

**Still needed.** The notebooks contain no labelled manual test set, precision/recall estimate, or independent screening benchmark for either screening route. Do not claim measured classification accuracy without an additional validation exercise.

## 4 What broad AMR terms were used in the search?

**Answer so far.** The active pair query adds one AMR/susceptibility clause:

```text
"Drug Resistance, Bacterial"[MeSH Terms]
OR "Drug Resistance, Microbial"[MeSH Terms]
OR "Microbial Sensitivity Tests"[MeSH Terms]
OR resistan*[Title/Abstract]
OR nonsusceptib*[Title/Abstract]
OR susceptib*[Title/Abstract]
OR "antibiotic sensitivity"[Title/Abstract]
OR "antimicrobial sensitivity"[Title/Abstract]
OR "drug sensitivity"[Title/Abstract]
```

The pathogen and antibiotic parts are separate synonym-aware clauses, both queried in Title/Abstract. Antibiotic names use canonical names, synonym variants, and split hyphen/slash forms. The AMR/susceptibility cues form a single OR clause, and publication dates are constrained to 2000/01/01–2026/12/31 for the active slice. Existing interval-cache rows were collected under the previous query and need to be refreshed before the saved dataset reflects this revised search.

The query vocabulary currently contains 245 canonical antibiotics and 422 antibiotic synonym rows. The pathogen run list contains 915 canonical names; `assets/bacteria_synonym_lookup.csv` currently contains 24,932 synonym rows. The earlier notebook record of 19,956 rows does not match the current asset, so report the vocabulary and lookup version from the final run after refreshing the query cache. `refresh_cached_intervals: true` is enabled in the active config; rerun collection after any query or vocabulary update.

The words “resistance”, “susceptibility”, “treatment”, “testing”, “mechanism”, “surveillance”, “stewardship”, and “policy” occur in the LLM screening policy, but they are not all independent PubMed retrieval keywords. This distinction should be stated clearly. Antibiotic classes and coarsened categories are used downstream for optional grouped analyses; the active main SI analysis is at the individual-antibiotic level.

## 5 Are “species–antibiotic combinations” the correct unit of analysis?

**Answer so far.** The implementation uses canonical `pathogen` and `antibiotic` fields and models one pathogen–antibiotic pair per year. The active analysis level is individual antibiotic, not antibiotic class. “Species–antibiotic pair” is understandable for bacterial examples, but “pathogen–antibiotic pair” is more faithful to the configured input list and should be used consistently in the paper. The phrase “resistance-species pair” in the Introduction is misleading and should be replaced with “pathogen–antibiotic pair” or “pathogen–antibiotic resistance association”, depending on the sentence.

The saved screened table contains 213 observed antibiotic labels; the saved SI fit contains 644 eligible pairs across 58 pathogens and 96 antibiotics; the saved interesting-pair shortlist contains 127 pairs across 19 pathogens and 48 antibiotics. These are different stages and should not be conflated. Regenerate the counts after a future collection refresh or screening change.

## 6 Are the model equation, method, and references sufficiently clear?

**Answer so far.** The code confirms the following model:

```text
log10(1 + C_ijt) = intercept
                 + beta_p log10(1 + Bp_it)
                 + beta_a log10(1 + Ba_jt)
                 + pathogen one-hot effects
                 + antibiotic one-hot effects
                 + error
```

The pooled baseline is fitted with `sklearn.linear_model.Ridge(alpha=1.0, fit_intercept=True)`. The expected log count is back-transformed as `10^yhat - 1`, clipped at zero, and SI is the residual on the log10 scale:

```text
SI_ijt = log10(1 + C_ijt) - expected_log_count_ijt
```

For the current saved output, yearly pair counts below 3 are treated as unavailable for SI fitting and interpolation. Eligibility requires at least 50 accepted publications in 2000–2025; the saved SI table contains 644 pairs after this and other background/coverage filters, and at least 10 valid years are required for pair-level forecasting. The 644 count follows the current System 1 screened parquet and supersedes the earlier 187-pair number.

Temporal modelling uses a degree-one LOESS smoother with span 0.6 on the log10-transformed pair counts and a separate LOESS curve on the expected log counts. Predicted SI is the difference between those two curves. If the LOESS implementation is unavailable or cannot fit, the code falls back to linear regression on the transformed counts. The current production outputs use no tail damping, so the fitted boundary slope is continued on the log scale outside the fitted interval.

**Still needed.** Add citations for ridge regression, LOESS, and any methodological precedent used to justify residualising against background publication attention. Also decide whether the paper wants to call the one-hot terms “pathogen- and antibiotic-specific effects” or simply “categorical fixed effects”.

## 7 What bounds should be added?

**Answer so far.** The code generates empirical ±1 standard-deviation bands around the smoothed log-count curve and propagates the same residual standard deviation to the SI curve. These are based on the sample standard deviation of the in-sample LOESS residuals. They are not confidence intervals or predictive intervals and do not include uncertainty in the pooled baseline, future background attention, model selection, or the choice of LOESS span.

The ranked output records this residual spread as `si_loess_residual_std` and `si_loess_residual_variance_mean`. The paper should call them “empirical residual bands” or “one-standard-deviation trend bands”, not statistical uncertainty intervals. The caption and Methods should state that the bands describe model residual variation only.

## 8 Is “volume” interpretable enough for readers?

**Answer so far.** The code uses literal publication counts at the input stage and predicted publication counts after smoothing. “Publication volume” is therefore interpretable, but it is less precise than “annual screened AMR publication count” or “predicted annual publication count”. The manuscript should reserve “scientific interest” for SI and use “publication count” for observed or predicted counts.

## 9 Are the phrases “irrespective of AMR” and “Needed?” necessary?

**Answer so far.** The retrieval query requires pathogen and antibiotic terms plus a resistance, nonsusceptibility, susceptibility, or microbial-sensitivity cue. The configured System 1 classifier then scores whether the paper supports the requested pair; the saved screened output applies the configured 0.75 threshold. This is a candidate-retrieval filter, not a claim that every retrieved paper discusses resistance to that exact antibiotic in that exact pathogen.

The word “Needed?” is not represented by a separate analysis parameter in the notebooks. It should be removed or replaced with a concrete statement describing whether the relevant filter or field is retained.

## 10 How should Figure 2 be described?

**Answer so far.** The notebooks support a narrower claim than the current draft wording: the workflow shows that broad PubMed retrieval followed by automated pair screening can recover sustained and changing publication signals for selected pathogen–antibiotic pairs. It does not, by itself, demonstrate that the method measures resistance prevalence or biological evolution.

The current saved SI outputs provide 644 modelled pairs and 127 pairs in the strictly filtered interesting shortlist. The shape-screen artifacts contain up to 10 convex U-shaped curves (the configured display cap), no concave inverted-U curves, and no decreasing-trend examples in the saved decreasing table. This is useful for finding candidate contrasts, but it is not yet a principled four-example selection. The four examples should be selected from the actual plotted pairs using an explicit rule and supported with external biological or surveillance references. Recheck these results after rebuilding the screen.

## 11 Should the validation, uncertainty, and prioritisation material be merged with the preceding Methods section?

**Answer so far.** The code separates these functions clearly: collection in notebook 04, candidate preparation and automated screening in notebooks 04a–04c, SI construction and filtering in notebook 05, and GLASS comparison in notebooks 06–07. For the paper, a coherent Methods sequence would be: retrieval and yearly aggregation; automated pair screening; pooled SI model; temporal smoothing and extrapolation; validation and empirical residual bands; prioritisation; GLASS matching.

The saved ranking fit uses 2000–2023, evaluates 2024–2025 as two holdout years, and extrapolates a 2025 reference curve to 2030. The 127 selected pairs all use `selection_mode = strict_filter`, so the configured fallback shortlist was not used. All 127 have two post-fit validation years; their mean absolute percentage SI errors range from about 0.9% to 96.1%, below the configured 100% cutoff. Recompute after a future collection refresh or screening change.

## 12 What additional references are required across the Introduction, Methods, and Results?

**Answer so far.** The notebooks cannot supply the missing literature references. They do establish which technical statements need to be cited: synonym-aware PubMed retrieval, the pathogen priority source, the AWaRe/antibiotic vocabulary source, ridge regression, LOESS smoothing, automated abstract classification, GLASS definitions, and the biological or surveillance basis for the Figure 2 examples. The draft’s “783 confirmed human pathogens” wording should be checked against the 915 configured pathogen names and 789 pathogen labels observed in candidate retrieval; the pipeline is not restricted to human clinical specimens. Likewise, do not imply that all retrieved records are human-infection abstracts.

The current draft still contains unresolved `REF` placeholders in the Introduction, Results, and Discussion. These should remain open until the final examples and exact GLASS comparison are fixed; otherwise the references may be attached to claims that change during revision.

## 13 What talking points should remain in the Discussion?

**Answer so far.** The notebooks support these discussion points:

- The method measures published scientific attention, not resistance prevalence, incidence, clinical burden, or transmission.
- The retrieval is broad and not restricted to human clinical specimens. The separate saved LLM category annotations include environmental, animal, and non-human contexts; System 1 itself does not assign these categories.
- A single abstract can contribute to several target pairs, and pair support can arise from a shared study panel or class-level context.
- The current method is intentionally low-resolution at the species–antibiotic level and does not record geography or publication lag.
- Neither the current System 1 screen nor the saved LLM screen has measured gold-standard accuracy in the notebooks; this must be presented as an unresolved validation gap.
- Notebook 08 can inspect suspicious pairs. A prior Escherichia coli–vancomycin inspection returned 17 accepted title rows from 2003–2026, including studies of intrinsic Gram-negative resistance, vancomycin-based interventions, environmental or abattoir isolates, and broader susceptibility panels. Treat this as an illustration from the saved LLM decisions, not as a verified count from the active System 1 screen; it shows why an accepted pair should not be interpreted as a prevalence estimate.

The final comment’s wording “Data is fuzzy but trends shine through” is a possible discussion message, but it should be rewritten more precisely and supported by the selected examples and validation results.
