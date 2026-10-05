# Questions from Michiel Vos to Resolve

These questions and unresolved points are extracted from the Word comments authored by **Vos, Michiel** in *AMR SI first draft.docx*. The answers below are a working draft based on the current notebooks, source code, and generated parquet outputs. They are ordered by priority and should be checked against the final intended run before being inserted into the manuscript.

## Important provenance issue

The current outputs contain a 2000–2026 collection, but the scientific-interest analysis is configured in separate windows: 2000–2023 for the validation/ranking fit, 2000–2025 for the publication-volume filter and displayed historical curves, and 2025–2030 for the five-year projection. Also, the active configuration carries forward `strict` LLM decisions only, whereas a notebook comment describes the accepted set as `strict + yes`. The manuscript and Supplementary Material should state one final, reproducible configuration.

## 1 What are all the steps and counts in Figure 1?

**Answer so far.** The current collection and screening outputs support the following pipeline:

1. The configured collection uses 915 pathogen names with positive pathogenicity index and 245 canonical antibiotics. The active query slice is 2000–2026.
2. The raw pair–PMID cache contains 197,354 pathogen–antibiotic–PMID associations, covering 37,007 unique PMIDs, 12,093 pair keys, 573 pathogen labels, and 198 antibiotic labels before yearly aggregation.
3. After publication-year assignment and deduplication to the LLM review unit, the candidate table contains 186,724 unique `(PMID, pathogen, antibiotic, year)` rows, covering 36,220 unique PMIDs. The yearly candidate table contains 41,531 pair–year rows and 8,893 unique pair keys.
4. The unified LLM decision table contains the same 186,724 review rows. Its decisions are: `strict` 54,757; `yes` 78,741; `no` 28,593; and `uncertain` 24,633. There are 448 rows with an empty abstract in the persisted table.
5. With the active `strict`-only setting, the screened dataset contains 54,757 accepted pair–PMID–year decisions, represented as 23,355 yearly rows, 4,328 unique pathogen–antibiotic pairs, 201 pathogens, and 196 antibiotics, covering 2000–2026.

The current notebooks do not provide a defensible count for “all approximately 24 million PubMed abstracts”; that number appears in the manuscript, not as an output of the active collection notebook. Figure 1 should therefore distinguish PubMed’s total indexed corpus from the candidate pair-level counts above. It should not call 197,354 pair–PMID associations “abstracts”, because one abstract can map to multiple target pairs.

**Decision needed.** Confirm whether the figure should show the raw pair–PMID cache, the 186,724 deduplicated LLM review rows, or both. The second Figure 1 step is correct only if it is labelled as pair-level deduplication rather than abstract-level deduplication.

## 2 What is the correct analysis period: 2000–2023 or 2000–2025?

**Answer so far.** Both periods are used, but for different purposes:

- PubMed collection currently covers 2000–2026.
- The accepted screened dataset spans 2000–2026.
- Pair eligibility is filtered by at least 50 accepted publications from 2000–2025.
- The pooled SI table contains years 2000–2026, but yearly counts below 3 are unavailable to the SI model.
- The validation/ranking fit uses 2000–2023.
- The displayed historical curve is fitted through 2025, with 2025 as the reference year and 2030 as the five-year forecast horizon.
- The 2024 and 2025 observations provide the two post-fit validation years for the 26 selected pairs. Although 2026 is present in the collection, it is not part of the main 2025 reference-year ranking output.

The notebooks do not record a separate execution timestamp for the collection run. `collection_run_min_date` and `collection_run_max_date` are query bounds, not the date on which the run was performed. A final run date should be added from the data-generation record before submission.

**Proposed manuscript wording.** “We collected records published from 2000 through 2026. Pair eligibility and the displayed historical trajectories used data through 2025. We fitted the validation/ranking model through 2023, evaluated it against 2024–2025, and extrapolated the 2025 trajectory to 2030.” This should be revised if the final analysis excludes 2026 entirely.

## 3 What model was used for the LLM screening, and where should the prompt and model details be provided?

**Answer so far.** The active configuration specifies OpenAI `gpt-5-nano` with `reasoning_effort="minimal"`, compact output, two worker threads, and a single unified screening pass. The user payload is deterministic JSON containing:

```json
{"target_pathogen":"...","target_antibiotic":"...","title":"...","abstract":"...","pmid":"..."}
```

The production system prompt is defined in `amr/screening/core.py` and displayed by `notebooks/03_query_examples.ipynb`. It requires the model to classify the pair into exactly one of `strict`, `yes`, `no`, or `uncertain`, assign a primary AMR category, and return only compact JSON with `decision`, `primary_category`, `target_pair_relation_supported`, and `is_amr_relevant`. The prompt says not to infer pair support from keyword co-occurrence alone; it allows explicit pair statements, shared study-panel context, and explicit antibiotic-class attribution.

The five primary categories are environmental/animal/non-human, clinical/outbreaks/reports, surveillance, mechanisms of resistance/genetics/laboratory studies, and other. The code also uses `NONE` for unsupported pairs.

The current persisted decision table contains rationale and evidence columns even though the active configuration requests compact output. This suggests that the parquet includes fields retained from an earlier explanatory or backfill run. The final manuscript should report the prompt version used for the actual accepted dataset and include that exact prompt in the Supplementary Material or a versioned repository file.

**Still needed.** The notebooks contain no labelled manual test set, precision/recall estimate, or independent screening benchmark. The LLM section can currently document model and prompt provenance, but it cannot claim measured classification accuracy without an additional validation exercise.

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

The query vocabulary currently contains 245 canonical antibiotics and 422 antibiotic synonym rows. The notebook output records 19,956 pathogen synonym rows, while the current `assets/bacteria_synonym_lookup.csv` contains 24,932 rows; this is another provenance discrepancy to resolve before reporting a final vocabulary size. The pathogen run list itself contains 915 canonical pathogen names.

The words “resistance”, “susceptibility”, “treatment”, “testing”, “mechanism”, “surveillance”, “stewardship”, and “policy” occur in the LLM screening policy, but they are not all independent PubMed retrieval keywords. This distinction should be stated clearly. Antibiotic classes and coarsened categories are used downstream for optional grouped analyses; the active main SI analysis is at the individual-antibiotic level.

## 5 Are “species–antibiotic combinations” the correct unit of analysis?

**Answer so far.** The implementation uses canonical `pathogen` and `antibiotic` fields and models one pathogen–antibiotic pair per year. The active analysis level is individual antibiotic, not antibiotic class. “Species–antibiotic pair” is understandable for the bacterial subset, but “pathogen–antibiotic pair” is more faithful to the configured input list and should be used consistently in the paper. The phrase “resistance-species pair” in the Introduction is misleading and should be replaced with “pathogen–antibiotic pair” or “pathogen–antibiotic resistance association”, depending on the sentence.

The accepted screened data contain 196 observed antibiotics, while the SI publication filter leaves 187 eligible pairs across 26 pathogens; the yearly SI table has 56 antibiotic values after the displayed-model joins and filtering. These are different stages and should not be conflated.

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

For the current notebook run, yearly pair counts below 3 are treated as unavailable for SI fitting and interpolation. Pair eligibility is then restricted to 187 of 4,328 accepted pairs with at least 50 accepted publications in 2000–2025. At least 10 valid years are required for the pair-level interpolation/ranking stage.

Temporal modelling uses a degree-one LOESS smoother with span 0.6 on the log10-transformed pair counts and a separate LOESS curve on the expected log counts. Predicted SI is the difference between those two curves. If the LOESS implementation is unavailable or cannot fit, the code falls back to linear regression on the transformed counts. The current production outputs use no tail damping, so the fitted boundary slope is continued on the log scale outside the fitted interval.

**Still needed.** Add citations for ridge regression, LOESS, and any methodological precedent used to justify residualising against background publication attention. Also decide whether the paper wants to call the one-hot terms “pathogen- and antibiotic-specific effects” or simply “categorical fixed effects”.

## 7 What bounds should be added?

**Answer so far.** The code generates empirical ±1 standard-deviation bands around the smoothed log-count curve and propagates the same residual standard deviation to the SI curve. These are based on the sample standard deviation of the in-sample LOESS residuals. They are not confidence intervals or predictive intervals and do not include uncertainty in the pooled baseline, future background attention, model selection, or the choice of LOESS span.

The selected output records this residual spread as `si_loess_residual_std` and `si_loess_residual_variance_mean`. The paper should call them “empirical residual bands” or “one-standard-deviation trend bands”, not statistical uncertainty intervals. The caption and Methods should state that the bands describe model residual variation only.

## 8 Is “volume” interpretable enough for readers?

**Answer so far.** The code uses literal publication counts at the input stage and predicted publication counts after smoothing. “Publication volume” is therefore interpretable, but it is less precise than “annual screened AMR publication count” or “predicted annual publication count”. The manuscript should reserve “scientific interest” for SI and use “publication count” for observed or predicted counts.

## 9 Are the phrases “irrespective of AMR” and “Needed?” necessary?

**Answer so far.** The retrieval query requires pathogen and antibiotic terms plus a resistance, nonsusceptibility, susceptibility, or microbial-sensitivity cue. The LLM then screens whether the paper supports the requested pair and is AMR-relevant. This is a candidate-retrieval filter, not a claim that every retrieved paper discusses resistance to that exact antibiotic in that exact pathogen.

The word “Needed?” is not represented by a separate analysis parameter in the notebooks. It should be removed or replaced with a concrete statement describing whether the relevant filter or field is retained.

## 10 How should Figure 2 be described?

**Answer so far.** The notebooks support a narrower claim than the current draft wording: the workflow shows that a broad PubMed retrieval followed by LLM pair screening can recover sustained and changing publication signals for selected pathogen–antibiotic pairs. It does not, by itself, demonstrate that the method measures resistance prevalence or biological evolution.

The current SI outputs provide 187 modelled pairs and 26 strictly eligible pairs for the main forecast plots. The notebook also finds three convex U-shaped SI curves and no concave inverted-U curves under the configured screen; it finds no decreasing-trend examples under that screen. This is useful for choosing contrasting examples, but it is not yet a principled four-example selection. The four examples should be selected from the actual plotted pairs using an explicit rule and supported with external biological or surveillance references.

## 11 Should the validation, uncertainty, and prioritisation material be merged with the preceding Methods section?

**Answer so far.** The code separates these functions clearly: collection and screening in notebook 04, SI construction and filtering in notebook 05, and GLASS comparison in notebooks 06–07. For the paper, a coherent Methods sequence would be: retrieval and yearly aggregation; LLM screening; pooled SI model; temporal smoothing and extrapolation; validation and empirical residual bands; prioritisation; GLASS matching.

The current ranking fit uses 2000–2023, evaluates 2024–2025 as two holdout years, and extrapolates a 2025 reference curve to 2030. The 26 selected pairs all use `selection_mode = strict_filter`, so the configured fallback shortlist was not used. All 26 have two post-fit validation years; their mean absolute percentage SI errors range from 3.0% to 71.5%, below the configured 100% cutoff.

## 12 What additional references are required across the Introduction, Methods, and Results?

**Answer so far.** The notebooks cannot supply the missing literature references. They do establish which technical statements need to be cited: synonym-aware PubMed retrieval, the pathogen priority source, the AWaRe/antibiotic vocabulary source, ridge regression, LOESS smoothing, LLM-assisted abstract classification, GLASS definitions, and the biological or surveillance basis for the Figure 2 examples.

The current draft still contains unresolved `REF` placeholders in the Introduction, Results, and Discussion. These should remain open until the final examples and exact GLASS comparison are fixed; otherwise the references may be attached to claims that change during revision.

## 13 What talking points should remain in the Discussion?

**Answer so far.** The notebooks support these discussion points:

- The method measures published scientific attention, not resistance prevalence, incidence, clinical burden, or transmission.
- The retrieval is broad and not restricted to human clinical specimens; the LLM categories explicitly include environmental, animal, and non-human contexts.
- A single abstract can contribute to several target pairs, and pair support can arise from a shared study panel or class-level context.
- The current method is intentionally low-resolution at the species–antibiotic level and does not record geography or publication lag.
- The current LLM screen has no measured gold-standard accuracy in the notebooks; this must be presented as an unresolved validation gap.
- The current pipeline can inspect suspicious pairs through notebook 08. For example, the Escherichia coli–vancomycin inspection returns 17 accepted title rows from 2003–2026, including studies of intrinsic Gram-negative resistance, vancomycin-based interventions, environmental or abattoir isolates, and broader susceptibility panels. This illustrates why an accepted pair should not be interpreted as a prevalence estimate.

The final comment’s wording “Data is fuzzy but trends shine through” is a possible discussion message, but it should be rewritten more precisely and supported by the selected examples and validation results.
