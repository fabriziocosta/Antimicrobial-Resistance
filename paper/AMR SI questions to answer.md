# Questions to Answer Before Submission

Internal working list for completing the first draft. Resolve these in order because later claims depend on the earlier decisions and evidence.

1. **What is the paper’s single primary research question and central claim?** Decide whether AMR-SI is presented as a measure of scientific attention only, a forecasting tool for literature trends, or evidence about emerging resistance itself.

2. **Can the literature-retrieval pipeline be reproduced exactly?** Specify the search date and query logic, synonym lists, pathogen and antibiotic source lists, date range, title/abstract restrictions, deduplication, and every exclusion step.

3. **What is the final dataset used for analysis?** Report the numbers of records retrieved, screened, retained, unique PMIDs, species–antibiotic pairs, years, and exclusions, including the missing placeholders in Results and Figure 1.

4. **How accurate and reproducible is the LLM screening step?** Give the model and run details, prompt, sampling or validation design, error rates, and representative false-positive or false-negative examples, including the proposed “bad paper” example.

5. **Is the AMR-SI model fully specified and internally consistent?** Resolve notation and terminology, the ridge-penalty parameter, zero handling, log-scale versus back-transformed quantities, inclusion thresholds, and the inconsistent references to 2023 and 2025.

6. **Does the temporal model justify the 2030 projections?** Explain the training window, LOESS and fallback rules, boundary-slope extrapolation, validation through 2025, uncertainty bands, and why these are trend projections rather than calibrated epidemiological forecasts.

7. **Which four contrasting examples demonstrate that AMR-SI recapitulates known trends?** Choose examples using an explicit rule, describe the biological and publication trends, and add literature citations that support each interpretation without cherry-picking.

8. **Which pathogen–antibiotic pairs should be prioritised for surveillance in 2030?** State the exact threshold and ranking rule, show the shortlisted pairs and scores, and test whether the shortlist changes under reasonable sensitivity analyses.

9. **How should similarities and discrepancies with the WHO GLASS list be interpreted?** Separate scope and coverage differences from genuine signals, identify plausible additions and false positives, and avoid treating GLASS as a direct gold standard for scientific attention.

10. **What can the method not infer?** State clearly that publication attention is not resistance prevalence, incidence, clinical burden, geography, or transmission; address publication lag, strain-level versus species-level evidence, gene versus phenotype, class-only mentions, and unusual pairs such as vancomycin-resistant *E. coli*.

11. **Can readers access the materials needed to audit or update the analysis?** Decide what to provide for the code, search queries, synonym tables, prompts, model settings, screened outputs, figures, and tables, subject to licensing and data-governance constraints.

12. **Does the Discussion and Conclusion state the novelty and practical use at the right strength?** Replace placeholders, define how this complements laboratory surveillance, and refine the final message so that “from eminence to evidence-based science” matches the evidence actually presented.
