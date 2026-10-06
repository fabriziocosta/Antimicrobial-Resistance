# Scientific-Interest Concept Discovery

This workflow extends the yearly scientific-interest (SI) analysis by testing whether recurring scientific concepts in AMR abstracts improve forecasts of later SI. It is designed to find broad and specialised **temporal predictors**. Predictive precedence is not proof that a concept causes scientific interest to rise.

For the research rationale and proposed analysis, see the [Scientific-Interest Concept Discovery research plan](si_concept_discovery_plan.md).

The six notebooks use the existing screened pair-year data, cached PubMed abstracts, and SI outputs. They do not change how the canonical SI series or high-interest ranking is produced.

## Workflow

Run notebook 05 first so that `outputs/parquets/yearly_si.parquet` is current. The abstract cache at the configured `paths.abstract_cache` must also be available.

| Notebook | Purpose |
| --- | --- |
| `10a_prepare_si_concept_corpus.ipynb` | Join screened pair-year PMID membership, abstracts, and observed SI; optionally limit the run to the top N pathogens; create the pair split and report abstract coverage and Jev request volume. |
| `10b_seed_and_refine_concepts.ipynb` | Start from the seed catalog, request optional child-concept hypotheses from development-only examples, and save a reviewed, versioned catalog. |
| `10c_classify_abstract_concepts.ipynb` | Classify unique abstracts against the current catalog and resume from the score cache. |
| `10d_evaluate_and_refine_concepts.ipynb` | Build yearly signals and evaluate candidate, population, conditional, and parent-versus-children models on development pairs. |
| `10f_automated_concept_refinement.ipynb` | Run a configurable Pareto-guided loop that expands selected parents, classifies new children, and evaluates them on development pairs. |
| `10e_confirm_si_concepts.ipynb` | Evaluate the frozen catalog on confirmation pairs and save the final tables and plot. |

Notebook 10f is stand-alone: it rebuilds the same corpus and deterministic pair split as 10a, so 10a and 10b are optional prerequisites. `RESET_CATALOG_TO_SEEDS=True` starts the run from the immutable nine-seed reference and mutates a separate working catalog through its iterations; `False` starts from the current working catalog and adds another set of iterations. Catalog versions are saved as immutable snapshots. Use 10a separately when you want to inspect corpus preparation without starting the automated cycle. Its setup cell exposes corpus paths and pathogen/split settings, model and classification controls, forecast settings, refinement parameters, and `ATTRIBUTION_NUM_PAIRS` (default 3). `N_ITER`, `K_PARENTS`, and `NUM_CHILDREN` set the number of refinement rounds, the maximum Pareto-front parents expanded per round, and the requested children per parent. It evaluates the starting catalog first; each round selects up to `K_PARENTS` unexpanded concepts from the non-dominated frontier for average gain and 90th-percentile gain, prioritising average gain among the frontier. For the current Pareto survivor population, it reports leave-one-out conditional gains (with versus without each concept). For every expanded parent it also compares the Pareto survivor population with that parent replaced by its child set, and reports individual-child replacement estimates while keeping the other Pareto survivor concepts fixed. Generated children are classified and evaluated before they can be selected in the next round. All proposal examples, forecasts, and selection criteria use development data only. At the end, 10f also reports exploratory concept attributions for the top `ATTRIBUTION_NUM_PAIRS` development pairs, selected by their latest observed SI at or before the forecast-origin end year. The 10a–10e sequence is the inspectable version of the same workflow: 10a prepares the corpus and split; 10b proposes/reviews the catalog; 10c classifies abstracts with the same Jev cache; 10d builds signals and evaluates candidates/populations; 10e confirms the frozen population on held-out pairs. In 10f these development stages are wired together in one repeated loop. It uses the same corpus builder, concept cache, signal builder, and rolling forecast/scoring helpers. Use 10a–10e to pause between stages and inspect intermediate artifacts; use 10f to automate repeated development rounds. Run 10e only after the final catalog is frozen; update its expected hash to the final catalog version.

## Corpus and concept labeling

Notebook 10a joins `yearly_amr_screened_pairs.parquet` to the observed rows in `yearly_si.parquet` on pathogen, antibiotic, and year. It explodes the pair-year PMID lists and deduplicates each `(pathogen, antibiotic, year, PMID)` association. An abstract linked to multiple pairs is classified once by PMID and its score is reused for each pair association.

The pair-level split is deterministic: 80% development and 20% confirmation, with seed 42. Split assignment applies to pathogen–antibiotic pairs. For concept proposals, any PMID that appears in both splits is excluded from the example pool, so the LLM sees only development-exclusive abstracts while proposing definitions. After a catalog is defined, the classifier scores abstracts from both splits so the frozen concepts can be evaluated on confirmation pairs. SI values and forecast results are not included in proposal or classification prompts.

The initial catalog contains nine seed concepts: surveillance, outbreak evidence, treatment failure, mechanism discovery, genomic evidence, diagnostics, hospital transmission, agricultural use, and policy or guideline change. Each concept has a stable ID, definition, and inclusion/exclusion criteria. LLM proposals are hypotheses: review them before adding them to the catalog.

Notebook 10a exposes `NUM_PATHOGENS`. `None` uses every pathogen represented in the SI corpus; an integer selects that many available pathogens in descending `pathogenic_index` order, matching the yearly collection priority. Every antibiotic represented for a selected pathogen is retained. The split is then built over the resulting pathogen–antibiotic pairs.

### Jev classification instructions

Notebook 10c sends one Jev Decisions API request per unique nonblank abstract, with one typed NOUL question per concept that needs a label. The request state contains only the title and abstract. Each concept question contains the common classification rubric plus the available concept record fields: `concept_id`, `name`, `definition`, `parent_id`, `scientific_rationale`, `inclusion_criteria`, `exclusion_criteria`, `representative_evidence`, `closest_competing_concept`, `difference`, `expected_temporal_relationship`, `testable_hypothesis`, and `rejection_conditions`. Empty or absent fields are omitted. The nine seed concepts currently have definitions and inclusion/exclusion criteria; reviewed child concepts can provide the additional rationale, evidence, and hypothesis fields.

The rubric asks Jev to:

- judge substantive evidence, analysis, methods, or synthesis in the supplied title/abstract, rather than a keyword mention, background-only statement, or future-work suggestion;
- require the concept's AMR connection and apply its inclusion/exclusion criteria;
- use `difference` and `closest_competing_concept` to distinguish adjacent concepts, while allowing multiple labels when each is independently supported;
- for a child concept, require evidence for the narrower child definition rather than inferring it from a parent match;
- use rationale, representative evidence, temporal relationship, hypothesis, and rejection conditions only to understand the intended construct, never as evidence about the current paper;
- avoid inferring future scientific-interest changes or causation from topic match, and avoid using pathogen, antibiotic, year, pair identity, SI, split, or outcome data, which are not sent.

The typed NOUL answer is Jev's confidence in the proposition that the abstract contains evidence for the concept under its operational definition. It is a model-assigned presence score, not a calibrated probability, paper importance score, forecast, or causal-effect estimate. Blank abstracts are skipped and are excluded from the concept-signal denominator. Notebook 10c is configured to classify the full selected corpus (`MAX_CALLS=None`) and resume from the cache if interrupted. Classification uses `system1.model` (default `typesafe/jev-1.13`) and the configured OpenRouter credentials. Child-concept proposal generation remains an optional configured LLM operation using development-only abstracts.

Notebook 10c displays a sample request state, the common rubric, and the full question for the most fully specified concept in the active catalog before any call. The rubric/template has a prompt version recorded in the score cache. Updating it intentionally causes labels to be recomputed under the new instructions; within one prompt version, successful labels remain cached independently by PMID and concept-definition hash, so unchanged concepts are reused when a catalog is expanded.

Scores are cached in `concept_scores.parquet`. Cache entries record PMID, concept ID, concept-definition hash, catalog hash, model, prompt version, probability, and status. Successful labels are reused independently per concept: adding a child or revising one definition only sends Jev the missing or changed concept questions. Reused labels are associated with the active catalog hash for downstream signals; failed labels can be retried. Catalog snapshots are written under `catalog_versions/` as well as to the current `concept_catalog.json` file.

## Signals and forecasts

For each pair, year, and concept, the signal is the mean abstract score among classifiable screened pair-year articles. Years without an article association have a concept signal of zero when constructing lag features. SI values are used as recorded: the concept workflow does not smooth or interpolate them. Forecast samples that lack an observed SI target or any required consecutive SI lag are skipped.

Notebook 10d and 10e use direct forecasts at horizons 1, 2, and 3 years, with a fixed lag length of three. Each horizon uses an expanding historical window. The model has pair-specific intercepts and shared lag slopes. In confirmation, shared slopes are learned on development pairs; a confirmation pair's intercept uses only its SI history available at the forecast origin. An origin is eligible only after at least ten SI observations for that pair are available. Training rows at every origin use targets no later than that origin.

Forecast loss is mean squared error. A positive gain means the candidate model has lower error than its reference model. Notebook 10d reports:

- **Generality:** mean forecast gain across pairs.
- **Specialist strength:** the 90th percentile of pair-level forecast gain.
- **Conditional gain:** improvement of the current population plus a candidate over the current population alone.
- **Parent-versus-children gain:** improvement from replacing a parent signal with its child signals.

Horizon-specific summaries are retained. The aggregate uses equal weights across horizons 1–3 and averages each pair's gains across available horizons before computing population scores. Concept survival and catalog edits are deliberate review decisions in notebook 10d; confirmation-pair results must not be used to make those decisions.

## Pair-specific exploratory attribution

Notebooks 10e and 10f save per-pair concept rankings after the catalog is frozen or the automated development loop completes. For each forecast horizon, a pooled linear panel model is fitted on development pairs with pair and origin-year intercepts, three lagged SI values, lagged publication volume, and the lagged prevalence signals for the complete frozen concept catalog. The model's shared coefficients are applied to each pair's concept history. The reported historical contribution contrasts that history with zero signal; a second contrast raises all three lagged yearly signals by 10 percentage points. The horizon-zero row averages the contributions across horizons 1, 2, and 3 and ranks concepts by absolute contribution. A pair must have at least five usable forecast origins; each origin must also satisfy `MIN_PRIOR_OBSERVATIONS` (default 10). Notebook 10e reports confirmation pairs. Notebook 10f reports up to `ATTRIBUTION_NUM_PAIRS` development pairs (default 3), ranked by their latest observed SI at or before `FORECAST_ORIGIN_END_YEAR`; its shared model is fit on development pairs and its selected pairs are drawn only from that partition. Each row includes the SI value and year used for selection. This is a model-based ranking, not a discovery of biological causation.

These estimates are labelled **assumption-dependent**: a causal interpretation requires a well-defined intervention on abstract prevalence, adequate support/positivity, and no unmeasured time-varying confounding. Available inputs do not capture all potential drivers such as AMR burden, policy changes, or surveillance intensity. The model adjusts for the other included concepts but that adjustment does not guarantee causal identification. Rank-deficient development fits, insufficient pair histories, and pairs without within-pair concept variation are retained with an explanatory status and left unranked as appropriate. Cluster-based intervals quantify coefficient uncertainty under the fitted model; they do not include classifier or concept-definition uncertainty. Keep these tables separate from rolling forecast gains, which answer a predictive question.

The concept-refinement phylogeny in notebook 10f labels nodes with compact codes such as `C001`, keeping long generated names and IDs out of the tree. The notebook displays a separate code-to-concept key below the plot; the companion `final_concept_phylogeny_nodes.csv` also records each full concept ID, name, definition, parent code, depth, and survivor/dead-end status. Blue paths lead to Pareto survivors, green nodes mark survivors, and `×` marks terminal dead ends. At the end of each refinement round, 10f prints cumulative OpenAI child-generation request and token counts. OpenAI's model API does not expose the account's remaining credit balance, so check the billing dashboard for that figure.

## Outputs

All concept-search artifacts are local under `outputs/concept_search/` and are ignored by Git. Notebook 10a creates:

- `pair_split.csv`: development/confirmation assignment for each pair;
- `pair_year_abstract_corpus.parquet`: pair-year-PMID membership, observed SI, and split, without duplicated abstract text;
- `unique_abstracts.parquet`: one title/abstract record per PMID.

Later notebooks create the current concept catalog and immutable catalog snapshots, `concept_scores.parquet`, `yearly_concept_signals.parquet`, development and confirmation forecast tables and summaries, per-pair assumption-dependent attribution tables (`pair_concept_assumption_dependent_horizons.parquet` and `pair_concept_assumption_dependent_ranking.parquet`), and figures under `figures/`. Notebook 10f stores run-specific proposals, Pareto frontiers, iteration history, forecast summaries, a readable phylogeny image and node index, plus `attribution_selected_pairs_by_si.csv`, `exploratory_pair_attribution_horizons.parquet`, `exploratory_pair_attribution_rankings.parquet`, and `exploratory_pair_attribution_top_concepts.png` under the run-specific `automated_refinement/` directory.

The helper implementation is in `amr/si/concepts.py`; notebook-facing corpus paths and loading are in `amr/pipelines/concepts.py`. Synthetic tests are in `tests/test_si_concepts.py` and `tests/test_system1.py`.

## Interpretation and limitations

- A forecast gain shows additional predictive information under this model, not a biological or social mechanism.
- Abstract-level labels may be ambiguous. Review examples and classification coverage before interpreting a concept.
- The split is by pathogen–antibiotic pair, not by pathogen. It tests generalisation to unseen pairs, including pairs involving pathogens present in development data.
- The workflow uses the current screened corpus and SI definition. Changes to upstream screening or SI outputs require rerunning notebook 10a and rebuilding the downstream artifacts.
- Repeated concept exploration can overfit development results. Freeze concept definitions before running notebook 10e, and report held-out results separately.
