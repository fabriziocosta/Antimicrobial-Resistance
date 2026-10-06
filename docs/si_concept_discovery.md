# Scientific-Interest Concept Discovery

This workflow extends the yearly scientific-interest (SI) analysis by testing whether recurring scientific concepts in AMR abstracts improve forecasts of later SI. It is designed to find broad and specialised **temporal predictors**. Predictive precedence is not proof that a concept causes scientific interest to rise.

For the research rationale and proposed analysis, see the [Scientific-Interest Concept Discovery research plan](si_concept_discovery_plan.md).

The five notebooks use the existing screened pair-year data, cached PubMed abstracts, and SI outputs. They do not change how the canonical SI series or high-interest ranking is produced.

## Workflow

Run notebook 05 first so that `outputs/parquets/yearly_si.parquet` is current. The abstract cache at the configured `paths.abstract_cache` must also be available.

| Notebook | Purpose |
| --- | --- |
| `10a_prepare_si_concept_corpus.ipynb` | Join screened pair-year PMID membership, abstracts, and observed SI; optionally limit the run to the top N pathogens; create the pair split and report abstract coverage and Jev request volume. |
| `10b_seed_and_refine_concepts.ipynb` | Start from the seed catalog, request optional child-concept hypotheses from development-only examples, and save a reviewed, versioned catalog. |
| `10c_classify_abstract_concepts.ipynb` | Classify unique abstracts against the current catalog and resume from the score cache. |
| `10d_evaluate_and_refine_concepts.ipynb` | Build yearly signals and evaluate candidate, population, conditional, and parent-versus-children models on development pairs. |
| `10e_confirm_si_concepts.ipynb` | Evaluate the frozen catalog on confirmation pairs and save the final tables and plot. |

Concept discovery is iterative. After classifying the seed catalog and reviewing development results, select a parent in notebook 10b, optionally ask the LLM for children, and manually review any proposal before changing the catalog. Run notebook 10c again for the revised catalog, then notebook 10d. Notebook 10d exposes `CURRENT_POPULATION`, `CANDIDATE_IDS`, and `PARENT_CHILD_REPLACEMENTS` so each round's comparisons are explicit. Repeat until the catalog is frozen. Notebook 10e is the final confirmation step; its confirmation switch is off and the expected catalog hash is blank by default to prevent accidental use before the catalog is frozen.

## Corpus and concept labeling

Notebook 10a joins `yearly_amr_screened_pairs.parquet` to the observed rows in `yearly_si.parquet` on pathogen, antibiotic, and year. It explodes the pair-year PMID lists and deduplicates each `(pathogen, antibiotic, year, PMID)` association. An abstract linked to multiple pairs is classified once by PMID and its score is reused for each pair association.

The pair-level split is deterministic: 80% development and 20% confirmation, with seed 42. Split assignment applies to pathogen–antibiotic pairs. For concept proposals, any PMID that appears in both splits is excluded from the example pool, so the LLM sees only development-exclusive abstracts while proposing definitions. After a catalog is defined, the classifier scores abstracts from both splits so the frozen concepts can be evaluated on confirmation pairs. SI values and forecast results are not included in proposal or classification prompts.

The initial catalog contains nine seed concepts: surveillance, outbreak evidence, treatment failure, mechanism discovery, genomic evidence, diagnostics, hospital transmission, agricultural use, and policy or guideline change. Each concept has a stable ID, definition, and inclusion/exclusion criteria. LLM proposals are hypotheses: review them before adding them to the catalog.

Notebook 10a exposes `NUM_PATHOGENS`. `None` uses every pathogen represented in the SI corpus; an integer selects that many available pathogens in descending `pathogenic_index` order, matching the yearly collection priority. Every antibiotic represented for a selected pathogen is retained. The split is then built over the resulting pathogen–antibiotic pairs.

The classifier sends one Jev Decisions API request per unique nonblank abstract, with a typed NOUL question for each concept. Jev sees only the title, abstract, concept definitions, and inclusion/exclusion criteria; it does not receive PMID, pathogen, antibiotic, SI, or split information. Scores from 0 to 1 are model judgments, not calibrated probabilities. Blank abstracts are excluded from the concept-signal denominator. Classification calls and child-concept proposals are disabled by default. Notebook 10c starts with a 25-request pilot; set `MAX_CALLS=None` to finish the corpus. Classification uses `system1.model` (default `typesafe/jev-1.13`) and the configured OpenRouter credentials. Child-concept proposal generation remains an optional configured LLM operation using development-only abstracts.

Scores are cached in `concept_scores.parquet`. Cache entries record PMID, concept ID, catalog hash, model, prompt version, probability, and status. Successful entries are reused; failed entries can be retried. Catalog snapshots are written under `catalog_versions/` as well as to the current `concept_catalog.json` file.

## Signals and forecasts

For each pair, year, and concept, the signal is the mean abstract score among classifiable screened pair-year articles. Years without an article association have a concept signal of zero when constructing lag features. SI values are used as recorded: the concept workflow does not smooth or interpolate them. Forecast samples that lack an observed SI target or any required consecutive SI lag are skipped.

Notebook 10d and 10e use direct forecasts at horizons 1, 2, and 3 years, with a fixed lag length of three. Each horizon uses an expanding historical window. The model has pair-specific intercepts and shared lag slopes. In confirmation, shared slopes are learned on development pairs; a confirmation pair's intercept uses only its SI history available at the forecast origin. An origin is eligible only after at least ten SI observations for that pair are available. Training rows at every origin use targets no later than that origin.

Forecast loss is mean squared error. A positive gain means the candidate model has lower error than its reference model. Notebook 10d reports:

- **Generality:** mean forecast gain across pairs.
- **Specialist strength:** the 90th percentile of pair-level forecast gain.
- **Conditional gain:** improvement of the current population plus a candidate over the current population alone.
- **Parent-versus-children gain:** improvement from replacing a parent signal with its child signals.

Horizon-specific summaries are retained. The aggregate uses equal weights across horizons 1–3 and averages each pair's gains across available horizons before computing population scores. Concept survival and catalog edits are deliberate review decisions in notebook 10d; confirmation-pair results must not be used to make those decisions.

## Outputs

All concept-search artifacts are local under `outputs/concept_search/` and are ignored by Git. Notebook 10a creates:

- `pair_split.csv`: development/confirmation assignment for each pair;
- `pair_year_abstract_corpus.parquet`: pair-year-PMID membership, observed SI, and split, without duplicated abstract text;
- `unique_abstracts.parquet`: one title/abstract record per PMID.

Later notebooks create the current concept catalog and immutable catalog snapshots, `concept_scores.parquet`, `yearly_concept_signals.parquet`, development and confirmation forecast tables and summaries, and figures under `figures/`. Child proposals are saved as `proposals_<parent-id>.json` for review.

The helper implementation is in `amr/si/concepts.py`; notebook-facing corpus paths and loading are in `amr/pipelines/concepts.py`. Synthetic tests are in `tests/test_si_concepts.py` and `tests/test_system1.py`.

## Interpretation and limitations

- A forecast gain shows additional predictive information under this model, not a biological or social mechanism.
- Abstract-level labels may be ambiguous. Review examples and classification coverage before interpreting a concept.
- The split is by pathogen–antibiotic pair, not by pathogen. It tests generalisation to unseen pairs, including pairs involving pathogens present in development data.
- The workflow uses the current screened corpus and SI definition. Changes to upstream screening or SI outputs require rerunning notebook 10a and rebuilding the downstream artifacts.
- Repeated concept exploration can overfit development results. Freeze concept definitions before running notebook 10e, and report held-out results separately.
