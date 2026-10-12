# Executive Summary: AMR Scientific-Interest Concept Search

## Purpose

The concept search asks whether themes reported in antimicrobial-resistance (AMR) abstracts—such as diagnostics, outbreak evidence, or treatment failure—provide an early signal of rising scientific interest in a pathogen–antibiotic pair. The goal is to find a small, interpretable set of concepts that improves forecasts of later literature attention. This is a predictive analysis; a forecast gain does not establish that a concept causes scientific interest to rise.

## Approach

The workflow in notebooks [10a–10f](../notebooks/) starts from the screened pair-year literature and the existing scientific-interest series. It classifies each unique abstract against operationally defined concepts, then averages article scores into yearly concept-prevalence signals for each pair. Nine broad concepts seed the search.

Pairs are assigned deterministically to development (80%) or confirmation (20%). Concept proposals and model selection use development data only. A rolling-origin forecast compares a scientific-interest-only baseline with models that add lagged concept signals. It predicts one, two, and three years ahead using three annual lags. Concepts are assessed both for average forecast improvement across pairs and for a robust upper-tail improvement, so a concept can be useful for a subset of pairs without being useful everywhere.

Notebook 10f automates repeated concept refinement: it proposes narrower child concepts for selected candidates, screens for semantic duplicates, classifies the new concepts, and evaluates them on development pairs. Notebook 10e is the held-out confirmation stage and should be run only after the concept catalog is frozen. Exploratory pair-level attributions are reported separately; they depend on stronger assumptions and are not identified causal effects.

## Current readout

The available automated development pilot is restricted to three pathogens and has 27 eligible development pairs in its saved Pareto summary. Outbreak evidence was the strongest seed concept on the two search objectives, with mean forecast gain of about −0.000044 and 90th-percentile gain of about +0.002622 in mean-squared-error units. The near-zero, slightly negative average alongside a positive upper tail is compatible with a possible specialist signal, but it is weak evidence at this sample size.

The pilot proposed and evaluated three outbreak-evidence subtypes: epidemiologically linked outbreaks and case clusters, molecularly defined resistant-organism clusters, and geographic or inter-setting clonal dissemination. The next refinement step found no eligible Pareto-front parent to expand. This is a useful proof of workflow, not evidence that the refined catalog is validated or that the concepts explain changes in attention.

## Implications and next steps

The workflow now supports a transparent path from abstract-level concept definitions to out-of-sample forecast evaluation. Its strongest current signal is that outbreak evidence may be more informative for a subset of pairs than as a general predictor. The available pilot does not show broad average predictive improvement.

Before making a substantive claim, run the search on the intended full pathogen set, review and freeze the final concept definitions using development results, then run notebook 10e on the untouched confirmation pairs. Report the horizon-specific results, pair counts, and both average and upper-tail gains. Treat any pair-specific attribution as exploratory and assumption-dependent.

## Scope and interpretation

The workflow uses model-assigned abstract-presence scores, which are not calibrated probabilities or measures of paper importance. Forecast results depend on the screened corpus, the current scientific-interest measure, concept definitions, and model specification. Pair-level splitting allows the same pathogen to occur in both partitions, so confirmation tests generalization to held-out pathogen–antibiotic pairs rather than entirely unseen pathogens. Repeated discovery rounds can overfit development results, which is why confirmation data must remain unused until the catalog is frozen.

For implementation details and output definitions, see [Scientific-Interest Concept Discovery](si_concept_discovery.md); for the research rationale, see [the concept-discovery plan](si_concept_discovery_plan.md).
