# Master’s Project Proposal: Discovering What Drives Scientific Interest in AMR

## Project overview

Scientific interest in an antimicrobial resistance (AMR) pathogen–antibiotic pair can rise for many reasons: a newly recognised resistance mechanism, an outbreak, a change in diagnostic practice, or increased surveillance, for example. This project will investigate whether concepts found in research abstracts help explain or anticipate changes in scientific interest. It will combine time-series forecasting with large language models (LLMs) to discover, classify, refine, and report candidate explanations.

The project builds on the existing AMR scientific-interest pipeline and the concept-discovery workflow in notebooks 10a–10f. The student will assess and improve this workflow, with particular attention to whether its concepts are meaningful, reproducible, and useful on held-out pathogen–antibiotic pairs.

## 1. How we define scientific interest (SI)

For each pathogen–antibiotic pair and year, the project counts screened AMR-related papers. A pair’s raw count is not a fair measure of its relative scientific attention on its own: it may rise because the overall literature about that pathogen or antibiotic has grown.

The current SI score adjusts for this background. The pipeline models the pair’s log publication count using the pathogen’s and antibiotic’s broader publication volumes, together with pathogen and antibiotic effects. SI is the residual:

> **SI = observed log publication count − expected log publication count**

Thus, positive SI means that the pair received more attention than expected given these background trends; negative SI means less. Pair-years below the configured minimum publication-count threshold are treated as unobserved for modelling. SI is an adjusted measure of relative research attention, not a measure of AMR prevalence, disease burden, or clinical risk.

## 2. Why investigate what is behind SI changes?

SI describes *how much* attention a pair receives relative to expectation, but not *why* that attention changes. Understanding the topics and events that precede increases could help explain different research trajectories and identify recurring patterns across pairs. Candidate drivers might include outbreak evidence, treatment failure, genomic evidence, new diagnostics, or policy changes.

The goal is not to assume that any one topic causes attention to rise. Instead, we will test whether abstract-derived signals add useful information about future SI after accounting for the pair’s own recent SI history. We will look for both:

- **General concepts** that contribute across many pathogen–antibiotic pairs.
- **Specialist concepts** that are informative for a smaller, coherent subset of pairs.

## 3. Granger-causality analysis: the predictive question

In a Granger-style analysis, a time series X is said to *Granger-predict* a time series Y if past values of X improve predictions of future Y beyond what can be predicted from Y’s own past. This is a test of temporal predictive value. It does not, by itself, show that X causes Y.

Here, Y is yearly SI for a pathogen–antibiotic pair. X is the yearly prevalence of a concept in abstracts associated with that pair. We will compare:

1. A baseline model using the pair’s previous SI values.
2. An augmented model using previous SI values **and** previous concept prevalence.

The current workflow uses a fixed three-year lag window and evaluates direct forecasts one, two, and three years ahead using rolling forecast origins. Forecast error is compared between the models. A positive forecast gain means the concept model predicts better under this setup. Results are summarised across pairs to distinguish average usefulness from strong performance in a subset, and evaluated on development pairs before a separate held-out confirmation set is used.

## 4. Using LLMs to discover and explain candidate concepts

LLMs will support two complementary tasks: generating candidate concepts and applying well-defined concepts consistently to abstracts. These tasks will be kept distinct so that a generated explanation is treated as a hypothesis to test, not as a result.

### Concept discovery with a generative LLM

The workflow starts with a reviewed seed catalog of broad concepts. A generative LLM can then be given development-only abstracts that score as positive for a selected parent concept and asked to propose more specific child concepts. Proposals should be scientifically coherent and detectable from title and abstract text, with clear definitions and inclusion and exclusion criteria. The student will review and revise these proposals before they enter the catalog.

The LLM that proposes concepts will not receive SI values, future outcomes, pair-level forecast results, or confirmation examples. This separation reduces the risk of shaping concept definitions to fit the outcomes they are later tested against.

### Abstract classification with System 1 LLMs

Once a concept definition has been reviewed, the project uses the configured System 1 classification service (currently Jev via the typed NOUL interface) to judge whether each abstract contains evidence for that concept. The classifier receives the title, abstract, and operational concept definition; it does not receive pair identity or SI outcomes. Its score is a model-assigned confidence in concept presence, not a calibrated probability of future interest or a causal effect.

Scores are aggregated into annual concept-prevalence signals for each pair. Classification results are cached by abstract and concept definition so that unchanged labels can be reused when the catalog is refined. The student can examine classification coverage, ambiguous cases, and consistency across concepts or model settings.

### Automated refinement and explanation

The automated workflow connects proposal, classification, and forecasting in repeated development rounds:

1. Score the current catalog and build yearly concept signals.
2. Evaluate each concept’s rolling forecast gain, including average gain across pairs and upper-tail gain for specialist performance.
3. Select promising concepts for refinement and ask the generative LLM for child-concept hypotheses.
4. Use System 1 to screen proposed concepts for semantic duplicates, classify new concepts, and measure their signals.
5. Test children as an alternative representation of their parent, and test candidate concepts for added predictive value alongside the current concept population.
6. Review the resulting concepts, forecasts, and pair-level summaries; freeze the selected catalog before held-out confirmation.

The automated report can rank concepts associated with a pair’s modelled forecast contribution. These attributions are exploratory and assumption-dependent: they are not identified causal effects. The main confirmatory result is whether the frozen concept signals improve held-out forecasts over the SI-history baseline.

## Proposed student contribution

The student will use the existing notebooks and code as a starting point, then document and evaluate the analysis with a manageable subset of pathogen–antibiotic pairs. A suitable project will:

- Check the SI and concept-signal definitions and explain their limitations.
- Review LLM-generated concept proposals and System 1 abstract classifications for clarity and reproducibility.
- Evaluate candidate concepts with the fixed rolling forecast design, reporting horizon-specific as well as aggregate results.
- Compare broad concepts with refined children and inspect whether predictive gains generalise to held-out pairs.
- Provide a careful interpretation that separates prediction, model-based attribution, and causal claims.

## Expected outcome

The project should produce a reviewed concept catalog, an auditable account of how LLM proposals and labels were generated, and an evaluation of whether abstract-derived concept signals improve predictions of SI. A useful outcome may be a small set of robust predictors, evidence that some concepts are pair-specific, or a finding that the current concepts and models add little beyond SI history. Each outcome will clarify what abstract content can—and cannot—tell us about changes in scientific attention to AMR.

## Project references

- SI implementation: `amr/si/core.py` and notebook `notebooks/05_model_scientific_interest.ipynb`.
- Concept-discovery workflow: `docs/si_concept_discovery.md` and research rationale in `docs/si_concept_discovery_plan.md`.
- Notebooks: `notebooks/10a_prepare_si_concept_corpus.ipynb` through `notebooks/10f_automated_concept_refinement.ipynb`.
