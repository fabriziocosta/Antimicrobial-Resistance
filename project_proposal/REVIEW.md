# Adversarial Review of the Current Proposal

## Overall judgment

The proposal is now much more coherent than earlier versions. It clearly states that the current implemented experiment is to predict future background-adjusted scientific interest (SI) from past SI or past pair-count trajectories, and that the proposed novelty is to show that additional literature-derived signals act as leading predictors of future SI. That is a substantial improvement.

The proposal is also more credible because it now emphasizes short-horizon forecasting, uses concrete dataset counts, includes an archived dense example, and explicitly allows antibiotic-class modeling when individual-drug series are too sparse. These changes resolve several earlier weaknesses.

Even so, a skeptical reviewer would still see this as a stronger scientometric forecasting proposal than a strong AMR early-warning proposal. The central remaining issue is not internal coherence anymore. It is external validity.

## Main strengths

- The target variable is now explicit and technically grounded: pair-level, background-adjusted SI.
- The current baseline is clearly defined as future-SI prediction from past SI.
- The novelty claim is much sharper: the contribution is to test whether external literature-derived signals improve prediction beyond that baseline.
- The proposal no longer depends on naive document-level co-occurrence; the LLM pair-disambiguation step is clearly explained.
- The forecasting scope is now credible because it prioritizes one-year to five-year horizons.
- The proposal now gives concrete scale numbers for PMIDs, pairs, pair-years, and usable trajectory counts.
- Antibiotic-class aggregation is a sensible and pragmatic answer to sparsity.

## Major concerns

### 1. The proposal still predicts scientific salience, not real-world AMR dynamics

The proposal now says this much more carefully, but the same basic limitation remains. It predicts future SI, which is a bibliometric quantity, not resistance prevalence, treatment failure burden, outbreak frequency, or surveillance-confirmed AMR emergence.

A skeptical reviewer will therefore say that the proposal is best understood as forecasting future scientific salience in AMR, not forecasting AMR in the epidemiological or clinical sense. The draft hints at this distinction, but it still leans on the language of “early warning” enough that the causal and public-health implications can sound overstated.

### 2. The external-validation problem remains unresolved

The proposal is now internally better defended, but it still does not show why improved future-SI prediction should matter beyond the literature itself. A reviewer will ask a simple question: if the model predicts that the literature will pay more attention to a pair in five years, why should that be considered useful unless it also connects to external AMR outcomes?

Without at least one validation layer against surveillance data, curated priority lists, or observed resistance trends, the proposal remains vulnerable to the critique that it models scientific attention as an end in itself.

### 3. “Causal predictors” is still a risky phrase

The proposal now uses Granger causality more carefully, but the causal language is still fragile. Granger causality identifies temporally ordered predictive relationships, not mechanistic causes. A hostile reviewer will push back if the proposal overinterprets these results as evidence of real causal drivers.

The safest version of the claim is that the project seeks leading predictors or temporally preceding predictive signals of future SI. If the text continues to imply more than that, reviewers will likely target it.

### 4. The candidate predictors may still be too endogenous to the target

Even with the sharper framing, a skeptical reviewer can still argue that the candidate predictors and the target all come from the same literature stream:

- SI is derived from screened publication counts.
- concept signals are extracted from those same screened abstracts.
- semantic signals are embeddings of those same screened abstracts.

This means the project may be learning richer encodings of publication dynamics rather than discovering genuinely separate sources of predictive information. The proposal needs to show not only that these signals precede SI, but that they add information not already recoverable from the history of SI itself.

### 5. Granger analysis may still be statistically unstable at the pair level

The proposal now gives the right trajectory counts, which helps. But those numbers also expose the problem:

- median years per pair is only 2
- only 2,078 pairs have at least 5 yearly observations
- only 1,120 pairs have at least 10 yearly observations

This is enough to support forecasting work, but it raises a serious question about the Granger component. Many pair-level series are simply too short for reliable lag-based inference. A reviewer will want to know:

- which subset of pairs are eligible for Granger testing
- what lag structure is feasible
- how stationarity and differencing will be handled
- how multiplicity will be controlled across many pair-signal tests

As written, the proposal presents Granger analysis as central, but it still reads as more statistically fragile than the forecasting baseline.

### 6. The transformer is more defensible now, but still not fully justified

The short-horizon framing and class-level aggregation both help the transformer case, but they do not eliminate the concern. A skeptical reviewer will still ask why a sequence transformer is the right next model for annual trajectories that are often short and heterogeneous.

The proposal should still expect this challenge:

- why a transformer rather than a smaller temporal model, boosted trees, or structured panel regressions?
- what is the effective sample size after horizon filtering, eligibility rules, and train/validation splits?
- how will the model avoid overfitting on a limited number of dense trajectories?

The current draft improves the situation by making the transformer a later-stage extension rather than the starting point. That was the right move. But the justification is not yet complete.

### 7. The role of `uncertain` examples remains underexplained

The proposal now reports both `yes`-only and `yes + uncertain` counts, which is useful. But the default pipeline still retains `uncertain` rows in the screened yearly dataset, and that deserves a stronger defense.

A reviewer will ask:

- why should `uncertain` publications contribute to SI?
- how sensitive are results to excluding them?
- is the broader dataset worth the added label noise?

This is now measurable, but the proposal does not yet say how this sensitivity will be tested.

## Secondary concerns

### 8. The dense figure is persuasive, but also selective

The proposal now includes a strong simplified-category penicillins figure to show that not all series are sparse. That is helpful. But a reviewer may note that the example is intentionally dense and class-aggregated, whereas the median pair still has only 2 years of support.

The figure works best as an existence proof, not as evidence that density is typical.

### 9. The class-level fallback is pragmatic, but it dilutes specificity

Allowing antibiotic classes is sensible and likely necessary. However, it also weakens the specificity of the target. The project’s attraction partly comes from its pair-level granularity. Moving to broad classes improves density, but risks collapsing clinically distinct antibiotic behaviors into one signal.

A reviewer may therefore ask when the unit of analysis is:

- individual antibiotic
- antibiotic class
- simplified category

and what scientific question each level is meant to answer.

### 10. The proposal still mixes “implemented now” and “future extension” quite often

This is much better than before, but the distinction still appears repeatedly throughout the draft. That repetition is understandable, because the proposal is trying to be precise. Still, the text occasionally reads as if it is justifying itself section by section rather than presenting one smooth argument.

## What the proposal now answers well

Compared with earlier drafts, the proposal now has strong answers to several reviewer questions:

- How many usable training examples remain after filtering and LLM disambiguation?
- Is the current experiment really future-SI-from-past-SI?
- Why is long-horizon forecasting not the main claim?
- How can the method cope with sparse individual-drug trajectories?
- Is there any evidence that some trajectories are genuinely dense?

Those were important gaps, and they are now handled much better.

## What would make the proposal substantially stronger now

### 1. Add one explicit external validation target

Even a modest external benchmark would help enormously. For example:

- resistance trend datasets
- curated WHO/CDC/ECDC priority organisms or watchlists
- known historical AMR concern escalations

Without this, the project remains internally interesting but externally vulnerable.

### 2. Add a sensitivity analysis for `yes` versus `yes + uncertain`

The proposal already gives the counts. It should now state that core forecasting and SI analyses will be repeated for both variants.

That would directly address the label-noise concern.

### 3. Define eligibility thresholds for each stage

The proposal should state explicitly something like:

- forecasting baseline: pairs with at least `k` yearly observations
- Granger analysis: pairs with at least `m` yearly observations
- transformer stage: pairs or classes meeting a denser threshold

This would make the modeling pipeline more concrete and statistically credible.

### 4. Narrow the causal rhetoric further

If the draft says “leading predictors” more often than “causal predictors,” it will be safer under review. The Granger framing can remain, but it should be handled conservatively.

### 5. Clarify the scientific role of class-level models

The proposal should explicitly state whether class-level forecasting is:

- a robustness analysis
- a fallback for sparse drugs
- a parallel modeling target
- or all three

Right now it is presented as sensible, but not yet fully operationalized.

## Bottom line

The current proposal is now credible as a thoughtful, technically grounded scientometric forecasting project for AMR literature. It is much improved over earlier versions, especially in its honesty about the baseline experiment, its use of real dataset counts, and its move to short-horizon forecasting.

Its main remaining weakness is no longer conceptual confusion. It is that the external meaning of the forecast is still underdetermined. Until the project shows why improved future-SI prediction connects to real-world AMR emergence, the strongest defensible interpretation is that it forecasts future scientific salience and investigates which literature-derived signals lead that salience.
