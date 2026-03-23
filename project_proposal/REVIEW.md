# Proposal Review

## Summary judgment

This is a strong proposal if it is judged as a scientometric prediction
project and a more vulnerable proposal if it is judged as a direct AMR
forecasting project. Its core idea is clear: model future
background-adjusted scientific interest for antibiotic-pathogen pairs
and test whether richer literature-derived signals improve that
prediction. That is a real and defensible research question.

The proposal is less convincing when it leans too close to claiming that
future scientific salience is already a proxy for future AMR burden. The
project may eventually support that broader interpretation, but the
proposal as written does not yet establish it.

## Major strengths

### 1. The proposal has a real target, not a vague ML objective

Many proposals in this space collapse into a generic ambition to “apply
AI to the literature.” This one does not. It defines a target variable,
scientific interest, in a precise way and grounds it in a
background-adjusted residual rather than raw publication volume. That is
important because it turns the problem into a coherent prediction task
instead of a loosely defined bibliometric exercise.

### 2. The data-construction workflow is serious

The proposal is strongest where it treats measurement as part of the
scientific problem. The use of pair-specific LLM adjudication after
broad retrieval is a good example. It directly addresses the fact that
simple co-mention is not enough and that the quality of the pair-level
signal depends on a credible screening step.

This gives the project a more defensible foundation than a naive
document-counting pipeline would have had.

### 3. The proposal is realistic about sparsity

The dataset section is one of the strongest parts of the document. It
shows counts, filtered trajectory lengths, and a practical fallback to
class-level aggregation. That matters because the proposal’s credibility
depends heavily on whether it appears to understand its own data regime.

It does.

### 4. The signal design is more concrete than in most literature-mining proposals

The concept-extraction stage now has identifiable signal families rather
than generic promises about latent NLP features. Treatment failure,
resistance mechanisms, epidemiological spread, surveillance signals, and
diagnostic signals are concrete enough that a reviewer can understand
what is being measured and why those variables might matter.

The same is true of the semantic-embedding section, which now gives a
clear reason for embeddings beyond feature inflation: they are intended
to support similarity-based generalization across related pathogens,
related antibiotics, and sparse pairs.

### 5. The modeling section is much more disciplined than a transformer-first pitch

The proposal’s decision to move from simpler models to more expressive
ones is good. Starting from linear or additive models, then considering
convolutional sequence models, and only using transformer-style models
if they are actually supported by the data is the right posture.

This also improves the proposal’s scientific tone, because it frames
model choice as something to be earned through evidence rather than
announced in advance.

## Main weaknesses

### 1. The project’s external meaning remains the largest unresolved issue

The proposal is clear that the immediate target is future scientific
interest. That is good. But the wider motivation still depends on the
idea that forecasting future scientific interest says something useful
about future AMR concern in the world, not just in the literature.

That bridge is plausible, but not yet demonstrated. Without an external
benchmark of some kind, the proposal remains vulnerable to a simple
criticism: that it forecasts the future attention of researchers rather
than the future trajectory of resistance itself.

This is not a fatal flaw, but it is the central one.

### 2. The Granger stage is still the shakiest part of the methodology

The proposal is persuasive when it uses Granger analysis as an
interpretability tool and signal-prioritization device. It is less
persuasive when it treats Granger analysis as if it were straightforward
to defend on short, filtered, noisy yearly series.

A reviewer is likely to ask:

- how many time points are truly required
- what lag structure is searched
- whether the analysis is run on SI levels or differences
- how zero-heavy or unstable trajectories are excluded
- how multiple testing is controlled

The proposal points in the right direction, but it still needs more
hard-edged operational detail here than it currently provides.

### 3. The distinction between individual-drug and class-level analyses needs sharper scientific framing

The class-level fallback is practical and probably necessary, but it
changes the meaning of the task. Predicting future salience for a single
antibiotic-pathogen pair is not the same scientific question as
predicting future salience for an antibiotic class against a pathogen.

The proposal knows that class-level aggregation helps with sparsity. It
does not yet say clearly enough what is gained scientifically and what
is lost when this switch happens.

### 4. The screening-rule sensitivity problem remains important

Including `uncertain` rows may be reasonable, but it is not a neutral
choice. It changes the evidence base from which SI is constructed. The
proposal is right to include a sensitivity plan, but this issue is
important enough that it should be framed as central to the validity of
the measurement model, not merely as a robustness check.

### 5. The proposal still relies on a few interpretive leaps

The strongest sections are concrete and operational. The weaker sections
move too quickly from:

- literature-derived signal
- to rising scientific salience
- to future AMR concern

Those links may all be reasonable, but they are not equivalent, and the
proposal is best when it keeps those distinctions explicit.

## What would most improve the proposal

### 1. Add one external validation target

This would improve the proposal more than any increase in model
complexity. Even a modest external anchor would help establish why
future-SI prediction matters.

Examples could include:

- surveillance trend data
- external AMR priority lists
- curated historical resistance escalations

### 2. Make eligibility rules explicit and numerical

The proposal should state concrete thresholds for:

- minimum yearly observations
- minimum publication support
- acceptable holdout error
- Granger eligibility
- when to switch from individual drugs to class-level models

That would turn the proposal from a plausible design into a much more
reviewable one.

### 3. Treat ablation as a central result, not a supplementary check

The proposal is right to commit to comparisons between:

- past-SI-only and multi-signal models
- `yes`-only and `yes + uncertain`
- individual-drug and class-level analyses
- linear, convolutional, and transformer-style models

Those comparisons should be framed as core outputs of the project
because they determine whether the richer modeling actually adds value
or merely adds flexibility.

## Final view

This proposal has genuine substance. It has a coherent target, a
credible measurement workflow, a realistic view of sparsity, and a more
disciplined modeling strategy than many proposals in this area.

Its main challenge is not that it lacks technical ideas. Its main
challenge is that the broader meaning of success still depends on an
external connection that the current design only argues for, rather
than directly demonstrates.

If the project adds one external validation layer and tightens the
eligibility rules for the more fragile stages, it becomes a much more
compelling proposal.
