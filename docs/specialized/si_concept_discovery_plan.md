# Research Plan: Discovering Precursors of Scientific Interest

This document describes the research rationale and proposed analysis. For the implemented notebook workflow, see [Scientific-Interest Concept Discovery](si_concept_discovery.md).

## Project extension

This project extends the existing **scientific interest** analysis for antibiotic–pathogen pairs.

The existing analysis provides, for each antibiotic–pathogen pair $i$, a yearly scientific-interest trajectory

$$
Y_i(t),
$$

where scientific interest measures how much literature attention that pair receives relative to an expected baseline.

The new objective is to explain **why scientific interest rises**.

We will extract candidate explanatory concepts from article abstracts, convert each concept into a yearly signal, and test whether that signal contains temporally prior information that improves prediction of future scientific interest.

The aim is not to identify one universal cause. The aim is to discover a **small, reusable vocabulary of precursor concepts** that explains scientific-interest trajectories across many antibiotic–pathogen pairs, while allowing different pairs to be explained by different subsets of concepts.

---

## 1. Basic objects

For each antibiotic–pathogen pair $i$:

$$
Y_i(t)
=
\text{scientific interest of pair }i\text{ in year }t.
$$

For each candidate concept $k$, classify every abstract associated with pair $i$ and year $t$.

A classifier may return a binary value or a probability

$$
p_{a,k}\in[0,1]
$$

for article $a$.

Aggregate these article-level scores into a yearly concept signal:

$$
X_{i,k}(t)
=
\frac{1}{N_i(t)}
\sum_{a\in(i,t)} p_{a,k}.
$$

Using prevalence rather than raw counts avoids making the concept signal mechanically proportional to the number of publications in that year.

Examples of initial broad concepts might include:

- surveillance;
- outbreak evidence;
- treatment failure;
- mechanism discovery;
- genomic evidence;
- diagnostics;
- hospital transmission;
- agricultural use;
- policy or guideline change.

These are only seed concepts. The vocabulary will be refined iteratively.

---

## 2. Minimal predictive setup

The statistical machinery should remain as fixed and simple as possible while the concept vocabulary evolves.

For each antibiotic–pathogen pair $i$, define a baseline autoregressive model using only previous scientific interest:

$$
M_0:
\qquad
Y_i(t)
=
\alpha_i
+
\sum_{\ell=1}^{L}
\beta_\ell Y_i(t-\ell)
+
\epsilon_{i,t}.
$$

For a candidate factor $k$, define the augmented model

$$
M_k:
\qquad
Y_i(t)
=
\alpha_i
+
\sum_{\ell=1}^{L}
\beta_\ell Y_i(t-\ell)
+
\sum_{\ell=1}^{L}
\gamma_{k,\ell} X_{i,k}(t-\ell)
+
\epsilon_{i,t}.
$$

Use one small fixed lag window globally, for example

$$
L=3.
$$

Do not tune the lag separately for each concept during the initial experiments.

The operational question is:

> Does the past trajectory of concept $k$ improve prediction of future scientific interest beyond what can already be predicted from the past trajectory of scientific interest itself?

This is the core Granger-style test.

---

## 3. Forecast horizons

Evaluate several forecast horizons separately:

$$
h\in\{1,2,3\}.
$$

For each historical cutoff $t$, forecast

$$
Y_i(t+1),\qquad
Y_i(t+2),\qquad
Y_i(t+3).
$$

Use rolling-origin evaluation so that every forecast is generated using only information available before the forecast origin.

Do not smooth using future observations before forecasting, because that would leak future information into the predictor.

For factor $k$, pair $i$, and horizon $h$, define forecast improvement

$$
\Delta_{i,k,h}
=
L_{0,i,h}
-
L_{k,i,h},
$$

where $L_0$ is the baseline forecast loss and $L_k$ is the loss after including the concept.

A positive $\Delta$ means that the concept improves prediction.

---

## 4. Combining forecast horizons

The horizon-specific results should remain visible, but concept search requires a single score.

For the minimal version, use equal weights:

$$
\Delta_{i,k}
=
\frac{1}{H}
\sum_{h=1}^{H}
\Delta_{i,k,h},
\qquad H=3.
$$

No discounting is required initially.

This gives each concept one forecasting-improvement score for each antibiotic–pathogen pair.

---

## 5. Generalist and specialist concepts

A concept should not be required to predict scientific interest for every antibiotic–pathogen pair.

Some causal precursors may be broad and recurring. Others may explain a smaller but coherent subset of cases.

For each candidate concept $k$, compute two population-level objectives.

### 5.1 Generality

$$
G_k
=
\frac{1}{N}
\sum_i
\Delta_{i,k}.
$$

This measures average forecast improvement across pairs.

### 5.2 Specialist strength

A raw maximum would be too sensitive to one unusual pair. Use a robust upper-tail score instead, such as

$$
S_k
=
Q_{0.9}
\left(
\{\Delta_{i,k}\}_i
\right),
$$

or the mean improvement among the top 10% of pairs.

This measures whether a concept is highly informative for a meaningful subset of pairs.

### 5.3 Selection frontier

Maintain candidate concepts on a Pareto frontier over

$$
(G_k,S_k).
$$

This preserves both:

- broad concepts with moderate effect across many pairs;
- narrower concepts with strong effect on a subset of pairs.

The search therefore maintains a **population of explanatory concepts**, rather than forcing all explanatory power into one factor.

---

## 6. Conditional contribution

Concepts can be correlated. Without an explicit redundancy criterion, the surviving population may fill with synonyms or near-duplicates.

Let $S_r$ be the set of surviving concepts at round $r$.

To evaluate an independent new candidate $k$, compare

$$
M_{S_r}:
\quad
Y
\sim
Y_{\mathrm{past}}
+
X_{S_r,\mathrm{past}}
$$

with

$$
M_{S_r+k}:
\quad
Y
\sim
Y_{\mathrm{past}}
+
X_{S_r,\mathrm{past}}
+
X_{k,\mathrm{past}}.
$$

Define the conditional forecast gain

$$
\Delta_{k\mid S_r}
=
L(M_{S_r})
-
L(M_{S_r+k}).
$$

A new concept survives only if it adds useful held-out predictive information beyond the concepts already in the population.

Pairwise correlations or pairwise conditional tests may still be used diagnostically, but the main selection test conditions on the whole surviving set.

---

## 7. Hierarchical concepts

The concept vocabulary is expected to become hierarchical.

For example:

$$
\text{surveillance}
\rightarrow
\{
\text{genomic surveillance},
\text{hospital surveillance},
\text{national surveillance},
\text{environmental surveillance}
\}.
$$

A child concept is structurally nested inside its parent. Testing the child while conditioning on the parent can suppress the child's signal because the parent already contains much of the same information.

Hierarchical refinement therefore requires a different test.

---

## 8. Testing a specialisation

Suppose concept $P$ is a surviving parent concept and the LLM proposes children
$$
C_1, \ldots, C_m.
$$

Do not test the children simply by adding them on top of $P$.

Instead compare two alternative representations.

### Parent representation

$$
M_P:
\quad
Y
\sim
Y_{\mathrm{past}}
+
X_{S_r\setminus P}
+
X_P.
$$

### Refined representation

$$
M_C:
\quad
Y
\sim
Y_{\mathrm{past}}
+
X_{S_r\setminus P}
+
X_{C_1}
+\cdots+
X_{C_m}.
$$

Define

$$
\Delta_{\mathrm{split}}
=
L(M_P)-L(M_C).
$$

If

$$
\Delta_{\mathrm{split}}>0
$$

consistently on held-out forecasts, the refined representation is preferred.

The parent can remain in the semantic ontology, but it need not remain as a predictor if its children provide the more informative representation.

This prevents hierarchical concepts from being penalised merely because they overlap with their parent.

---

## 9. LLM-guided concept refinement

The LLM is used to propose scientifically meaningful refinements of the current vocabulary.

The LLM does **not** decide whether a concept is useful.

Its role is to generate hypotheses. The forecasting experiment decides whether those hypotheses survive.

### Instructions to the LLM

Given an existing concept and scientific abstracts associated with that concept, propose scientifically meaningful refinements.

A proposed concept should:

1. represent a coherent scientific phenomenon, process, mechanism, practice, event, or form of evidence;
2. be operationally detectable from an abstract;
3. be distinguishable from existing parent and sibling concepts;
4. recur across multiple papers rather than explain one unusual example;
5. be defined without reference to future scientific-interest values;
6. use an appropriate level of abstraction;
7. admit clear inclusion and exclusion criteria;
8. be falsifiable as a useful category;
9. have a plausible temporal relationship to scientific interest that can subsequently be tested.

For every proposed concept, return:

- **Name**
- **Definition**
- **Parent concept**
- **Scientific rationale**
- **Why the parent concept is insufficient**
- **Inclusion criteria**
- **Exclusion criteria**
- **Representative textual evidence or patterns**
- **Closest competing concept**
- **How it differs from that competing concept**
- **Expected temporal relationship to scientific interest**
- **Testable hypothesis**
- **Conditions under which it should be rejected, merged, or revised**

The LLM must not see future outcomes for individual antibiotic–pathogen pairs when defining a concept.

---

## 10. Round-by-round search

Let

$$
S_r
$$

be the surviving concept population at round $r$.

Each round proceeds as follows.

### Step 1 — Score current concepts

Classify abstracts for every surviving concept and construct yearly trajectories

$$
X_{i,k}(t).
$$

### Step 2 — Evaluate predictive usefulness

For every concept, evaluate rolling forecasts at 1, 2, and 3 years.

Compute:

- mean forecast improvement across horizons;
- generality across antibiotic–pathogen pairs;
- robust specialist strength;
- conditional gain given the rest of the surviving population.

### Step 3 — Select concepts for refinement

Prefer concepts that:

- improve forecasting across many pairs;
- strongly improve forecasting for a coherent subset of pairs;
- still contribute conditional information;
- show heterogeneous behaviour suggesting that the umbrella concept may contain several distinct processes.

### Step 4 — Ask the LLM for refinements

For selected parents, provide the associated abstracts and ask for scientifically coherent child concepts.

The LLM proposes hypotheses only from the scientific content.

### Step 5 — Score the child concepts

Classify the same corpus using the proposed children and construct their yearly trajectories.

### Step 6 — Compare parent and child representations

For hierarchical refinements, evaluate

$$
M_P
\quad\text{versus}\quad
M_C.
$$

If the children improve held-out forecasting, prefer the refined representation.

### Step 7 — Add independent concepts conditionally

For concepts that are not children of an existing parent, evaluate their incremental value conditional on the full current surviving set.

### Step 8 — Update the population

Construct

$$
S_{r+1}
$$

from the concepts that remain useful, non-redundant, and scientifically interpretable.

Repeat.

---

## 11. Concept-search operations

The search process may perform four basic operations.

### Specialise

Split a broad concept into more specific child concepts.

$$
P
\rightarrow
\{C_1,\ldots,C_m\}.
$$

### Merge

If two concepts are semantically similar and produce almost identical predictive signals, ask whether they represent one higher-level phenomenon.

### Add

Introduce a new concept that captures information not represented by the existing vocabulary.

### Remove

Discard concepts that add no held-out predictive information or are redundant with the surviving population.

---

## 12. Avoiding overfitting during concept discovery

Repeated concept generation can overfit the historical record.

Separate **concept discovery** from **concept confirmation**.

Possible partitions include:

- discovery antibiotic–pathogen pairs versus held-out pairs;
- discovery historical periods versus later confirmation periods;
- nested cross-validation over pairs and years.

The concept hierarchy may evolve using the discovery data.

Once a concept definition is frozen, its predictive value should be measured on data that were not used to formulate or refine that concept.

The final claim should therefore concern concepts that generalise beyond the examples that generated them.

---

## 13. Reverse-direction check

For every surviving concept, also test whether scientific interest predicts the future concept trajectory:

$$
Y
\rightarrow
X_k.
$$

This helps distinguish:

- concepts that tend to precede increased scientific interest;
- concepts that mostly appear after scientific interest has already risen;
- concepts with bidirectional temporal relationships.

The result should still be interpreted as evidence of temporal predictive precedence rather than proof of mechanistic causation.

---

## 14. Minimal implementation

The first implementation should deliberately avoid unnecessary machinery.

Use:

- yearly time series;
- one fixed lag length, initially $L=3$;
- linear autoregressive models;
- 1-, 2-, and 3-year forecasts;
- rolling-origin evaluation;
- one fixed forecast loss;
- equal weighting across horizons;
- generalist and specialist population scores;
- conditional gain against the surviving set;
- parent-versus-children replacement tests for hierarchical concepts;
- held-out confirmation.

Do not initially require:

- nonlinear forecasting models;
- smoothing;
- interpolation;
- learned horizon weights;
- pair-specific lag tuning;
- complicated VAR structures;
- elaborate causal graphical models.

The scientific complexity should initially reside in the **concept vocabulary**, not in the forecasting machinery.

---

## 15. Overall pipeline

**Inputs:** existing scientific-interest trajectories and article abstracts.  
↓  
**Candidate scientific concepts**  
↓  
**Abstract-level concept classification**  
↓  
Yearly concept signals: `X_{i,k}(t)`  
↓  
**Rolling forecasts and Granger-style evaluation**  
↓  
**Generality, specialist strength, and conditional gain**  
↓  
**LLM-guided specialisation, merging, or addition**  
↓  
**Parent-versus-children model comparison**  
↓  
**Updated concept population**  
↓  
**Repeat**

The final output is a compact, hierarchical vocabulary of recurring **precursors of scientific interest**, together with evidence about:

- which concepts improve prediction;
- at which forecast horizons;
- for which antibiotic–pathogen pairs;
- whether they act as broad or specialised precursors;
- which refinements outperform their parent concepts;
- which concepts add information beyond the rest of the vocabulary.

The project therefore extends the existing scientific-interest analysis from measuring **where attention emerges** to discovering **which recurring scientific processes tend to precede that emergence**.
