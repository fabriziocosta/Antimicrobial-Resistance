# SI Driver Classification Analysis

This document describes the analysis added to `notebooks/02_model_scientific_interest.ipynb` to explain predicted scientific-interest changes using LLM abstract-classification metadata.

The analysis is descriptive. It does not change the SI model, the forecast, or the ranking thresholds. It helps interpret why a pathogen-antibiotic pair is rising by showing which types of AMR papers make up the observed count curve behind the SI forecast.

## Inputs

The analysis uses two canonical outputs from the active workflow:

- `yearly_amr_screened_pairs.parquet`: yearly AMR-positive counts used for SI modeling.
- `yearly_amr_unified_decisions.parquet`: one row per reviewed `(PMID, pathogen, antibiotic, year)` candidate, including LLM category metadata.

The required LLM category columns are:

- `unified_decision`
- `amr_primary_category`
- `amr_pair_centrality`
- `amr_needs_manual_review`

The notebook defaults to:

```python
CATEGORY_INCLUDE_DECISIONS = ("yes", "uncertain")
DRIVER_LOOKBACK_YEARS = 5
```

This means the driver analysis follows the same inclusion logic as the screened SI dataset: LLM `yes` and `uncertain` rows are counted, while `no` rows are excluded.

## Category Count Curves

For each reviewed abstract, the LLM assigns one primary category:

- `RESISTANCE_PREVALENCE_SURVEILLANCE`
- `RESISTANCE_MECHANISM`
- `TREATMENT_EFFECTIVENESS`
- `TREATMENT_REUSE_OR_SALVAGE`
- `DIAGNOSTIC_STEWARDSHIP`
- `GUIDELINE_POLICY_STEWARDSHIP`
- `ONE_HEALTH_ENVIRONMENTAL_RESERVOIR`
- `CO_OCCURRENCE_OR_WEAK_SIGNAL`

The notebook aggregates these rows into yearly category counts:

```text
group by pathogen, antibiotic, year, amr_primary_category
count reviewed PMIDs
```

The resulting table answers:

> For this pathogen-antibiotic pair, which kind of AMR literature is increasing over time?

The category curves are saved as figure files in:

```text
figures/interesting_yearly_si_driver_plots/
```

Each plot marks `FIT_END_YEAR`, so recent observed category changes can be compared with the forecast window.

## Driver Summary Table

For every pair in `interesting_df`, the notebook computes a recent driver summary over the last `DRIVER_LOOKBACK_YEARS` ending at `FIT_END_YEAR`.

It compares:

- the recent window: `FIT_END_YEAR - DRIVER_LOOKBACK_YEARS + 1` through `FIT_END_YEAR`
- the previous window of the same length immediately before it

The summary includes:

- `driver_recent_total`: number of included reviewed abstracts in the recent window
- `driver_previous_total`: number of included reviewed abstracts in the previous window
- `driver_count_change`: recent total minus previous total
- `driver_dominant_category`: most common recent category
- `driver_dominant_category_count`: recent count for that category
- `driver_fastest_growing_category`: category with the largest recent-minus-previous increase
- `driver_fastest_growing_category_change`: size of that category increase
- `driver_weak_signal_share`: recent share classified as `CO_OCCURRENCE_OR_WEAK_SIGNAL`
- `driver_manual_review_share`: recent share marked for manual review
- `driver_central_count`: recent count with pair centrality `central`
- `driver_partial_count`: recent count with pair centrality `partial`
- `driver_incidental_count`: recent count with pair centrality `incidental`

The enriched table is saved to:

```text
yearly_si_driver_explanations.parquet
```

## Interpretation

The SI forecast ranks pairs based on observed screened AMR counts and background-adjusted SI trends. The driver analysis explains the composition of those observed counts.

Examples:

- A rising pair dominated by `RESISTANCE_MECHANISM` suggests increasing molecular or genomic explanation of resistance.
- A rising pair dominated by `RESISTANCE_PREVALENCE_SURVEILLANCE` suggests increasing reporting of resistance rates, prevalence, outbreaks, or epidemiology.
- A rising pair dominated by `TREATMENT_EFFECTIVENESS` suggests clinical outcome or therapy-evaluation literature.
- A high `CO_OCCURRENCE_OR_WEAK_SIGNAL` share suggests the SI signal may be partly driven by broad co-occurrence rather than pair-centered AMR evidence.
- A high `driver_manual_review_share` flags pairs where the explanation should be checked before drawing strong conclusions.
- High `driver_central_count` relative to `driver_incidental_count` makes the driver explanation more trustworthy.

The category analysis is therefore best used as an audit and explanation layer for SI drivers, not as a replacement for the SI score.

## Pseudocode

```text
load yearly_amr_screened_pairs.parquet
build SI dataset
rank future high-interest pairs

if yearly_amr_unified_decisions.parquet exists and has category columns:
  load unified decisions
else:
  skip driver analysis

category_rows = unified decisions where unified_decision in CATEGORY_INCLUDE_DECISIONS

category_counts = group category_rows by:
  pathogen, antibiotic, year, amr_primary_category

for each interesting pathogen-antibiotic pair:
  recent_window = final DRIVER_LOOKBACK_YEARS ending at FIT_END_YEAR
  previous_window = same length immediately before recent_window

  recent_rows = category_rows for pair in recent_window
  previous_rows = category_rows for pair in previous_window

  compute:
    recent total
    previous total
    count change
    dominant recent category
    category with largest recent-minus-previous increase
    weak-signal share
    manual-review share
    centrality counts

merge driver summary onto interesting_df
save yearly_si_driver_explanations.parquet

for each top interesting pair:
  plot yearly stacked category counts
  mark FIT_END_YEAR
  save plot to figures/interesting_yearly_si_driver_plots/
```

## Failure Behavior

If the unified decision table is missing or does not yet contain the required `amr_*` category columns, notebook 02 prints a warning and continues with the standard SI modeling workflow.

This keeps old SI runs reproducible while allowing richer explanations once the category-enriched LLM screen has been run.
