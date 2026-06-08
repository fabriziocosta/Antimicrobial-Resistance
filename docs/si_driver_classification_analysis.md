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
- `amr_target_pair_relation_supported`

The notebook defaults to:

```python
CATEGORY_INCLUDE_DECISIONS = ("yes", "uncertain")
DRIVER_LOOKBACK_YEARS = 5
```

This means the driver analysis follows the same inclusion logic as the screened SI dataset: LLM `yes` and `uncertain` rows are counted, while `no` rows are excluded.

## Category Share Curves

The fine-grained LLM categories can also be read through a coarser split when the individual categories are too detailed for interpretation. This coarser view keeps the original category labels intact in the data and plots, but groups them conceptually into fewer higher-level drivers:

| Coarse concept | Fine categories |
| --- | --- |
| Environmental animal non-human | `ENVIRONMENTAL_ANIMAL_NON_HUMAN` |
| Clinical outbreaks and reports | `CLINICAL_OUTBREAKS_REPORTS` |
| Surveillance | `SURVEILLANCE` |
| Mechanisms genetics and lab studies | `MECHANISMS_OF_RESISTANCE_GENETICS_LAB_STUDIES` |
| Other | `OTHER` |
| Unsupported or missing evidence | `NONE`, `MISSING_LLM_INFORMATION` |

The reusable lookup for this split is saved in:

```text
assets/amr_driver_category_coarsening.csv
```

Notebook plotting can use either level through `DRIVER_CATEGORY_LEVEL = "original"` or `DRIVER_CATEGORY_LEVEL = "coarse"`. The `save_category_driver_plots` helper keeps `"original"` as the default API behavior, while the notebooks can opt into the coarser grouping by passing `driver_category_level="coarse"` and the lookup path.

The simplified categories are already close to the display-level grouping. The coarsening lookup mainly supplies stable ordering, colors, and the shared `Unsupported or missing evidence` bucket for `NONE` and `MISSING_LLM_INFORMATION`.

`Unsupported or missing evidence` captures cases where the pair-centered evidence is thin, incidental, ambiguous, or unavailable. `NONE` means the abstract mentions the pathogen and antibiotic without a strong pair-centered AMR contribution, while `MISSING_LLM_INFORMATION` means category metadata is absent or came from an LLM failure. A high share of this group should be interpreted as an uncertainty flag rather than as a substantive biological or clinical driver.

For each reviewed abstract, the LLM assigns one primary category:

- `SURVEILLANCE`: The abstract mainly reports how often resistance or reduced susceptibility is observed for the target pathogen-antibiotic pair. This includes resistance rates, susceptibility frequencies, incidence, prevalence, temporal trends, geographic spread, outbreak burden, surveillance studies, antibiograms, epidemiology, and routine monitoring outputs. A rise in this category means the pair is increasingly appearing in descriptive resistance-burden literature.

- `MECHANISMS_OF_RESISTANCE_GENETICS_LAB_STUDIES`: The abstract mainly explains why or how resistance occurs biologically for the target pair. This includes resistance genes, beta-lactamases, carbapenemases, ESBLs, KPC, NDM, OXA, CTX-M, plasmids, integrons, transposons, horizontal gene transfer, mutations, porin loss, efflux pumps, target modification, and genomic analyses focused on resistance. This category takes priority over prevalence/surveillance when an abstract both reports rates and explains the mechanism.

- `CLINICAL_OUTBREAKS_REPORTS`: The abstract mainly concerns human clinical cases, patient cohorts, infections, case reports, case series, hospital events, clinical burden, treatment experience, patient outcomes, or outbreak investigations.

- `ENVIRONMENTAL_ANIMAL_NON_HUMAN`: The abstract mainly concerns animals, livestock, companion animals, wildlife, aquaculture, veterinary medicine, food production, farms, slaughterhouses, wastewater, sewage, rivers, soil, water, environmental sampling, non-human reservoirs, or One Health transmission.

- `OTHER`: The abstract is AMR-relevant for the explicitly supported target pair but does not fit the other four substantive categories. This includes diagnostics, policy, stewardship, guidelines, prescribing behaviour, antibiotic consumption, reviews, modelling, economic burden, education, and broad AMR discussions.

- `NONE`: The abstract mentions both the target pathogen and the target antibiotic or class, but the target pair is not central, the evidence is too thin to assign a stronger category, or the relationship is incidental. It is also used when the abstract is ambiguous, lacks enough detail, or contains AMR-related keywords without a clear pair-centered contribution. A high share of this category means the SI signal may be driven partly by broad co-occurrence rather than strong pathogen-antibiotic AMR evidence.

- `MISSING_LLM_INFORMATION`: The category metadata needed for driver interpretation is missing or comes from a prior failed LLM call. Rows with blank `amr_primary_category` values, or rows whose `unified_rationale` starts with `LLM failure:`, are assigned to this category for plotting and summaries. In plots, if the observed SI input count for a pair-year is larger than the number of classified category rows, the unclassified remainder is also added to this category. It is shown in gray so missing classifier information is visually distinct from true weak co-occurrence evidence.

The notebook first aggregates these rows into yearly category counts:

```text
group by pathogen, antibiotic, year, amr_primary_category
count reviewed PMIDs
```

For plotting, the yearly category counts are normalized by the total included reviewed PMIDs in that year for the pair. Each stacked bar therefore sums to 100% when at least one included publication is present.

When `si_df` is passed to the plotting helper and contains a `count` column, the plot compares the classified category total with the observed pair-year publication count. Any positive difference is added as gray `MISSING_LLM_INFORMATION`. This means a year with observed publications but no available category classifications is plotted as a 100% gray bar rather than disappearing.

The resulting table answers:

> For this pathogen-antibiotic pair, which kind of AMR literature is taking a larger or smaller share of the yearly publication mix?

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
- `driver_fastest_growing_category`: category with the largest recent-minus-previous share increase
- `driver_fastest_growing_category_change`: size of that category share increase, in fraction units
- `driver_weak_signal_share`: recent share classified as `NONE`
- `driver_target_pair_supported_share`: recent share with `amr_target_pair_relation_supported = true`
- `driver_target_pair_supported_count`: recent count with explicit target-pair support
- `driver_target_pair_unsupported_count`: recent count without explicit target-pair support

The enriched table is saved to:

```text
yearly_si_driver_explanations.parquet
```

## Reported Value Computation

This section defines how each reported value in the category-share plots and driver summary table is computed.

### Shared Preprocessing

Both the yearly category share curves and the driver summary start from `yearly_amr_unified_decisions.parquet`.

1. Normalize `unified_decision` by stripping whitespace and lowercasing.
2. Keep rows where `unified_decision` is in `CATEGORY_INCLUDE_DECISIONS`; by default this keeps `yes` and `uncertain` and excludes `no`.
3. Convert `year` to an integer and drop rows with missing `pathogen`, `antibiotic`, or `year`.
4. Normalize `amr_primary_category`:
   - if `amr_primary_category` is missing or blank, set it to `MISSING_LLM_INFORMATION`;
   - if `unified_rationale` starts with `LLM failure:`, set it to `MISSING_LLM_INFORMATION`;
   - otherwise use the stripped `amr_primary_category` value.

### Yearly Category Plot Values

The raw count for a given `(pathogen, antibiotic, year, category)` is:

```text
count = number of retained reviewed PMID rows with that pathogen, antibiotic, year, and normalized category
```

For plots with `si_df["count"]`, compute the missing classified count as:

```text
classified_total = sum of normalized category counts for that pathogen, antibiotic, and year
observed_total = observed pair-year publication count from si_df
missing_llm_count = max(observed_total - classified_total, 0)
```

`missing_llm_count` is added to `MISSING_LLM_INFORMATION` before fractions are computed.

The plotted value is the within-year fraction:

```text
year_total = sum of category counts after adding missing_llm_count
category_fraction = count / year_total
```

If `year_total` is `0`, all category fractions for that year are plotted as `0`.

The plot is a pivot of the fraction table:

```text
index   = year
columns = normalized amr_primary_category
values  = category_fraction
```

If a plot start or end year is supplied, missing years are reindexed into the plot with zero fractions. The vertical dashed line is drawn after `FIT_END_YEAR` when that year is present in the plotted year index.

### Driver Summary Window Definitions

For each `(pathogen, antibiotic)` pair in `interesting_df`, the driver summary uses two adjacent windows:

```text
recent_start  = FIT_END_YEAR - DRIVER_LOOKBACK_YEARS + 1
recent_end    = FIT_END_YEAR
previous_start = FIT_END_YEAR - (2 * DRIVER_LOOKBACK_YEARS) + 1
previous_end   = recent_start - 1
```

With the notebook defaults `FIT_END_YEAR = 2023` and `DRIVER_LOOKBACK_YEARS = 5`:

```text
recent window   = 2019 through 2023
previous window = 2014 through 2018
```

Only retained rows for the current pair are used.

### Driver Summary Fields

- `driver_recent_total`: number of retained rows for the pair in the recent window.

- `driver_previous_total`: number of retained rows for the pair in the previous window.

- `driver_count_change`: `driver_recent_total - driver_previous_total`.

- `driver_dominant_category`: normalized category with the largest row count in the recent window. Ties are broken alphabetically by category name. If the recent window has no rows, this is `None`.

- `driver_dominant_category_count`: recent-window count for `driver_dominant_category`. If the recent window has no rows, this is `0`.

- `driver_fastest_growing_category`: category with the largest share increase from the previous window to the recent window. For every category seen in either window, compute:

```text
recent_category_share = recent_category_count / driver_recent_total
previous_category_share = previous_category_count / driver_previous_total
category_change = recent_category_share - previous_category_share
```

If one window has no retained rows, all category shares for that window are treated as `0`. The category with the largest `category_change` is reported. Ties are broken alphabetically by category name. If neither window has category rows, this is `None`.

- `driver_fastest_growing_category_change`: the share-change value for `driver_fastest_growing_category`, in fraction units. For example, `0.25` means a 25 percentage-point increase. If neither window has category rows, this is `0.0`.

- `driver_weak_signal_share`: recent-window share of retained rows whose normalized category is `NONE`.

```text
driver_weak_signal_share = recent weak-signal row count / driver_recent_total
```

If `driver_recent_total` is `0`, this is `0.0`. Rows assigned to `MISSING_LLM_INFORMATION` are not counted as weak-signal rows.

- `driver_target_pair_supported_share`: recent-window mean of `amr_target_pair_relation_supported` after coercing object/string values such as `true`, `1`, and `yes` to `True`; other string values are treated as `False`. If `driver_recent_total` is `0`, this is `0.0`.

- `driver_target_pair_supported_count`: number of recent-window rows where the target pathogen-antibiotic relation is explicitly supported.

- `driver_target_pair_unsupported_count`: number of recent-window rows where the target pathogen-antibiotic relation is not explicitly supported.

## Interpretation

The SI forecast ranks pairs based on observed screened AMR counts and background-adjusted SI trends. The driver analysis explains the composition of those observed counts.

Examples:

- A rising pair dominated by `MECHANISMS_OF_RESISTANCE_GENETICS_LAB_STUDIES` suggests increasing molecular or genomic explanation of resistance.
- A rising pair dominated by `SURVEILLANCE` suggests increasing reporting of resistance rates, prevalence, outbreaks, or epidemiology.
- A rising pair dominated by `CLINICAL_OUTBREAKS_REPORTS` suggests clinical cases, outcomes, burden, or outbreak literature.
- A high `NONE` share suggests the SI signal may be partly driven by broad co-occurrence rather than pair-centered AMR evidence.
- A high `driver_target_pair_supported_share` makes the driver explanation more trustworthy.

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
category_rows.amr_primary_category = normalized category:
  MISSING_LLM_INFORMATION for blank categories or prior LLM failures
  original stripped category otherwise

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
    category with largest recent-minus-previous share increase
    weak-signal share
    manual-review share
    centrality counts

merge driver summary onto interesting_df
save yearly_si_driver_explanations.parquet

for each top interesting pair:
  plot yearly stacked category shares
  mark FIT_END_YEAR
  save plot to figures/interesting_yearly_si_driver_plots/
```

## Failure Behavior

If the unified decision table is missing or does not yet contain the required `amr_*` category columns, notebook 02 prints a warning and continues with the standard SI modeling workflow.

This keeps old SI runs reproducible while allowing richer explanations once the category-enriched LLM screen has been run.
