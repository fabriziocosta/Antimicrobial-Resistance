# AMR Exploratory Quadrants Site: LLM Build Handoff

This document is a self-contained build brief for recreating the interactive AMR scientific-interest (SI) quadrant explorer in this repository. It describes the intended user experience, data contract, calculations, implementation behavior, visual design, validation checks, and publishing requirements. Treat the current implementation in `sites/amr-si-explorer/` as a reference, and treat the data files listed below as the source of truth for the current published figure.

## Goal

Build a responsive, shareable web page that lets collaborators explore the predicted future scientific-interest score and SI change for bacteria–antibiotic pairs. Users should be able to find a pair, inspect its model and validation fields, compare it to the quadrant thresholds, and open the PubMed records from 2025 for that pair.

The main chart is a scatter plot:

- One circle represents one bacterium–antibiotic pair.
- X axis: `future_predicted_SI` (predicted future SI).
- Y axis: `future_minus_latest_smoothed_SI` (future predicted SI minus smoothed SI at the latest available year).
- Initial quadrant thresholds: X = 0 and Y = 0.
- Keep drawing every circle regardless of whether its name can be placed without collision.
- Draw a name only when it fits without overlapping other names, points, or chart boundaries. When a label cannot be placed legibly, suppress the label and retain the circle.
- Hovering a circle shows all available pair information. Clicking a circle selects the pair, updates a compact summary, and loads that pair's 2025 PMID list below the chart.
- The selected-pair summary is followed by a collapsible `plot_single_pair_si`-style yearly plot, then the collapsible 2025 PMID list. Keep this order: pair info → yearly plot → PMIDs.
- Precompute the yearly plot inputs for every scatter pair and store them as static Site data so the browser does not need to fit LOESS models or access the analysis repository.

## Repository inputs

Use the checked-in outputs first; do not rerun expensive forecasting or PubMed collection just to recreate the site.

| Path | Purpose |
| --- | --- |
| `outputs/figures/future_si_vs_current_si_quadrant.csv` | Current plotted pair records. Despite the filename, the interactive explorer's axes are future predicted SI and future-minus-latest-smoothed SI. It currently contains 324 rows and 69 columns. |
| `outputs/figures/plotted_pair_pmids_2025.csv` | Pair-to-PMID mapping for the plot, with `bacteria`, `antibiotic`, `year`, and `PMID` columns. The current artifact has 14,479 rows, all for year 2025. |
| `outputs/parquets/yearly_amr_screened_pairs.parquet` | Yearly screened-pair publication data; useful for regenerating the PMID mapping if the CSV is absent. |
| `notebooks/06_glass_interactive_si_shift.ipynb` | Notebook context for creating `latest_count_shift_df`, the interactive plot, the plotted-pair PMID list, and the Excel PMID output. |
| `amr/pipelines/glass_si.py` | Python plotting implementation and expected SI field semantics. |
| `sites/amr-si-explorer/dist/index.html` | Existing UI and Plotly interaction reference. |
| `sites/amr-si-explorer/dist/pairs.json` | Existing static web data bundle and its field shape. |
| `sites/amr-si-explorer/dist/pair_timeseries.json` | Precomputed yearly observed SI, forecast and uncertainty, plus annual publication-count tables for every plotted pair. |
| `sites/amr-si-explorer/scripts/build_pair_timeseries.py` | Regenerates the static single-pair plotting tables from the yearly SI dataset and current pipeline configuration. |
| `sites/amr-si-explorer/.openai/hosting.json` | Existing Sites project ID and static output configuration. Preserve these values when updating this Site. |

The project has an existing public Sites deployment. When rebuilding within this repository, retain its current Site identity and audience. Do not create a second Site unless explicitly asked.

## Data preparation

Create a static JSON file at `sites/amr-si-explorer/dist/pairs.json` so the page can load quickly without a backend. Use the CSV inputs above as the reproducible current source. The JSON top level should be:

```json
{
  "count": 324,
  "records": [
    {
      "pathogen": "Escherichia coli",
      "antibiotic": "Vancomycin",
      "future_predicted_SI": 0.130678955,
      "future_minus_latest_smoothed_SI": 0.121947312,
      "pmids_2025": ["39714140", "39828825"]
    }
  ]
}
```

The example's PMIDs are illustrative only; load all matching IDs from the current PMID input. Preserve all 69 plot-record columns from `future_si_vs_current_si_quadrant.csv`, using the CSV headers as JSON field names. Numeric values should remain numeric, missing values should become JSON `null`, and booleans should remain booleans. Add one field per plotted pair:

- `pmids_2025`: an array of PMID strings belonging to that exact `pathogen` + `antibiotic` pair for calendar year 2025.
- Match the PMID CSV's `bacteria` column to the plot CSV's `pathogen` column and match `antibiotic` exactly after trimming whitespace. Filter to `year == 2025` even if the supplied file currently contains only 2025.
- De-duplicate PMIDs per pair while preserving a stable order (numeric ascending is acceptable).
- Include `pmids_2025: []` for a pair with no matching PMID records. Do not use null for this field.

The current expected data size is 324 pairs and 14,479 pair-PMID links. Use this as a sanity check, not as a hard-coded limit. The published data is current through latest available year 2025, with forecast year 2028.

## Page structure

Use a single-page, buildless HTML/CSS/JavaScript implementation unless the deployment environment requires a framework. Plotly.js is suitable for the scatter chart. Keep the application static: bundle all plotted records locally and do not require user authentication or a server-side database.

Recommended page layout:

1. Header with the title **Predicted future SI vs SI shift**, a short explanation, and a badge showing the latest available year and forecast year.
2. Small summary metrics: visible pair count, visible GLASS-match count, and forecast year.
3. Main workspace with a filter panel and a large chart.
4. Selected-pair detail card under the chart.
5. A collapsible 2025 PMID section at the bottom of the page, below the chart and selected-pair summary.

The page should work at desktop, tablet, and mobile widths. On narrow screens, stack the filter panel above the plot; keep labels and filter controls usable, and allow the plot to scroll/zoom using Plotly.

## Plot behavior and visual encoding

### Axes and thresholds

- X data: `future_predicted_SI`.
- Y data: `future_minus_latest_smoothed_SI`.
- X axis label: `Predicted future SI`.
- Y axis label: `Future predicted SI − latest smoothed SI`.
- The horizontal and vertical quadrant thresholds default to exactly zero. The X threshold belongs at x=0 and the Y threshold belongs at y=0. Do not place either initial line at 0.5 or at the median.
- Provide numeric threshold inputs (step 0.01) and an **Apply thresholds** action. Moving thresholds changes the dashed lines and shaded regions; it must not alter point coordinates or hide points.
- Provide a **Reset filters** action that restores both thresholds to zero, clears search and selections, clears GLASS-only and optional-label toggles, and restores all pairs.
- Draw threshold lines over the plot as dark dotted/dashed lines. Use subtle translucent fills for the below-threshold bands and a blended fill where the bands overlap. Keep the plot background light and gridlines low contrast.
- Calculate ranges from finite plotted values and the thresholds, with modest padding. This is important: the zero threshold must remain visible even if all plotted values lie on one side of zero.

### Circles and labels

- Every filtered record remains represented by a marker, even when no label can be shown.
- Use open/hollow circles by default. Highlight GLASS matches with a thicker/darker outline and slightly larger marker. A selected point should receive a clear extra outline.
- The current reference supports an optional **Label selected high-shift pairs** control. If retained, label a small capped set (10 is a reasonable default), prioritizing GLASS matches and higher `future_minus_latest_smoothed_SI` values.
- Before rendering a text name, estimate its screen-space bounding box and suppress it if the box intersects another label, another marker, or falls outside the plotting area. This is only label suppression: never suppress the corresponding circle. Recompute label eligibility when filters, axes, thresholds, viewport dimensions, or zoom change.
- Keep pathogen and antibiotic names on separate lines for labels. Use a small, readable font and avoid opaque boxes unless they are needed for contrast.
- Do not render labels for every point by default; dense central regions must remain readable.

### Marker encoding

- Circle size/outline indicates GLASS membership: non-GLASS pairs use a thin neutral outline; GLASS pairs use a heavier dark outline and modestly larger size.
- If marker color encodes `post_fit_count_abs_pct_error_mean`, use a light sequential scale and include a labeled legend/colorbar. If the design instead uses hollow circles throughout, retain the mean absolute count error in tooltips and the selected-pair detail card. Avoid encoding error in both color and size.
- Include a small legend or text so collaborators understand what the different circle outlines mean.

## Filters and controls

Implement these controls and make each one update the chart and visible-count metrics without reloading the page:

- Free-text search across bacterium/pathogen and antibiotic, case-insensitive.
- Bacterium select menu, with an “All bacteria” option.
- Antibiotic select menu, with an “All antibiotics” option. When a bacterium is selected, it is helpful to limit available antibiotics to those present for that bacterium.
- GLASS-only checkbox, based on `is_glass_pair`.
- Optional pair-label checkbox described above.
- X/Y quadrant threshold inputs and apply button, initially 0 and 0.
- Reset filters button.
- Download filtered CSV button, exporting the currently visible rows and relevant plot fields. Quote/escape values safely in CSV.

Filtering must not destroy a selected record silently: if the selected pair is filtered out, clear the selection, detail card, and PMID panel state.

## Hover tooltip: include all pair information

The tooltip is the main path to the complete data record. Include every field present in that pair's JSON record (including newly added fields if the CSV later grows), grouped into readable sections. At minimum, show the following groups and fields when available:

### Pair identity and GLASS status

- Bacterium/pathogen and antibiotic.
- GLASS match flag, `appears_in_GLASS`, `is_glass_pair`, and `glass_group_column`.
- Fit `status` and `error`.

### Years and counts

- `reference_year`, `latest_available_year`, `raw_latest_available_year`, `future_year`, and `num_future_years`.
- `current_si_source`, `current_count`, `latest_available_count`, `latest_count_source`, `latest_available_predicted_count`, `future_predicted_count`, `count_change`, `total_publications`, and `n_pair_years`.

### Scientific-interest values and ranks

- `observed_current_SI`, `current_predicted_SI`, `latest_available_SI`, `latest_available_predicted_SI`, `latest_smoothed_SI`, `future_predicted_SI`.
- `future_minus_latest_SI`, `future_minus_latest_smoothed_SI`, and `si_change` (see definitions below).
- `current_rank`, `future_rank`, `rank_change`, and `abs_rank_change` when present.

### Count validation

- `latest_count_pct_error`, `latest_count_abs_pct_error`, `post_fit_count_pct_error_mean`, `post_fit_count_abs_pct_error_mean`, `n_count_error_years`, and `count_error_display`.

### SI validation

- `n_post_fit_observed_years`, `latest_post_fit_observed_year`, `latest_post_fit_observed_SI`, `latest_post_fit_predicted_SI`, `latest_post_fit_si_error`, `latest_post_fit_si_abs_error`, `latest_post_fit_si_pct_error`, `post_fit_si_error_mean`, `post_fit_si_abs_error_mean`, `post_fit_si_pct_error_mean`, and `post_fit_si_abs_pct_error_mean`.

### Model settings

- `fit_start_year`, `fit_end_year`, `curve_fit_end_year`, `post_fit_validation_end_year`, `min_amr_pair_publication_count`, `min_num_years_with_valid_counts`, `loess_frac`, `loess_degree`, `tail_damping_years`, `validation_loess_frac`, `validation_loess_degree`, `validation_tail_damping_years`, `si_loess_residual_variance_mean`, and `si_loess_residual_std`.

Render missing values as `n/a`, not `undefined`, `NaN`, or blank labels. Format counts with separators and SI values to a consistent precision (3–4 decimal places). Format count errors explicitly as percentages. Escape untrusted text before inserting it into HTML; do not interpolate raw record values into unsafe markup.

## SI change definitions

These names are easy to confuse; keep their definitions explicit in the interface and tooltips:

- `si_change` = `future_predicted_SI - current_predicted_SI`. The current-predicted value may be the model's predicted value at the reference/current year, and it is not necessarily the observed latest SI.
- `future_minus_latest_SI` = `future_predicted_SI - latest_available_SI`, where `latest_available_SI` is the latest observed SI.
- `future_minus_latest_smoothed_SI` = `future_predicted_SI - latest_smoothed_SI`. This is the Y coordinate used by the explorer and the displayed **SI shift**.

Never label the chart's Y axis as just `si_change`: the plotted change is specifically future SI minus latest smoothed SI.

## Click selection and selected-pair card

Clicking a marker selects the pair. Clicking a different marker changes the selected pair. The selected point should be highlighted and the detail card should show:

- Bacterium and antibiotic.
- Reference year → forecast year.
- Latest available observed SI and count, including count source when available.
- Future predicted SI.
- Plotted SI shift (`future_minus_latest_smoothed_SI`).
- Future predicted count.
- Mean absolute post-fit count error.
- Rank change where available.
- GLASS status badge when applicable.

Hover should remain available while selection is active. Keep a helpful empty state before the user chooses a point.

## Collapsible 2025 PMID panel

Add a native, accessible collapsible section (`<details><summary>…</summary>…</details>` is suitable) at the bottom, after the chart and pair details.

- Initial state: closed.
- Empty state before a point is selected: explain that selecting a plotted pair loads the 2025 PubMed IDs.
- On selection, update the summary to include the selected bacterium and antibiotic, and show a count of its 2025 PMIDs.
- Render the IDs as a compact responsive grid with as many columns as can fit in the available width. A CSS grid using `repeat(auto-fill, minmax(76px–90px, 1fr))` is suitable. Keep each cell compact, readable, and keyboard-focusable.
- Each PMID is a link to `https://pubmed.ncbi.nlm.nih.gov/{PMID}/`, with `target="_blank"` and `rel="noopener noreferrer"` so it opens the PubMed entry in a new tab safely.
- Use a bounded height with internal scrolling for very large lists; do not truncate or omit IDs from the data.
- If a selected pair has zero PMIDs, display a clear “No 2025 PMIDs are available for this pair” message.
- Updating the selected point must replace the old pair's PMIDs immediately; never leave stale links from the previous selection.

## Accessibility, responsiveness, and safety

- Associate visible labels with each search/select/checkbox/number input.
- Ensure controls, collapsible summary, links, and buttons are keyboard operable with visible focus styles.
- Do not communicate GLASS status, selected state, or thresholds by color alone; use outline weight, text, or an accessible description as well.
- Keep text contrast readable and chart touch interactions usable on mobile.
- Escape displayed record data and build PMID URLs from validated PMID strings. Use safe link attributes for new tabs.
- Keep the data file local and static; the page does not need to fetch PubMed or any private source at runtime.

## Implementation outline

1. Read this file, the two CSV inputs, and the existing `sites/amr-si-explorer/dist/index.html` before changing the project.
2. Prepare or regenerate `dist/pairs.json` from the current plot CSV plus 2025 PMID CSV. Keep the 69 source plot columns and add `pmids_2025`.
3. Build the responsive page with semantic HTML, CSS, and JavaScript. Plotly.js can be loaded from its official CDN if network access is acceptable; otherwise vendor/pin it as a local static asset.
4. Render the filtered points as one marker trace, with text labels handled independently so label filtering never filters markers.
5. Re-render after filter and threshold changes; attach click selection to marker points and update the detail card and PMID panel.
6. Generate `dist/pair_timeseries.json` by running `python sites/amr-si-explorer/scripts/build_pair_timeseries.py` from the repository root. It stores, for every scatter pair, annual observed SI and count rows, precomputed predicted SI and ±1 SD bounds, predicted publication counts, and the pair/pathogen/antibiotic publication-count series. Use `amr.si.core.fit_predict_yearly_si_loess`, the model called by `plot_single_pair_si`; do not fit a model in the browser.
7. Add a collapsible yearly chart immediately after the pair detail card and before the PMID section. When the user selects a point, bind its matching precomputed record; render the chart when expanded. Use a year X axis, SI left Y axis, logarithmic publication-count right Y axis, observed SI markers/line, predicted SI line and uncertainty ribbon, three annual publication-count bar series, predicted pair-count curve, zero SI reference, and vertical fit-end marker. If no pair is selected, show an instruction. If a pair's time series is absent, show an explicit unavailable state.
8. Verify field mappings, threshold positions, no-label/no-circle regressions, filter behavior, PMID links, selected-pair chart data, collapse/expand behavior, and mobile layout.
9. Preserve the existing hosting manifest and public audience. For a hosted edit, use the repository's Sites source helper, push the exact finished source state, save a version from that commit and static `dist` output, then deploy the saved version. Never deploy an unsaved working tree.

## Acceptance checklist

- [ ] The site loads all plotted pair data from the static JSON file and displays the current pair count.
- [ ] The horizontal and vertical threshold lines initially cross at exactly (0, 0); threshold fields also show zero.
- [ ] Axis values come from `future_predicted_SI` and `future_minus_latest_smoothed_SI` respectively.
- [ ] Filtering changes visible markers and counts consistently.
- [ ] Every filtered pair still has a circle, including pairs whose names are hidden because of collisions.
- [ ] No pair name is rendered where it overlaps another label, marker, or plot boundary.
- [ ] GLASS pair styling is visible and represented in a legend, tooltip, or explanatory text.
- [ ] Hover exposes all fields available in the data record, grouped and formatted clearly.
- [ ] Click selection highlights the marker and updates the detail card.
- [ ] Selecting a pair updates the collapsible yearly plot; it appears after the pair information and before PMIDs.
- [ ] Every plotted pair has precomputed observed, predicted, uncertainty, and annual count data in the static Site assets.
- [ ] The yearly chart includes observed SI, predicted SI with ±1 SD, yearly pair/pathogen/antibiotic counts, predicted pair counts, a zero reference, and the fit-end marker.
- [ ] The yearly chart remains collapsed until opened and resizes correctly when expanded.
- [ ] The bottom PMID section is collapsible, responsive, and updates to exactly the selected pair's 2025 records.
- [ ] The PMID grid uses available page width effectively; each link opens the matching PubMed record in a new tab.
- [ ] A pair with no PMIDs receives an explicit empty-state message.
- [ ] Reset restores the default zero thresholds and unfiltered plot.
- [ ] The layout remains usable on desktop and mobile widths.
- [ ] The existing Site identity and public sharing configuration remain intact when publishing from this repository.

## Current reference values

Use these only as sanity checks against the repository's current artifacts:

- 324 plotted pairs.
- 14,479 pair-PMID links for year 2025.
- Latest available data year: 2025.
- Forecast year: 2028.
- Pair key columns: `pathogen` and `antibiotic`.
- Plot coordinates: `future_predicted_SI` and `future_minus_latest_smoothed_SI`.
- Default quadrant thresholds: `(0, 0)`.

