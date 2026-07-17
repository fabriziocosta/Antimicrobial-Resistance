# Mick answer package — 17 July 2026

This folder contains the draft response and supporting files for Mick's questions
about the AMR paper methods, model, and main figure.

## Start here

- `draft_email.md`: concise response ready for review.
- `methods_note.md`: reproducibility details and limitations behind the answers.
- `current_llm_prompt.txt`: snapshot of the current system prompt and user-prompt
  template used by the active code.

## GLASS and antibiotic inputs

- `original_GLASS_list_from_Richard.xlsx`: original GLASS workbook received from
  Richard.
- `normalised_GLASS_list.csv`: normalised GLASS table used for matching in the
  analysis.
- `antibiotic_list_used.xlsx`: source antibiotic workbook loaded by the active
  preparation workflow.
- `antibiotic_coarsening_lookup.csv`: cleaned canonical antibiotic vocabulary and
  analysis categories generated from the workbook.

The reply makes the provenance and transformation explicit: Richard supplied the
original GLASS list, and Fabrizio normalised it for analysis.

## Figure and raw data

- `main_figure_revised.pdf`: enlarged vector figure with readable pathogen and
  antibiotic labels on a selective subset of points.
- `main_figure_revised.html`: interactive version with full names and values on
  hover.
- `main_figure_raw_data.csv`: the 139-row table underlying the supplied plot.

The revised figure retains the existing eligibility rule, including the configured
0–75% mean absolute count-error range. It labels all GLASS pairs and then the most
prominent remaining pairs, up to 50 labels in total, rather than removing additional
points solely to reduce label crowding.

## Checks before sending

1. Review whether the 75% count-error ceiling is acceptable for the manuscript or
   whether a stricter sensitivity figure is also needed.
