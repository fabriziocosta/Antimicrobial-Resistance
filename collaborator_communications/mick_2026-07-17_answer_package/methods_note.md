# Methods note supporting the response

## Candidate retrieval

The active workflow creates a grid of canonical pathogen–antibiotic pairs. Pathogens
come from `assets/counts_extended2.csv` and are ordered using `pathogenic_index`;
the current collection configuration retains the 50 highest-ranked pathogens above
zero. Antibiotics come from the complete active vocabulary prepared from
`assets/antibiotic_list3.xlsx`.

PubMed retrieval expands pathogen and antibiotic names with synonym tables. Each
pair query also includes the broad title/abstract constraint:

```text
"antimicrobial resistance"[Title/Abstract]
OR "antibiotic resistance"[Title/Abstract]
```

There is no separate human-infection keyword clause in the active query. The
human-disease element is present in the provenance and ranking of the pathogen list,
not as an additional abstract keyword filter.

## Abstract screening

Screening operates on each `(PMID, pathogen, antibiotic, year)` combination, because
one paper may support one target pair but not another. Every retrieved candidate
combination is considered:

- combinations with abstracts are passed to the LLM unless a completed cached
  decision is reused;
- combinations without abstracts receive `uncertain`/`NONE` defaults without an LLM
  call;
- accepted downstream counts use the configured accepted decision set, currently
  `strict`.

The unified prompt tests whether the target pair shares a supported AMR,
susceptibility, treatment, testing, mechanism, surveillance, or policy context. It
also assigns one primary setting/category: clinical/outbreak, surveillance,
mechanism/genetics/laboratory, environmental/animal/non-human, other, or none.

## Prompt and model provenance

The active configuration specifies OpenAI `gpt-5-nano` with minimal reasoning
effort. The prompt sent on 8 June is not the latest code version: repository history
shows prompt/schema changes later on 8 June and further prompt changes on 9 and
11 June.

The final prompt included in this package was used for the complete data collection.
The resulting dataset was therefore screened consistently with the same final system
prompt and user-prompt template.

## GLASS-list provenance and normalisation

Richard supplied the original GLASS workbook; Fabrizio did not construct the
original list. Fabrizio normalised its pathogen and antibiotic names so they could
be matched consistently against the canonical names used by the analysis. Both the
original workbook and the resulting normalised CSV are included in this package,
allowing the transformation to be checked directly.

The separate antibiotic source workbook and cleaned/coarsened antibiotic vocabulary
are also included because they define the antibiotic names used by the PubMed and SI
pipelines.

## Main figure

The supplied raw table has 139 pathogen–antibiotic pairs. The revised plot:

- retains the existing 0–75% mean absolute count-error inclusion range;
- remains a vector PDF and is enlarged to 1800×1200;
- uses a genus initial plus species epithet and the full antibiotic name;
- labels all GLASS pairs, followed by the largest positive SI shifts and counts, up
  to 50 labels;
- retains full pair names and numerical values in the interactive hover text.

The x-axis is predicted future scientific interest. The y-axis is predicted future
SI minus smoothed SI at the latest available year. Colour represents mean absolute
count percentage error, and outlined points identify pairs matched to GLASS.
