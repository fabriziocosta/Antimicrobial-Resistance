# Collaborator Communications

This internal log records collaborator questions, verified project facts, proposed
responses, promised deliverables, and follow-up work. It is a working summary rather
than a verbatim email archive. Add new entries in chronological order, with the newest
entry first.

## Entry template

### YYYY-MM-DD — Collaborator: Subject

**Incoming questions**

1. Summarise each question or request.

**Verified project facts**

- Record facts confirmed from the active repository, generated outputs, or another
  named source.

**Proposed response**

- Draft concise answers, explicitly labelling anything that remains to be checked.

**Attachments and deliverables**

- List files to send or work promised.

**Unresolved items and meeting follow-ups**

- Record checks, decisions, and discussion points that remain open.

---

## 2026-07-17 — Mick: Methods, model, and main-figure questions

### Incoming questions

1. Can we send the WHO antibiotic list used in the analysis so it can be
   independently checked?
2. Is the latest prompt the version sent on Monday 8 June 2026 at 16:05? If not, can
   we send the most recent version?
3. How did human infection enter the initial filtering: through an LLM prompt, a
   human-infection keyword list, or the existing infection keywords?
4. Are all abstracts captured by the keyword filters processed?
5. Has the project moved on from Anthropic Claude 3.5 Sonnet?
6. For the attached main figure, can abbreviated pathogen labels such as `Sta` be
   replaced by readable names such as *S. aureus*, can the figure be enlarged, should
   high-error pairs be filtered if necessary, and can the raw data be supplied?

### Verified project facts

- **GLASS-list provenance:** Richard supplied the original GLASS list. Fabrizio
  normalised its pathogen and antibiotic names for matching; he did not construct
  the original list.
- **Antibiotic input:** The active workflow also loads canonical antibiotic names
  and metadata from `assets/antibiotic_list3.xlsx`. It generates
  `assets/antibiotic_synonym_lookup.csv` for query expansion and
  `assets/antibiotic_coarsening_lookup.csv` for canonical names and analysis
  categories.
- **Prompt history:** The prompt sent on 8 June is not the latest repository version.
  The screening schema changed later on 8 June and the unified prompt was revised
  again on 9 and 11 June. The final prompt was used for the complete data collection,
  so all resulting data were screened consistently with that version. The prompt is
  defined in `amr/screening/core.py`, together with the PMID-level user-prompt
  builder.
- **Human-infection element:** Human relevance enters the active workflow initially
  through the pathogen source and ranking, which use the human-pathogenicity fields
  `n_causation` and `pathogenic_index`. The active PubMed pair query does not apply a
  separate human-infection keyword list. It combines pathogen terms, antibiotic
  terms, and a broad AMR text constraint. The later unified LLM screen distinguishes
  human clinical/outbreak papers from surveillance, mechanism, and
  environmental/animal/non-human papers.
- **Abstract coverage:** Every candidate `(PMID, pathogen, antibiotic, year)`
  combination produced by retrieval is considered by the screening workflow.
  Candidate rows with an abstract are submitted to the LLM unless a completed cached
  decision is reused. Rows without an abstract receive a deterministic `uncertain`
  classification and are excluded from the accepted `strict`/`yes` counts used
  downstream.
- **Current configured model:** The active pipeline configuration uses OpenAI
  `gpt-5-nano` with minimal reasoning effort rather than Claude 3.5 Sonnet.
- **Figure feasibility and raw data:** The plotting workflow supports readable names,
  adjustable width and height, selective text labels, and count-error filtering. The
  plotted table is exported as `outputs/figures/latest_count_shift_df.csv`. The
  attached temporary figure is a vector PDF, so it can also be enlarged without
  raster degradation.

### Proposed response

- Send the original GLASS workbook received from Richard and the normalised GLASS
  table used for matching. Explain that Richard supplied the original and Fabrizio
  only normalised it. Also distinguish these from the antibiotic source workbook,
  synonym lookup, and cleaned/coarsened analysis vocabulary.
- Explain that the 8 June prompt is no longer the latest. Send the exact current
  system prompt and user-prompt template, and confirm that this final prompt was
  used for the complete data collection.
- Explain that the human element came from the pathogen source/ranking rather than a
  separate human-infection query filter. Candidate retrieval uses pathogen,
  antibiotic, and broad AMR terms; the LLM then categorises the evidence setting.
- Confirm that all retrieved candidate combinations enter screening, while noting
  the cached-decision and missing-abstract behaviour.
- Confirm that the active configuration has moved to `gpt-5-nano`, but do not imply
  that older results were generated by it until the result provenance check is
  complete.
- Confirm that the figure can be enlarged and relabelled. Use readable scientific
  names, retain all pairs passing the current eligibility/error filter, and label
  only GLASS pairs plus the most important or extreme points to reduce overlap.

### Attachments and deliverables

- Prepared response package:
  [`mick_2026-07-17_answer_package/`](mick_2026-07-17_answer_package/README.md).
- Original GLASS workbook from Richard and the normalised GLASS table.
- Antibiotic source workbook and cleaned antibiotic lookup.
- Plain-text or Markdown copy of the current system prompt and user-prompt template.
- Larger publication-ready PDF with readable pathogen names and selective labels.
- `latest_count_shift_df.csv`, containing the rows plotted in the revised figure.
- Short methods note covering retrieval, screening, missing abstracts, prompt
  version, model provenance, and the figure's error threshold.

### Unresolved items and meeting follow-ups

- Confirm whether the current inclusion ceiling of 75% mean absolute count error is
  appropriate for the paper or whether a stricter sensitivity figure should also be
  produced.
- Discuss the methods wording and any remaining figure-design preferences at the
  Monday meeting.
