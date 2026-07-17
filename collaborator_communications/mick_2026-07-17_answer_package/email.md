# Draft email

Hi Mick,

Thanks — answers below, and I have included the relevant files.

1. **Antibiotic list:** I received the original GLASS list from Richard and then
   normalised it for use in the analysis; I did not construct the original list. I
   have attached both the original workbook Richard sent me and the normalised
   version used for matching, so you can compare them directly.

2. **Prompt:** The version sent on Monday 8 June at 16:05 is not the latest
   repository version. The screening schema changed later on 8 June and the prompt
   was revised again on 9 and 11 June. I have attached the final system prompt and
   the user-prompt template. This final prompt was used for the complete data
   collection, so the resulting dataset was screened consistently with one prompt
   version.

3. **Human-infection element:** This entered initially through the pathogen list,
   which was ranked using evidence of human pathogenicity (`n_causation` and
   `pathogenic_index`). We did not add a separate human-infection keyword clause to
   the active PubMed searches. Retrieval uses pathogen terms, antibiotic terms, and
   broad AMR terms. The subsequent LLM screen determines whether the pair is
   genuinely supported and classifies the setting, including human clinical,
   surveillance, mechanism, and environmental/animal categories.

4. **Abstract coverage:** Yes — all candidate pathogen–antibiotic–PMID combinations
   captured by the retrieval filters enter the screening workflow. Combinations
   with an abstract are processed by the LLM unless a completed cached decision
   already exists. Records without an abstract are marked `uncertain` and are not
   included in the accepted downstream counts.

5. **Model:** The active pipeline is now configured to use OpenAI `gpt-5-nano` with
   minimal reasoning effort rather than Claude 3.5 Sonnet.

6. **Main figure:** Yes, all of those changes are possible. I have attached a larger
   vector version using readable species and antibiotic names on a selective set of
   labels, plus an interactive version and the raw 139-row plotted dataset. I kept
   all pairs passing the current error criterion rather than filtering extra points
   only for readability. The present ceiling is 75% mean absolute count error; we
   can discuss whether a stricter sensitivity version would be preferable for the
   paper.

Happy to go through the methods and figure-threshold points on Monday.

Cheers,

Fabrizio
