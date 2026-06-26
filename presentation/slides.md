This presentation outlines our approach to identifying emerging antimicrobial resistance research priorities from the biomedical literature. We use a large language model to identify relevant evidence for pathogen-antibiotic pairs, calculate a Scientific Interest signal that adjusts for their general publication popularity, and forecast which pairs are gaining attention. We compare the results with WHO GLASS priorities and examine the categories of evidence driving each signal, with particular interest in non-GLASS pairs that currently receive limited attention but are increasing rapidly. The approach is intended as an early-warning tool to support expert review, surveillance and further investigation, rather than to assess clinical risk directly.

# Antimicrobial resistance: finding tomorrow's research priorities

- **AMR is changing quickly.** Antimicrobial resistance threatens our ability to treat common infections, and the most important research question is often which pathogen-antibiotic problems are beginning to matter before they are widely recognised.
- **The literature contains early signals.** Biomedical papers capture laboratory findings, clinical observations, surveillance studies and outbreak reports that can reveal emerging attention before formal priority lists are updated.
- **Our aim is prioritisation.** We combine large language models, publication counts and forecasting to identify pathogen-antibiotic pairs that deserve expert review, not to make direct clinical risk assessments.

---

# 1. Why AMR matters

- **Antibiotics support modern care.** Routine medicine, surgery, cancer treatment and intensive care all depend on antibiotics remaining effective against common and severe bacterial infections.
- **Resistance changes outcomes.** A familiar infection can become difficult or impossible to treat when resistance removes reliable treatment options and forces clinicians toward less effective or more toxic alternatives.
- **Earlier warning creates options.** Detecting emerging resistance patterns sooner can help focus surveillance, research funding and prevention efforts before a problem becomes widespread.

---

# 2. Why AMR is difficult to anticipate

- **AMR is many problems.** Each pathogen-antibiotic combination can follow a different trajectory, so a single organism or drug name is not enough to describe the risk.
- **Evidence is fragmented.** Signals are spread across laboratory studies, clinical reports, surveillance programmes, outbreak descriptions and country-specific datasets that are difficult to compare manually.
- **Popularity hides novelty.** Well-known pathogens and antibiotics attract large publication volumes regardless of new risk, while important early signals may be buried before robust surveillance data exist.

---

# 3. What is GLASS?

- **GLASS is the WHO reference system.** The Global Antimicrobial Resistance and Use Surveillance System provides a shared international framework for collecting and comparing AMR surveillance data.
- **GLASS defines recognised priorities.** Its pathogen-antibiotic combinations reflect expert judgement about important current threats and give our analysis a practical benchmark.
- **Benchmarking tests the signal.** We ask whether scientific literature signals can recover known GLASS priorities and also point to additional combinations that may deserve attention.

---

# 4. Why go beyond expert panels?

- **Experts remain essential.** Priority-setting panels bring clinical judgement and public health context, but they are periodic, resource-intensive and necessarily selective.
- **The literature moves continuously.** New studies appear every day across many diseases, settings and countries, creating a stream of evidence that formal review cycles cannot fully monitor.
- **Automation supports review.** A reproducible scan can compare thousands of combinations consistently and give experts an objective early-warning tool for prioritisation.

---

# 5. Scientific Interest: our signal

- **Scientific Interest adjusts attention.** We compare the observed AMR literature for each pathogen-antibiotic pair with the number of papers expected from the general popularity of that pathogen and antibiotic.
- **Positive SI marks unusual focus.** When Scientific Interest is above expectation, the pair is receiving more AMR-specific research attention than its background publication profile would predict.
- **Rising SI marks acceleration.** A growing Scientific Interest signal suggests that attention is increasing for a specific pair, although it measures research activity rather than resistance prevalence or clinical severity.

---

# 6. How large language models make this possible

- **Search retrieves candidates.** Broad PubMed searches first collect papers that may mention a pathogen-antibiotic pair, accepting that many hits will be incidental or irrelevant.
- **The model filters evidence.** A large language model reads each title and abstract in context and decides whether it contains genuine pair-level AMR evidence rather than a coincidental mention.
- **Structured outputs enable scale.** The resulting classifications let us count relevant papers by pair and year, audit decisions, update the time series and test different models or prompts.

---

# 7. Turning the signal into research priorities

- **Forecasting separates level and growth.** We compare expected future Scientific Interest with the recent increase in Scientific Interest to identify pairs that are both notable and changing quickly.
- **Emerging pairs need attention.** Non-GLASS pairs with low current attention but rapid growth are especially useful candidates for expert review, surveillance planning and follow-up investigation.
- **Evidence categories explain drivers.** Grouping papers into surveillance, clinical cases, mechanisms, environmental or animal evidence and unsupported evidence helps distinguish substantive signals from uncertainty.

---

# 8. Where AI experts and doctors can collaborate

- **Map emerging clinical questions.** Doctors can define the pathogen, treatment and syndrome questions that matter in practice, while AI experts use LLMs to extract matching evidence from the literature at scale.
- **Build predictive evidence signals.** AI experts can fit forecasting models on structured literature-derived data, while doctors assess whether rising signals are biologically plausible and clinically meaningful.
- **Prioritise action together.** Joint review can turn model outputs into practical next steps for surveillance, laboratory validation, guideline discussion or targeted research funding.
