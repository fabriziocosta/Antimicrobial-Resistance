# Modeling Scientific Interest to Detect Future Antimicrobial Resistance Problems

## Executive Summary

Antimicrobial resistance (AMR) research traditionally focuses on biological mechanisms: identifying genetic mutations in bacteria, characterizing biochemical pathways of resistance, and experimentally studying antibiotic action. While these approaches provide deep mechanistic insight, they are inherently slow, expensive, and difficult to scale across large numbers of pathogen-antibiotic combinations. Moreover, predicting when resistance will become clinically important remains extremely challenging because resistance often arises through complex and stochastic processes such as mutation, horizontal gene transfer, and ecological selection pressures.

This project proposes a complementary perspective: instead of directly predicting biological resistance mechanisms or epidemiological AMR burden, we model scientific attention toward antibiotic-pathogen pairs as a measurable signal that aggregates expert knowledge and emerging concerns across the global research community. The goal is therefore to forecast future scientific interest and to assess whether that signal can help detect future AMR problems indirectly.

Scientific attention can be quantified through publication data, which provides a large-scale, structured, and continuously updated record of the topics researchers consider important. Bibliometric indicators derived from publication counts and trends have long been used to measure research activity and emerging fields.

In the current codebase, scientific interest is already operationalized more specifically than raw publication volume. For each antibiotic-pathogen pair and year, the pipeline estimates how much publication attention would be expected given the general prominence of the pathogen and the antibiotic in the literature, and then measures scientific interest as the excess attention above or below that expectation.

Under this definition, a pair has high SI when it is receiving more attention than would be expected from background publication patterns alone.

Unlike biological measurements, which require laboratory experiments or genomic sequencing, scientific attention signals are immediately observable, inexpensive to collect, and available retrospectively across decades of research literature. These signals also implicitly integrate many types of expert knowledge simultaneously, including clinical observations, epidemiological developments, mechanistic discoveries, and public health concerns.

By modeling these signals over time, we aim to capture the collective sensing ability of the scientific community. When multiple research groups begin investigating a specific antibiotic-pathogen combination, this often reflects early warning signals such as treatment failures, unusual resistance patterns, or new molecular mechanisms.

The central hypothesis of the project is that scientific attention dynamics contain predictive information about emerging AMR concerns. In the current implementation, the experiment is specifically to predict future background-adjusted scientific interest (SI) from past SI alone. The proposed extension is to test whether additional literature-derived signals act as leading predictors that improve future-SI prediction beyond this autoregressive baseline. These additional signals include:

- Conceptual indicators of AMR-related phenomena such as treatment failure and resistance mechanisms.
- Semantic representations of how pathogens and antibiotics are discussed in the literature.
- Contextual descriptors of pathogens and antibiotics.

These signals can be processed using modern machine learning models that learn temporal patterns across multiple information streams.

The resulting system aims to identify early indicators of rising concern for specific antibiotic-pathogen pairs. Importantly, this approach does not attempt to replace mechanistic biological research or direct epidemiological surveillance. Instead, it provides a complementary surveillance layer that operates at the level of scientific discourse and bibliometric response, with the premise that rising scientific interest may flag future AMR problems before they are fully recognized.

The advantages of this approach are threefold:

1. Measurability. Scientific interest can be directly quantified from publication data, providing a consistent and scalable signal.
2. Knowledge aggregation. Literature attention implicitly integrates diverse forms of expert knowledge across disciplines.
3. Predictive smoothing. Aggregated attention signals may be more stable and predictable than individual biological events, which often arise through stochastic processes.

If successful, this framework could provide a data-driven early-warning signal for emerging antimicrobial resistance concerns, helping guide research prioritization, surveillance efforts, and downstream hypothesis generation. In practice, the most credible forecasting setting is a short-horizon one, especially one-year to five-year prediction windows. Any stronger claim about policy or surveillance deployment would require additional validation against external AMR outcomes.

In summary, while biological resistance mechanisms remain essential for understanding AMR, modeling the dynamics of scientific interest offers a complementary and scalable way to forecast future scientific salience and to test whether that salience can serve as an early indicator of future AMR problems.

## Literature Review

The relevant literature falls into three main blocks: AMR bibliometrics, scientometric work on emergence and research fronts, and predictive bibliometrics.

### 1. AMR bibliometrics exists, but it is mostly descriptive

There are many recent AMR bibliometric studies, but they are overwhelmingly framed as mapping the field, identifying hotspots, visualizing collaboration, and describing thematic evolution using tools such as VOSviewer, CiteSpace, and Bibliometrix, usually on Web of Science Core Collection or Scopus data. Representative examples cover antimicrobial stewardship, carbapenem-resistant Enterobacteriaceae, phage therapy for bacterial infection, MRSA bacteremia, and regional AMR landscapes. These studies typically report prolific countries, journals, keyword clusters, co-citations, and burst terms rather than building predictive models of future pair-level attention.

A representative pattern is visible in the MRSA bacteremia bibliometric study, which retrieved publications from 2006 to 2024 from WoSCC and used VOSviewer, CiteSpace, and Bibliometrix for data visualization and trend analysis. The phage-therapy bibliometric study similarly positions itself as a visualized analysis of global trends and hotspots. The carbapenem-resistant Enterobacteriaceae study reports hotspots such as antibacterial-agent development, early etiological detection, and genome sequencing techniques. These are useful precedents because they show that AMR researchers already accept bibliometric evidence as a way to characterize the field, but they are less useful as direct methodological precedents for forecasting future pair-level attention.

There are also narrower AMR bibliometric slices, such as AMR education, artificial intelligence for antimicrobial drug discovery, and drug repurposing against ESKAPE pathogens. Again, these studies mostly characterize growth, collaboration, and thematic structure rather than predicting future attention trajectories for a concrete antibiotic-pathogen pair.

### 2. Scientometrics has a mature literature on emergence and research fronts

Outside AMR, there is a well-developed scientometric literature on how to detect emerging topics, research fronts, and transient patterns.

One foundational paper is Chen's CiteSpace II, which presents citation-burst and structural-network methods for detecting and visualizing emerging trends and transient patterns in scientific literature. That paper underpins a large amount of later hotspot and frontier work that identifies emerging areas through burst keywords, co-citation clusters, and centrality patterns.

Another foundational reference is Rotolo, Hicks, and Martin's paper on what counts as an emerging technology. This is important for the framing of the proposal because it separates the concept of emergence from any single indicator and emphasizes attributes such as novelty, growth, coherence, and impact. That supports the use of publication activity or literature-derived signals as proxies for emergence without claiming they fully capture the phenomenon.

More operational models followed. Wang proposes a bibliometric model for identifying emerging research topics using transparent indicator-based criteria. Porter and colleagues develop emergence scoring to identify frontier R&D topics and key players. Xu and colleagues propose multidimensional scientometric indicators for detecting emerging research topics. Yang and colleagues propose a knowledge-ecology method for emerging topic detection. Huang and colleagues use dynamic co-word networks plus link prediction and machine learning for emerging-topic identification. Collectively, these papers show that scientometrics already supports a move from purely descriptive mapping toward formalized detection of emergence.

### 3. There is a smaller but real line of work on prediction, including ML-based prediction

The literature gets closer to the present proposal when it stops asking what is currently prominent and starts asking what will become important.

A direct precedent is Liang et al., combining deep neural networks with bibliometric indicators for emerging research topic prediction. This is close in spirit to the present proposal because it combines bibliometric indicators with a neural model for prediction rather than stopping at retrospective mapping.

Another related direction is Min, Bu, and Sun on predicting scientific breakthroughs based on knowledge-structure variations. That work predicts later breakthrough status from structural properties of citation networks, showing that predictive scientometrics is feasible when the target is future scientific salience rather than present-day description.

There is also a smaller stream of bibliometric work that explicitly uses time-series forecasting on publication counts. These studies use methods such as ARIMA or polynomial regression to forecast publication trends after the bibliometric mapping stage. This is methodologically weaker than the intended architecture of this project, but it supports the broader legitimacy of forecasting bibliometric signals.

### Closest prior work

The closest methodological neighbors are not AMR bibliometrics themselves, but the general scientometric emergence and prediction papers:

- Chen (2006) for burst-based and network-based detection of emerging fronts.
- Wang (2018) for indicator-based identification of emerging research topics.
- Porter et al. (2019) for emergence scoring.
- Xu et al. (2021) for multidimensional indicators.
- Liang et al. (2021) for combining neural models with bibliometric indicators for prediction.
- Huang et al. (2021) for dynamic co-word networks plus machine learning.
- Min et al. (2021) for forecasting future high-impact or breakthrough developments from literature-network structure.

These studies provide methodological support for the claim that bibliometric signals can be used not only to summarize a field, but also to infer and predict emergence.

### Apparent novelty of the proposed AMR setup

The novelty claim here is an inference from the literature summarized above, not a formal proof of novelty.

First, the target is unusually specific: future scientific interest in a concrete antibiotic-pathogen pair, rather than a whole field, disease area, or generic topic cluster. In the present codebase, that target is not raw publication count alone but a background-adjusted pair-level residual scientific-interest score. The AMR bibliometric papers reviewed here are mostly field-level or subfield-level mappings.

Second, the proposed signals are richer than standard bibliometrics. Most AMR bibliometric papers rely on publication counts, citations, co-authorship, co-citation, and keyword co-occurrence. The current codebase already goes one step beyond raw counts by estimating expected pair attention from pooled background activity and using the residual as the core SI outcome. The proposed next step is not merely to forecast SI from its own history, but to show that abstract-level concept scores, pair-specific semantic context vectors, and related features contain leading predictive information about future SI.

Third, the framing is not simply to identify hotspots, but to use literature dynamics as an early-warning proxy for emerging AMR concern. Conceptually, that places the project closer to scientometric emergence detection than to conventional AMR bibliometrics.

### Positioning the project

A defensible related-work narrative is:

1. AMR bibliometrics is already established as a valid way to quantify and visualize the AMR research landscape.
2. Scientometrics has already moved beyond description toward formal detection of emerging topics and, in some cases, prediction of future salience.
3. What remains missing is evidence that pair-level AMR scientific interest can be predicted better by using external literature-derived signals than by using past SI alone. In particular, the gap is to show that abstract-derived concepts, semantic context, and related features behave as leading predictors of future background-adjusted SI.

### Compact bibliography for related work

For methodological precedent:

- Chen, *CiteSpace II: Detecting and visualizing emerging trends and transient patterns in scientific literature*.
- Rotolo, Hicks, and Martin, *What Is an Emerging Technology?*
- Wang, *A bibliometric model for identifying emerging research topics*.
- Porter et al., *Emergence scoring to identify frontier R&D topics and key players*.
- Xu et al., *Multidimensional scientometric indicators for the detection of emerging research topics*.
- Liang et al., *Combining deep neural network and bibliometric indicator for emerging research topic prediction*.
- Huang et al., *Tracking the dynamics of co-word networks for emerging topic identification*.
- Min et al., *Predicting scientific breakthroughs based on knowledge structure variations*.

For AMR-domain precedent:

- Sweileh et al., *Bibliometric analysis of peer-reviewed literature on antimicrobial stewardship*.
- Zhong et al., *Bibliometric analysis of carbapenem-resistant Enterobacteriaceae research, 2010-2020*.
- Maimaiti et al., *Global trends and hotspots of phage therapy for bacterial infection*.
- Lin et al., *Global insights into MRSA bacteremia: a bibliometric analysis and future outlook*.
- Borromeo et al., *Southeast-Asian landscape of antimicrobial resistance research*.

## Research Objectives

This proposal is organized around a clear distinction between the current implemented baseline and the proposed methodological extension. The current codebase already supports pair extraction, LLM-based disambiguation, yearly SI construction, and LOESS-based SI forecasting. The proposed project extends that baseline with richer abstract-derived signals, stronger predictive models, and causal-dependency analysis.

The overarching goal of this project is to develop a predictive framework that leverages signals embedded in the biomedical literature to anticipate emerging antimicrobial resistance concerns. Rather than focusing solely on biological mechanisms of resistance, the project investigates whether patterns of scientific attention toward antibiotic-pathogen pairs can serve as an early indicator of future research and clinical importance.

The central hypothesis is that scientific attention dynamics contain predictive information about emerging AMR challenges because the scientific community collectively responds to early signals such as treatment failures, unusual resistance phenotypes, novel genetic mechanisms, and epidemiological outbreaks. These signals often appear in the literature before the broader significance of a resistance problem becomes widely recognized. In operational terms, the project treats pair-level scientific interest as a background-adjusted residual signal rather than as raw publication count alone. The current baseline experiment is therefore to forecast future SI from past SI, while the novelty claim is that additional signals may function as leading predictors of future SI.

To investigate this hypothesis, the project pursues five specific objectives.

### Objective 1. Quantify scientific attention toward antibiotic-pathogen pairs

The first objective is to construct measurable signals representing scientific attention in the AMR literature. Publication activity has long been used as a quantitative proxy for research activity and knowledge production.

For each antibiotic-pathogen pair, the project will extract yearly publication counts and related bibliometric indicators from the biomedical literature. Using the currently implemented pipeline, these counts will be background-adjusted using yearly pathogen and antibiotic marginal publication volumes to estimate an expected log count for each pair-year. Scientific interest will then be defined as the difference between observed and expected log counts. This SI series will serve as the core baseline target for forecasting. Where individual antibiotics are too sparse or fragmented, the same framework can also be applied at a coarser level by aggregating antibiotics into broader therapeutic classes.

### Objective 2. Extract conceptual AMR signals from scientific abstracts

Scientific abstracts often contain early indicators of emerging resistance concerns, such as mentions of treatment failure, newly discovered resistance mechanisms, epidemiological spread, or diagnostic challenges.

Using large language models and expert-defined concept categories, the project will extract graded signals representing the presence of such concepts in each abstract. These conceptual signals will be aggregated into time-series indicators for each antibiotic-pathogen pair.

### Objective 3. Model semantic context of pathogens and antibiotics

The language used to describe pathogens and antibiotics evolves over time as new discoveries are made. Advances in natural language processing allow scientific documents to be represented as high-dimensional semantic vectors capturing contextual meaning.

The project will construct semantic embeddings summarizing how specific pathogens and antibiotics are discussed in the literature over time. These representations provide contextual signals beyond simple publication counts.

### Objective 4. Develop multi-signal temporal prediction models

The extracted signals will be integrated into machine-learning models designed to predict future scientific attention.

The existing codebase already contains a baseline forecasting system that fits LOESS curves to yearly log publication counts for each pair, extrapolates future counts, and then converts those forecasts into future SI by subtracting the interpolated expected-count baseline. Functionally, this is a past-SI or past-counts baseline for future-SI prediction. The proposed project extends this baseline in two stages. First, Granger-causality analysis is used to identify which literature-derived signals consistently precede changes in scientific attention or SI. Second, those prioritized signals are used in a sequence-transformer model designed to predict future publication activity and future SI for antibiotic-pathogen pairs. The emphasis is on short-horizon forecasting, especially one-year to five-year prediction, rather than distant extrapolation over multiple decades.

### Objective 5. Evaluate predictive value and early-warning capability

The final objective is to assess whether literature-derived signals provide predictive value beyond simple trend extrapolation.

Model performance will be evaluated using historical data to determine:

- Predictive accuracy of future scientific attention and future SI.
- Lead time for detecting emerging AMR concerns.
- Generalization across antibiotic-pathogen pairs.

If successful, the framework could provide a scalable method for identifying emerging AMR threats earlier than traditional retrospective analyses.

## Methodology

The methodology combines an implemented baseline workflow with a proposed set of extensions. The implemented workflow defines the data, the screened pair-construction process, the SI target, and the current forecasting benchmark. That current benchmark is a future-SI prediction experiment based on past SI or past pair-count trajectories alone. The proposed extensions add concept-level features, semantic representations, Granger-based dependency analysis, and a stronger sequence-transformer predictor to test whether external signals improve on that baseline.

### 4.1 Data sources and corpus construction

The primary data source for the project will be the biomedical literature indexed in PubMed. PubMed provides access to millions of abstracts covering biomedical and life sciences research.

The dataset will be constructed through automated retrieval of abstracts mentioning antibiotics, bacterial pathogens, or antimicrobial resistance. Standardized vocabularies such as Medical Subject Headings (MeSH) will be used to identify relevant entities and ensure consistent mapping of biological terms.

In the current codebase, candidate antibiotic-pathogen records are first generated through PubMed title/abstract queries that combine antibiotic names with pathogen names and pathogen synonyms. This retrieval stage is intentionally broad and is designed to maximize recall rather than to resolve the true relation at query time. As a result, initial candidate rows should be understood as potential pair mentions rather than confirmed AMR evidence. This distinction is important: the project does not treat simple co-occurrence of a pathogen and an antibiotic in the same abstract as evidence of a meaningful AMR relationship.

Each publication record will include:

- Publication year.
- Abstract text.
- Indexed keywords and MeSH terms.
- Bibliographic metadata.

These data will be used to construct longitudinal signals describing the evolution of research attention toward specific antibiotic-pathogen pairs. In addition to modeling individual antibiotics, the archived workflow also shows that the same downstream ranking machinery can be applied to coarser antibiotic abstractions such as classes or simplified categories. This provides an important fallback when individual-drug trajectories are too sparse for stable inference.

### 4.1.1 Pair disambiguation and LLM-based screening

Because co-mention in PubMed does not guarantee that a paper actually reports resistance for the target pair, the project uses an explicit disambiguation step after retrieval. This step is designed specifically to address the main validity problem in bibliometric AMR mining: many abstracts mention multiple organisms, multiple drugs, susceptibility testing panels, or background resistance issues without asserting that the queried pathogen-antibiotic pair is the actual object of AMR evidence.

The current pipeline first flattens yearly pair rows into one row per `(PMID, pathogen, antibiotic, year)` candidate. For each candidate row, the system fetches the PubMed title and abstract and sends the LLM a prompt containing:

- The PMID.
- The exact target pathogen.
- The exact target antibiotic.
- The title.
- The abstract text.

The LLM is instructed to judge the target pair only, ignoring other organisms or drugs mentioned in the abstract. In effect, the model is used as a relation-resolution and relevance-screening layer, not merely as a topic tagger. The unified screening prompt requires that two conditions both hold for a `yes` decision:

1. The abstract explicitly reports that the specified pathogen shows resistance, reduced susceptibility, or treatment failure due to resistance for the specified antibiotic.
2. The study is scientifically relevant to real-world AMR interest, such as clinical, surveillance, epidemiological, environmental, food, veterinary, or naturally occurring strain contexts, rather than being primarily a laboratory model-system experiment where AMR is incidental.

The LLM returns one of three labels:

- `yes`: the target pair is explicitly supported and relevant.
- `no`: the target pair is not supported or the study is not relevant.
- `uncertain`: the abstract is missing, ambiguous, or insufficiently clear.

This means pair disambiguation is performed at the reviewed combination level, not at the PMID level alone. A single PMID can therefore be retained for one pathogen-antibiotic pair and rejected for another if the abstract supports only one of them. The screened yearly dataset is then reconstructed by keeping only the reviewed `(PMID, pathogen, antibiotic, year)` combinations whose unified decision is in the accepted set, with the current default retaining `yes` and `uncertain` rows for downstream yearly aggregation.

This design matters methodologically because it makes the downstream SI series closer to a curated pair-level evidence signal than to a raw co-occurrence count. The proposal therefore treats LLM screening as part of the measurement model for scientific interest: it determines which publications count toward a pair's AMR-relevant attention trajectory in the first place.

### 4.1.2 Dataset scale after pair-level filtering

One practical reviewer question is how many usable examples remain after pair-level filtering and LLM disambiguation. The current archived and refactored outputs are consistent on this point. Using the unified decision table and the yearly screened pair tables, the post-screening dataset has the following scale.

#### Table 1. Corpus size after LLM pair disambiguation

| Inclusion rule | Unique PMIDs | Pathogens | Antibiotics | Unique pairs | Pair-years | Pair-PMID-year rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `yes` only | 50,583 | 566 | 210 | 8,915 | 38,507 | 128,794 |
| `yes` + `uncertain` (current default) | 55,980 | 577 | 210 | 9,330 | 41,284 | 142,883 |

The current SI workflow is built on the `yes` + `uncertain` screened dataset, which yields 9,330 pathogen-antibiotic pairs across 41,278 valid SI pair-years after background merging.

#### Table 2. Usable SI trajectory volume for forecasting

| Criterion | Count |
| --- | ---: |
| Total pairs in yearly SI dataset | 9,330 |
| Median observed years per pair | 2 |
| Mean observed years per pair | 4.42 |
| Pairs with at least 2 yearly observations | 4,732 |
| Pairs with at least 3 yearly observations | 3,296 |
| Pairs with at least 5 yearly observations | 2,078 |
| Pairs with at least 10 yearly observations | 1,120 |
| Pairs with at least 20 yearly observations | 456 |

These counts show that the full corpus is broad, but long trajectories are concentrated in a smaller subset of pairs. This matters for model choice, evaluation design, and interpretation of Granger-style dependency tests.

The archive also contains short-horizon rank-shift outputs for a five-year forecasting setting, for example `si_yearly_rank_shift_simplified_categories_fit_1970_2020_ref_2020_future_2025.parquet`. This is a more credible evaluation regime than 20-year extrapolation and reinforces the case for emphasizing one-year to five-year prediction windows in the proposal.

#### Table 3. Approximate supervised training examples from yearly SI trajectories

| Forecast setup | Approximate examples |
| --- | ---: |
| 1-year-ahead future-SI examples | 31,948 |
| 3-year-ahead future-SI examples | 23,920 |
| 5-year-ahead future-SI examples | 19,301 |

These counts should be interpreted as pair-year training opportunities derived from the screened yearly SI panel, not as independent biological events. They provide an upper bound on the supervised examples available before additional splitting constraints, rolling validation windows, or feature-availability filters are applied.

#### Figure 1. Example of a dense yearly SI trajectory

![Archived yearly SI plot for Escherichia coli and Penicillins](../archive/legacy_assets/interesting_yearly_si_pngs_simplified_categories/Escherichia_coli__Penicillins.png)

The archived simplified-category SI plot for *Escherichia coli* and Penicillins is a useful counterexample to the concern that all trajectories are sparse. It also illustrates why coarser antibiotic representations can be valuable: when related drugs are aggregated into a therapeutic class, the resulting signal is often denser, smoother, and more interpretable than many individual-drug trajectories. This is exactly why the proposal treats antibiotic-class modeling as a credible companion to individual-antibiotic prediction, especially for short-horizon forecasting and robustness analysis.

### 4.2 Scientific interest time-series construction

For each antibiotic-pathogen pair, the project will compute yearly publication counts representing research attention.

Additional bibliometric indicators may include:

- Normalized publication counts.
- Citation-based measures.
- Journal impact indicators.

The current codebase defines the core yearly SI target through a pooled background-adjusted baseline. Specifically, yearly pair counts are merged with yearly pathogen background counts and yearly antibiotic background counts. A pooled ridge regression model is then fit on log-transformed counts using:

- `log10(1 + bg_pathogen_n)`
- `log10(1 + bg_antibiotic_n)`
- Pathogen dummy variables
- Antibiotic dummy variables

This yields an expected log pair count, `expected_log_count_global`, and scientific interest is defined as:

**SI = log10(1 + pair count) - expected_log_count_global**

These SI values form the baseline time series used for forecasting future attention dynamics. This definition is important because it measures pair-specific excess attention relative to general background prominence, not just absolute publication volume. The same construction can also be applied to broader antibiotic classes when a coarser and denser representation is preferable to individual-drug modeling.

### 4.3 Concept extraction from abstracts

To capture higher-level AMR signals, the project will identify conceptual indicators within scientific abstracts.

Examples of conceptual categories include:

- Clinical treatment failure.
- Emergence of resistance mechanisms.
- Epidemiological spread.
- Diagnostic innovation.
- Therapeutic development.
- Surveillance and public health alerts.

A language model will evaluate the presence of each concept in an abstract and assign a graded score representing the strength of association. Scores will be aggregated annually to produce concept-specific time-series signals.

This approach allows the system to detect early changes in how antibiotic-pathogen interactions are discussed in the literature.

The existing unified LLM filter also provides a foundation for this stage because it already performs abstract-level, pair-specific judgment rather than relying on raw co-occurrence. In practice, this means later concept extraction can be applied to a cleaner corpus of pair-resolved AMR-relevant abstracts, reducing contamination from papers that merely mention the antibiotic and pathogen without asserting a relevant resistance relationship.

This is also an important answer to a likely reviewer criticism. The project is not based on naive document-level co-mention counts. Instead, it uses broad retrieval for recall and then applies pair-specific LLM adjudication to decide whether a paper contributes evidence to the focal antibiotic-pathogen trajectory.

### 4.4 Semantic representation of literature context

In addition to explicit conceptual signals, the project will model latent semantic patterns in the literature.

Abstracts mentioning a given pathogen or antibiotic will be embedded using biomedical language models. Embeddings will be averaged annually to generate vector representations capturing the evolving discourse surrounding each entity.

These semantic signals may capture subtle changes in scientific narratives that precede increases in publication activity.

### 4.5 Predictive modeling

The extracted signals will be integrated within a predictive modeling framework with two layers: a current benchmark model and a proposed multi-signal extension.

The current implemented baseline is simpler. It fits a LOESS curve to yearly log counts for each antibiotic-pathogen pair, uses damped linear tails for extrapolation when forecasting beyond the observed range, and then derives predicted SI by subtracting the interpolated expected log-count baseline from the predicted log count. This provides a concrete, already operational benchmark for future work.

The proposed extension is to move beyond this count-only forecasting baseline in a staged way. Granger-causality analysis is first used to identify signals that show consistent predictive temporal relationships with future attention or SI. These results then inform feature prioritization for a sequence-transformer model that can integrate heterogeneous signals over longer temporal contexts than the current baseline.

Sequence transformers are well suited to this setting because they can model long-range dependencies across multiple signal streams, support flexible conditioning on heterogeneous features, and provide a natural framework for learning interactions among attention dynamics, abstract-derived concepts, semantic context, and biological descriptors. In this sense, the proposal should be understood as an extension of an existing SI-forecasting pipeline rather than as a fully greenfield modeling effort.

Conceptually, the model integrates:

- Historical scientific attention.
- Conceptual AMR indicators.
- Semantic context vectors.
- Pathogen descriptors.
- Antibiotic descriptors.

The system would predict future publication counts and future SI at time horizons of one to five years, with the current LOESS-based SI forecaster serving as the baseline comparator and the transformer serving as a later-stage, more expressive model. For pairs with limited support, the same short-horizon prediction task can also be defined at the antibiotic-class level to improve density and robustness.

### 4.6 Evaluation and benchmarking

The predictive framework will be evaluated using retrospective validation.

Baseline models will include:

- Linear trend extrapolation.
- Autoregressive time-series models.
- Nonparametric smoothing approaches, including the existing LOESS-based forecaster.

Performance will be measured using predictive accuracy on both counts and SI, along with the ability to detect emerging attention trajectories earlier than baseline models. Because the codebase already computes post-fit holdout errors for predicted SI, the evaluation framework can be anchored in directly observed pair-level forecasting error rather than only qualitative hotspot identification.

Crucially, the evaluation should test whether abstract-derived signals improve prediction beyond publication momentum alone. That means the central benchmark is not simply whether the proposed model forecasts well in absolute terms, but whether it outperforms autoregressive and trend-based baselines built only from historical pair counts or historical SI. The main novelty claim of the project is therefore not that SI is forecastable at all, but that there exist causal or leading predictors in the literature that improve future-SI prediction beyond past-SI-only models. Evaluation should prioritize short horizons of one to five years, where the forecasting task is most credible, and can be repeated at both the individual-antibiotic and antibiotic-class levels.

### Expected methodological contribution

The proposed framework combines four methodological elements that are rarely integrated in AMR analysis:

1. Pair-level LLM-based disambiguation of AMR evidence from broad literature retrieval.
2. Background-adjusted SI as a pair-specific bibliometric target.
3. Semantic and concept-level signals extracted from biomedical text.
4. A staged temporal modeling pipeline that moves from interpretable Granger-based signal prioritization to sequence-transformer forecasting.

Together, these components provide a scalable approach to forecasting emerging AMR research priorities and identifying antibiotic-pathogen pairs likely to become significant in the near future.

## Granger-Causality Analysis

A key question in the proposed framework is not only whether literature-derived signals can predict future scientific attention toward antibiotic-pathogen pairs, but also which signals are most informative and how they influence one another over time. Understanding these dependencies is important both for interpretability and for validating whether the proposed signals capture meaningful early indicators of emerging AMR concerns.

To address this question, the project will employ Granger causality analysis, a statistical framework for identifying predictive dependencies between time series.

### 5.1 Conceptual overview

Granger causality provides a formal method for testing whether past values of one time series improve the prediction of another time series beyond what can be achieved using the past of the target variable alone.

Formally, a variable X is said to Granger-cause a variable Y if including past observations of X significantly improves the prediction of Y compared with a model that uses only past values of Y.

It is important to emphasize that Granger causality does not establish true mechanistic causation. Instead, it identifies predictive temporal dependencies, which are often useful indicators of underlying relationships.

In the context of AMR literature analysis, this framework can be used to determine whether certain signals extracted from the literature systematically precede increases in scientific attention or increases in background-adjusted scientific interest.

### 5.2 Application to literature-derived signals

The project will construct several types of time-series signals for each antibiotic-pathogen pair:

- Scientific attention signals, including yearly publication counts and the background-adjusted SI series already defined in the current codebase.
- Conceptual signals extracted from abstracts, such as treatment failure, resistance mechanisms, and epidemiological spread.
- Semantic context signals, including embedding-based representations of how pathogens or antibiotics are discussed in the literature.
- Contextual descriptors related to pathogen characteristics or antibiotic properties.

Granger causality tests can be used to evaluate whether any of these signals consistently precede increases in scientific attention or SI.

In practical terms, this involves comparing two models:

1. A model predicting future scientific attention or future SI using only past values of the target signal.
2. A model predicting future scientific attention or future SI using past target values plus past values of another signal.

If the second model significantly improves predictive performance, the additional signal is said to Granger-cause the target signal. In the context of this proposal, that comparison formalizes the central novelty claim: whether signals other than past SI provide additional predictive value for future SI.

### 5.3 Example 1. Resistance mechanism discovery preceding research attention

Consider a conceptual signal representing mentions of novel resistance mechanisms in abstracts.

Let:

- `A_t` represent scientific attention or SI for a specific antibiotic-pathogen pair at year `t`.
- `M_t` represent the frequency of abstracts describing new resistance mechanisms.

If increases in `M_t` systematically occur before increases in `A_t`, the following regression model may reveal a Granger-causal relationship:

```text
A_t = alpha + sum(beta_i * A_{t-i}) + sum(gamma_i * M_{t-i}) + epsilon_t
```

If the coefficients on the lagged `M` terms are statistically significant, this suggests that discoveries of resistance mechanisms tend to precede increases in scientific attention.

This would provide quantitative evidence that mechanistic discoveries act as early signals of emerging AMR concerns.

### 5.4 Example 2. Epidemiological signals predicting research activity

Another conceptual signal may represent mentions of epidemiological spread or outbreaks.

Let:

- `E_t` represent the frequency of abstracts describing epidemiological spread.

If epidemiological mentions precede increased publication activity or increased SI, a Granger test may detect that lagged values of `E` improve prediction of the attention signal.

Such a result would support the hypothesis that clinical or public-health observations drive subsequent increases in research activity.

### 5.5 Example 3. Semantic shifts in scientific discourse

Semantic embeddings allow detection of changes in how pathogens or antibiotics are discussed in the literature.

For instance, a semantic dimension might capture increasing emphasis on:

- Multidrug resistance.
- Treatment failure.
- Hospital-acquired infection.

Changes in these embedding components may precede increases in publication volume or pair-level SI.

Granger analysis can test whether shifts in semantic context vectors predict subsequent increases in attention signals.

This approach enables detection of latent narrative changes in the literature before they manifest as large increases in publication counts.

### 5.6 Multi-variable Granger analysis

Beyond pairwise testing, the project will explore multivariate Granger causality, where multiple signals are considered simultaneously.

Vector autoregressive (VAR) models provide a natural framework for such analysis.

Using VAR models, it becomes possible to examine networks of predictive dependencies among signals such as:

- Resistance mechanism discovery.
- Epidemiological spread.
- Diagnostic innovation.
- Treatment failure.
- Scientific attention.
- Background-adjusted SI.

This analysis can reveal predictive pathways in the evolution of scientific attention, helping to identify which signals consistently act as early indicators.

### 5.7 Relationship with the predictive model

Granger causality analysis serves two complementary roles in the proposed framework.

First, it provides interpretability by identifying which literature-derived signals precede increases in scientific attention or SI.

Second, it helps guide feature selection and model design by highlighting which signals carry predictive information beyond past-SI dynamics alone.

Signals identified as Granger-causal can be prioritized in the later sequence-transformer stage of the predictive architecture described in the methodology. In this way, Granger analysis acts as an interpretability and feature-selection bridge between the current LOESS benchmark and the more powerful transformer-based predictor.

### 5.8 Expected insights

Applying Granger causality in this context can help answer questions such as:

- Do reports of treatment failure precede increases in research attention?
- Do discoveries of resistance genes predict future publication activity?
- Do shifts in scientific discourse anticipate emerging AMR concerns?
- Do some signals predict changes in background-adjusted SI more strongly than they predict raw publication volume?

If consistent predictive dependencies are identified, this would provide stronger evidence that scientific literature contains measurable early signals of emerging antimicrobial resistance issues.

Such findings would support the broader goal of the project: developing a data-driven early-warning system based on the dynamics of scientific attention.

## References

1. World Health Organization. *Antimicrobial resistance: global report on surveillance*. WHO, 2014. https://www.who.int/publications/i/item/9789241564748
2. Pritchard, A. *Statistical bibliography or bibliometrics?* Journal of Documentation, 1969. https://doi.org/10.1108/eb026482
3. Borromeo et al. *Southeast-Asian landscape of antimicrobial resistance research*. https://pubmed.ncbi.nlm.nih.gov/40983143/
4. Sweileh et al. *Bibliometric analysis of peer-reviewed literature on antimicrobial stewardship*. https://pubmed.ncbi.nlm.nih.gov/33397377/
5. Lin et al. *Global insights into MRSA bacteremia: a bibliometric analysis and future outlook*. https://pubmed.ncbi.nlm.nih.gov/39911705/
6. Trends in antimicrobial resistance education research. https://pubmed.ncbi.nlm.nih.gov/40677968/
7. Chen. *CiteSpace II: Detecting and visualizing emerging trends and transient patterns in scientific literature*. https://onlinelibrary.wiley.com/doi/10.1002/asi.20317
8. Rotolo, Hicks, Martin. *What Is an Emerging Technology?* https://arxiv.org/abs/1503.00673
9. Wang. *A bibliometric model for identifying emerging research topics*. https://asistdl.onlinelibrary.wiley.com/doi/10.1002/asi.23930
10. Liang et al. *Combining deep neural network and bibliometric indicator for emerging research topic prediction*. https://www.sciencedirect.com/science/article/pii/S0306457321001072
11. Min, Bu, Sun. *Predicting scientific breakthroughs based on knowledge structure variations*. https://www.sciencedirect.com/science/article/pii/S0040162520313287
12. Forecasting the future of library and information science with bibliometric time-series analysis. https://pmc.ncbi.nlm.nih.gov/articles/PMC7745590/
13. Porter et al. *Emergence scoring to identify frontier R&D topics and key players*. https://www.sciencedirect.com/science/article/pii/S0040162517314804
14. Xu et al. *Multidimensional scientometric indicators for the detection of emerging research topics*. https://www.sciencedirect.com/science/article/pii/S0040162520313160
15. Huang et al. *Tracking the dynamics of co-word networks for emerging topic identification*. https://www.sciencedirect.com/science/article/pii/S0040162521003760
16. Zhong et al. *Bibliometric analysis of carbapenem-resistant Enterobacteriaceae research, 2010-2020*. https://pubmed.ncbi.nlm.nih.gov/34237952/
17. Maimaiti et al. *Global trends and hotspots of phage therapy for bacterial infection*. https://pubmed.ncbi.nlm.nih.gov/36699585/
18. Lee J. et al. *BioBERT: a pre-trained biomedical language representation model for biomedical text mining*. https://doi.org/10.1093/bioinformatics/btz682
19. Vaswani A. et al. *Attention Is All You Need*. https://arxiv.org/abs/1706.03762
20. National Center for Biotechnology Information (NCBI). *PubMed database*. https://pubmed.ncbi.nlm.nih.gov
21. Taşkin Z. et al. *Forecasting publication trends using time series analysis*. https://doi.org/10.1007/s11192-020-03410-3
22. Cleveland W.S. *Robust locally weighted regression and smoothing scatterplots*. https://doi.org/10.1080/01621459.1979.10481038
23. Granger C.W.J. *Investigating causal relations by econometric models and cross-spectral methods*. https://doi.org/10.2307/1912791
24. Sims C.A. *Macroeconomics and reality*. https://doi.org/10.2307/1912017
