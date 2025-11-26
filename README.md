# Pathogen Atlas — Antimicrobial Resistance (AMR)

Utilities and notebooks to collect PubMed evidence for antimicrobial resistance, apply LLM-assisted filtering of abstracts, debias and visualize time trends, and run simple predictive experiments.

## Repository Structure

- `amr/`
  - `config.py` — Shared constants (rate limits, retries, endpoints).
  - `pubmed/` — Query builders and Entrez helpers
    - `queries.py` — antibiotic/pathogen query construction and synonym handling
    - `search.py` — esearch/esummary, pagination, date parsing
    - `cache.py` — metadata cache and parquet merge utilities
  - `pipelines/`
    - `collection.py` — high-level data collection orchestrators (interval queries, yearly counts)
  - `features/`
    - `venues.py` — venue histogram to vector utilities
  - `filtering.py` — PubMed efetch, LLM prompt/decision flow, retry/backoff, logging, and resume support.
  - `debiasing.py` — Interval labeling and expected-vs-observed adjustments (z-scores) for antibiotics/classes.
  - `visualization.py` — Heatmaps and plotting utilities (raw and debiased).
  - `predictive.py` — Feature engineering from time bins and RandomForest experiments (optionally with embeddings).
  - `llm/` — Language model abstraction with OpenAI and Ollama backends (`complete`, `embed`).
  - `data_collection.py` — legacy compatibility shim re-exporting the above.
- `llm.py` — legacy shim re-exporting `amr.llm` (kept for compatibility).
- Notebooks: `1_collect_data.ipynb`, `2_filter.ipynb`, `2_1_iterative_filter.ipynb`, `3_display.ipynb`, `4_predictive.ipynb`, `0_temp.ipynb`.
- Data artifacts (examples): `pmids_results.parquet`, `pmids_results_positive.parquet`, `pmids_results_filtered.parquet`, `venue_index.parquet`, `bacteria_synonym_lookup.csv`, `antibiotic_list*.xlsx`.

## Installation

Requires Python 3.10+.

Install core dependencies (adjust as needed):

```bash
pip install \
  pandas numpy requests biopython pyarrow \
  matplotlib seaborn scikit-learn \
  jupyter
```

Optional (model backends and embeddings):

```bash
# OpenAI
pip install openai
# Torch for embedding tensors (if you use llm.embed)
pip install torch --index-url https://download.pytorch.org/whl/cpu  # or appropriate CUDA/OS build
```

## Credentials and Environment

- OpenAI backend: set `OPENAI_API_KEY` in your environment.
- Ollama backend (local): set `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, and `OLLAMA_EMBED_MODEL` if you want to override defaults.
- NCBI E-utilities: provide a contact email when querying and respect rate limits (~3 req/s); see examples below.

## Quick Start (Library)

### 1) Collect counts and build queries

```python
from amr.pubmed.queries import (
    prepare_antibiotic_pubmed_query_tiab,
    build_pathogen_tiab_query,
)
from amr.pubmed.search import esearch_count_for_year
import pandas as pd

# Example lookup of synonyms loaded from CSV (optional)
bacteria_synonym_df = pd.read_csv("bacteria_synonym_lookup.csv")

pathogen = "Klebsiella pneumoniae"
antibiotic = "co-trimoxazole"  # handles hyphen/ slash terms

pathogen_clause = build_pathogen_tiab_query(pathogen, bacteria_synonym_df)
antibiotic_clause = prepare_antibiotic_pubmed_query_tiab(antibiotic)
term = f"({pathogen_clause}) AND ({antibiotic_clause})"

# Count per year (provide your email, rate-limit per NCBI policy)
counts = []
for year in range(2010, 2025):
    n = esearch_count_for_year(term, year, email="your.name@example.com")
    counts.append({"year": year, "count": n})
counts_df = pd.DataFrame(counts)
```

### 2) LLM-assisted filtering of abstracts

```python
import os
import pandas as pd
from amr.filtering import (
    fetch_pubmed_abstract, prepare_pmids_to_review, run_llm_filter
)
from amr.llm import OpenAILanguageModel, OpenAIConfig

# Load a Parquet produced by your collection step with columns like:
# ["pathogen", "antibiotic", "pmids", "start_date", "end_date"]
pmid_df = pd.read_parquet("pmids_results.parquet")
long_df = prepare_pmids_to_review(pmid_df, max_pmids=100)

# Configure LLM backend (OpenAI example)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "<your-key>")
llm = OpenAILanguageModel(OpenAIConfig(model="gpt-4o-mini", max_tokens=256, temperature=0))

results = run_llm_filter(
    long_df,
    llm,
    email="your.name@example.com",  # forwarded to efetch
    save_every=50,
    progress_every=10,
    output_path="pmids_results_filtered.parquet",
    log_path="pmid_filter.log",
    verbose=1,
)

# results -> DataFrame with pmid, decision (yes/no/uncertain), rationale
```

Notes:
- `run_llm_filter` supports `resume_from`, `seen_pmids`, and `max_llm_calls` for incremental processing.
- Abstracts are fetched via efetch; failures/backoffs are logged and skipped gracefully.

### 3) Debiasing and visualization

```python
import pandas as pd
from amr.debiasing import debias_amr_marginal
from amr.visualization import plot_heatmap_for_pathogen

amr_df = pd.read_parquet("pmids_results_positive.parquet")
class_df = pd.read_excel("antibiotic_list.xlsx")  # contains antibiotic->class mapping

# Compute expected vs observed at antibiotic/class level with z-scores
ab_z, class_z = debias_amr_marginal(amr_df, class_df, interval_method="end")

# Plot raw counts for a pathogen (grouped by antibiotic class)
plot_heatmap_for_pathogen(
    amr_df,
    pathogen="Klebsiella pneumoniae",
    class_df=class_df,
    group_column="class",
    log_scale=True,
)
```

### 4) Predictive experiments

```python
import pandas as pd
from amr.predictive import build_data_and_target, random_forest_experiment_with_embeddings

amr_df = pd.read_parquet("pmids_results_positive.parquet")

# Turn AMR time-binned counts into features/targets
X, Y, feat_df, tgt_df = build_data_and_target(amr_df)

# Simple RandomForest baseline across pathogens
_ = random_forest_experiment_with_embeddings(amr_df, data_df=None, embedding_cols=None, n_repeats=3)
```

## Notebook Workflow

- `1_collect_data.ipynb` — Build queries, collect PMIDs and counts over time.
- `2_filter.ipynb` / `2_1_iterative_filter.ipynb` — Fetch abstracts and run LLM filtering, with resume/logging.
- `3_display.ipynb` — Visualize raw and debiased trends (by antibiotic/class).
- `4_predictive.ipynb` — Train/evaluate simple models using time-binned features.

## Data Expectations

The collection step typically produces a long-format Parquet with rows per `(pathogen, antibiotic)` and a list of PMIDs per time interval. Functions like `prepare_pmids_to_review` expect columns:

- `pathogen` (string)
- `antibiotic` (string)
- `pmids` (list/array of strings or ints)
- `start_date`, `end_date` (datetime-like)

Filtered outputs include per-PMID decisions: `pmid`, `decision` in `{yes, no, uncertain}`, and `rationale`.

## Tips & Troubleshooting

- Respect NCBI rate limits: the helpers sleep between requests and implement retries with exponential backoff.
- Provide a valid contact email to Entrez/efetch functions when possible.
- If using OpenAI, ensure `OPENAI_API_KEY` is set. For local models, point `OllamaLanguageModel` to your `OLLAMA_BASE_URL`.
- Parquet IO: ensure `pyarrow` is installed for best compatibility.

## Disclaimer

This code assists literature exploration and is not a substitute for systematic review or clinical guidance. Validate findings with domain experts.
