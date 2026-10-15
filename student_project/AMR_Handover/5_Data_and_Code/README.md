# AMR robustness: analysis package (September 2026)

Everything in the executive summary and supplementary material is regenerated from `output/amr.sqlite`.

## To reproduce (from the amr_pipeline folder)
1. Copy the database, then add the no-LLM baseline to the copy:
   `INSERT INTO screening(pmid,pair_id,llm_key,temperature,replicate,verdict,category,raw,model)
    SELECT pmid,pair_id,'accept_all',0.1,0,'strict','NA','','baseline' FROM screening WHERE llm_key='ref_sonnet5';`
2. `python build_si.py --db <copy> --outdir <si_dir>`   (default settings)
3. Edit the two paths at the top of `scripts/analyse_all.py`, then run it: writes `results/*.csv` and `facts.json`
4. `python scripts/make_pub_figures.py`: writes every figure

## results/
- `abstract_metrics.csv`, `perclass.csv`: per-paper agreement vs Sonnet 5 (T = 0.10 only; the documents quote the official `metrics_prf.py` values, which majority-vote across temperatures and differ by < 0.01)
- `binary_agreement.csv`: accept/reject agreement under each rule
- `pair_metrics.csv`: rho, quadrant changes, top-5, ±1 SD band, emerging overlap, both rules, all systems incl. the no-LLM baseline
- `model_vs_model.csv`: candidate-vs-candidate quadrant disagreement
- `temperature.csv`, `verdict_mix.csv`, `ladder.csv`
- `tokens.csv`, `costs.csv`: cost per paper / per 1,000 / full corpus
- `quadrant_strict_only.csv`, `quadrant_strict_or_yes.csv`: forecast SI level and shift per pair and system
- `facts.json`: headline counts
