#!/usr/bin/env python3
"""
analyse_all.py -- one script that recomputes every number used in the
executive summary, supplementary appendix and cost analysis, straight from
output/amr.sqlite plus the build_si.py quadrant tables.

Outputs (all in ./results):
  abstract_metrics.csv       per-model abstract-level agreement vs reference
  perclass.csv               per-class P/R/F1
  binary_agreement.csv       accept/reject agreement under each rule
  pair_metrics.csv           pair-level: rho, flips, top-5, band pass, emerging Jaccard
  model_vs_model.csv         quadrant disagreement between candidate models
  temperature.csv            T0.10 vs T0.15 self-agreement
  verdict_mix.csv            verdict shares per model (full corpus)
  ladder.csv                 size ladder on the 3 shared pairs
  tokens.csv                 measured token usage per call
  costs.csv                  cost per paper / per 1k / full corpus per model
  facts.json                 headline numbers for the documents
"""
from __future__ import annotations
import json, sqlite3, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DB = Path("/home/claude/work/PROJECT/amr_pipeline/output/amr.sqlite")
SI = sorted(Path("/home/claude/work/si_baseline").iterdir())[-1]   # includes accept_all baseline
OUT = ROOT / "results"; OUT.mkdir(exist_ok=True)
sys.path.insert(0, "/home/claude/work/PROJECT/amr_pipeline")
import prompt as P  # the exact screening prompt

REF = "ref_sonnet5"
CANDS = ["gemflash", "gpt4nano", "gpt35", "gemma3_4b_or", "gemma3_270m"]
NICE = {"ref_sonnet5": "Claude Sonnet 5 (reference)",
        "gemflash": "Gemini 3.5 Flash-Lite", "gpt4nano": "GPT-4.1-nano",
        "gpt35": "GPT-3.5-turbo", "gemma3_4b_or": "Gemma 3 4B",
        "gemma3_270m": "Gemma 3 270M"}
LABEL = {"strict": "Confident", "yes": "Probable", "no": "No", "uncertain": "No answer"}
con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
facts = {}

# ------------------------------------------------------------------ corpus
facts["pairs"] = con.execute("select count(*) from pairs").fetchone()[0]
facts["pathogens"], facts["antibiotics"] = con.execute(
    "select count(distinct pathogen), count(distinct antibiotic) from pairs").fetchone()
facts["unique_abstracts"] = con.execute("select count(*) from abstracts").fetchone()[0]
facts["pair_abstract_links"] = con.execute(
    "select count(distinct pair_id||'|'||pmid) from pair_pmids").fetchone()[0]
facts["year_range"] = list(con.execute("select min(year),max(year) from pair_pmids").fetchone())
facts["count_cells"] = con.execute("select count(*) from counts").fetchone()[0]
facts["screening_rows"] = con.execute("select count(*) from screening").fetchone()[0]
facts["models_attempted"] = con.execute("select count(distinct llm_key) from screening").fetchone()[0]
facts["systems_attempted"] = con.execute(
    "select count(*) from (select distinct llm_key,temperature from screening)").fetchone()[0]
facts["ref_via_openrouter"] = con.execute(
    "select count(*) from screening where llm_key='ref_sonnet5' and model like 'anthropic/%'").fetchone()[0]

# ------------------------------------------------------- abstract level
scr = pd.read_sql_query(
    "select pmid,pair_id,llm_key,temperature,verdict,raw from screening", con)
scr["verdict"] = scr["verdict"].fillna("uncertain")
ref = scr[scr.llm_key == REF].drop_duplicates(["pmid", "pair_id"]).set_index(["pmid", "pair_id"])["verdict"]
facts["ref_screens"] = int(len(ref))

def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    cats = np.union1d(a, b)
    po = (a == b).mean()
    pe = sum((a == c).mean() * (b == c).mean() for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else np.nan

def at(key, t=0.1):
    d = scr[(scr.llm_key == key) & (np.isclose(scr.temperature, t))]
    return d.drop_duplicates(["pmid", "pair_id"]).set_index(["pmid", "pair_id"])["verdict"]

pool = ref.index
for k in CANDS:
    pool = pool.intersection(at(k).index)
facts["shared_pool"] = int(len(pool))
truth = ref.loc[pool]

rows, pcrows, brows = [], [], []
for k in CANDS:
    pred = at(k).loc[pool]
    pc = []
    for v in ["strict", "yes", "no", "uncertain"]:
        tp = int(((truth == v) & (pred == v)).sum()); fp = int(((truth != v) & (pred == v)).sum())
        fn = int(((truth == v) & (pred != v)).sum())
        prec = tp / (tp + fp) if tp + fp else np.nan
        rec = tp / (tp + fn) if tp + fn else np.nan
        f1 = 2 * prec * rec / (prec + rec) if (prec == prec and rec == rec and prec + rec) else np.nan
        pc.append(dict(model=k, verdict=v, label=LABEL[v], support=int((truth == v).sum()),
                       predicted=int((pred == v).sum()), precision=prec, recall=rec, f1=f1))
    pc = pd.DataFrame(pc); pcrows.append(pc)
    core = pc[pc.verdict.isin(["strict", "yes", "no"])]
    rows.append(dict(model=k, name=NICE[k], n=len(pool),
                     accuracy=float((truth == pred).mean()), kappa=kappa(truth, pred),
                     macro_f1_core=core.f1.mean(skipna=True), core_classes=int(core.f1.notna().sum()),
                     no_answer_rate=float((pred == "uncertain").mean())))
    for rule, acc in [("strict_only", {"strict"}), ("strict_or_yes", {"strict", "yes"})]:
        ta, pa = truth.isin(acc), pred.isin(acc)
        brows.append(dict(model=k, rule=rule, agreement=float((ta == pa).mean()),
                          kappa=kappa(ta, pa), ref_accept_rate=float(ta.mean()),
                          model_accept_rate=float(pa.mean())))
am = pd.DataFrame(rows); am.to_csv(OUT / "abstract_metrics.csv", index=False)
pd.concat(pcrows).to_csv(OUT / "perclass.csv", index=False)
bn = pd.DataFrame(brows); bn.to_csv(OUT / "binary_agreement.csv", index=False)

# ------------------------------------------------------ temperature check
trows = []
for k in ["gemflash", "gpt4nano", "gpt35"]:
    a, b = at(k, 0.1), at(k, 0.15)
    ix = a.index.intersection(b.index)
    trows.append(dict(model=k, n=len(ix), agreement=float((a.loc[ix] == b.loc[ix]).mean()),
                      kappa=kappa(a.loc[ix], b.loc[ix])))
pd.DataFrame(trows).to_csv(OUT / "temperature.csv", index=False)

# ------------------------------------------------------------ verdict mix
vm = (scr[np.isclose(scr.temperature, 0.1) & scr.llm_key.isin([REF] + CANDS)]
      .groupby(["llm_key", "verdict"]).size().unstack(fill_value=0))
vm = vm.div(vm.sum(axis=1), axis=0)
vm.to_csv(OUT / "verdict_mix.csv")

# ------------------------------------------------------------ size ladder
LP = ("klebsiella_pneumoniae__ciprofloxacin", "acinetobacter_baumannii__tigecycline",
      "escherichia_coli__ceftazidime")
lad = scr[scr.pair_id.isin(LP) & np.isclose(scr.temperature, 0.1)]
refn = lad[lad.llm_key == REF].drop_duplicates(["pmid", "pair_id"]).shape[0]
lrows = []
SIZES = {"gemma3_270m": ("Gemma 3", 0.27, "local"), "gemma3_1b": ("Gemma 3", 1.0, "local"),
         "gemma3_4b": ("Gemma 3", 4.0, "local"), "gemma3_4b_or": ("Gemma 3", 4.0, "OpenRouter"),
         "qwen3_06b": ("Qwen 3", 0.6, "local"), "qwen3_17b": ("Qwen 3", 1.7, "local")}
for k, (fam, size, route) in SIZES.items():
    g = lad[lad.llm_key == k]
    if g.empty: continue
    n = len(g)
    lrows.append(dict(model=k, family=fam, params_b=size, route=route, n=n,
                      coverage=n / refn, usable=1 - (g.verdict == "uncertain").mean(),
                      strict_share=(g.verdict == "strict").mean(),
                      no_share=(g.verdict == "no").mean()))
pd.DataFrame(lrows).to_csv(OUT / "ladder.csv", index=False)
facts["ladder_ref_abstracts"] = int(refn)

# ------------------------------------------------------------- pair level
def quad(level, shift):
    return np.where(level >= 0, np.where(shift >= 0, "HR", "HF"), np.where(shift >= 0, "EM", "LF"))

prow, mvm = [], []
for rule in ["strict_only", "strict_or_yes"]:
    q = pd.read_csv(SI / rule / "quadrant.csv")
    q["q"] = quad(q.si_level, q.si_shift)
    L = q.pivot(index="pair_id", columns="system", values="si_level")
    S = q.pivot(index="pair_id", columns="system", values="si_shift")
    Q = q.pivot(index="pair_id", columns="system", values="q")
    SD = q.pivot(index="pair_id", columns="system", values="resid_sd")
    R = f"{REF}@T0.1"
    ref_em = set(Q.index[Q[R] == "EM"])
    for sysname in sorted(c for c in L.columns if c != R):
        rho = L[R].corr(L[sysname], method="spearman")
        flips = float((Q[R] != Q[sysname]).mean())
        top5 = len(set(L[R].nlargest(5).index) & set(L[sysname].nlargest(5).index))
        band = float(((L[sysname] >= L[R] - SD[R]) & (L[sysname] <= L[R] + SD[R])).mean())
        em = set(Q.index[Q[sysname] == "EM"])
        jac = len(ref_em & em) / len(ref_em | em) if (ref_em | em) else np.nan
        k, t = sysname.split("@T")
        prow.append(dict(rule=rule, system=sysname, model=k, temperature=float(t), n_pairs=len(L),
                         spearman=rho, quadrant_flips=flips, top5_overlap=top5, band_pass=band,
                         ref_emerging=len(ref_em), model_emerging=len(em),
                         emerging_shared=len(ref_em & em), emerging_jaccard=jac))
    cands = [f"{c}@T0.1" for c in ["gemflash", "gpt4nano", "gpt35", "gemma3_4b_or"]]
    for i, a in enumerate(cands):
        for b in cands[i + 1:]:
            mvm.append(dict(rule=rule, a=a, b=b, disagree=float((Q[a] != Q[b]).mean())))
pm = pd.DataFrame(prow); pm.to_csv(OUT / "pair_metrics.csv", index=False)
mv = pd.DataFrame(mvm); mv.to_csv(OUT / "model_vs_model.csv", index=False)
facts["model_vs_model"] = {r: [float(mv[mv.rule == r].disagree.min()), float(mv[mv.rule == r].disagree.max())]
                           for r in ["strict_only", "strict_or_yes"]}
facts["ref_emerging"] = {r: int(pm[pm.rule == r].ref_emerging.iloc[0]) for r in ["strict_only", "strict_or_yes"]}

# ---------------------------------------------------------------- tokens
abs_ = pd.read_sql_query("select pmid,title,abstract from abstracts", con).set_index("pmid")
pairs = pd.read_sql_query("select * from pairs", con).set_index("pair_id")
refrows = scr[scr.llm_key == REF].drop_duplicates(["pmid", "pair_id"])[["pmid", "pair_id"]]
sys_chars = len(P.SYSTEM)
def in_chars(r):
    a = abs_.loc[r.pmid]; p = pairs.loc[r.pair_id]
    return sys_chars + len(P.build_user_prompt(p.pathogen, p.antibiotic, a.title or "", a.abstract or ""))
chars = refrows.apply(in_chars, axis=1)
CPT = 4.0                                   # standard ~4 characters per token for English text
tok_in = chars / CPT + 12                  # + chat-template overhead per message pair
facts["tokens_in_mean"] = float(tok_in.mean())
facts["tokens_in_median"] = float(tok_in.median())
facts["tokens_in_p95"] = float(tok_in.quantile(.95))
facts["corpus_screens"] = int(len(refrows))

out_tok = (scr[np.isclose(scr.temperature, 0.1) & scr.llm_key.isin([REF] + CANDS)]
           .assign(t=lambda d: d.raw.fillna("").str.len() / CPT)
           .groupby("llm_key").t.mean())
trow = [dict(model=k, tokens_in=facts["tokens_in_mean"], tokens_out=float(out_tok[k])) for k in [REF] + CANDS]
pd.DataFrame(trow).to_csv(OUT / "tokens.csv", index=False)

# ----------------------------------------------------------------- costs
# USD per 1M tokens (input, output). Source: config.py PRICES_PER_MTOK, list
# prices recorded Jul-Aug 2026 from OpenAI / Anthropic / OpenRouter.
PRICE = {"ref_sonnet5": (3.00, 15.00), "gemflash": (0.30, 2.50), "gpt4nano": (0.10, 0.40),
         "gpt35": (0.50, 1.50), "gemma3_4b_or": (0.05, 0.10), "gemma3_270m": (0.0, 0.0)}
ROUTE = {"ref_sonnet5": "Anthropic Batch API (50% off)", "gemflash": "OpenRouter",
         "gpt4nano": "OpenAI", "gpt35": "OpenAI", "gemma3_4b_or": "OpenRouter",
         "gemma3_270m": "Local GPU (LM Studio)"}
# wall-clock throughput from run timestamps (T=0.1 only)
ts = pd.read_sql_query("select llm_key, min(run_at) a, max(run_at) b, count(*) n from screening "
                       "where abs(temperature-0.1)<1e-9 group by llm_key", con)
ts["hours"] = (pd.to_datetime(ts.b) - pd.to_datetime(ts.a)).dt.total_seconds() / 3600
ts = ts.set_index("llm_key")

N = facts["corpus_screens"]
crow = []
am_i = am.set_index("model"); pm_y = pm[(pm.rule == "strict_or_yes") & (pm.temperature == 0.1)].set_index("model")
pm_s = pm[(pm.rule == "strict_only") & (pm.temperature == 0.1)].set_index("model")
for k in [REF] + CANDS:
    pin, pout = PRICE[k]
    ti, to = facts["tokens_in_mean"], float(out_tok[k])
    if k == REF:
        to_bill = 60.0   # adaptive thinking is billed as output; max_tokens=60 caps it -> use the cap
        per_std = (ti * pin + to_bill * pout) / 1e6
        per = per_std * 0.5
    else:
        to_bill = to
        per_std = per = (ti * pin + to_bill * pout) / 1e6
    crow.append(dict(model=k, name=NICE[k], route=ROUTE[k], price_in=pin, price_out=pout,
                     tokens_in=ti, tokens_out_billed=to_bill,
                     cost_per_paper=per, cost_per_paper_standard=per_std,
                     cost_per_1k=per * 1000, cost_full_corpus=per * N,
                     cost_per_pair_avg=per * N / facts["pairs"],
                     relative_to_ref=None,
                     kappa=1.0 if k == REF else am_i.loc[k, "kappa"],
                     macro_f1=1.0 if k == REF else am_i.loc[k, "macro_f1_core"],
                     rho_soy=1.0 if k == REF else pm_y.loc[k, "spearman"],
                     rho_so=1.0 if k == REF else pm_s.loc[k, "spearman"],
                     flips_soy=0.0 if k == REF else pm_y.loc[k, "quadrant_flips"],
                     flips_so=0.0 if k == REF else pm_s.loc[k, "quadrant_flips"],
                     wall_hours=float(ts.loc[k, "hours"]) if k in ts.index else np.nan))
cs = pd.DataFrame(crow)
refc = cs.loc[cs.model == REF, "cost_per_paper"].iloc[0]
cs["relative_to_ref"] = cs.cost_per_paper / refc
cs.to_csv(OUT / "costs.csv", index=False)

# the sweep as actually run (T=0.10 and 0.15 for three cloud models)
spent = 0.0
for k, reps in {"ref_sonnet5": 1, "gemflash": 2, "gpt4nano": 2, "gpt35": 2, "gemma3_4b_or": 1}.items():
    spent += cs.set_index("model").loc[k, "cost_per_paper"] * N * reps
facts["study_cost_estimate"] = spent
facts["costs"] = cs.set_index("model")[["cost_per_paper", "cost_per_1k", "cost_full_corpus", "relative_to_ref"]].to_dict("index")

json.dump(facts, open(OUT / "facts.json", "w"), indent=2, default=float)
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
print(json.dumps(facts, indent=1, default=float))
for f in ["abstract_metrics", "binary_agreement", "temperature", "ladder", "model_vs_model", "costs"]:
    print(f"\n== {f}"); print(pd.read_csv(OUT / f"{f}.csv").round(4).to_string(index=False))
print("\n== pair_metrics"); print(pm.round(3).to_string(index=False))
print("\n== verdict_mix"); print(vm.round(3))
