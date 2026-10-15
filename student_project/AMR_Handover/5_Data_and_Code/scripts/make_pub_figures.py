#!/usr/bin/env python3
"""Publication figures for the executive summary and supplementary appendix.
Reads ./results/*.csv (from analyse_all.py) and the official metrics_prf output.
Writes PDF (vector, for LaTeX) and PNG (300 dpi, for Word) to ./figs/."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

R = Path(__file__).resolve().parent / "results"
F = Path(__file__).resolve().parent / "figs"; F.mkdir(exist_ok=True)
OFFICIAL = Path("/home/claude/work/PROJECT/amr_pipeline/output")

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
C_SO, C_SOY = ORANGE, BLUE         # strict_only / strict_or_yes, fixed everywhere
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9.5,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.labelsize": 8.5,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": "#c9c8c3", "axes.linewidth": 0.8, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 8, "savefig.dpi": 300,
    "pdf.fonttype": 42,
})
SHORT = {"ref_sonnet5": "Sonnet 5 (ref.)", "gemflash": "Gemini 3.5 Flash-Lite",
         "gpt4nano": "GPT-4.1-nano", "gpt35": "GPT-3.5-turbo",
         "gemma3_4b_or": "Gemma 3 4B", "gemma3_270m": "Gemma 3 270M", "accept_all": "No LLM (accept all)"}
ORDER = ["gemflash", "gpt4nano", "gpt35", "gemma3_4b_or", "gemma3_270m"]

def save(fig, name):
    fig.savefig(F / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(F / f"{name}.png", bbox_inches="tight", facecolor="white", dpi=300)
    plt.close(fig)

cs = pd.read_csv(R / "costs.csv").set_index("model")
off = pd.read_csv(OFFICIAL / "metrics_prf_summary.csv").set_index("llm_key")
pm = pd.read_csv(R / "pair_metrics.csv")
pm1 = pm[pm.temperature == 0.1]
so = pm1[pm1.rule == "strict_only"].set_index("model")
soy = pm1[pm1.rule == "strict_or_yes"].set_index("model")

# local model has $0 API cost; draw it on a floor tick so a log axis works
FLOOR = 0.01
def xcost(m):
    c = cs.loc[m, "cost_per_1k"]
    return c if c > 0 else FLOOR

# ================================================================ FIG A
# Cost vs performance: two panels, shared log-x (USD per 1,000 papers)
def cost_perf(name, width=7.2, height=3.0, compact=False):
    fig, axes = plt.subplots(1, 2, figsize=(width, height), sharex=True)
    models = ["ref_sonnet5"] + ORDER
    # --- panel 1: abstract-level kappa
    ax = axes[0]
    for m in models:
        x = xcost(m)
        y = 1.0 if m == "ref_sonnet5" else off.loc[m, "kappa"]
        ax.scatter(x, y, s=46, color=INK if m == "ref_sonnet5" else BLUE,
                   edgecolor="white", linewidth=1.6, zorder=4)
        off_pts = {"ref_sonnet5": (-8, -2, "right"), "gemflash": (8, 0, "left"),
                   "gpt4nano": (-8, 8, "right"), "gpt35": (8, -2, "left"),
                   "gemma3_4b_or": (8, 8, "left"), "gemma3_270m": (8, -10, "left")}[m]
        ax.annotate(SHORT[m], (x, y), xytext=off_pts[:2], textcoords="offset points",
                    fontsize=7.4, color=INK2, ha=off_pts[2], va="center")
    ax.axhspan(0.41, 0.60, color="#f3f2ee", zorder=0)
    ax.text(0.0105, 0.585, "moderate κ (0.41–0.60)", fontsize=6.8, color=MUTED, va="top")
    ax.set_ylim(-0.2, 1.1)
    ax.set_ylabel("Cohen's κ vs reference\n(73,845 abstracts)")
    ax.set_title("a  Per-paper agreement")
    # --- panel 2: pair-level rank agreement under both rules
    ax = axes[1]
    for m in ORDER:
        x = xcost(m)
        y1 = so.loc[m, "spearman"]; y2 = soy.loc[m, "spearman"]
        if np.isnan(y1):
            ax.scatter(x, -0.02, marker="x", s=30, color=C_SO, zorder=4, linewidth=1.6)
            ax.annotate("undefined\n(constant SI)", (x, -0.02), xytext=(x * 1.25, 0.02),
                        fontsize=6.6, color=MUTED, va="center")
            y1p = -0.02
        else:
            ax.scatter(x, y1, s=40, color=C_SO, edgecolor="white", linewidth=1.5, zorder=4)
            y1p = y1
        ax.plot([x, x], [y1p, y2], color="#c9c8c3", lw=1.2, zorder=2)
        ax.scatter(x, y2, s=40, color=C_SOY, edgecolor="white", linewidth=1.5, zorder=5)
        lab = {"gemflash": "Gemini FL", "gpt4nano": "GPT-4.1n", "gpt35": "GPT-3.5",
               "gemma3_4b_or": "Gemma 4B", "gemma3_270m": "Gemma 270M"}[m]
        if m == "gemflash":
            ax.annotate(lab, (x, y2), xytext=(0, 7), textcoords="offset points",
                        fontsize=6.6, color=INK2, ha="center", va="bottom")
        elif m == "gpt4nano":
            ax.annotate(lab, (x, y1), xytext=(7, 0), textcoords="offset points",
                        fontsize=6.6, color=INK2, ha="left", va="center")
        elif m == "gemma3_270m":
            ax.annotate(lab, (x, y2), xytext=(7, 0), textcoords="offset points",
                        fontsize=6.6, color=INK2, ha="left", va="center")
        else:
            ax.annotate(lab, (x, y1), xytext=(0, -8), textcoords="offset points",
                        fontsize=6.6, color=INK2, ha="center", va="top")
    ax.scatter(xcost("ref_sonnet5"), 1.0, s=46, color=INK, edgecolor="white", linewidth=1.6, zorder=5)
    for rule_tab, c in ((so, C_SO), (soy, C_SOY)):
        b = rule_tab.loc["accept_all", "spearman"]
        ax.axhline(b, color=c, lw=1.0, ls=(0, (4, 3)), alpha=0.8, zorder=1)
    ax.text(3.6, soy.loc["accept_all", "spearman"] - 0.035, "no-LLM baseline", fontsize=6.4,
            color=MUTED, ha="right", va="top")
    ax.text(3.6, so.loc["accept_all", "spearman"] - 0.035, "no-LLM baseline", fontsize=6.4,
            color=MUTED, ha="right", va="top")
    ax.set_ylim(-0.12, 1.14)
    ax.set_ylabel("Spearman ρ of pair priority\n(89 pairs, forecast SI level)")
    ax.set_title("b  Pair-level ranking")
    h = [Line2D([], [], marker="o", ls="", color=C_SO, label="strict-only rule"),
         Line2D([], [], marker="o", ls="", color=C_SOY, label="strict-or-yes rule")]
    ax.legend(handles=h, loc="lower right")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlim(0.006, 4)
        ax.set_xticks([0.01, 0.1, 1])
        ax.set_xticklabels(["$0\n(local)", "$0.10", "$1.00"])
        ax.set_xlabel("API cost per 1,000 papers screened (USD, log)")
    fig.tight_layout(w_pad=2.2)
    save(fig, name)
cost_perf("fig_cost_performance")

# ================================================================ FIG B
# Rule effect: three dumbbell panels (rho, quadrant flips, band pass)
RE_ORDER = ORDER + ["accept_all"]
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), sharey=True)
ypos = np.arange(len(RE_ORDER))[::-1]
for ax, col, title, xl, lim in [
        (axes[0], "spearman", "a  Rank agreement (ρ)", "Spearman ρ", (-0.05, 1.02)),
        (axes[1], "quadrant_flips", "b  Quadrant changes", "share of 89 pairs", (0, 0.8)),
        (axes[2], "band_pass", "c  Inside ref. ±1 SD band", "share of 89 pairs", (0, 1.02))]:
    ax.axhspan(-0.5, 0.5, color="#f3f2ee", zorder=0)
    for y, m in zip(ypos, RE_ORDER):
        a, b = so.loc[m, col], soy.loc[m, col]
        if np.isnan(a):
            ax.scatter(lim[0] + 0.02, y, marker="x", color=C_SO, s=26, linewidth=1.5, zorder=4)
            ax.text(lim[0] + 0.07, y + 0.28, "undef.", fontsize=6.5, color=MUTED)
            a = lim[0] + 0.02
        else:
            ax.scatter(a, y, color=C_SO, s=34, edgecolor="white", linewidth=1.4, zorder=4)
        ax.plot([a, b], [y, y], color="#c9c8c3", lw=1.3, zorder=2)
        ax.scatter(b, y, color=C_SOY, s=34, edgecolor="white", linewidth=1.4, zorder=5)
    ax.set_xlim(*lim); ax.set_title(title); ax.set_xlabel(xl)
    ax.grid(axis="y", visible=False)
axes[0].set_yticks(ypos); axes[0].set_yticklabels([SHORT[m] for m in RE_ORDER])
h = [Line2D([], [], marker="o", ls="", color=C_SO, label="strict-only"),
     Line2D([], [], marker="o", ls="", color=C_SOY, label="strict-or-yes")]
fig.legend(handles=h, loc="lower center", ncol=2, bbox_to_anchor=(0.55, -0.06))
fig.tight_layout(rect=(0, 0.06, 1, 1))
save(fig, "fig_rule_effect")

# ================================================================ FIG C
# model-vs-model disagreement
mv = pd.read_csv(R / "model_vs_model.csv")
mv["label"] = mv.a.str.split("@").str[0].map(SHORT) + "  vs  " + mv.b.str.split("@").str[0].map(SHORT)
piv = mv.pivot(index="label", columns="rule", values="disagree").sort_values("strict_only")
fig, ax = plt.subplots(figsize=(5.6, 2.4))
y = np.arange(len(piv))
for i, (lab, r) in enumerate(piv.iterrows()):
    ax.plot([r.strict_or_yes, r.strict_only], [i, i], color="#c9c8c3", lw=1.3, zorder=2)
    ax.scatter(r.strict_only, i, color=C_SO, s=34, edgecolor="white", linewidth=1.4, zorder=4)
    ax.scatter(r.strict_or_yes, i, color=C_SOY, s=34, edgecolor="white", linewidth=1.4, zorder=5)
ax.set_yticks(y); ax.set_yticklabels(piv.index, fontsize=7.4)
ax.set_xlim(0, 0.6); ax.set_xlabel("share of 89 pairs placed in different quadrants")
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.set_title("Candidate vs candidate: quadrant disagreement")
ax.grid(axis="y", visible=False)
ax.legend(handles=h, loc="lower right")
fig.tight_layout(); save(fig, "fig_model_vs_model")

# ================================================================ FIG D
# verdict mix (100% stacked horizontal bars)
vm = pd.read_csv(R / "verdict_mix.csv").set_index("llm_key")
rows = ["ref_sonnet5"] + ORDER
cols = [("strict", "Confident", BLUE), ("yes", "Probable", AQUA),
        ("no", "No", ORANGE), ("uncertain", "No answer", "#b9b8b2")]
fig, ax = plt.subplots(figsize=(6.4, 2.9))
yy = np.arange(len(rows))[::-1]
for y, m in zip(yy, rows):
    left = 0
    for v, lab, c in cols:
        w = vm.loc[m, v] if v in vm.columns else 0
        ax.barh(y, w, left=left, height=0.62, color=c, edgecolor="white", linewidth=1.2)
        if w > 0.08:
            ax.text(left + w / 2, y, f"{w:.0%}", ha="center", va="center", fontsize=6.8,
                    color="white" if c in (BLUE,) else INK)
        left += w
ax.set_yticks(yy); ax.set_yticklabels([SHORT[m] for m in rows])
ax.set_xlim(0, 1); ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.set_xlabel("share of 73,857 screening decisions (T = 0.10)")
ax.grid(False)
fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in cols],
           labels=[l for _, l, _ in cols], ncol=4, loc="lower center", bbox_to_anchor=(0.55, 0.0))
ax.set_title("Verdict mix: models agree on relevance, disagree on confidence")
fig.tight_layout(rect=(0, 0.08, 1, 1)); save(fig, "fig_verdict_mix")

# ================================================================ FIG E
# per-class F1
pc = pd.read_csv(OFFICIAL / "metrics_prf_perclass.csv")
pc = pc[pc.verdict.isin(["strict", "yes", "no"])]
fig, ax = plt.subplots(figsize=(6.4, 2.4))
w = 0.24
for j, (v, lab, c) in enumerate(cols[:3]):
    vals = [pc[(pc.llm_key == m) & (pc.verdict == v)].f1.fillna(0).iloc[0] for m in ORDER]
    xs = np.arange(len(ORDER)) + (j - 1) * w
    ax.bar(xs, vals, width=w * 0.9, color=c, label=lab)
ax.set_xticks(np.arange(len(ORDER))); ax.set_xticklabels([SHORT[m] for m in ORDER], fontsize=7.6)
ax.set_ylim(0, 1); ax.set_ylabel("F1 vs reference")
ax.legend(ncol=3, loc="upper right"); ax.set_title("Per-class F1 (reference = Claude Sonnet 5)")
ax.grid(axis="x", visible=False)
fig.tight_layout(); save(fig, "fig_perclass_f1")

# ================================================================ FIG F
# size ladder (3 shared pairs)
lad = pd.read_csv(R / "ladder.csv")
lad = lad[(lad.coverage >= 0.5)]
fig, ax = plt.subplots(figsize=(4.6, 2.6))
for fam, c in [("Gemma 3", BLUE), ("Qwen 3", ORANGE)]:
    g = lad[(lad.family == fam) & (lad.route == "local")].sort_values("params_b")
    ax.plot(g.params_b, g.usable, "--", color=c, lw=1.2, alpha=0.7)
    ax.plot(g.params_b, g.usable, "o", color=c, ms=7, mec="white", mew=1.4, label=fam)
    for _, r in g.iterrows():
        ax.annotate(f"{r.usable:.0%}", (r.params_b, r.usable), xytext=(0, -12),
                    textcoords="offset points", ha="center", fontsize=7, color=INK2)
ax.set_xscale("log"); ax.set_xticks([0.27, 0.6, 1.7, 4]); ax.set_xticklabels(["270M", "0.6B", "1.7B", "4B"])
ax.set_ylim(0, 1.12); ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.set_xlabel("parameters (log)"); ax.set_ylabel("abstracts with a parseable verdict")
ax.set_title("Format collapse is family-specific")
ax.text(1.0, 0.22, "Gemma 3 1B: only 22 screens\n(not plotted)", fontsize=6.8, color=MUTED, ha="center")
ax.legend(loc="lower right")
fig.tight_layout(); save(fig, "fig_size_ladder")

print("figures:", sorted(p.name for p in F.glob("*.png")))

# ================================================================ CLASSIC (2003-style) cost figure for the exec summary
def cost_perf_classic(name="fig_cost_performance_classic"):
    import matplotlib as mpl
    with mpl.rc_context({
        "font.family": "Liberation Serif", "font.size": 10.5, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.titlelocation": "center", "axes.edgecolor": "black", "axes.linewidth": 0.8,
        "axes.labelcolor": "black", "xtick.color": "black", "ytick.color": "black",
        "axes.spines.top": True, "axes.spines.right": True, "axes.grid": True,
        "grid.color": "black", "grid.linewidth": 0.4, "grid.linestyle": "-",
        "axes.facecolor": "#C0C0C0", "legend.frameon": True, "legend.edgecolor": "black",
        "legend.fancybox": False, "legend.framealpha": 1.0, "legend.facecolor": "white", "axes.labelsize": 10.5}):
        NAVY, MAG, BLK = "#000080", "#993366", "#000000"
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3), sharex=True)
        models = ["ref_sonnet5"] + ORDER
        ax = axes[0]
        for m in models:
            x = xcost(m); y = 1.0 if m == "ref_sonnet5" else off.loc[m, "kappa"]
            ax.scatter(x, y, s=34, marker="D", color=BLK if m == "ref_sonnet5" else NAVY, zorder=4, linewidth=0)
            o = {"ref_sonnet5": (-7, -1, "right"), "gemflash": (7, 0, "left"), "gpt4nano": (-7, 8, "right"),
                 "gpt35": (7, -2, "left"), "gemma3_4b_or": (7, 8, "left"), "gemma3_270m": (7, -9, "left")}[m]
            ax.annotate(SHORT[m], (x, y), xytext=o[:2], textcoords="offset points", fontsize=9.2, ha=o[2], va="center")
        ax.set_ylim(-0.2, 1.1); ax.set_ylabel("Cohen's kappa vs reference")
        ax.set_title("(a) Per-paper agreement", fontfamily="Liberation Sans")
        ax = axes[1]
        for m in ORDER:
            x = xcost(m); y1 = so.loc[m, "spearman"]; y2 = soy.loc[m, "spearman"]
            if not np.isnan(y1):
                ax.scatter(x, y1, s=30, marker="s", color=MAG, zorder=4, linewidth=0)
            ax.scatter(x, y2, s=34, marker="D", color=NAVY, zorder=5, linewidth=0)
            lab = {"gemflash": "Gemini FL", "gpt4nano": "GPT-4.1n", "gpt35": "GPT-3.5",
                   "gemma3_4b_or": "Gemma 4B", "gemma3_270m": "Gemma 270M"}[m]
            if m == "gemflash":
                ax.annotate(lab, (x, y2), xytext=(0, 7), textcoords="offset points", fontsize=8.8, ha="center", va="bottom")
            elif m == "gemma3_270m":
                ax.annotate(lab, (x, y2), xytext=(7, 0), textcoords="offset points", fontsize=8.8, ha="left", va="center")
            elif m == "gpt4nano":
                ax.annotate(lab, (x, y1), xytext=(7, 0), textcoords="offset points", fontsize=8.8, ha="left", va="center")
            else:
                ax.annotate(lab, (x, y1), xytext=(0, -8), textcoords="offset points", fontsize=8.8, ha="center", va="top")
        ax.scatter(xcost("ref_sonnet5"), 1.0, s=34, marker="D", color=BLK, zorder=5, linewidth=0)
        ax.axhline(soy.loc["accept_all", "spearman"], color=NAVY, lw=0.9, ls="--")
        ax.axhline(so.loc["accept_all", "spearman"], color=MAG, lw=0.9, ls="--")
        ax.text(0.0068, so.loc["accept_all", "spearman"] + 0.02, "no LLM", fontsize=8.4, ha="left", va="bottom", style="italic")
        ax.text(0.0068, soy.loc["accept_all", "spearman"] + 0.02, "no LLM", fontsize=8.4, ha="left", va="bottom", style="italic")
        ax.set_ylim(-0.12, 1.14); ax.set_ylabel("Spearman rho of pair ranking")
        ax.set_title("(b) Pair-level ranking", fontfamily="Liberation Sans")
        h = [Line2D([], [], marker="s", ls="", color=MAG, label="strict-only"),
             Line2D([], [], marker="D", ls="", color=NAVY, label="strict-or-yes")]
        ax.legend(handles=h, loc="lower right", fontsize=9.2)
        for ax in axes:
            ax.set_xscale("log"); ax.set_xlim(0.006, 4); ax.set_xticks([0.01, 0.1, 1])
            ax.set_xticklabels(["$0 (local)", "$0.10", "$1.00"])
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            ax.set_xlabel("API cost per 1,000 papers (USD, log scale)")
        fig.tight_layout(w_pad=2.0)
        save(fig, name)
cost_perf_classic()
