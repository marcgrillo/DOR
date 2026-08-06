"""
Generate the Empirical-evidence figures and summary statistics for the DOR paper.

Design decisions (see paper Section 'Empirical evidence'):
  * Only the steps T in {1,3,5,10} are reported (the elicitation horizons of interest).
  * The 'initial reference set size' factor is DROPPED: because the data-generating
    process does not change, it has the same role as the per-step subset size k.
    We therefore keep ONLY the configurations in which the initial subset size equals
    the per-step subset size (AR == K) and report a single factor 'subset size k'.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

CSV = "grid_results_full.csv"
OUT = "paper_figs"
os.makedirs(OUT, exist_ok=True)

STEPS = [1, 3, 5, 10]
C_FTRL, C_BAYES = "#1f4fd8", "#e8810c"      # blue / orange

df = pd.read_csv(CSV)
# --- filter: drop the redundant initial-reference-set factor (keep AR == K) ---
df = df[df.AR_Active_Ratio == df.K_Subset_Size].copy()
df = df[df.Step_T.isin(STEPS)].copy()

METRICS = [("ASRS", "ASR"), ("ASPS", "ASP"), ("AIOS", "AIOS")]

def stat(sub, col):
    v = sub[col].to_numpy()
    n = len(v)
    if n == 0:
        return np.nan, np.nan
    return v.mean(), 1.96 * v.std() / np.sqrt(n)

# ----------------------------------------------------------------------------
# 1. print summary table (for the LaTeX annex table + in-text numbers)
# ----------------------------------------------------------------------------
def line(label, sub):
    out = [label]
    for col, _ in METRICS:
        for meth in ["FTRL", "BAYES"]:
            mu, ci = stat(sub[sub.Method == meth], col)
            out.append(f"{mu:.3f}")
    return out

factors = [
    ("all", None, None),
    ("|A|", "F1_Alternatives", [10, 20, 35, 50]),
    ("m",   "F2_Criteria",     [3, 5, 7, 9]),
    ("T",   "Step_T",          [1, 3, 5, 10]),
    ("k",   "K_Subset_Size",   [3, 4, 5]),
    ("Inc", "Inconsistency_Pct", [0, 15, 35, 50]),
]

print("\n=== SUMMARY (AR==K, steps {1,3,5,10}) ===")
print("rows FTRL:", (df.Method=='FTRL').sum(), " rows BAYES:", (df.Method=='BAYES').sum())
hdr = ["config", "ASR_F","ASR_B","ASP_F","ASP_B","AIOS_F","AIOS_B"]
print("\t".join(hdr))
# overall with CI
for meth in ["FTRL","BAYES"]:
    s=df[df.Method==meth]
    print(f"CI_{meth}", *[f"{stat(s,c)[1]:.4f}" for c,_ in METRICS])
for fname, fcol, levels in factors:
    if fcol is None:
        print("\t".join(line("all", df)))
    else:
        for lv in levels:
            print("\t".join(line(f"{fname}={lv}", df[df[fcol] == lv])))

# ----------------------------------------------------------------------------
# 2. Overall bar chart
# ----------------------------------------------------------------------------
plt.rcParams.update({"font.size": 20})
fig, ax = plt.subplots(figsize=(11, 6.2))
x = np.arange(len(METRICS))
w = 0.36
for i, meth in enumerate(["FTRL", "BAYES"]):
    mus, cis = [], []
    for col, _ in METRICS:
        mu, ci = stat(df[df.Method == meth], col)
        mus.append(mu); cis.append(ci)
    pos = x + (i - 0.5) * w
    color = C_FTRL if meth == "FTRL" else C_BAYES
    bars = ax.bar(pos, mus, w, yerr=cis, capsize=6, color=color,
                  label=f"{meth}-DOR", edgecolor="black", linewidth=0.6)
    for p, mu in zip(pos, mus):
        ax.text(p, mu + 0.018, f"{mu:.3f}", ha="center", va="bottom",
                fontsize=17, color=color, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([lbl for _, lbl in METRICS], fontsize=22)
ax.set_ylim(0, 1.0)
ax.set_ylabel("Mean metric value", fontsize=21)
ax.legend(fontsize=20, loc="upper left", framealpha=0.95)
ax.grid(True, axis="y", linestyle="--", alpha=0.45)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_overall.png"), dpi=200)
plt.close()

# ----------------------------------------------------------------------------
# 3. Consolidated factor-effects grid: rows = metric, cols = factor
# ----------------------------------------------------------------------------
grid_factors = [
    ("F1_Alternatives", [10, 20, 35, 50], r"Number of alternatives $|A|$"),
    ("F2_Criteria",     [3, 5, 7, 9],     r"Number of criteria $m$"),
    ("Step_T",          [1, 3, 5, 10],    r"Elicitation steps $T$"),
    ("K_Subset_Size",   [3, 4, 5],        r"Subset size $k$"),
    ("Inconsistency_Pct", [0, 15, 35, 50], r"Inconsistency $|F_{inc}|$ (\%)" if False else "Inconsistency (%)"),
]

plt.rcParams.update({"font.size": 15})
nrow, ncol = len(METRICS), len(grid_factors)
fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.3 * nrow), sharex="col")

for r, (col, mlabel) in enumerate(METRICS):
    # shared y range across the row
    row_vals = []
    for fcol, levels, _ in grid_factors:
        for meth in ["FTRL", "BAYES"]:
            for lv in levels:
                mu, _ = stat(df[(df.Method == meth) & (df[fcol] == lv)], col)
                row_vals.append(mu)
    lo, hi = np.nanmin(row_vals), np.nanmax(row_vals)
    pad = 0.08 * (hi - lo + 1e-9)
    for c, (fcol, levels, flabel) in enumerate(grid_factors):
        ax = axes[r, c]
        xs = np.arange(len(levels))
        for meth, color, ls, mk in [("FTRL", C_FTRL, "-", "o"),
                                    ("BAYES", C_BAYES, "--", "s")]:
            mus, cis = [], []
            for lv in levels:
                mu, ci = stat(df[(df.Method == meth) & (df[fcol] == lv)], col)
                mus.append(mu); cis.append(ci)
            mus, cis = np.array(mus), np.array(cis)
            ax.plot(xs, mus, ls, color=color, marker=mk, markersize=7,
                    linewidth=2.4, label=f"{meth}-DOR")
            ax.fill_between(xs, mus - cis, mus + cis, color=color, alpha=0.18)
        ax.set_xticks(xs)
        ax.set_xticklabels(levels, fontsize=13)
        ax.set_ylim(lo - pad, hi + pad)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        if r == 0:
            ax.set_title(flabel, fontsize=16)
        if c == 0:
            ax.set_ylabel(mlabel, fontsize=19, fontweight="bold")
        ax.tick_params(axis="y", labelsize=13)
        ax.tick_params(axis="x", labelsize=13)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=18,
           framealpha=0.95, bbox_to_anchor=(0.5, 1.03))
fig.tight_layout(rect=[0, 0, 1, 0.985])
plt.savefig(os.path.join(OUT, "fig_factors.png"), dpi=200, bbox_inches="tight")
plt.close()

print("\nWrote paper_figs/fig_overall.png and paper_figs/fig_factors.png")
