"""Emit the main-text summary table (tab:overall_summary), replacing the old
overall bar figure. Upper block: the four B-DOR arms over the whole grid.
Lower block: proposed vs existing DOR methods after the first session (T=1).
Each block is followed by one-sided paired Wilcoxon signed-rank tests, paired
at the level of the simulated decision problem (identical data for all arms,
replications shared by the two methods being compared).
"""
import os

import numpy as np
from scipy.stats import wilcoxon

import make_paper_figs_v1 as m

OUT = os.path.join("paper_tables", "summary_table_v1.tex")

LBL = {
    'FTRL_DOR_ADJJ1': 'FTRL-DOR',
    'BAYES_DOR_ADJJ1': 'BAYES-DOR',
    'SMAA_DOR_ADJJ1': 'SMAA-DOR',
    'DORLP_DOR_ADJJ1': 'DOR (LP)',
    'FTRLDIR_DOR_ADJJ1': r'FTRL$_{\text{dir}}$',
    'BAYESDIR_DOR_ADJJ1': r'BAYES$_{\text{dir}}$',
}
METS = ['asrs', 'asps', 'aios']

UPPER_ARMS = ['FTRL_DOR_ADJJ1', 'FTRLDIR_DOR_ADJJ1',
              'BAYES_DOR_ADJJ1', 'BAYESDIR_DOR_ADJJ1']
UPPER_TESTS = [('FTRL_DOR_ADJJ1', 'FTRLDIR_DOR_ADJJ1'),
               ('BAYES_DOR_ADJJ1', 'BAYESDIR_DOR_ADJJ1')]
LOWER_ARMS = ['FTRL_DOR_ADJJ1', 'BAYES_DOR_ADJJ1',
              'SMAA_DOR_ADJJ1', 'DORLP_DOR_ADJJ1']
# like-for-like: the point estimator against the LP, the sampling-based
# estimator against SMAA-DOR
LOWER_TESTS = [('FTRL_DOR_ADJJ1', 'DORLP_DOR_ADJJ1'),
               ('BAYES_DOR_ADJJ1', 'SMAA_DOR_ADJJ1')]

CAPTION = (
    r"Summary of the two comparisons. Upper block: mean of each metric over "
    r"the whole simulation grid, with $95\%$ confidence intervals, for the "
    r"four B-DOR algorithms (subscript dir: card counts ignored); the FTRL "
    r"algorithms use $20$ replications per grid cell, the Bayesian ones $5$. Lower "
    r"block: the same means after the first elicitation session only "
    r"($T=1$), for the proposed estimators and the existing DOR methods, "
    r"SMAA-DOR \citep{corrente2025deck} and the original DOR linear "
    r"programme \citep[LP,][]{barbati2024deck}. Each Wilcoxon row reports "
    r"the $p$-value of a one-sided paired Wilcoxon signed-rank test of the "
    r"hypothesis that the first method scores higher than the second, "
    r"paired on the same simulated decision problems and restricted to the "
    r"replications shared by the two methods; in the lower block, each "
    r"proposed estimator is compared with the existing method of the same "
    r"type (the point estimator FTRL-DOR with the LP, the sampling-based "
    r"BAYES-DOR with SMAA-DOR).")


def paired(results, arm_a, arm_b, metric, T=None):
    """Per-table paired values under AR==K; tables index the same datasets."""
    xs, ys = [], []
    for c, arms in results.items():
        p = m.parse_config(c)
        if p['AR'] != p['K'] or arm_a not in arms or arm_b not in arms:
            continue
        da = np.array(arms[arm_a][metric])
        db = np.array(arms[arm_b][metric])
        n = min(da.shape[1], db.shape[1])
        for t in ([T] if T else m.T_LEVELS):
            xs.extend(da[t - 1][:n])
            ys.extend(db[t - 1][:n])
    xs, ys = np.array(xs), np.array(ys)
    ok = ~(np.isnan(xs) | np.isnan(ys))
    return xs[ok], ys[ok]


def pfmt(p):
    if p < 0.001:
        return r"$<0.001$"
    if p > 0.999:
        return r"$>0.999$"
    return f"${p:.3f}$"


def mean_row(results, arm, T=None):
    cells = []
    for met in METS:
        filt = {'T': T} if T else {}
        mean, ci = m.mean_ci(m.pool(results, arm, met, **filt))
        cells.append(f"${mean:.3f}{{\\scriptstyle\\pm{ci:.3f}}}$")
    return f"{LBL[arm]} & " + " & ".join(cells) + r" \\"


def test_row(results, arm_a, arm_b, T=None):
    cells = []
    for met in METS:
        x, y = paired(results, arm_a, arm_b, met, T=T)
        d = x - y
        if np.all(d == 0):
            cells.append("--")
            p = float('nan')
        else:
            p = wilcoxon(x, y, alternative='greater').pvalue
            cells.append(pfmt(p))
        print(f"  {LBL[arm_a]} > {LBL[arm_b]}  {met}  n={len(x)}  "
              f"mean diff {np.mean(d):+.4f}  p={p:.2e}")
    return (f"\\quad {LBL[arm_a]} $>$ {LBL[arm_b]} (Wilcoxon $p$) & "
            + " & ".join(cells) + r" \\")


def main():
    results = np.load(m.NPZ, allow_pickle=True)['results'].item()
    lines = [
        r"\begin{table}[htp]",
        r"    \centering",
        f"    \\caption{{{CAPTION}}}",
        r"    \label{tab:overall_summary}",
        r"\begin{tabularx}{\textwidth}{l|CCC}",
        r"\toprule",
        r"&ASR&ASP&AIO\\",
        r"\midrule",
        r"\multicolumn{4}{l}{\itshape Whole grid, all horizons "
        r"(Section~\ref{sec:sim_results})}\\[2pt]",
    ]
    for arm in UPPER_ARMS:
        lines.append(mean_row(results, arm))
    print("Upper-block tests (whole grid):")
    for a, b in UPPER_TESTS:
        lines.append(test_row(results, a, b))
    lines += [
        r"\midrule",
        r"\multicolumn{4}{l}{\itshape First session only ($T=1$), proposed "
        r"vs existing methods (Section~\ref{sec:sim_baselines})}\\[2pt]",
    ]
    for arm in LOWER_ARMS:
        lines.append(mean_row(results, arm, T=1))
    print("Lower-block tests (T=1):")
    for a, b in LOWER_TESTS:
        lines.append(test_row(results, a, b, T=1))
    lines += [r"\bottomrule", r"\end{tabularx}", r"\end{table}"]

    tex = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
