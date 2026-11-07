"""Write Table 3 of the paper (tab:overall_summary): overall means and Wilcoxon tests.

Reads the aggregated grid results (192 AR == K configurations) and writes one LaTeX table
of ASR, ASP and AIO, each cell the mean +- half-width of the normal 95% CI.
Upper block: FTRL-DOR, FTRLdir, BAYES-DOR and BAYESdir over the whole grid (T in
{1, 3, 5, 10}); each family is read on the replications it shares with its direction-only
partner (20 per cell for the FTRL family, 5 for the BAYES family), followed by one-sided
paired Wilcoxon signed-rank tests FTRL-DOR > FTRLdir and BAYES-DOR > BAYESdir.
Lower block: FTRL-DOR, BAYES-DOR, SMAA-DOR and DOR after the first session only (T = 1),
20 replications per cell, with the tests FTRL-DOR > DOR and BAYES-DOR > SMAA-DOR. Tests
pair the two methods per replication (identical data), on the replications both have.

Paper:   Table 3 (its numbers are quoted in Sec. 5.3, 'Overall comparison', and Sec. 5.5,
         'Overall picture')
Inputs:  results/grid_results_joint_v1.npz
Outputs: outputs/tables/summary_table_v1.tex; test details on stdout
Runtime: a few seconds

Usage:
    python scripts/paper/make_summary_table.py

(formerly make_summary_table_v1.py)
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, so `import bdor` works without installing

import numpy as np
from scipy.stats import wilcoxon

from bdor.paths import TABLES_DIR
from bdor.results import PARTNER, T_LEVELS, load_results, mean_ci, parse_config, pool

OUT = os.path.join(TABLES_DIR, "summary_table_v1.tex")

LBL = {
    'FTRL_DOR_ADJJ1': 'FTRL-DOR',
    'BAYES_DOR_ADJJ1': 'BAYES-DOR',
    'SMAA_DOR_ORIG': 'SMAA-DOR',
    'DORLP_DOR_ORIG': 'DOR',
    'FTRLDIR_DOR_ADJJ1': r'FTRL$_{\text{dir}}$',
    'BAYESDIR_DOR_ADJJ1': r'BAYES$_{\text{dir}}$',
}
METS = ['asrs', 'asps', 'aios']

UPPER_ARMS = ['FTRL_DOR_ADJJ1', 'FTRLDIR_DOR_ADJJ1',
              'BAYES_DOR_ADJJ1', 'BAYESDIR_DOR_ADJJ1']
UPPER_TESTS = [('FTRL_DOR_ADJJ1', 'FTRLDIR_DOR_ADJJ1'),
               ('BAYES_DOR_ADJJ1', 'BAYESDIR_DOR_ADJJ1')]
LOWER_ARMS = ['FTRL_DOR_ADJJ1', 'BAYES_DOR_ADJJ1',
              'SMAA_DOR_ORIG', 'DORLP_DOR_ORIG']
# like-for-like: the point estimator against DOR, the sampling-based
# estimator against SMAA-DOR
LOWER_TESTS = [('FTRL_DOR_ADJJ1', 'DORLP_DOR_ORIG'),
               ('BAYES_DOR_ADJJ1', 'SMAA_DOR_ORIG')]

CAPTION = (
    r"Summary of the two comparisons. Upper block: mean of each metric over "
    r"the whole simulation grid, with $95\%$ confidence intervals, for the "
    r"four B-DOR algorithms (subscript dir: card counts ignored); each "
    r"algorithm is read on the replications it shares with its own "
    r"direction-only version, $20$ per grid cell for the FTRL family and $5$ "
    r"for the Bayesian one. Lower "
    r"block: the same means after the first elicitation session only "
    r"($T=1$), for the proposed estimators and the existing DOR methods, "
    r"SMAA-DOR \citep{corrente2025deck} and the original DOR method "
    r"\citep{barbati2024deck}, all four on the same $20$ replications per "
    r"grid cell. Each Wilcoxon row reports "
    r"the $p$-value of a one-sided paired Wilcoxon signed-rank test of the "
    r"hypothesis that the first method scores higher than the second, "
    r"paired on the same simulated decision problems and restricted to the "
    r"replications shared by the two methods; in the lower block, each "
    r"proposed estimator is compared with the existing method of the same "
    r"type (the point estimator FTRL-DOR with DOR, the sampling-based "
    r"BAYES-DOR with SMAA-DOR).")


def paired(results, arm_a, arm_b, metric, T=None):
    """Paired per-replication values of two arms over the AR == K configurations.

    Replication i of a configuration is the same simulated decision problem for every
    arm, so each cell contributes its first ``min(n_a, n_b)`` replications; pairs with a
    NaN on either side are dropped.

    Parameters
    ----------
    results : dict
        Grid results (``bdor.results.load_results``).
    arm_a, arm_b : str
        Arm keys.
    metric : {'asrs', 'asps', 'aios'}
    T : int, optional
        Use only the values after session ``T``; default: all of ``T_LEVELS``.

    Returns
    -------
    x, y : ndarray, shape (n_pairs,)
        Values of ``arm_a`` and ``arm_b``.
    """
    xs, ys = [], []
    for c, arms in results.items():
        p = parse_config(c)
        if p['AR'] != p['K'] or arm_a not in arms or arm_b not in arms:
            continue
        da = np.array(arms[arm_a][metric])
        db = np.array(arms[arm_b][metric])
        n = min(da.shape[1], db.shape[1])
        for t in ([T] if T else T_LEVELS):
            xs.extend(da[t - 1][:n])
            ys.extend(db[t - 1][:n])
    xs, ys = np.array(xs), np.array(ys)
    ok = ~(np.isnan(xs) | np.isnan(ys))
    return xs[ok], ys[ok]


def pfmt(p):
    """Format a p-value for the table (bounded at 0.001 and 0.999)."""
    if p < 0.001:
        return r"$<0.001$"
    if p > 0.999:
        return r"$>0.999$"
    return f"${p:.3f}$"


def mean_row(results, arm, T=None, match=None):
    """LaTeX row of mean +- 95% CI of every metric for ``arm`` (optionally at horizon
    ``T`` only, and cut to the replications shared with the arm ``match``)."""
    cells = []
    for met in METS:
        filt = {'T': T} if T else {}
        mean, ci = mean_ci(pool(results, arm, met, match=match, **filt))
        cells.append(f"${mean:.3f}{{\\scriptstyle\\pm{ci:.3f}}}$")
    return f"{LBL[arm]} & " + " & ".join(cells) + r" \\"


def test_row(results, arm_a, arm_b, T=None):
    """LaTeX row of one-sided paired Wilcoxon p-values (``arm_a`` > ``arm_b``) for every
    metric; also prints n, the mean difference and the raw p-value."""
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
    """Build the table, write it to ``OUT`` and print the test details."""
    results = load_results()
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
        # each family read on the replications it shares with its own
        # direction-only version, so the ablation stays a paired comparison
        lines.append(mean_row(results, arm, match=PARTNER.get(arm)))
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
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(tex)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
