"""Emit the appendix tables of the experimental section: one overall table plus
one table per experimental factor, covering all six compared arms. Reuses the
pooling of make_paper_figs_v1 so every entry matches the figures exactly."""
import os

import numpy as np

import make_paper_figs_v1 as m

OUT = os.path.join("paper_tables", "appendix_tables_v1.tex")

ARMS = ['FTRL_DOR_ADJJ1', 'BAYES_DOR_ADJJ1', 'SMAA_DOR_ADJJ1',
        'DORLP_DOR_ADJJ1', 'FTRLDIR_DOR_ADJJ1', 'BAYESDIR_DOR_ADJJ1']
HEADER = r"FTRL&BAYES&SMAA&LP&FTRL$_{\text{dir}}$&BAYES$_{\text{dir}}$"
METS = [('asrs', 'ASR'), ('asps', 'ASP'), ('aios', 'AIO')]

# (label, filter key, levels, header symbol, caption description, row formatter)
FACTORS = [
    ('tab:app_alt', 'F1', [10, 20, 35, 50], r'$|A|$',
     r'the number of alternatives $|A|$', lambda lv: f"${lv}$"),
    ('tab:app_crit', 'F2', [3, 5, 7, 9], r'$m$',
     r'the number of criteria $m$', lambda lv: f"${lv}$"),
    ('tab:app_steps', 'T', [1, 3, 5, 10], r'$T$',
     r'the number of elicitation steps $T$', lambda lv: f"${lv}$"),
    ('tab:app_sub', 'K', [3, 4, 5], r'$k$',
     r'the subset size $k$', lambda lv: f"${lv}$"),
    ('tab:app_inc', 'Inc', [0, 15, 35, 50], r'$|F_{\text{inc}}|$',
     r'the share of inconsistent preferences $|F_{\text{inc}}|$',
     lambda lv: f"${lv}\\%$"),
]

CAP_OVERALL = (
    r"Mean of each metric over the whole simulation grid, for the six compared "
    r"methods: the proposed FTRL-DOR and BAYES-DOR, SMAA-DOR "
    r"\citep{corrente2025deck}, the original DOR linear programme "
    r"\citep[LP,][]{barbati2024deck} and the two direction-only ablations "
    r"(subscript dir: card counts ignored). Each entry reports the mean with "
    r"the half-width of its $95\%$ confidence interval as a subscript "
    r"(\,$\text{mean}{\scriptstyle\pm\text{c.i.}}$\,). The point estimators "
    r"(FTRL-DOR, LP, FTRL$_{\text{dir}}$) use $20$ replications per grid "
    r"cell, the sampling-based methods (BAYES-DOR, SMAA-DOR, "
    r"BAYES$_{\text{dir}}$) use $5$.")


def cell(results, arm, met, fac=None, lv=None):
    if fac is None:
        v = m.pool(results, arm, met)
    elif fac == 'T':
        v = m.pool(results, arm, met, T=lv)
    else:
        v = m.pool(results, arm, met, **{fac: lv})
    mean, ci = m.mean_ci(v)
    return f"${mean:.3f}{{\\scriptstyle\\pm{ci:.3f}}}$"


def overall_table(results):
    lines = [
        r"\begin{table}[htp]",
        r"    \centering",
        f"    \\caption{{{CAP_OVERALL}}}",
        r"    \label{tab:app_overall}",
        r"\begin{tabularx}{\textwidth}{l|CCCCCC}",
        r"\toprule",
        f"&{HEADER}\\\\",
        r"\midrule",
    ]
    for met, name in METS:
        row = " & ".join(cell(results, a, met) for a in ARMS)
        lines.append(f"{name} & {row} \\\\")
    lines += [r"\bottomrule", r"\end{tabularx}", r"\end{table}"]
    return "\n".join(lines)


def factor_table(results, label, fac, levels, sym, desc, fmt):
    cap = (f"Mean of each metric at each level of {desc}, averaged over all "
           r"the other factors of the grid. Methods and conventions as in "
           r"Table~\ref{tab:app_overall}.")
    lines = [
        r"\begin{table}[htp]",
        r"    \centering",
        f"    \\caption{{{cap}}}",
        f"    \\label{{{label}}}",
        r"\begin{tabularx}{\textwidth}{ll|CCCCCC}",
        r"\toprule",
        f"&{sym}&{HEADER}\\\\",
        r"\midrule",
    ]
    for b, (met, name) in enumerate(METS):
        for i, lv in enumerate(levels):
            head = f"\\multirow{{{len(levels)}}}{{*}}{{{name}}}" if i == 0 else ""
            row = " & ".join(cell(results, a, met, fac, lv) for a in ARMS)
            lines.append(f"{head}&{fmt(lv)} & {row} \\\\")
        lines.append(r"\midrule" if b < len(METS) - 1 else r"\bottomrule")
    lines += [r"\end{tabularx}", r"\end{table}"]
    return "\n".join(lines)


def main():
    results = np.load(m.NPZ, allow_pickle=True)['results'].item()
    blocks = [overall_table(results)]
    for label, fac, levels, sym, desc, fmt in FACTORS:
        blocks.append(factor_table(results, label, fac, levels, sym, desc, fmt))
    tex = "\n\n".join(blocks) + "\n"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(tex)
    print(tex)
    print(f"% wrote {OUT}")


if __name__ == "__main__":
    main()
