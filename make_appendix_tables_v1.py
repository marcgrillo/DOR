"""Emit the appendix tables of the experimental section, two groups:
  1. the four B-DOR arms (FTRL/BAYES, full and direction-only) over the whole
     grid: one overall table plus one table per experimental factor;
  2. the single-session comparison (T = 1) of the proposed estimators against
     the one-shot existing DOR methods (SMAA-DOR, LP): one overall table plus
     one table per remaining factor.
Reuses the pooling of make_paper_figs_v1 so every entry matches the figures.
"""
import os

import numpy as np

import make_paper_figs_v1 as m

OUT = os.path.join("paper_tables", "appendix_tables_v1.tex")

ARMS_BDOR = ['FTRL_DOR_ADJJ1', 'BAYES_DOR_ADJJ1',
             'FTRLDIR_DOR_ADJJ1', 'BAYESDIR_DOR_ADJJ1']
HEADER_BDOR = r"FTRL&BAYES&FTRL$_{\text{dir}}$&BAYES$_{\text{dir}}$"
ARMS_BASE = ['FTRL_DOR_ADJJ1', 'BAYES_DOR_ADJJ1',
             'SMAA_DOR_ADJJ1', 'DORLP_DOR_ADJJ1']
HEADER_BASE = r"FTRL&BAYES&SMAA&LP"
METS = [('asrs', 'ASR'), ('asps', 'ASP'), ('aios', 'AIO')]

# (label suffix, filter key, levels, header symbol, caption description, fmt)
FACTORS = [
    ('alt', 'F1', [10, 20, 35, 50], r'$|A|$',
     r'the number of alternatives $|A|$', lambda lv: f"${lv}$"),
    ('crit', 'F2', [3, 5, 7, 9], r'$m$',
     r'the number of criteria $m$', lambda lv: f"${lv}$"),
    ('steps', 'T', [1, 3, 5, 10], r'$T$',
     r'the number of elicitation steps $T$', lambda lv: f"${lv}$"),
    ('sub', 'K', [3, 4, 5], r'$k$',
     r'the subset size $k$', lambda lv: f"${lv}$"),
    ('inc', 'Inc', [0, 15, 35, 50], r'$|F_{\text{inc}}|$',
     r'the share of inconsistent preferences $|F_{\text{inc}}|$',
     lambda lv: f"${lv}\\%$"),
]

CAP_OVERALL_BDOR = (
    r"Mean of each metric over the whole simulation grid, for the four B-DOR "
    r"arms: FTRL-DOR and BAYES-DOR with the full likelihood, and their "
    r"direction-only versions (subscript dir: card counts ignored). Each "
    r"entry reports the mean with the half-width of its $95\%$ confidence "
    r"interval as a subscript (\,$\text{mean}{\scriptstyle\pm\text{c.i.}}$\,)."
    r" The FTRL arms use $20$ replications per grid cell, the Bayesian arms "
    r"$5$.")

CAP_OVERALL_T1 = (
    r"Single-session comparison of Section~\ref{sec:sim_baselines}: mean of "
    r"each metric after the first elicitation session ($T=1$), averaged over "
    r"the whole grid, for the proposed estimators and the existing DOR "
    r"methods: SMAA-DOR \citep{corrente2025deck} and the original DOR linear "
    r"programme \citep[LP,][]{barbati2024deck}. Each entry reports the mean "
    r"with the half-width of its $95\%$ confidence interval as a subscript. "
    r"FTRL-DOR and the LP use $20$ replications per grid cell, BAYES-DOR and "
    r"SMAA-DOR $5$.")


def cell(results, arm, met, fac=None, lv=None, t_fix=None):
    filt = {}
    if fac == 'T':
        filt['T'] = lv
    elif fac is not None:
        filt[fac] = lv
    if t_fix is not None:
        filt['T'] = t_fix
    v = m.pool(results, arm, met, **filt)
    mean, ci = m.mean_ci(v)
    return f"${mean:.3f}{{\\scriptstyle\\pm{ci:.3f}}}$"


def overall_table(results, arms, header, cap, label, t_fix=None):
    cols = 'C' * len(arms)
    lines = [
        r"\begin{table}[htp]",
        r"    \centering",
        f"    \\caption{{{cap}}}",
        f"    \\label{{{label}}}",
        f"\\begin{{tabularx}}{{\\textwidth}}{{l|{cols}}}",
        r"\toprule",
        f"&{header}\\\\",
        r"\midrule",
    ]
    for met, name in METS:
        row = " & ".join(cell(results, a, met, t_fix=t_fix) for a in arms)
        lines.append(f"{name} & {row} \\\\")
    lines += [r"\bottomrule", r"\end{tabularx}", r"\end{table}"]
    return "\n".join(lines)


def factor_table(results, arms, header, label, ref, fac, levels, sym, desc,
                 fmt, t_fix=None):
    lead = ("Single-session comparison ($T=1$): mean of each metric at each "
            f"level of {desc}" if t_fix is not None
            else f"Mean of each metric at each level of {desc}")
    cap = (f"{lead}, averaged over all the other factors of the grid. "
           f"Methods and conventions as in Table~\\ref{{{ref}}}.")
    cols = 'C' * len(arms)
    lines = [
        r"\begin{table}[htp]",
        r"    \centering",
        f"    \\caption{{{cap}}}",
        f"    \\label{{{label}}}",
        f"\\begin{{tabularx}}{{\\textwidth}}{{ll|{cols}}}",
        r"\toprule",
        f"&{sym}&{header}\\\\",
        r"\midrule",
    ]
    for b, (met, name) in enumerate(METS):
        for i, lv in enumerate(levels):
            head = f"\\multirow{{{len(levels)}}}{{*}}{{{name}}}" if i == 0 else ""
            row = " & ".join(cell(results, a, met, fac, lv, t_fix=t_fix)
                             for a in arms)
            lines.append(f"{head}&{fmt(lv)} & {row} \\\\")
        lines.append(r"\midrule" if b < len(METS) - 1 else r"\bottomrule")
    lines += [r"\end{tabularx}", r"\end{table}"]
    return "\n".join(lines)


def main():
    results = np.load(m.NPZ, allow_pickle=True)['results'].item()
    blocks = []

    # group 1: the four B-DOR arms over the whole grid
    blocks.append(overall_table(results, ARMS_BDOR, HEADER_BDOR,
                                CAP_OVERALL_BDOR, "tab:app_overall"))
    for suff, fac, levels, sym, desc, fmt in FACTORS:
        blocks.append(factor_table(results, ARMS_BDOR, HEADER_BDOR,
                                   f"tab:app_{suff}", "tab:app_overall",
                                   fac, levels, sym, desc, fmt))

    # group 2: single-session comparison against the existing DOR methods
    blocks.append(overall_table(results, ARMS_BASE, HEADER_BASE,
                                CAP_OVERALL_T1, "tab:app_t1_overall", t_fix=1))
    for suff, fac, levels, sym, desc, fmt in FACTORS:
        if fac == 'T':
            continue
        blocks.append(factor_table(results, ARMS_BASE, HEADER_BASE,
                                   f"tab:app_t1_{suff}", "tab:app_t1_overall",
                                   fac, levels, sym, desc, fmt, t_fix=1))

    tex = "\n\n".join(blocks) + "\n"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(tex)
    print(tex)
    print(f"% wrote {OUT}")


if __name__ == "__main__":
    main()
