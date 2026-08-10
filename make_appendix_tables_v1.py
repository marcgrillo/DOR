"""Emit the appendix tables of the experimental section, two groups:
  1. the four B-DOR arms (FTRL/BAYES, full and direction-only) over the whole
     grid: one table per experimental factor;
  2. the single-session comparison (T = 1) of the proposed estimators against
     the one-shot existing DOR methods (SMAA-DOR, LP): one table per
     remaining factor.
The overall means live in the main text (tab:overall_summary, emitted by
make_summary_table_v1.py); captions anchor their conventions there.
Reuses the pooling of make_paper_figs_v1 so every entry matches the figures.
"""
import os

import numpy as np
from scipy.stats import wilcoxon

import make_paper_figs_v1 as m

OUT = os.path.join("paper_tables", "appendix_tables_v1.tex")

ARMS_BDOR = ['FTRL_DOR_ADJJ1', 'BAYES_DOR_ADJJ1',
             'FTRLDIR_DOR_ADJJ1', 'BAYESDIR_DOR_ADJJ1']
HEADER_BDOR = r"FTRL&BAYES&FTRL$_{\text{dir}}$&BAYES$_{\text{dir}}$"
ARMS_BASE = ['FTRL_DOR_ADJJ1', 'BAYES_DOR_ADJJ1',
             'SMAA_DOR_ADJJ1', 'DORLP_DOR_ADJJ1']
HEADER_BASE = r"FTRL&BAYES&SMAA&LP"
METS = [('asrs', 'ASR'), ('asps', 'ASP'), ('aios', 'AIO')]

# Per-level one-sided paired Wilcoxon tests appended to each factor table.
# B-DOR tables: within-family only -- each estimator against its own
# direction-only version (the per-level value of the card counts).
TESTS_BDOR = [
    ('FTRL_DOR_ADJJ1', 'FTRLDIR_DOR_ADJJ1', r"$p_{\mathrm{F>F_d}}$"),
    ('BAYES_DOR_ADJJ1', 'BAYESDIR_DOR_ADJJ1', r"$p_{\mathrm{B>B_d}}$"),
]
# T=1 tables: each proposed estimator against the existing method of the same
# type -- point estimator vs the LP, sampling-based vs SMAA-DOR.
TESTS_BASE = [
    ('FTRL_DOR_ADJJ1', 'DORLP_DOR_ADJJ1', r"$p_{\mathrm{F>L}}$"),
    ('BAYES_DOR_ADJJ1', 'SMAA_DOR_ADJJ1', r"$p_{\mathrm{B>S}}$"),
]

NOTE_BDOR = (
    r" The last two columns report the $p$-values of one-sided paired "
    r"Wilcoxon signed-rank tests of each estimator against its own "
    r"direction-only version, at that factor level "
    r"(F $=$ FTRL-DOR, F$_{\mathrm{d}}$ $=$ FTRL$_{\text{dir}}$, B $=$ "
    r"BAYES-DOR, B$_{\mathrm{d}}$ $=$ BAYES$_{\text{dir}}$); pairing as in "
    r"Table~\ref{tab:overall_summary}.")
NOTE_BASE = (
    r" The last two columns report the $p$-values of one-sided paired "
    r"Wilcoxon signed-rank tests of each proposed estimator against the "
    r"existing method of the same type --- the point estimator FTRL-DOR "
    r"against the linear programme, the sampling-based BAYES-DOR against "
    r"SMAA-DOR --- at that factor level (F $=$ FTRL-DOR, B $=$ BAYES-DOR, "
    r"S $=$ SMAA-DOR, L $=$ LP); pairing as in "
    r"Table~\ref{tab:overall_summary}, values close to $1$ indicate that "
    r"the existing method is ahead.")

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


def paired(results, arm_a, arm_b, metric, fac=None, lv=None, t_fix=None):
    """Per-table paired values under AR==K, restricted to one factor level;
    tables with the same index share the same simulated decision problem."""
    xs, ys = [], []
    for c, arms in results.items():
        p = m.parse_config(c)
        if p['AR'] != p['K'] or arm_a not in arms or arm_b not in arms:
            continue
        if fac is not None and fac != 'T' and p[fac] != lv:
            continue
        da = np.array(arms[arm_a][metric])
        db = np.array(arms[arm_b][metric])
        n = min(da.shape[1], db.shape[1])
        if t_fix is not None:
            ts = [t_fix]
        elif fac == 'T':
            ts = [lv]
        else:
            ts = m.T_LEVELS
        for t in ts:
            xs.extend(da[t - 1][:n])
            ys.extend(db[t - 1][:n])
    xs, ys = np.array(xs), np.array(ys)
    ok = ~(np.isnan(xs) | np.isnan(ys))
    return xs[ok], ys[ok]


def ptest(results, arm_a, arm_b, metric, fac, lv, t_fix=None):
    x, y = paired(results, arm_a, arm_b, metric, fac=fac, lv=lv, t_fix=t_fix)
    d = x - y
    if np.all(d == 0):
        return "--"
    p = wilcoxon(x, y, alternative='greater').pvalue
    print(f"    {arm_a.split('_')[0]}>{arm_b.split('_')[0]} {metric} "
          f"{fac}={lv}{' T=1' if t_fix else ''}: n={len(x)} "
          f"diff {np.mean(d):+.4f} p={p:.2e}")
    if p < 0.001:
        return r"$<0.001$"
    if p > 0.999:
        return r"$>0.999$"
    return f"${p:.3f}$"


def factor_table(results, arms, header, label, ref, fac, levels, sym, desc,
                 fmt, t_fix=None, blocknote="", tests=(), note=""):
    lead = ("Single-session comparison ($T=1$): mean of each metric at each "
            f"level of {desc}" if t_fix is not None
            else f"Mean of each metric at each level of {desc}")
    cap = (f"{lead}, averaged over all the other factors of the grid. "
           f"Methods and conventions as in Table~\\ref{{{ref}}}{blocknote}."
           f"{note}")
    cols = 'C' * len(arms) + '|' + 'c' * len(tests)
    test_header = "".join(f"&{lbl}" for _, _, lbl in tests)
    lines = [
        r"\begin{table}[htp]",
        r"    \centering",
        f"    \\caption{{{cap}}}",
        f"    \\label{{{label}}}",
        f"\\begin{{tabularx}}{{\\textwidth}}{{ll|{cols}}}",
        r"\toprule",
        f"&{sym}&{header}{test_header}\\\\",
        r"\midrule",
    ]
    for b, (met, name) in enumerate(METS):
        for i, lv in enumerate(levels):
            head = f"\\multirow{{{len(levels)}}}{{*}}{{{name}}}" if i == 0 else ""
            row = " & ".join(cell(results, a, met, fac, lv, t_fix=t_fix)
                             for a in arms)
            trow = "".join(
                " & " + ptest(results, a, b_, met, fac, lv, t_fix=t_fix)
                for a, b_, _ in tests)
            lines.append(f"{head}&{fmt(lv)} & {row}{trow} \\\\")
        lines.append(r"\midrule" if b < len(METS) - 1 else r"\bottomrule")
    lines += [r"\end{tabularx}", r"\end{table}"]
    return "\n".join(lines)


def main():
    results = np.load(m.NPZ, allow_pickle=True)['results'].item()
    blocks = []

    # group 1: the four B-DOR algorithms over the whole grid
    for suff, fac, levels, sym, desc, fmt in FACTORS:
        print(f"tab:app_{suff}")
        blocks.append(factor_table(results, ARMS_BDOR, HEADER_BDOR,
                                   f"tab:app_{suff}", "tab:overall_summary",
                                   fac, levels, sym, desc, fmt,
                                   blocknote=" (upper block)",
                                   tests=TESTS_BDOR, note=NOTE_BDOR))

    # group 2: single-session comparison against the existing DOR methods
    for suff, fac, levels, sym, desc, fmt in FACTORS:
        if fac == 'T':
            continue
        print(f"tab:app_t1_{suff}")
        blocks.append(factor_table(results, ARMS_BASE, HEADER_BASE,
                                   f"tab:app_t1_{suff}", "tab:overall_summary",
                                   fac, levels, sym, desc, fmt, t_fix=1,
                                   blocknote=" (lower block)",
                                   tests=TESTS_BASE, note=NOTE_BASE))

    tex = "\n\n".join(blocks) + "\n"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(tex)
    print(tex)
    print(f"% wrote {OUT}")


if __name__ == "__main__":
    main()
