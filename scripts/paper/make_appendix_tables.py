"""Write Supplementary Tables B.1-B.9: metric means at each factor level, with Wilcoxon tests.

Reads the aggregated grid results (192 AR == K configurations) and writes nine LaTeX
tables; every cell is the mean +- half-width of the normal 95% CI of ASR, ASP or AIO at one
level of one factor, averaged over all the other factors.
Tables B.1-B.5 (labels tab:app_alt, _crit, _steps, _sub, _inc): FTRL-DOR, BAYES-DOR, FTRLdir
and BAYESdir by |A|, m, T, k and Finc (T in {1, 3, 5, 10}); each family is read on the
replications it shares with its direction-only partner (20 per cell for FTRL, 5 for BAYES),
with per-level one-sided paired Wilcoxon tests FTRL-DOR > FTRLdir and BAYES-DOR > BAYESdir.
Tables B.6-B.9 (labels tab:app_t1_alt, _crit, _sub, _inc): FTRL-DOR, BAYES-DOR, SMAA-DOR and
DOR at T = 1 by |A|, m, k and Finc, 20 replications per cell, with per-level tests
FTRL-DOR > DOR and BAYES-DOR > SMAA-DOR. The pooling (bdor.results.pool) is the one of
make_figures.py, so every entry matches Figures 2 and 3.

Paper:   Supplementary Tables B.1-B.9 (Appendix B), quoted in the factor paragraphs of
         Sec. 5.3 and 5.5; the captions refer to Table 3 (make_summary_table.py)
Inputs:  results/grid_results_joint_v1.npz
Outputs: outputs/tables/appendix_tables_v1.tex; test details and the LaTeX on stdout
Runtime: a few seconds

Usage:
    python scripts/paper/make_appendix_tables.py

(formerly make_appendix_tables_v1.py)
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, so `import bdor` works without installing

import numpy as np
from scipy.stats import wilcoxon

from bdor.paths import TABLES_DIR
from bdor.results import PARTNER, T_LEVELS, load_results, mean_ci, parse_config, pool

OUT = os.path.join(TABLES_DIR, "appendix_tables_v1.tex")

ARMS_BDOR = ['FTRL_DOR_ADJJ1', 'BAYES_DOR_ADJJ1',
             'FTRLDIR_DOR_ADJJ1', 'BAYESDIR_DOR_ADJJ1']
HEADER_BDOR = r"FTRL&BAYES&FTRL$_{\text{dir}}$&BAYES$_{\text{dir}}$"
ARMS_BASE = ['FTRL_DOR_ADJJ1', 'BAYES_DOR_ADJJ1',
             'SMAA_DOR_ORIG', 'DORLP_DOR_ORIG']
HEADER_BASE = r"FTRL&BAYES&SMAA&DOR"
METS = [('asrs', 'ASR'), ('asps', 'ASP'), ('aios', 'AIO')]

# Per-level one-sided paired Wilcoxon tests appended to each factor table.
# B-DOR tables: within-family only -- each estimator against its own
# direction-only version (the per-level value of the card counts).
TESTS_BDOR = [
    ('FTRL_DOR_ADJJ1', 'FTRLDIR_DOR_ADJJ1', r"$p_{\mathrm{F>F_d}}$"),
    ('BAYES_DOR_ADJJ1', 'BAYESDIR_DOR_ADJJ1', r"$p_{\mathrm{B>B_d}}$"),
]
# T=1 tables: each proposed estimator against the existing method of the same
# type -- the point estimator against DOR (the DOR linear programme), the
# sampling-based one against SMAA-DOR.
TESTS_BASE = [
    ('FTRL_DOR_ADJJ1', 'DORLP_DOR_ORIG', r"$p_{\mathrm{F>D}}$"),
    ('BAYES_DOR_ADJJ1', 'SMAA_DOR_ORIG', r"$p_{\mathrm{B>S}}$"),
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
    r"against DOR, the sampling-based BAYES-DOR against "
    r"SMAA-DOR --- at that factor level (F $=$ FTRL-DOR, B $=$ BAYES-DOR, "
    r"S $=$ SMAA-DOR, D $=$ DOR); pairing as in "
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


def cell(results, arm, met, fac=None, lv=None, t_fix=None, match=None):
    """LaTeX cell 'mean +- 95% CI' of ``arm``/``met`` at level ``lv`` of factor ``fac``.

    ``fac`` is a key of ``bdor.results.parse_config`` or 'T'; ``t_fix`` restricts the
    pool to one horizon; ``match`` cuts every configuration to the replications shared
    with that arm (see ``bdor.results.pool``).
    """
    filt = {}
    if fac == 'T':
        filt['T'] = lv
    elif fac is not None:
        filt[fac] = lv
    if t_fix is not None:
        filt['T'] = t_fix
    v = pool(results, arm, met, match=match, **filt)
    mean, ci = mean_ci(v)
    return f"${mean:.3f}{{\\scriptstyle\\pm{ci:.3f}}}$"


def paired(results, arm_a, arm_b, metric, fac=None, lv=None, t_fix=None):
    """Paired per-replication values of two arms at one factor level (AR == K only).

    Replication i of a configuration is the same simulated decision problem for every
    arm, so each configuration contributes its first ``min(n_a, n_b)`` replications;
    pairs with a NaN on either side are dropped. The horizons are ``t_fix`` if given,
    else ``lv`` when ``fac == 'T'``, else all of ``T_LEVELS``.

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
            ts = T_LEVELS
        for t in ts:
            xs.extend(da[t - 1][:n])
            ys.extend(db[t - 1][:n])
    xs, ys = np.array(xs), np.array(ys)
    ok = ~(np.isnan(xs) | np.isnan(ys))
    return xs[ok], ys[ok]


def ptest(results, arm_a, arm_b, metric, fac, lv, t_fix=None):
    """LaTeX cell with the p-value of the one-sided paired Wilcoxon signed-rank test
    ``arm_a`` > ``arm_b`` at one factor level ('--' if all differences are zero); the
    test details are printed."""
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
                 fmt, t_fix=None, blocknote="", tests=(), note="",
                 match_map=None):
    """LaTeX source of one appendix table: metric blocks by the levels of ``fac``.

    Parameters
    ----------
    results : dict
        Grid results (``bdor.results.load_results``).
    arms : list of str
        Arms, one column each; ``header`` holds their LaTeX column titles.
    label, ref : str
        LaTeX label of this table and of the table its caption refers to.
    fac, levels, sym, desc, fmt
        One entry of ``FACTORS``: filter key, levels, header symbol, caption text and
        level formatter.
    t_fix : int, optional
        Restrict every cell to this horizon (the T = 1 tables).
    blocknote, note : str
        Caption fragments (block of Table 3 referred to; explanation of the tests).
    tests : sequence of (arm_a, arm_b, column title)
        One-sided paired Wilcoxon tests ``arm_a`` > ``arm_b``, one column each.
    match_map : dict, optional
        ``{arm: partner}`` for the replication matching of ``cell``.
    """
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
            row = " & ".join(cell(results, a, met, fac, lv, t_fix=t_fix,
                                  match=(match_map or {}).get(a))
                             for a in arms)
            trow = "".join(
                " & " + ptest(results, a, b_, met, fac, lv, t_fix=t_fix)
                for a, b_, _ in tests)
            lines.append(f"{head}&{fmt(lv)} & {row}{trow} \\\\")
        lines.append(r"\midrule" if b < len(METS) - 1 else r"\bottomrule")
    lines += [r"\end{tabularx}", r"\end{table}"]
    return "\n".join(lines)


def main():
    """Build Tables B.1-B.9, write them to ``OUT`` and print them with the test details."""
    results = load_results()
    blocks = []

    # group 1: the four B-DOR algorithms over the whole grid
    for suff, fac, levels, sym, desc, fmt in FACTORS:
        print(f"tab:app_{suff}")
        blocks.append(factor_table(results, ARMS_BDOR, HEADER_BDOR,
                                   f"tab:app_{suff}", "tab:overall_summary",
                                   fac, levels, sym, desc, fmt,
                                   blocknote=" (upper block)",
                                   tests=TESTS_BDOR, note=NOTE_BDOR,
                                   match_map=PARTNER))

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
