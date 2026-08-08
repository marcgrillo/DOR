"""
LaTeX tables for the joint-v1 campaign (P7/P8 manuscript material).

Reads grid_results_joint_v1.npz and writes to paper_figs/:
  1. tab_simulation_result_joint_v1.tex  - drop-in replacement for the paper's
     tab:simulation_result (marginal means +- 95% CI subscripts by factor
     level; FTRL/BAYES ADJACENT arms = the primary model).
  2. tab_ablation_supplement_joint_v1.tex - supplement: adjacent (proper
     likelihood) vs all-pairs (composite) arms, overall and by inconsistency.
  3. tab_baselines_joint_v1.tex - baselines/ablations (SMAA-DOR, DOR-LP,
     direction-only) vs the primary arms; rows skipped for arms not yet in
     the npz.

Pooling convention (matches the paper's table): AR == K configs only,
T in {1,3,5,10}, all tables pooled.

Run:  C:\\Users\\marco\\anaconda3\\python.exe make_paper_tables_joint_v1.py
"""
import os
import re
import numpy as np

NPZ = "grid_results_joint_v1.npz"
OUT_DIR = "paper_figs"
METRICS = ['asrs', 'asps', 'aios']
T_LEVELS = [1, 3, 5, 10]


def parse_config(c):
    m = re.match(r"F1=(\d+)_F2=(\d+)_S=(\d+)_AR=(\d+)_K=(\d+)_Inc=(\d+)", c)
    return {k: int(v) for k, v in zip(['F1', 'F2', 'S', 'AR', 'K', 'Inc'], m.groups())}


def pool(results, arm, metric, T=None, **filters):
    """All pooled values for an arm/metric under AR==K and the given filters."""
    vals = []
    for c, arms in results.items():
        p = parse_config(c)
        if p['AR'] != p['K'] or arm not in arms:
            continue
        if any(p[k] != v for k, v in filters.items()):
            continue
        data = np.array(arms[arm][metric])
        for t in ([T] if T else T_LEVELS):
            vals.extend(data[t - 1])
    return np.array(vals)


def cell(v):
    if v.size == 0:
        return "--"
    return "$%.3f{\\scriptstyle\\pm%.3f}$" % (v.mean(), 1.96 * v.std() / np.sqrt(len(v)))


def factor_rows(results, arms):
    """(label, row_head, filters, T) tuples in the paper's table order."""
    rows = [("all", None, {}, None)]
    for label, key, levels in [
        ("N. of alternatives - $\\left|A\\right|$", 'F1', [10, 20, 35, 50]),
        ("N. criteria - $m$", 'F2', [3, 5, 7, 9]),
        ("N. of steps -  $T$", 'T', T_LEVELS),
        ("Subset size - $k$", 'K', [3, 4, 5]),
        ("\\% of inconsistency", 'Inc', [0, 15, 35, 50]),
    ]:
        for i, lv in enumerate(levels):
            head = label if i == 0 else None
            if key == 'T':
                rows.append((str(lv), (head, len(levels)), {}, lv))
            else:
                rows.append((str(lv), (head, len(levels)), {key: lv}, None))
    return rows


def write_marginal_table(results, arms, col_names, caption, label, path):
    ncols = len(arms) * len(METRICS)
    lines = [
        "\\begin{table}[htp]",
        "    \\centering",
        f"    \\caption{{{caption}}}",
        f"    \\label{{{label}}}",
        "\\begin{tabularx}{\\textwidth}{l|c|" + "C" * ncols + "}",
        "\\toprule",
        "\\multirow{2}{*}{Configuration}&&" + "&".join(
            f"\\multicolumn{{{len(arms)}}}{{c}}{{{m}}}" for m in ['ASR', 'ASP', 'AIO']) + "\\\\",
    ]
    cml = []
    for i in range(len(METRICS)):
        a = 3 + i * len(arms)
        cml.append(f"\\cmidrule(lr){{{a}-{a + len(arms) - 1}}}")
    lines.append("".join(cml))
    lines.append("&&" + "&".join(col_names * len(METRICS)) + "\\\\")
    lines.append("\\midrule")

    first_group_done = False
    for name, head, filters, T in factor_rows(results, arms):
        cells = [cell(pool(results, a, m, T=T, **filters))
                 for m in METRICS for a in arms]
        if name == "all":
            lines.append("\\multicolumn{2}{l|}{all} & " + " & ".join(cells) + " \\\\")
            continue
        head_label, span = head
        if head_label is not None:
            lines.append("\\midrule")
            lines.append(f"\\multirow{{{span}}}{{*}}{{{head_label}}}&{name} & "
                         + " & ".join(cells) + " \\\\")
        else:
            lines.append(f"&{name} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabularx}", "\\end{table}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {path}")


def write_overall_by_inc_table(results, arms, col_names, caption, label, path):
    """Compact comparison, transposed: one metric block of rows (all + by-Inc)
    per metric, methods as columns -- fits \\textwidth even with 6 methods."""
    present = [a for a in arms if any(a in v for v in results.values())]
    if len(present) < len(arms):
        missing = [a for a in arms if a not in present]
        print(f"NOTE: arms not yet in npz, skipped: {missing}")
    names = [n for a, n in zip(arms, col_names) if a in present]
    if not present:
        print(f"SKIP {path}: no requested arms present")
        return
    lines = [
        "\\begin{table}[htp]",
        "    \\centering",
        f"    \\caption{{{caption}}}",
        f"    \\label{{{label}}}",
        "\\begin{tabularx}{\\textwidth}{ll|" + "C" * len(present) + "}",
        "\\toprule",
        "&$|F_{\\text{inc}}|$&" + "&".join(names) + "\\\\",
        "\\midrule",
    ]
    for mi, (metric, mname) in enumerate(zip(METRICS, ['ASR', 'ASP', 'AIO'])):
        if mi:
            lines.append("\\midrule")
        lines.append(f"\\multirow{{5}}{{*}}{{{mname}}}&all & " + " & ".join(
            cell(pool(results, a, metric)) for a in present) + " \\\\")
        for inc in [0, 15, 35, 50]:
            lines.append(f"&${inc}\\%$ & " + " & ".join(
                cell(pool(results, a, metric, Inc=inc)) for a in present) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabularx}", "\\end{table}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = np.load(NPZ, allow_pickle=True)['results'].item()
    print(f"Configs: {len(results)}")

    write_marginal_table(
        results,
        arms=['FTRL_DOR_ADJJ1', 'BAYES_DOR_ADJJ1'],
        col_names=['FTRL', 'BAYES'],
        caption=("Simulation results: marginal mean of each metric over the grid, broken down "
                 "by factor level, for FTRL-DOR and BAYES-DOR under the adjacent-gap likelihood. "
                 "Each entry reports the mean with the half-width of its $95\\%$ confidence "
                 "interval as a subscript (\\,$\\text{mean}{\\scriptstyle\\pm\\text{c.i.}}$\\,). "
                 "The top row (\\emph{all}) averages over the whole grid."),
        label="tab:simulation_result",
        path=os.path.join(OUT_DIR, "tab_simulation_result_joint_v1.tex"),
    )

    write_overall_by_inc_table(
        results,
        arms=['FTRL_DOR_ADJJ1', 'FTRL_DOR_ALLJ1', 'BAYES_DOR_ADJJ1', 'BAYES_DOR_ALLJ1'],
        col_names=['FTRL', 'FTRL$_{\\text{ap}}$', 'BAYES', 'BAYES$_{\\text{ap}}$'],
        caption=("Ablation: the adjacent-gap likelihood (primary model) against the all-pairs "
                 "composite construction (subscript ap) on identical data. The composite "
                 "re-uses each elicited card count up to $k-1$ times, which buys a small "
                 "amount of point accuracy at the cost of an over-concentrated posterior."),
        label="tab:ablation_composite",
        path=os.path.join(OUT_DIR, "tab_ablation_supplement_joint_v1.tex"),
    )

    write_overall_by_inc_table(
        results,
        arms=['FTRL_DOR_ADJJ1', 'BAYES_DOR_ADJJ1', 'SMAA_DOR_ADJJ1',
              'DORLP_DOR_ADJJ1', 'FTRLDIR_DOR_ADJJ1', 'BAYESDIR_DOR_ADJJ1'],
        col_names=['FTRL', 'BAYES', 'SMAA', 'LP', 'FTRL$_{\\text{dir}}$', 'BAYES$_{\\text{dir}}$'],
        caption=("Comparison of the proposed estimators (FTRL-DOR, BAYES-DOR) with SMAA-DOR "
                 "\\citep{corrente2025deck}, the original DOR linear programme "
                 "\\citep[LP,][]{barbati2024deck} and the direction-only ablation of "
                 "Section~\\ref{sec:sim_direction} (subscript dir: card counts ignored), on the "
                 "same simulated data. Mean of each metric with $95\\%$ confidence interval, "
                 "over the whole grid (\\emph{all}) and by inconsistency level."),
        label="tab:baselines",
        path=os.path.join(OUT_DIR, "tab_baselines_joint_v1.tex"),
    )


if __name__ == "__main__":
    main()
