"""
Compare full pairwise expansion vs adjacent-only expansion of subranks
for BAYES-DOR and FTRL-DOR.

Full ('all'):      A > B > C  ->  (A,B), (A,C), (B,C), eta = summed cards.
Adjacent ('adjacent'): A > B > C  ->  (A,B), (B,C), eta = cards between them.

Runs the ADJACENT variants (resumable: overwrite=False) on the same grid
configs used for the SMAA comparison, then compares against the existing
*_PASSIVE samples on the shared tables. Writes figs/adjacent_comparison.png
and adjacent_comparison.csv.

Run:  python compare_adjacent.py   (from inside DOR/)
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

from common.utils import read_dataset, parse_subfold_string
from experiments.runner import run_batch_experiments
from mcda.models import PiecewiseLinearTransformer

F1, F2, NUM_STEPS = 10, 3, 10
INC_LEVELS = [0, 15, 35, 50]
HM = 5  # tables shared by all methods (BAYES was run on the first 5)
NUM_SUBINT = 3
ETA_MAX = 5

# sub_fold -> (label, color, linestyle); color = method, linestyle = expansion
METHODS = {
    "BAYES_DOR_PASSIVE": ("Bayes-DOR (all pairs)", "#2a78d6", "-"),
    "BAYES_DOR_ADJACENT": ("Bayes-DOR (adjacent)", "#2a78d6", "--"),
    "FTRL_DOR_PASSIVE": ("FTRL-DOR (all pairs)", "#eb6834", "-"),
    "FTRL_DOR_ADJACENT": ("FTRL-DOR (adjacent)", "#eb6834", "--"),
}
RUN_METHODS = ["BAYES_DOR_ADJACENT", "FTRL_DOR_ADJACENT"]

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"


def run_adjacent_variants():
    for inc in INC_LEVELS:
        fold = f"grid_F1={F1}_F2={F2}_S={NUM_STEPS}_AR=3_K=3_Inc={inc}"
        for sub_fold in RUN_METHODS:
            alg_name, _ = parse_subfold_string(sub_fold)
            print(f"\n>>> {sub_fold} on {fold}")
            run_batch_experiments(
                F1=[F1], F2=[F2], sub_fold=sub_fold, dataset_folds=[fold],
                alg=alg_name, overwrite=False, hm=HM, k_subset=(3, 3),
                num_steps=NUM_STEPS, eta_max=ETA_MAX, expansion='adjacent'
            )


def calc_poi(t, sam):
    sam = np.atleast_2d(sam)
    vals = t @ sam[:, : t.shape[1]].T
    count = np.zeros((t.shape[0], t.shape[0]))
    for col in range(vals.shape[1]):
        s = vals[:, col]
        count += s[:, None] >= s[None, :]
    return count / vals.shape[1]


def calc_rai(t, sam):
    sam = np.atleast_2d(sam)
    vals = t @ sam[:, : t.shape[1]].T
    count = np.zeros((t.shape[0], t.shape[0]))
    for col in range(vals.shape[1]):
        ranks = np.argsort(np.argsort(-vals[:, col]))
        count[np.arange(t.shape[0]), ranks] += 1
    return count / vals.shape[1]


def step_metrics(t, sam, ranking):
    f1 = t.shape[0]
    rai = calc_rai(t, sam)
    poi = calc_poi(t, sam)
    asrs = np.mean([rai[np.argwhere(ranking == k)[0, 0], k] for k in range(f1)])
    aios = rai[np.argwhere(ranking == 0)[0, 0], 0]
    pref_mask = ranking[:, None] > ranking[None, :]
    asps = np.sum(poi.T * pref_mask) * 2 / (f1 * (f1 - 1))
    mean_u = (t @ np.atleast_2d(sam)[:, : t.shape[1]].T).mean(axis=1)
    tau = kendalltau(-mean_u, ranking).statistic
    return {"asrs": asrs, "aios": aios, "asps": asps, "tau": tau}


def main():
    run_adjacent_variants()

    rows = []  # inc, method, step, table, metric, value
    for inc in INC_LEVELS:
        fold = f"grid_F1={F1}_F2={F2}_S={NUM_STEPS}_AR=3_K=3_Inc={inc}"
        tables, rankings, _, _ = read_dataset(os.path.join("datasets", fold), F1, F2, NUM_STEPS)
        cfg_dir = os.path.join("samples", fold, f"f1_{F1}_f2_{F2}_nsteps_{NUM_STEPS}")
        for i in range(HM):
            transformer = PiecewiseLinearTransformer.from_equal_intervals(tables[i], NUM_SUBINT)
            t_vec = transformer.transform(tables[i])
            ranking = np.array(rankings[i])
            for sub_fold in METHODS:
                for j in range(1, NUM_STEPS + 1):
                    path = os.path.join(cfg_dir, sub_fold, f"table_{i}", f"{j}.npy")
                    if not os.path.exists(path):
                        print(f"missing: {path}")
                        continue
                    sam = np.load(path)
                    for metric, val in step_metrics(t_vec, sam, ranking).items():
                        rows.append((inc, sub_fold, j, i, metric, val))
        print(f"Inc={inc} metrics done")

    os.makedirs("figs", exist_ok=True)
    with open("adjacent_comparison.csv", "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["inc", "method", "step", "table", "metric", "value"])
        wcsv.writerows(rows)

    # ---- aggregate: mean +- se over tables ----
    data = {}
    for inc, m, j, i, metric, v in rows:
        data.setdefault((inc, m, metric), {}).setdefault(j, []).append(v)

    metric_names = {
        "tau": "Kendall $\\tau$ (barycenter ranking)",
        "asrs": "ASRS (RAI at true ranks)",
        "asps": "ASPS (pairwise success)",
        "aios": "AIOS (RAI of true best at rank 1)",
    }
    metric_order = ["tau", "asrs", "asps", "aios"]

    fig, axes = plt.subplots(len(metric_order), len(INC_LEVELS),
                             figsize=(13, 11), sharex=True, sharey="row")
    fig.patch.set_facecolor(SURFACE)
    steps = np.arange(1, NUM_STEPS + 1)

    for r, metric in enumerate(metric_order):
        for c, inc in enumerate(INC_LEVELS):
            ax = axes[r, c]
            ax.set_facecolor(SURFACE)
            for sub_fold, (label, color, ls) in METHODS.items():
                per_step = data.get((inc, sub_fold, metric), {})
                mean = np.array([np.mean(per_step.get(j, [np.nan])) for j in steps])
                se = np.array([
                    np.std(per_step.get(j, [np.nan]), ddof=1) / np.sqrt(len(per_step.get(j, [1])))
                    if len(per_step.get(j, [])) > 1 else 0.0 for j in steps])
                ax.plot(steps, mean, color=color, lw=2, ls=ls, marker="o", ms=4, label=label)
                ax.fill_between(steps, mean - se, mean + se, color=color, alpha=0.12, lw=0)
            ax.grid(True, color=GRID, lw=0.8)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            for spine in ["left", "bottom"]:
                ax.spines[spine].set_color(GRID)
            ax.tick_params(colors=MUTED, labelsize=8)
            if r == 0:
                ax.set_title(f"Inconsistency {inc}%", fontsize=10, color=INK2)
            if c == 0:
                ax.set_ylabel(metric_names[metric], fontsize=9, color=INK2)
            if r == len(metric_order) - 1:
                ax.set_xlabel("Elicitation step", fontsize=9, color=INK2)
                ax.set_xticks(steps)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 0.99))
    fig.suptitle(f"Full pairwise vs adjacent-only likelihood  "
                 f"(F1={F1}, F2={F2}, AR=3, K=3, {HM} tables, mean $\\pm$ SE)",
                 fontsize=11, color=INK, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join("figs", "adjacent_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"saved {out} and adjacent_comparison.csv")

    # ---- console summary at final step ----
    print(f"\nFinal step ({NUM_STEPS}) means:")
    header = f"{'metric':6s} {'Inc':>4s} " + " ".join(f"{METHODS[m][0]:>22s}" for m in METHODS)
    print(header)
    for metric in metric_order:
        for inc in INC_LEVELS:
            vals = []
            for m in METHODS:
                per_step = data.get((inc, m, metric), {})
                vals.append(np.mean(per_step.get(NUM_STEPS, [np.nan])))
            print(f"{metric:6s} {inc:4d} " + " ".join(f"{v:22.3f}" for v in vals))


if __name__ == "__main__":
    main()
