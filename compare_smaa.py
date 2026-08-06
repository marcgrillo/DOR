"""
Compare BAYES-DOR / FTRL-DOR with the SMAA-DOR baseline (arXiv:2405.04313).

Loads the per-step samples of each method from samples/<config>/, computes
RAI/POI-based metrics against the ground truth on the tables shared by all
methods, and writes a comparison figure + CSV.

Run:  python compare_smaa.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

from common.utils import read_dataset
from mcda.models import PiecewiseLinearTransformer

F1, F2, NUM_STEPS = 10, 3, 10
INC_LEVELS = [0, 15, 35, 50]
HM = 5  # tables shared by all methods (BAYES was run on the first 5)
NUM_SUBINT = 3

METHODS = {  # sub_fold -> (label, color)  [fixed categorical order]
    "BAYES_DOR_PASSIVE": ("Bayes-DOR", "#2a78d6"),
    "FTRL_DOR_PASSIVE": ("FTRL-DOR", "#eb6834"),
    "SMAA_DOR_PASSIVE": ("SMAA-DOR", "#1baf7a"),
}

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"


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
        print(f"Inc={inc} done")

    import csv
    os.makedirs("figs", exist_ok=True)
    with open("smaa_comparison.csv", "w", newline="") as f:
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
            for sub_fold, (label, color) in METHODS.items():
                per_step = data.get((inc, sub_fold, metric), {})
                mean = np.array([np.mean(per_step.get(j, [np.nan])) for j in steps])
                se = np.array([
                    np.std(per_step.get(j, [np.nan]), ddof=1) / np.sqrt(len(per_step.get(j, [1])))
                    if len(per_step.get(j, [])) > 1 else 0.0 for j in steps])
                ax.plot(steps, mean, color=color, lw=2, marker="o", ms=4, label=label)
                ax.fill_between(steps, mean - se, mean + se, color=color, alpha=0.15, lw=0)
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
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, 0.99))
    fig.suptitle(f"Bayes-DOR vs FTRL-DOR vs SMAA-DOR  "
                 f"(F1={F1}, F2={F2}, AR=3, K=3, {HM} tables, mean $\\pm$ SE)",
                 fontsize=11, color=INK, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join("figs", "smaa_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"saved {out} and smaa_comparison.csv")

    # ---- console summary at final step ----
    print(f"\nFinal step ({NUM_STEPS}) means:")
    header = f"{'metric':6s} {'Inc':>4s} " + " ".join(f"{METHODS[m][0]:>10s}" for m in METHODS)
    print(header)
    for metric in metric_order:
        for inc in INC_LEVELS:
            vals = [np.mean(data[(inc, m, metric)][NUM_STEPS]) for m in METHODS]
            print(f"{metric:6s} {inc:4d} " + " ".join(f"{v:10.3f}" for v in vals))


if __name__ == "__main__":
    main()
