"""
Set-valued DOR levels: tau-sensitivity experiment (extension of 2026-08-06).

The DOR protocol places cards between SETS of mutually indifferent
alternatives, not only between single alternatives.  The generator now merges
alternatives whose TRUE utility gap is below merge_threshold (tau) into one
level (single-linkage); the likelihood uses level-CENTROID features
(PreferenceSampler.add_gap).  tau = 0 reduces exactly to the singleton model.

Design: folds sets_F1=10_F2=3_S=10_AR=3_K=3_Inc={inc}_Tau={tau} for
tau in {0, 0.02, 0.05, 0.10} x Inc in {0,15,35,50}.  For each Inc the global
RNG is seeded identically before every tau's generation; tau only affects the
merging step AFTER all random draws, so all tau variants share identical
tables, true utilities and noise -> perfectly PAIRED comparisons across tau.

Arms (joint-v1 engine, adjacent-only, theta_0 = 0): FTRL_DOR_SETJ1 (20 tables),
BAYES_DOR_SETJ1 (5 tables).  Outputs: sets_tau_comparison.csv +
figs/sets_tau/ figures + console summary.

Run:  C:\\Users\\marco\\anaconda3\\python.exe compare_sets_tau.py
"""
import os
import json
import glob
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from generate_datasets import DatasetGenerator
from experiments.runner import run_batch_experiments
from experiments.metrics import BenchmarkRunner
from common.utils import parse_subfold_string
import common.utils as utils

F1, F2, STEPS, AR, K = 10, 3, 10, 3, 3
TAUS = [0.0, 0.02, 0.05, 0.10]
INCS = [0, 15, 35, 50]
HM_0 = 20
ETA_MAX = 5
GEN_SEED = 20260807  # + inc: same tables/noise for every tau within an Inc
RUN_SEED = 30260807
ARMS = [('FTRL_DOR_SETJ1', True), ('BAYES_DOR_SETJ1', False)]
METRICS = ['asrs', 'asps', 'aios']
OUT_DIR = os.path.join('figs', 'sets_tau')
CSV_PATH = 'sets_tau_comparison.csv'


def fold_name(inc, tau):
    return f"sets_F1={F1}_F2={F2}_S={STEPS}_AR={AR}_K={K}_Inc={inc}_Tau={tau:g}"


def generate_all():
    for inc in INCS:
        for tau in TAUS:
            fold = fold_name(inc, tau)
            json_path = os.path.join('datasets', fold, f"f1_{F1}__f2_{F2}__nsteps_{STEPS}.json")
            if os.path.exists(json_path):
                print(f"{fold}: dataset exists, skipping generation")
                continue
            # Identical RNG stream for every tau within this Inc (tau acts only
            # after the draws), so tau variants are paired run-by-run.
            np.random.seed(GEN_SEED + inc)
            gen = DatasetGenerator(output_dir=os.path.join('datasets', fold), n_runs=HM_0)
            gen.generate_batch(
                F1=[F1], F2=[F2], num_steps=STEPS, k_subset=(AR, K), n_max=ETA_MAX,
                target_consistency=1.0 - inc / 100.0, plot_utilities=False,
                merge_threshold=tau
            )


def level_stats(inc, tau):
    """Mean #levels per subrank and share of multi-member levels."""
    json_path = os.path.join('datasets', fold_name(inc, tau), f"f1_{F1}__f2_{F2}__nsteps_{STEPS}.json")
    data = json.load(open(json_path))
    n_levels, n_multi, n_total = [], 0, 0
    for run in data['runs']:
        for sr in run['subranks']:
            lv = sr.get('levels', [[i] for i in sr['items']])
            n_levels.append(len(lv))
            n_multi += sum(1 for l in lv if len(l) > 1)
            n_total += len(lv)
    return float(np.mean(n_levels)), n_multi / max(n_total, 1)


def run_all():
    for inc in INCS:
        for tau_idx, tau in enumerate(TAUS):
            fold = fold_name(inc, tau)
            for arm_idx, (sub_fold, all_tables) in enumerate(ARMS):
                alg_name, _ = parse_subfold_string(sub_fold)
                hm = HM_0 if all_tables else int(max(1, HM_0 / 4))
                seed_base = RUN_SEED + inc * 1000 + tau_idx * 100 + arm_idx * 10
                t0 = time.time()
                run_batch_experiments(
                    F1=[F1], F2=[F2], sub_fold=sub_fold, dataset_folds=[fold],
                    alg=alg_name, overwrite=False, hm=hm, k_subset=(AR, K),
                    num_steps=STEPS, eta_max=ETA_MAX, expansion='adjacent',
                    seed_base=seed_base, log_loss=True
                )
                runner = BenchmarkRunner(
                    dataset_fold=fold, sub_fold=sub_fold, num_subint=3, hm=hm,
                    F1=[F1], F2=[F2], NUM_STEPS_LIST=[STEPS], eta_max=ETA_MAX
                )
                runner.compute_metrics('poi', force=True)
                runner.compute_metrics('rai', force=True)
                runner.compute_asrs(force=True)
                runner.compute_asps(force=True)
                runner.compute_aios(force=True)
                print(f"{fold} / {sub_fold}: done in {time.time() - t0:.1f}s")


def collect():
    rows = []
    for inc in INCS:
        for tau in TAUS:
            fold = fold_name(inc, tau)
            s_mean, multi_share = level_stats(inc, tau)
            for sub_fold, all_tables in ARMS:
                hm = HM_0 if all_tables else int(max(1, HM_0 / 4))
                for metric in METRICS:
                    data = utils.load_test_results(metric, fold, sub_fold, F1, F2, STEPS)
                    data = np.array(data)  # steps x tables
                    for j in range(data.shape[0]):
                        for i in range(data.shape[1]):
                            rows.append((inc, tau, sub_fold, metric, j + 1, i,
                                         data[j, i], s_mean, multi_share))
    import csv
    with open(CSV_PATH, 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['Inc', 'Tau', 'Method', 'Metric', 'Step', 'Table', 'Value',
                     'MeanLevelsPerSubrank', 'MultiLevelShare'])
        wr.writerows(rows)
    print(f"Wrote {CSV_PATH} ({len(rows)} rows)")
    return rows


def plot(rows):
    os.makedirs(OUT_DIR, exist_ok=True)
    arr = np.array(rows, dtype=object)
    style = {'FTRL_DOR_SETJ1': dict(color='#1f4fd8', marker='o'),
             'BAYES_DOR_SETJ1': dict(color='#e8810c', marker='s')}
    label = {'FTRL_DOR_SETJ1': 'FTRL-DOR', 'BAYES_DOR_SETJ1': 'BAYES-DOR'}
    for metric in METRICS:
        fig, axes = plt.subplots(1, len(INCS), figsize=(4 * len(INCS), 3.4), sharey=True)
        for ax, inc in zip(axes, INCS):
            for sub_fold, _ in ARMS:
                means, ses = [], []
                for tau in TAUS:
                    m = ((arr[:, 0] == inc) & (arr[:, 1] == tau) & (arr[:, 2] == sub_fold)
                         & (arr[:, 3] == metric) & (arr[:, 4] == STEPS))
                    vals = arr[m, 6].astype(float)
                    means.append(vals.mean())
                    ses.append(vals.std() / np.sqrt(len(vals)))
                means = np.array(means); ses = np.array(ses)
                st = style[sub_fold]
                ax.errorbar(TAUS, means, yerr=1.96 * ses, label=label[sub_fold],
                            capsize=3, **st)
            ax.set_title(f"Inc = {inc}%")
            ax.set_xlabel(r"merge threshold $\tau$")
            ax.grid(alpha=0.3)
            ax.set_xticks(TAUS)
        axes[0].set_ylabel(f"{metric.upper()} (final step)")
        axes[0].legend(fontsize=8)
        fig.suptitle(f"Set-valued DOR levels: {metric.upper()} vs merge threshold "
                     f"(F1={F1}, F2={F2}, AR=K={K}, T={STEPS}; paired datasets per Inc)")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f"tau_{metric}.png"), dpi=150)
        plt.close(fig)
        print(f"Wrote {OUT_DIR}/tau_{metric}.png")


def summary(rows):
    arr = np.array(rows, dtype=object)
    print("\n===== FINAL-STEP MEANS (T=10) =====")
    print(f"{'Inc':>4} {'tau':>5} {'s_mean':>7} {'multi%':>7}", end="")
    for sub_fold, _ in ARMS:
        for metric in METRICS:
            print(f" {label_short(sub_fold)}_{metric}", end="")
    print()
    for inc in INCS:
        for tau in TAUS:
            m0 = (arr[:, 0] == inc) & (arr[:, 1] == tau) & (arr[:, 4] == STEPS)
            if not m0.any():
                continue
            s_mean = arr[m0, 7][0]; multi = arr[m0, 8][0]
            print(f"{inc:>4} {tau:>5g} {s_mean:>7.2f} {100*multi:>6.1f}%", end="")
            for sub_fold, _ in ARMS:
                for metric in METRICS:
                    m = m0 & (arr[:, 2] == sub_fold) & (arr[:, 3] == metric)
                    print(f" {arr[m, 6].astype(float).mean():>10.3f}", end="")
            print()


def label_short(sub_fold):
    return 'F' if sub_fold.startswith('FTRL') else 'B'


if __name__ == '__main__':
    generate_all()
    run_all()
    rows = collect()
    plot(rows)
    summary(rows)
