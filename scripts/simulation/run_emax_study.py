"""
e_max sensitivity experiment (2026-08-13).

The grid campaign fixes the card-scale cap at e_max = 5 (ETA_MAX). Hypothesis:
a finer scale carries more strength-of-preference information, so performance
should improve. This script reruns ONE fixed grid configuration with
e_max = 50 under the SAME generating mechanism (utilities scaled by e_max,
cards = clip(round(noisy scaled diff), 0, e_max), sigma re-tuned to the SAME
target consistency given e_max) and compares FTRL-DOR / BAYES-DOR
(joint-v1 engine: theta_0 = 0 joint CLM, adjacent-only likelihood).

Design: folds emax_F1=10_F2=3_S=10_AR=3_K=3_Inc={inc}_Emax={emax} for
emax in {5, 50} x Inc in {0, 35}. For each Inc the global RNG is seeded
identically before each emax's generation; n_max only rescales the (identical)
draws, so both emax variants share identical tables, true utilities and noise
z-scores -> perfectly PAIRED comparisons across emax. NOTE the mechanism's
consistency definition: sigma is tuned so P(exact card) = 1 - Inc/100 GIVEN
e_max, so at Inc > 0 the e_max = 50 DM is also less noisy on the utility
scale (finer bins with the same per-bin hit rate) - that is part of the
mechanism, and the Inc = 0 folds (sigma = 0 in both) isolate the pure
quantization-granularity effect.

Arms: FTRL_DOR_EMXJ1 / BAYES_DOR_EMXJ1, 10 tables each, T=10 steps.
Outputs: emax_comparison.csv, figs/emax/emax_evolution.png, console summary
with final-step means and paired Wilcoxon (emax=50 vs 5 across tables).

Run:  C:\\Users\\marco\\anaconda3\\python.exe compare_emax.py   (from DOR/)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, so `import bdor` works without installing
import os
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

from bdor.datagen import DatasetGenerator
from bdor.runner import run_batch_experiments
from bdor.metrics import BenchmarkRunner
from bdor.utils import parse_subfold_string
import bdor.utils as utils
from bdor.paths import DATASETS_DIR, RESULTS_DIR, DIAGNOSTICS_DIR

F1, F2, STEPS, AR, K = 10, 3, 10, 3, 3
EMAXES = [5, 50]
INCS = [0, 35, 50]
HM = 10
GEN_SEED = 20260813  # + inc: same tables/true utilities/noise z-scores for both emax
RUN_SEED = 40260813
ARMS = ['FTRL_DOR_EMXJ1', 'BAYES_DOR_EMXJ1']
METRICS = ['asrs', 'asps', 'aios']
OUT_DIR = os.path.join(DIAGNOSTICS_DIR, 'emax')
CSV_PATH = os.path.join(RESULTS_DIR, 'emax_comparison.csv')

# chart chrome + validated categorical slots 1/2 (same hexes as compare_adjacent)
BAYES_C, FTRL_C = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"


def fold_name(inc, emax):
    return f"emax_F1={F1}_F2={F2}_S={STEPS}_AR={AR}_K={K}_Inc={inc}_Emax={emax}"


def generate_all():
    for inc in INCS:
        for emax in EMAXES:
            fold = fold_name(inc, emax)
            json_path = os.path.join(DATASETS_DIR, fold, f"f1_{F1}__f2_{F2}__nsteps_{STEPS}.json")
            if os.path.exists(json_path):
                print(f"{fold}: dataset exists, skipping generation")
                continue
            # Identical RNG stream for both emax within this Inc (n_max only
            # rescales the draws), so emax variants are paired run-by-run.
            np.random.seed(GEN_SEED + inc)
            gen = DatasetGenerator(output_dir=os.path.join(DATASETS_DIR, fold), n_runs=HM)
            gen.generate_batch(
                F1=[F1], F2=[F2], num_steps=STEPS, k_subset=(AR, K), n_max=emax,
                target_consistency=1.0 - inc / 100.0, plot_utilities=False
            )


def card_stats(inc, emax):
    """Sigma and card-count distribution: how much strength info the DM emits."""
    json_path = os.path.join(DATASETS_DIR, fold_name(inc, emax),
                             f"f1_{F1}__f2_{F2}__nsteps_{STEPS}.json")
    data = json.load(open(json_path))
    cards = [c for run in data['runs'] for sr in run['subranks'] for c in sr['cards']]
    cards = np.array(cards)
    return {
        'sigma': data['metadata']['sigma'],
        'mean': cards.mean(),
        'zero_share': np.mean(cards == 0),
        'cap_share': np.mean(cards == emax),
        'distinct': len(np.unique(cards)),
    }


def run_all():
    for inc in INCS:
        for emax_idx, emax in enumerate(EMAXES):
            fold = fold_name(inc, emax)
            for arm_idx, sub_fold in enumerate(ARMS):
                alg_name, _ = parse_subfold_string(sub_fold)
                seed_base = RUN_SEED + inc * 1000 + emax_idx * 100 + arm_idx * 10
                t0 = time.time()
                run_batch_experiments(
                    F1=[F1], F2=[F2], sub_fold=sub_fold, dataset_folds=[fold],
                    alg=alg_name, overwrite=False, hm=HM, k_subset=(AR, K),
                    num_steps=STEPS, eta_max=emax, expansion='adjacent',
                    seed_base=seed_base, log_loss=True
                )
                runner = BenchmarkRunner(
                    dataset_fold=fold, sub_fold=sub_fold, num_subint=3, hm=HM,
                    F1=[F1], F2=[F2], NUM_STEPS_LIST=[STEPS], eta_max=emax
                )
                runner.compute_metrics('poi', force=True)
                runner.compute_metrics('rai', force=True)
                runner.compute_asrs(force=True)
                runner.compute_asps(force=True)
                runner.compute_aios(force=True)
                print(f"{fold} / {sub_fold}: done in {time.time() - t0:.1f}s", flush=True)


def collect():
    rows = []
    for inc in INCS:
        for emax in EMAXES:
            fold = fold_name(inc, emax)
            for sub_fold in ARMS:
                for metric in METRICS:
                    data = np.array(utils.load_test_results(metric, fold, sub_fold, F1, F2, STEPS))
                    for j in range(data.shape[0]):
                        for i in range(data.shape[1]):
                            rows.append((inc, emax, sub_fold, metric, j + 1, i, data[j, i]))
    import csv
    with open(CSV_PATH, 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['Inc', 'Emax', 'Method', 'Metric', 'Step', 'Table', 'Value'])
        wr.writerows(rows)
    print(f"Wrote {CSV_PATH} ({len(rows)} rows)")
    return rows


def series_vals(arr, inc, emax, sub_fold, metric, step):
    m = ((arr[:, 0] == inc) & (arr[:, 1] == emax) & (arr[:, 2] == sub_fold)
         & (arr[:, 3] == metric) & (arr[:, 4] == step))
    # table index in col 5 orders the paired vectors identically on both sides
    sub = arr[m]
    return sub[np.argsort(sub[:, 5].astype(int)), 6].astype(float)


def plot(rows):
    os.makedirs(OUT_DIR, exist_ok=True)
    arr = np.array(rows, dtype=object)
    metric_names = {'asrs': 'ASRS (RAI at true ranks)',
                    'asps': 'ASPS (pairwise success)',
                    'aios': 'AIOS (true best at rank 1)'}
    # color = method (identity), linestyle/marker = e_max (secondary encoding)
    series = [('FTRL_DOR_EMXJ1', 5, FTRL_C, '--', 'o', 'FTRL-DOR, $e_{max}=5$'),
              ('FTRL_DOR_EMXJ1', 50, FTRL_C, '-', 's', 'FTRL-DOR, $e_{max}=50$'),
              ('BAYES_DOR_EMXJ1', 5, BAYES_C, '--', 'o', 'BAYES-DOR, $e_{max}=5$'),
              ('BAYES_DOR_EMXJ1', 50, BAYES_C, '-', 's', 'BAYES-DOR, $e_{max}=50$')]
    steps = np.arange(1, STEPS + 1)

    fig, axes = plt.subplots(len(METRICS), len(INCS),
                             figsize=(5.2 * len(INCS), 3.1 * len(METRICS)),
                             sharex=True, sharey='row')
    fig.patch.set_facecolor(SURFACE)
    for r, metric in enumerate(METRICS):
        for c, inc in enumerate(INCS):
            ax = axes[r, c]
            ax.set_facecolor(SURFACE)
            for sub_fold, emax, color, ls, mk, label in series:
                mean = np.array([series_vals(arr, inc, emax, sub_fold, metric, j).mean()
                                 for j in steps])
                se = np.array([series_vals(arr, inc, emax, sub_fold, metric, j).std(ddof=1)
                               / np.sqrt(HM) for j in steps])
                ax.plot(steps, mean, color=color, lw=2, ls=ls, marker=mk, ms=4, label=label)
                ax.fill_between(steps, mean - se, mean + se, color=color, alpha=0.12, lw=0)
            ax.grid(True, color=GRID, lw=0.8)
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
            for spine in ['left', 'bottom']:
                ax.spines[spine].set_color(GRID)
            ax.tick_params(colors=MUTED, labelsize=8)
            if r == 0:
                ax.set_title(f"Inconsistency {inc}%", fontsize=10, color=INK2)
            if c == 0:
                ax.set_ylabel(metric_names[metric], fontsize=9, color=INK2)
            if r == len(METRICS) - 1:
                ax.set_xlabel("Elicitation step", fontsize=9, color=INK2)
                ax.set_xticks(steps)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"Card-scale cap $e_{{max}}$: 5 vs 50  "
                 f"(F1={F1}, F2={F2}, AR=K={K}, T={STEPS}, {HM} paired tables, mean $\\pm$ SE)",
                 fontsize=11, color=INK, y=1.045)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(OUT_DIR, 'emax_evolution.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {out}")


def summary(rows):
    arr = np.array(rows, dtype=object)
    print("\n===== DGP per fold (sigma is on the e_max-scaled utility axis) =====")
    print(f"{'Inc':>4} {'emax':>5} {'sigma':>8} {'sigma/emax':>10} {'mean_cards':>10} "
          f"{'zero%':>6} {'cap%':>6} {'distinct':>8}")
    for inc in INCS:
        for emax in EMAXES:
            s = card_stats(inc, emax)
            print(f"{inc:>4} {emax:>5} {s['sigma']:>8.3f} {s['sigma'] / emax:>10.4f} "
                  f"{s['mean']:>10.2f} {100 * s['zero_share']:>5.1f}% "
                  f"{100 * s['cap_share']:>5.1f}% {s['distinct']:>8d}")

    print(f"\n===== FINAL-STEP (T={STEPS}) PAIRED COMPARISON, {HM} tables =====")
    print(f"{'Inc':>4} {'method':>10} {'metric':>6} {'emax=5':>8} {'emax=50':>8} "
          f"{'delta':>8} {'wilcoxon_p':>10}")
    for inc in INCS:
        for sub_fold in ARMS:
            name = 'FTRL' if sub_fold.startswith('FTRL') else 'BAYES'
            for metric in METRICS:
                v5 = series_vals(arr, inc, 5, sub_fold, metric, STEPS)
                v50 = series_vals(arr, inc, 50, sub_fold, metric, STEPS)
                try:
                    p = wilcoxon(v50, v5).pvalue
                except ValueError:
                    p = float('nan')
                print(f"{inc:>4} {name:>10} {metric:>6} {v5.mean():>8.3f} {v50.mean():>8.3f} "
                      f"{v50.mean() - v5.mean():>+8.3f} {p:>10.4f}")


if __name__ == '__main__':
    generate_all()
    run_all()
    rows = collect()
    plot(rows)
    summary(rows)
