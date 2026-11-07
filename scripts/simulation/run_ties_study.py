"""Ties study: alternatives closer than tau in true value share a level of the declared order.

For merge threshold tau in {0, 0.02, 0.05, 0.10} x Finc in {0, 15, 35, 50}% on |A| = 10, m = 3,
k = 3, T = 10, e_max = 5 (20 replications per fold, the same data for every tau of a Finc),
runs FTRL_DOR_SETJ1 on all 20 replications and BAYES_DOR_SETJ1 on the first 5, and collects
ASR / ASP / AIO per session; scripts/paper/make_figures.py draws Figure 4 from the CSV.
Resumable: existing datasets and per-session sample files are reused; the metric files are
recomputed on every run (force=True). No command-line options: the design is set below.

Paper:   Figure 4 and the Sec. 5.6 numbers (both computed from the CSV by
         scripts/paper/make_figures.py)
Inputs:  none (generates data/datasets/sets_F1=10_F2=3_S=10_AR=3_K=3_Inc=<Finc>_Tau=<tau>/)
Outputs: results/sets_tau_comparison.csv, outputs/diagnostics/sets_tau/tau_<metric>.png
         (metric = asrs, asps, aios); stdout: final-session means per fold;
         intermediate states and metrics under data/samples/ and data/metrics/
Runtime: about 18 min from scratch, one process on a desktop CPU (per fold about 15 s for
         FTRL-DOR and 50 s for BAYES-DOR); about a minute when all sample files exist.

Usage:
    python scripts/simulation/run_ties_study.py

(formerly compare_sets_tau.py)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, so `import bdor` works without installing
import os
import csv
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from bdor.datagen import DatasetGenerator
from bdor.runner import run_batch_experiments
from bdor.metrics import BenchmarkRunner
from bdor.utils import parse_subfold_string
import bdor.utils as utils
from bdor.paths import DATASETS_DIR, RESULTS_DIR, DIAGNOSTICS_DIR

F1, F2, STEPS, AR, K = 10, 3, 10, 3, 3  # |A|, m, sessions T, subset size of session 1 / later
TAUS = [0.0, 0.02, 0.05, 0.10]          # merge thresholds on the true-utility gap
INCS = [0, 15, 35, 50]                  # nominal inconsistency Finc (%)
HM_0 = 20                               # replications per fold (= n_runs of each dataset)
ETA_MAX = 5
GEN_SEED = 20260807  # + Finc: same tables/noise for every tau within a Finc
RUN_SEED = 30260807  # base of the inference seeds (formula in run_all)
# (arm, uses all HM_0 replications); False -> only the first HM_0 / 4 = 5
ARMS = [('FTRL_DOR_SETJ1', True), ('BAYES_DOR_SETJ1', False)]
METRICS = ['asrs', 'asps', 'aios']      # ASR, ASP, AIO (paper eqs. (23)-(25))
OUT_DIR = os.path.join(DIAGNOSTICS_DIR, 'sets_tau')
CSV_PATH = os.path.join(RESULTS_DIR, 'sets_tau_comparison.csv')


def fold_name(inc, tau):
    """Dataset folder name of one fold, e.g. ``sets_F1=10_F2=3_S=10_AR=3_K=3_Inc=35_Tau=0.05``."""
    return f"sets_F1={F1}_F2={F2}_S={STEPS}_AR={AR}_K={K}_Inc={inc}_Tau={tau:g}"


def generate_all():
    """Generate the 16 datasets of the study, skipping those that already exist.

    Before each tau the global NumPy RNG is seeded with ``GEN_SEED + Finc``. Merging consumes
    no random numbers, so the four tau folds of one Finc share tables, true utilities,
    presented subsets and noise draws, and differ only in which alternatives share a level
    (hence in the declared levels and cards).
    In bdor.datagen, consecutive alternatives of the true-utility order whose true-utility gap
    is below tau are merged into one level (single linkage); the DM orders the levels by the
    mean of their members' perceived utilities (eq. (21)) and places between consecutive
    levels the rounded difference of these means, capped at e_max. tau = 0 gives singleton
    levels, as in the main grid. The likelihood then uses level-centroid features (eq. (8)).
    """
    for inc in INCS:
        for tau in TAUS:
            fold = fold_name(inc, tau)
            json_path = os.path.join(DATASETS_DIR, fold, f"f1_{F1}__f2_{F2}__nsteps_{STEPS}.json")
            if os.path.exists(json_path):
                print(f"{fold}: dataset exists, skipping generation")
                continue
            # Same seed for every tau of this Finc: paired folds (see docstring).
            np.random.seed(GEN_SEED + inc)
            gen = DatasetGenerator(output_dir=os.path.join(DATASETS_DIR, fold), n_runs=HM_0)
            gen.generate_batch(
                F1=[F1], F2=[F2], num_steps=STEPS, k_subset=(AR, K), n_max=ETA_MAX,
                target_consistency=1.0 - inc / 100.0, plot_utilities=False,
                merge_threshold=tau
            )


def level_stats(inc, tau):
    """Level statistics of one fold's dataset, over all replications and sessions.

    Returns
    -------
    mean_levels : float
        Mean number of levels per session (k = 3 without ties).
    multi_share : float
        Share of multi-member levels among all levels. This is a share of levels, not of
        sessions: with k = 3 a session with a tie has exactly one such level, so the share of
        sessions with a tie is ``multi_share * mean_levels``.
    """
    json_path = os.path.join(DATASETS_DIR, fold_name(inc, tau),
                             f"f1_{F1}__f2_{F2}__nsteps_{STEPS}.json")
    with open(json_path) as f:
        data = json.load(f)
    n_levels, n_multi, n_total = [], 0, 0
    for run in data['runs']:
        for sr in run['subranks']:
            lv = sr.get('levels', [[i] for i in sr['items']])
            n_levels.append(len(lv))
            n_multi += sum(1 for l in lv if len(l) > 1)
            n_total += len(lv)
    return float(np.mean(n_levels)), n_multi / max(n_total, 1)


def run_all():
    """Run both arms on the 16 folds and recompute their metric files.

    FTRL_DOR_SETJ1 runs on all HM_0 = 20 replications, BAYES_DOR_SETJ1 on the first 5; both
    with the settings of the paper's grid arms (adjacent-gap observations, eta_max = 5,
    prequential log-loss recorded) and
    ``seed_base = RUN_SEED + 1000 * Finc + 100 * (tau index) + 10 * (arm index)``. Existing
    per-session states are reused; PWI ('poi' in the code), RAI and ASR / ASP / AIO are then
    recomputed with force=True.
    """
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
    """Gather the per-session metrics of every fold and arm and write ``CSV_PATH``.

    Returns
    -------
    list of tuple
        One row ``(Finc, tau, arm, metric, session, replication, value, mean_levels,
        multi_share)`` per metric value, with session counted from 1 and replication from 0;
        the last two entries repeat the fold's level_stats(). The CSV holds the same rows
        under the header
        ``Inc,Tau,Method,Metric,Step,Table,Value,MeanLevelsPerSubrank,MultiLevelShare``.
    """
    rows = []
    for inc in INCS:
        for tau in TAUS:
            fold = fold_name(inc, tau)
            s_mean, multi_share = level_stats(inc, tau)
            for sub_fold, _ in ARMS:
                for metric in METRICS:
                    data = utils.load_test_results(metric, fold, sub_fold, F1, F2, STEPS)
                    data = np.array(data)  # steps x tables
                    for j in range(data.shape[0]):
                        for i in range(data.shape[1]):
                            rows.append((inc, tau, sub_fold, metric, j + 1, i,
                                         data[j, i], s_mean, multi_share))
    with open(CSV_PATH, 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['Inc', 'Tau', 'Method', 'Metric', 'Step', 'Table', 'Value',
                     'MeanLevelsPerSubrank', 'MultiLevelShare'])
        wr.writerows(rows)
    print(f"Wrote {CSV_PATH} ({len(rows)} rows)")
    return rows


def plot(rows):
    """Plot each metric at the last session against tau, one panel per Finc.

    Points are means over replications with 1.96 x SE error bars (SE from the ddof=0 standard
    deviation); ``rows`` as returned by collect(). Writes
    outputs/diagnostics/sets_tau/tau_<metric>.png. Figure 4 of the paper is a different plot,
    pooled over Finc and sessions, drawn by scripts/paper/make_figures.py.
    """
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
        out = os.path.join(OUT_DIR, f"tau_{metric}.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Wrote {out}")


def label_short(sub_fold):
    """One-letter method tag of the summary header: 'F' for FTRL arms, 'B' otherwise."""
    return 'F' if sub_fold.startswith('FTRL') else 'B'


def summary(rows):
    """Print, per fold, the level statistics and the last-session means of both arms.

    ``rows`` as returned by collect(); column ``s_mean`` is the mean number of levels per
    session and ``multi%`` the share of multi-member levels (see level_stats).
    """
    arr = np.array(rows, dtype=object)
    print(f"\n===== FINAL-STEP MEANS (T={STEPS}) =====")
    print(f"{'Inc':>4} {'tau':>5} {'s_mean':>7} {'multi%':>7}", end="")
    for sub_fold, _ in ARMS:
        for metric in METRICS:
            print(f" {label_short(sub_fold) + '_' + metric:>10}", end="")
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


if __name__ == '__main__':
    import argparse
    argparse.ArgumentParser(description=__doc__.split('\n\n')[0].strip(),
                            epilog='Takes no options: runs the whole (resumable) study.').parse_args()
    generate_all()
    run_all()
    rows = collect()
    plot(rows)
    summary(rows)
