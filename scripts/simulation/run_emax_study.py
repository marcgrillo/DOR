"""e_max study: card-scale cap e_max = 5 versus 50 on one fixed configuration.

Runs FTRL-DOR and BAYES-DOR (arms FTRL_DOR_EMXJ1 / BAYES_DOR_EMXJ1, eta_max = e_max; NUTS with
200 warm-up + 200 draws) on |A| = 10, m = 3, k = 3, T = 10 for e_max in {5, 50} x Finc in
{0, 35, 50}%, 10 replications per fold, paired across e_max; sigma is re-tuned per e_max so
that P(exact card count) = 1 - Finc ('card-matched' noise, see generate_all). Resumable:
existing datasets and per-session sample files are reused; the metric files are recomputed on
every run (force=True). run_emax_flipmatch.py builds on these folds. No command-line options.

Paper:   Sec. 5.4, consistent-DM numbers (the Finc = 0 rows, paired Wilcoxon at T = 10)
Inputs:  none (generates data/datasets/emax_F1=10_F2=3_S=10_AR=3_K=3_Inc=<Finc>_Emax=<e_max>/)
Outputs: results/emax_comparison.csv, outputs/diagnostics/emax/emax_evolution.png;
         stdout: card statistics per fold and paired Wilcoxon tests at T = 10;
         intermediate states and metrics under data/samples/ and data/metrics/
Runtime: about 40 min from scratch, one process on a desktop CPU (BAYES-DOR at e_max = 50
         takes about 10 min per fold); about a minute when all sample files exist.

Usage:
    python scripts/simulation/run_emax_study.py

(formerly compare_emax.py)
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
from scipy.stats import wilcoxon

from bdor.datagen import DatasetGenerator
from bdor.runner import run_batch_experiments
from bdor.metrics import BenchmarkRunner
from bdor.utils import parse_subfold_string
import bdor.utils as utils
from bdor.paths import DATASETS_DIR, RESULTS_DIR, DIAGNOSTICS_DIR

# Module-level constants: run_emax_flipmatch.py imports them.
F1, F2, STEPS, AR, K = 10, 3, 10, 3, 3  # |A|, m, sessions T, subset size of session 1 / later
EMAXES = [5, 50]                        # card-scale caps compared
INCS = [0, 35, 50]                      # nominal inconsistency Finc (%)
HM = 10                                 # replications per fold (= n_runs of each dataset)
GEN_SEED = 20260813  # + Finc: both e_max folds of a Finc share tables, utilities, noise draws
RUN_SEED = 40260813  # base of the inference seeds (formula in run_all)
ARMS = ['FTRL_DOR_EMXJ1', 'BAYES_DOR_EMXJ1']
METRICS = ['asrs', 'asps', 'aios']      # ASR, ASP, AIO (paper eqs. (23)-(25))
OUT_DIR = os.path.join(DIAGNOSTICS_DIR, 'emax')
CSV_PATH = os.path.join(RESULTS_DIR, 'emax_comparison.csv')

# Plot colours: one per method, plus neutral tones for text, ticks, grid and background
# (the neutral tones are reused by run_emax_flipmatch.py).
BAYES_C, FTRL_C = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"


def fold_name(inc, emax):
    """Dataset folder name of one fold, e.g. ``emax_F1=10_F2=3_S=10_AR=3_K=3_Inc=35_Emax=50``."""
    return f"emax_F1={F1}_F2={F2}_S={STEPS}_AR={AR}_K={K}_Inc={inc}_Emax={emax}"


def generate_all():
    """Generate the six datasets of the study, skipping those that already exist.

    Before each e_max the global NumPy RNG is seeded with ``GEN_SEED + Finc``. The generator
    (bdor.datagen) consumes the same random numbers whatever ``n_max`` is, so the two e_max
    folds of one Finc share tables, true utilities, presented subsets and standard-normal
    noise draws, replication by replication. sigma solves eq. (22) separately for each e_max
    (gaps scaled by e_max), so that P(exact card count) = 1 - Finc on both card scales. At
    Finc > 0 this makes the e_max = 50 DM about ten times less noisy on the utility scale,
    with far fewer reversed orders (3.5% vs 21.0% of adjacent pairs at Finc = 50); the control
    that holds the ordinal noise fixed is run_emax_flipmatch.py. At Finc = 0 (sigma = 0) both
    folds declare the same orders and differ only in the card grid.
    """
    for inc in INCS:
        for emax in EMAXES:
            fold = fold_name(inc, emax)
            json_path = os.path.join(DATASETS_DIR, fold, f"f1_{F1}__f2_{F2}__nsteps_{STEPS}.json")
            if os.path.exists(json_path):
                print(f"{fold}: dataset exists, skipping generation")
                continue
            # Same seed for both e_max values of this Finc: paired folds (see docstring).
            np.random.seed(GEN_SEED + inc)
            gen = DatasetGenerator(output_dir=os.path.join(DATASETS_DIR, fold), n_runs=HM)
            gen.generate_batch(
                F1=[F1], F2=[F2], num_steps=STEPS, k_subset=(AR, K), n_max=emax,
                target_consistency=1.0 - inc / 100.0, plot_utilities=False
            )


def card_stats(inc, emax):
    """Noise scale and card-count distribution of one fold's dataset.

    Parameters
    ----------
    inc, emax : int
        Finc (%) and e_max of the fold.

    Returns
    -------
    dict
        ``sigma``: noise scale of eq. (21), on the e_max-scaled utility axis; ``mean``: mean
        card count per gap; ``zero_share`` / ``cap_share``: share of gaps with 0 / e_max
        cards; ``distinct``: number of distinct card counts. Pooled over all replications
        and sessions.
    """
    json_path = os.path.join(DATASETS_DIR, fold_name(inc, emax),
                             f"f1_{F1}__f2_{F2}__nsteps_{STEPS}.json")
    with open(json_path) as f:
        data = json.load(f)
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
    """Run both arms on the six folds and recompute their metric files.

    bdor.runner.run_batch_experiments runs the T = 10 sessions of every replication
    (adjacent-gap observations, eta_max = e_max, prequential log-loss recorded) with
    ``seed_base = RUN_SEED + 1000 * Finc + 100 * (e_max index) + 10 * (arm index)``, from which
    bdor.simulation derives one seed per replication and session. FTRL-DOR is the MAP of
    PreferenceSampler.find_map; BAYES-DOR runs NUTS with the 200 warm-up iterations and 200
    draws set in bdor.simulation.get_sampler_state. Existing per-session states are reused;
    PWI ('poi' in the code), RAI and ASR / ASP / AIO are then recomputed with force=True.
    """
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
    """Gather the per-session metrics of every fold and arm and write ``CSV_PATH``.

    Returns
    -------
    list of tuple
        One row ``(Finc, e_max, arm, metric, session, replication, value)`` per metric value,
        with session counted from 1 and replication from 0. The CSV holds the same rows under
        the header ``Inc,Emax,Method,Metric,Step,Table,Value``.
    """
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
    with open(CSV_PATH, 'w', newline='') as f:
        wr = csv.writer(f, lineterminator='\n')
        wr.writerow(['Inc', 'Emax', 'Method', 'Metric', 'Step', 'Table', 'Value'])
        wr.writerows(rows)
    print(f"Wrote {CSV_PATH} ({len(rows)} rows)")
    return rows


def series_vals(arr, inc, emax, sub_fold, metric, step):
    """Metric values of one fold, arm and session, ordered by replication.

    Parameters
    ----------
    arr : ndarray of object, shape (n_rows, 7)
        The rows returned by collect().
    inc, emax, sub_fold, metric, step
        Values to match in the first five columns.

    Returns
    -------
    ndarray of float, shape (n_replications,)
        Sorted by replication index, so the vectors for e_max = 5 and 50 are paired.
    """
    m = ((arr[:, 0] == inc) & (arr[:, 1] == emax) & (arr[:, 2] == sub_fold)
         & (arr[:, 3] == metric) & (arr[:, 4] == step))
    # table index in col 5 orders the paired vectors identically on both sides
    sub = arr[m]
    return sub[np.argsort(sub[:, 5].astype(int)), 6].astype(float)


def plot(rows):
    """Plot the mean +- SE over replications of each metric per session.

    One panel per metric (rows) and Finc (columns), colour = method, line style = e_max;
    ``rows`` as returned by collect(). Writes outputs/diagnostics/emax/emax_evolution.png.
    """
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
    """Print the card statistics of each fold and the final-session paired comparison.

    For every Finc, arm and metric: the means at e_max = 5 and 50, their difference and the
    two-sided Wilcoxon signed-rank p-value over the paired replications (NaN if SciPy raises
    ValueError). ``rows`` as returned by collect().
    """
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
    import argparse
    argparse.ArgumentParser(description=__doc__.split('\n\n')[0].strip(),
                            epilog='Takes no options: runs the whole (resumable) study.').parse_args()
    generate_all()
    run_all()
    rows = collect()
    plot(rows)
    summary(rows)
