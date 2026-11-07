"""Noise-scaled control of the e_max study at Finc = 50%: sigma held fixed on the utility scale.

run_emax_study.py tunes sigma per e_max on the card scale, so at Finc = 50 its e_max = 50 fold
reverses far fewer adjacent pairs than the e_max = 5 one (3.5% vs 21.0%). This script adds the
fold ..._Inc=50_Emax=50_Sigma=util, with sigma tuned at e_max = 5 and multiplied by 10
(= e_max / 5): its declared orders equal those of the e_max = 5 fold (verify() asserts it) and
only the card grid is finer. Run run_emax_study.py first (its folds and constants are reused).
Resumable: datasets and per-session sample files are reused; metrics are recomputed (force=True).

Paper:   Sec. 5.4, noise-scaled control (sigma of eq. (21) times ten): ASR gain and paired
         Wilcoxon p-value of e50_flip over e5_inc50 at T = 10
Inputs:  data/datasets/emax_F1=10_F2=3_S=10_AR=3_K=3_Inc=50_Emax=5/ and the data/metrics/
         files of the four emax_* folds with Finc in {0, 50}, all written by run_emax_study.py
Outputs: results/emax_flipmatch.csv, outputs/diagnostics/emax/emax_flipmatch.png;
         stdout: pairing check, final-session means and paired Wilcoxon tests;
         data/{datasets,samples,metrics}/emax_F1=10_F2=3_S=10_AR=3_K=3_Inc=50_Emax=50_Sigma=util/
Runtime: about 11 min from scratch, one process on a desktop CPU (BAYES-DOR about 9 min);
         about a minute when the sample files exist.

Usage:
    python scripts/simulation/run_emax_flipmatch.py

(formerly compare_emax_flipmatch.py)
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
# Sibling script (importable because Python puts this script's folder on sys.path): design
# constants, seeds, fold names and plot colours of the e_max study.
from run_emax_study import (F1, F2, STEPS, AR, K, HM, GEN_SEED, RUN_SEED, ARMS,
                            METRICS, fold_name, INK, INK2, MUTED, GRID, SURFACE)

INC = 50
FM_FOLD = f"emax_F1={F1}_F2={F2}_S={STEPS}_AR={AR}_K={K}_Inc={INC}_Emax=50_Sigma=util"
OUT_DIR = os.path.join(DIAGNOSTICS_DIR, 'emax')
CSV_PATH = os.path.join(RESULTS_DIR, 'emax_flipmatch.csv')

# Compared variants: variant -> (dataset fold, eta_max). The three Finc = 50 variants share
# tables, true utilities and noise draws (same seed), so they are compared pairwise:
#   e5_inc50   e_max = 5                                     21.0% reversed adjacent pairs
#   e50_card   e_max = 50, card-matched sigma                 3.5%
#   e50_flip   e_max = 50, sigma fixed on the utility axis   21.0%, same orders as e5_inc50
# e50_flip vs e5_inc50 isolates the finer card grid under the same ordinal noise (the
# comparison reported in the paper); e50_flip vs e50_card shows how much of the card-matched
# gain comes from the lower noise. The two Finc = 0 folds are a noise-free reference (other
# tables, not paired with the rest).
VARIANTS = {
    'e5_inc50': (fold_name(INC, 5), 5),
    'e50_card': (fold_name(INC, 50), 50),
    'e50_flip': (FM_FOLD, 50),
    'e5_noinc': (fold_name(0, 5), 5),
    'e50_noinc': (fold_name(0, 50), 50),
}
PAIRED = ['e5_inc50', 'e50_card', 'e50_flip']  # same seed -> same tables


class UtilityScaleSigmaGenerator(DatasetGenerator):
    """DatasetGenerator whose noise scale is fixed on the utility axis, not on the card axis.

    ``_tune_sigma`` solves eq. (22) on the e_max = 5 scale and multiplies the solution by
    ``n_max / 5``. The parent tuner is called with ``n_max=5``, so the calibration consumes the
    same random numbers as for the e_max = 5 fold and returns the same sigma. With the same
    seed, the perceived utilities of eq. (21) are then ``n_max / 5`` times those of the
    e_max = 5 fold (up to floating-point rounding): the declared orders coincide and only the
    card grid is finer.
    """

    def _tune_sigma(self, f1, f2, target_consistency, n_max, num_intervals=3,
                    utility_type='exponential'):
        """Return the e_max = 5 solution of eq. (22) times ``n_max / 5`` (0 if consistent)."""
        sigma5 = super()._tune_sigma(f1, f2, target_consistency, 5,
                                     num_intervals=num_intervals,
                                     utility_type=utility_type)
        return sigma5 * (n_max / 5.0)


def generate():
    """Generate the flip-matched dataset, unless it already exists.

    Uses the seed of both Finc = 50 folds of run_emax_study.py (``GEN_SEED + 50``).
    """
    json_path = os.path.join(DATASETS_DIR, FM_FOLD, f"f1_{F1}__f2_{F2}__nsteps_{STEPS}.json")
    if os.path.exists(json_path):
        print(f"{FM_FOLD}: dataset exists, skipping generation")
        return
    np.random.seed(GEN_SEED + INC)  # same stream as both existing Inc=50 folds
    gen = UtilityScaleSigmaGenerator(output_dir=os.path.join(DATASETS_DIR, FM_FOLD), n_runs=HM)
    gen.generate_batch(
        F1=[F1], F2=[F2], num_steps=STEPS, k_subset=(AR, K), n_max=50,
        target_consistency=1.0 - INC / 100.0, plot_utilities=False
    )


def load_fold(fold):
    """Load the dataset JSON of one emax_* fold (dict with 'metadata' and 'runs')."""
    path = os.path.join(DATASETS_DIR, fold, f"f1_{F1}__f2_{F2}__nsteps_{STEPS}.json")
    with open(path) as f:
        return json.load(f)


def adj_flip_rate(data):
    """Percentage of declared adjacent pairs that reverse the true order.

    Counts, over all replications and sessions of a dataset (as returned by load_fold), the
    consecutive items of each declared order whose upper item has the lower true utility.
    """
    flips = tot = 0
    for run in data['runs']:
        tu = np.array(run['true_utilities'])
        for sr in run['subranks']:
            items = sr['items']
            for p in range(len(items) - 1):
                flips += bool(tu[items[p]] < tu[items[p + 1]])
                tot += 1
    return 100.0 * flips / tot


def verify():
    """Check that the flip-matched fold declares the same orders as the e_max = 5, Finc = 50 fold.

    Prints the check, the realized adjacent flip rates and both sigmas; raises AssertionError
    if any declared order differs (the pairing would then be broken).
    """
    fm = load_fold(FM_FOLD)
    e5 = load_fold(fold_name(INC, 5))
    same = all(a['items'] == b['items']
               for ra, rb in zip(fm['runs'], e5['runs'])
               for a, b in zip(ra['subranks'], rb['subranks']))
    r_fm, r_e5 = adj_flip_rate(fm), adj_flip_rate(e5)
    print(f"declared orders identical to e_max=5 fold: {same}")
    print(f"adjacent flips: flip-matched e_max=50 = {r_fm:.1f}%, e_max=5 = {r_e5:.1f}%")
    print(f"sigma: {fm['metadata']['sigma']:.3f} (= 10 x {e5['metadata']['sigma']:.3f})")
    assert same, "pairing broken: declared orders differ from the e_max=5 fold"


def run_arms():
    """Run both arms on the flip-matched fold and recompute its metric files.

    Same settings as run_emax_study.run_all with eta_max = 50 and
    ``seed_base = RUN_SEED + 1000 * Finc + 300 + 10 * (arm index)``; the offset 300 keeps these
    seeds apart from those of the e_max study folds (offsets 0 and 100).
    """
    for arm_idx, sub_fold in enumerate(ARMS):
        alg_name, _ = parse_subfold_string(sub_fold)
        seed_base = RUN_SEED + INC * 1000 + 300 + arm_idx * 10
        t0 = time.time()
        run_batch_experiments(
            F1=[F1], F2=[F2], sub_fold=sub_fold, dataset_folds=[FM_FOLD],
            alg=alg_name, overwrite=False, hm=HM, k_subset=(AR, K),
            num_steps=STEPS, eta_max=50, expansion='adjacent',
            seed_base=seed_base, log_loss=True
        )
        runner = BenchmarkRunner(
            dataset_fold=FM_FOLD, sub_fold=sub_fold, num_subint=3, hm=HM,
            F1=[F1], F2=[F2], NUM_STEPS_LIST=[STEPS], eta_max=50
        )
        runner.compute_metrics('poi', force=True)
        runner.compute_metrics('rai', force=True)
        runner.compute_asrs(force=True)
        runner.compute_asps(force=True)
        runner.compute_aios(force=True)
        print(f"{FM_FOLD} / {sub_fold}: done in {time.time() - t0:.1f}s", flush=True)


def collect():
    """Gather the per-session metrics of the five variants and write ``CSV_PATH``.

    Returns
    -------
    dict
        ``vals[variant][arm][metric]``: array of shape (sessions, replications). The CSV has
        one row per value, header ``Variant,Method,Metric,Step,Table,Value`` (Step counted
        from 1, Table from 0).
    """
    vals = {}
    rows = []
    for variant, (fold, _) in VARIANTS.items():
        vals[variant] = {}
        for sub_fold in ARMS:
            vals[variant][sub_fold] = {}
            for metric in METRICS:
                data = np.array(utils.load_test_results(metric, fold, sub_fold, F1, F2, STEPS))
                vals[variant][sub_fold][metric] = data
                for j in range(data.shape[0]):
                    for i in range(data.shape[1]):
                        rows.append((variant, sub_fold, metric, j + 1, i, data[j, i]))
    with open(CSV_PATH, 'w', newline='') as f:
        wr = csv.writer(f, lineterminator='\n')
        wr.writerow(['Variant', 'Method', 'Metric', 'Step', 'Table', 'Value'])
        wr.writerows(rows)
    print(f"Wrote {CSV_PATH} ({len(rows)} rows)")
    return vals


def plot(vals):
    """Plot the mean +- SE over replications of each metric per session for the five variants.

    One panel per metric (rows) and method (columns); SE bands only for the paired Finc = 50
    variants. ``vals`` as returned by collect(). Writes outputs/diagnostics/emax/emax_flipmatch.png.
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    metric_names = {'asrs': 'ASRS (RAI at true ranks)',
                    'asps': 'ASPS (pairwise success)',
                    'aios': 'AIOS (true best at rank 1)'}
    # The three Finc = 50 variants in distinct colours; the two noise-free references in
    # muted grey, told apart by line style.
    series = [
        ('e50_flip', "#2a78d6", '-', 's', 2.2, '$e_{max}=50$, flip-matched (21% flips)'),
        ('e50_card', "#eb6834", '-', 'o', 2.2, '$e_{max}=50$, card-matched (3.5% flips)'),
        ('e5_inc50', "#1baf7a", '-', '^', 2.2, '$e_{max}=5$ (21% flips)'),
        ('e50_noinc', MUTED, '-', None, 1.4, '$e_{max}=50$, no noise (ref)'),
        ('e5_noinc', MUTED, '--', None, 1.4, '$e_{max}=5$, no noise (ref)'),
    ]
    steps = np.arange(1, STEPS + 1)
    method_label = {'FTRL_DOR_EMXJ1': 'FTRL-DOR', 'BAYES_DOR_EMXJ1': 'BAYES-DOR'}

    fig, axes = plt.subplots(len(METRICS), len(ARMS),
                             figsize=(5.4 * len(ARMS), 3.1 * len(METRICS)),
                             sharex=True, sharey='row')
    fig.patch.set_facecolor(SURFACE)
    for r, metric in enumerate(METRICS):
        for c, sub_fold in enumerate(ARMS):
            ax = axes[r, c]
            ax.set_facecolor(SURFACE)
            for variant, color, ls, mk, lw, label in series:
                data = vals[variant][sub_fold][metric]  # steps x tables
                mean = data.mean(axis=1)
                se = data.std(axis=1, ddof=1) / np.sqrt(data.shape[1])
                ax.plot(steps, mean, color=color, lw=lw, ls=ls, marker=mk, ms=4, label=label)
                if variant in PAIRED:
                    ax.fill_between(steps, mean - se, mean + se, color=color, alpha=0.12, lw=0)
            ax.grid(True, color=GRID, lw=0.8)
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
            for spine in ['left', 'bottom']:
                ax.spines[spine].set_color(GRID)
            ax.tick_params(colors=MUTED, labelsize=8)
            if r == 0:
                ax.set_title(method_label[sub_fold], fontsize=10, color=INK2)
            if c == 0:
                ax.set_ylabel(metric_names[metric], fontsize=9, color=INK2)
            if r == len(METRICS) - 1:
                ax.set_xlabel("Elicitation step", fontsize=9, color=INK2)
                ax.set_xticks(steps)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"Ordinal noise held fixed: flip-matched $e_{{max}}=50$ at Inc={INC}%  "
                 f"(F1={F1}, F2={F2}, AR=K={K}, T={STEPS}, {HM} tables, mean $\\pm$ SE)",
                 fontsize=11, color=INK, y=1.06)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(OUT_DIR, 'emax_flipmatch.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {out}")


def summary(vals):
    """Print the final-session means of every variant and the paired Wilcoxon tests.

    The tests are two-sided Wilcoxon signed-rank tests over the paired replications of
    e50_flip vs e5_inc50 (the paper's comparison) and e50_flip vs e50_card (NaN if SciPy
    raises ValueError). ``vals`` as returned by collect().
    """
    def pval(a, b):
        try:
            return wilcoxon(a, b).pvalue
        except ValueError:
            return float('nan')

    print(f"\n===== FINAL-STEP (T={STEPS}) MEANS, {HM} tables =====")
    header = f"{'method':>10} {'metric':>6}" + "".join(f" {v:>10}" for v in VARIANTS)
    print(header)
    for sub_fold in ARMS:
        name = 'FTRL' if sub_fold.startswith('FTRL') else 'BAYES'
        for metric in METRICS:
            cells = "".join(f" {vals[v][sub_fold][metric][-1].mean():>10.3f}" for v in VARIANTS)
            print(f"{name:>10} {metric:>6}{cells}")

    print(f"\n===== PAIRED WILCOXON at T={STEPS} (Inc={INC} variants, same tables) =====")
    print(f"{'method':>10} {'metric':>6} {'flip50-e5':>10} {'p':>8} {'flip50-card50':>13} {'p':>8}")
    for sub_fold in ARMS:
        name = 'FTRL' if sub_fold.startswith('FTRL') else 'BAYES'
        for metric in METRICS:
            flip = vals['e50_flip'][sub_fold][metric][-1]
            e5 = vals['e5_inc50'][sub_fold][metric][-1]
            card = vals['e50_card'][sub_fold][metric][-1]
            print(f"{name:>10} {metric:>6} "
                  f"{flip.mean() - e5.mean():>+10.3f} {pval(flip, e5):>8.4f} "
                  f"{flip.mean() - card.mean():>+13.3f} {pval(flip, card):>8.4f}")


if __name__ == '__main__':
    import argparse
    argparse.ArgumentParser(description=__doc__.split('\n\n')[0].strip(),
                            epilog='Takes no options: runs the whole (resumable) study; '
                                   'run_emax_study.py must have run first.').parse_args()
    generate()
    verify()
    run_arms()
    vals = collect()
    plot(vals)
    summary(vals)
