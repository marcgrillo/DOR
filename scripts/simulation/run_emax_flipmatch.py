"""
Flip-matched e_max control (2026-08-17, follow-up to compare_emax.py).

The card-matched injection of compare_emax.py holds P(exact card) fixed given
e_max, which at e_max=50 all but removes ordinal noise (realized adjacent
flips at Inc=50: 21.0% for e_max=5 vs 3.5% for e_max=50). This control
matches the ORDINAL noise instead: sigma is tuned on the e_max=5 scale
(bit-identically to the Inc=50/Emax=5 fold, same calibration RNG stream) and
scaled by e_max/5, i.e. held fixed on the utility axis. With the same
generation seed the noisy perceived utilities are IDENTICAL up to the x10
scale factor, so the declared orders -- and hence the 21.0% realized adjacent
flip rate -- match the e_max=5 fold run-by-run; only the card grid is finer.

Comparisons (Inc=50 variants share the same 10 tables -> paired Wilcoxon):
  e_max=50 flip-matched vs e_max=5              pure granularity under REAL ordinal noise
  e_max=50 flip-matched vs e_max=50 card-matched size of the injection artifact
  Inc=0 folds shown as no-inconsistency reference (different tables, unpaired).

Run:  C:\\Users\\marco\\anaconda3\\python.exe compare_emax_flipmatch.py   (from DOR/)
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
from run_emax_study import (F1, F2, STEPS, AR, K, HM, GEN_SEED, RUN_SEED, ARMS,
                          METRICS, fold_name, INK, INK2, MUTED, GRID, SURFACE)

INC = 50
FM_FOLD = f"emax_F1={F1}_F2={F2}_S={STEPS}_AR={AR}_K={K}_Inc={INC}_Emax=50_Sigma=util"
OUT_DIR = os.path.join(DIAGNOSTICS_DIR, 'emax')
CSV_PATH = os.path.join(RESULTS_DIR, 'emax_flipmatch.csv')

# variant -> (dataset fold, eta_max)
VARIANTS = {
    'e5_inc50': (fold_name(INC, 5), 5),
    'e50_card': (fold_name(INC, 50), 50),
    'e50_flip': (FM_FOLD, 50),
    'e5_noinc': (fold_name(0, 5), 5),
    'e50_noinc': (fold_name(0, 50), 50),
}
PAIRED = ['e5_inc50', 'e50_card', 'e50_flip']  # same seed -> same tables


class UtilityScaleSigmaGenerator(DatasetGenerator):
    """sigma tuned on the e_max=5 scale, then scaled by n_max/5 (fixed on the
    utility axis). The parent tuner is called with n_max=5 so the calibration
    consumes the identical RNG stream as the e_max=5 fold (pairing)."""

    def _tune_sigma(self, f1, f2, target_consistency, n_max, num_intervals=3,
                    utility_type='exponential'):
        sigma5 = super()._tune_sigma(f1, f2, target_consistency, 5,
                                     num_intervals=num_intervals,
                                     utility_type=utility_type)
        return sigma5 * (n_max / 5.0)


def generate():
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
    path = os.path.join(DATASETS_DIR, fold, f"f1_{F1}__f2_{F2}__nsteps_{STEPS}.json")
    return json.load(open(path))


def adj_flip_rate(data):
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
    """The flip-matched fold must declare the same orders as e_max=5, Inc=50."""
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
    """vals[variant][sub_fold][metric] = array (steps x tables)."""
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
    import csv
    with open(CSV_PATH, 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['Variant', 'Method', 'Metric', 'Step', 'Table', 'Value'])
        wr.writerows(rows)
    print(f"Wrote {CSV_PATH} ({len(rows)} rows)")
    return vals


def plot(vals):
    os.makedirs(OUT_DIR, exist_ok=True)
    metric_names = {'asrs': 'ASRS (RAI at true ranks)',
                    'asps': 'ASPS (pairwise success)',
                    'aios': 'AIOS (true best at rank 1)'}
    # validated categorical slots 1-3 for the Inc=50 variants; muted ink for
    # the no-inconsistency references (identity also carried by linestyle)
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

            def pval(a, b):
                try:
                    return wilcoxon(a, b).pvalue
                except ValueError:
                    return float('nan')
            print(f"{name:>10} {metric:>6} {flip.mean() - e5.mean():>+10.3f} {pval(flip, e5):>8.4f} "
                  f"{flip.mean() - card.mean():>+13.3f} {pval(flip, card):>8.4f}")


if __name__ == '__main__':
    generate()
    verify()
    run_arms()
    vals = collect()
    plot(vals)
    summary(vals)
