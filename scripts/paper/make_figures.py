"""Build Figures 2-5 of the paper and print the numbers quoted with them.

Final stage of the simulation pipeline: reads the aggregated grid results and the
ties-study CSV written by scripts/simulation/ and draws the four simulation-study figures.
Figures 2 and 3 pool the 192 AR == K configurations with bdor.results.pool (error bars:
normal 95% confidence intervals); Figure 5 also refits the best-fixed-in-hindsight
comparator (bdor.regret.hindsight_loss) on the two dataset folders committed under
data/datasets/.

Paper:   Figure 2 (fig_factors_bdor_v1.png): FTRL-DOR, BAYES-DOR, FTRLdir and BAYESdir by
           |A|, m, T, k and Finc, each family on the replications it shares with its
           direction-only partner (20 per cell for FTRL, 5 for BAYES).
         Figure 3 (fig_baselines_t1_v1.png): FTRL-DOR, BAYES-DOR, SMAA-DOR and DOR at
           T = 1 by |A|, m, k and Finc, 20 replications per cell.
         Figure 4 (fig_sets_tau_v1.png): ties study, metrics against the merge threshold
           tau for |A| = 10, m = 3, k = 3 (FTRL-DOR 20, BAYES-DOR 5 replications).
         Figure 5 (fig_logloss_v1.png): regret against n_T for |A| = 10, m = 3, k = 5,
           Finc = 0% and 50%, first five replications.
         Printed: levels per session and ASR per tau (Sec. 5.6); final cumulative loss,
           hindsight loss and regret (Sec. 5.7); HMC diagnostics of BAYES-DOR (Sec. 5.1,
           sampler health); l1 norms of FTRL-DOR fits (a check, not quoted in the paper).
Inputs:  results/grid_results_joint_v1.npz, results/sets_tau_comparison.csv,
         data/datasets/grid_F1=10_F2=3_S=10_AR=5_K=5_Inc={0,50}/ (Figure 5),
         data/samples/grid_*/ (l1 check only; skipped when absent)
Outputs: outputs/figures/fig_factors_bdor_v1.png, fig_baselines_t1_v1.png,
         fig_sets_tau_v1.png, fig_logloss_v1.png; numbers on stdout
Runtime: about 10 s on a desktop CPU (Figure 5 loads JAX for ten small L-BFGS-B fits; the
         l1 check globs data/samples and is slower on a cold disk cache)

Usage:
    python scripts/paper/make_figures.py [--only STEP [STEP ...]]

    STEP is one of fig2, fig3, fig4, diagnostics (HMC and l1 checks), fig5. The selected
    steps always run in this order; the default runs all of them.

(formerly make_paper_figs_v1.py)
"""
import argparse
import csv
import glob
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, so `import bdor` works without installing

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import ScalarFormatter

from bdor.paths import FIGURES_DIR, RESULTS_DIR, SAMPLES_DIR
from bdor.results import (METRICS, MET_LABEL, T_LEVELS, PARTNER, load_results, pool,
                          mean_ci)

OUT = str(FIGURES_DIR)  # read at call time, so a caller may redirect the figures
DPI = 200
STEPS = ('fig2', 'fig3', 'fig4', 'diagnostics', 'fig5')

# Okabe-Ito colour-blind-safe palette: blue / sky blue for the FTRL family,
# orange / light orange for the BAYES family, bluish green and reddish purple
# for the two competing baselines.
STYLE = {
    'FTRL_DOR_ADJJ1':  dict(color='#0072B2', ls='-',  marker='o', label='FTRL-DOR'),
    'BAYES_DOR_ADJJ1': dict(color='#E69F00', ls='--', marker='s', label='BAYES-DOR'),
    'SMAA_DOR_ORIG':  dict(color='#009E73', ls='-.', marker='^', label='SMAA-DOR'),
    'DORLP_DOR_ORIG': dict(color='#CC79A7', ls=':',  marker='D', label='DOR'),
    'FTRLDIR_DOR_ADJJ1':  dict(color='#7FC4EC', ls='-',  marker='o',
                               label=r'FTRL$_{\mathrm{dir}}$'),
    'BAYESDIR_DOR_ADJJ1': dict(color='#F3C673', ls='--', marker='s',
                               label=r'BAYES$_{\mathrm{dir}}$'),
}
# The paper's two comparisons: the four B-DOR arms over the full grid (Figure 2),
# and the proposed estimators against the one-shot existing DOR methods at T = 1
# (Figure 3).
ARMS_BDOR = ['FTRL_DOR_ADJJ1', 'BAYES_DOR_ADJJ1',
             'FTRLDIR_DOR_ADJJ1', 'BAYESDIR_DOR_ADJJ1']
ARMS_BASELINES = ['FTRL_DOR_ADJJ1', 'BAYES_DOR_ADJJ1',
                  'SMAA_DOR_ORIG', 'DORLP_DOR_ORIG']

plt.rcParams.update({
    'font.size': 15,
    'axes.titlesize': 17,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 16,
    'grid.linestyle': '--',
    'grid.alpha': 0.5,
    'lines.linewidth': 2.4,
    'lines.markersize': 7,
})


# --------------------------------------------------------- Figures 2 and 3
def _fig_factors_group(results, arms, fname, T_fix=None, match_map=None):
    """Draw one marginal-effects figure: factors as rows, metrics as columns.

    In every panel each level of the row factor gets one bar per arm (mean and 95% CI
    of the metric, pooled over all the other factors), sorted best to worst within
    the level group; each panel has its own vertical scale.

    Parameters
    ----------
    results : dict
        Grid results, as returned by ``bdor.results.load_results``.
    arms : list of str
        Arms to draw (keys of ``STYLE``).
    fname : str
        File name, written under ``OUT``.
    T_fix : int, optional
        Restrict every panel to this horizon and drop the T row (Figure 3, T = 1).
    match_map : dict, optional
        ``{arm: partner}``: each arm is cut to the replications it shares with its
        partner (``PARTNER`` for Figure 2), so paired bars rest on the same decision
        problems.
    """
    factors = [('F1', [10, 20, 35, 50], "Number of\nalternatives $|A|$"),
               ('F2', [3, 5, 7, 9], "Number of\ncriteria $m$"),
               ('T', T_LEVELS, "Elicitation\nsteps $T$"),
               ('K', [3, 4, 5], "Subset size $k$"),
               ('Inc', [0, 15, 35, 50], "Inconsistency (%)")]
    if T_fix is not None:
        factors = [f for f in factors if f[0] != 'T']
    fig, axes = plt.subplots(len(factors), len(METRICS),
                             figsize=(13.5, 3.1 * len(factors)))
    n_arms = len(arms)
    bw = min(0.24, 0.82 / n_arms)
    offs = (np.arange(n_arms) - (n_arms - 1) / 2) * bw
    for r, (fac, levels, title) in enumerate(factors):
        for c, metric in enumerate(METRICS):
            ax = axes[r, c]
            xs = np.arange(len(levels))
            lo, hi = np.inf, -np.inf
            for j, lv in enumerate(levels):
                filt = dict(T=lv) if fac == 'T' else {fac: lv}
                if T_fix is not None:
                    filt['T'] = T_fix
                stats = {arm: mean_ci(pool(results, arm, metric,
                                           match=(match_map or {}).get(arm),
                                           **filt))
                         for arm in arms}
                # within each level group, bars sorted from best to worst
                order = sorted(arms, key=lambda a: -stats[a][0])
                for pos, arm in enumerate(order):
                    mean, ci = stats[arm]
                    ax.bar(j + offs[pos], mean, width=bw * 0.9, yerr=ci,
                           capsize=2.5, color=STYLE[arm]['color'],
                           error_kw=dict(lw=1.1, capthick=1.1))
                    lo = min(lo, mean - ci)
                    hi = max(hi, mean + ci)
            pad = 0.12 * (hi - lo)
            ax.set_ylim(max(0.0, lo - pad), hi + pad)
            ax.set_xticks(xs)
            ax.set_xticklabels([str(lv) for lv in levels], fontsize=15)
            ax.tick_params(axis='y', labelsize=14)
            if r == 0:
                ax.set_title(MET_LABEL[metric], fontsize=20, fontweight='bold', pad=10)
            if c == 0:
                ax.set_ylabel(title, fontsize=16)
            ax.grid(axis='y')
            ax.set_axisbelow(True)
    handles = [Patch(facecolor=STYLE[a]['color'], label=STYLE[a]['label'])
               for a in arms]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.0),
               ncol=min(4, n_arms), fontsize=18, frameon=True, columnspacing=2.0)
    top = 1.0 - 0.62 / (3.1 * len(factors))
    fig.tight_layout(rect=[0, 0, 1, top])
    fig.savefig(os.path.join(OUT, fname), dpi=DPI)
    plt.close(fig)
    print(f"Wrote {os.path.join(OUT, fname)}")


def fig_factors_bdor(results):
    """Figure 2: the four B-DOR arms over the whole grid, each family read on the
    replications it shares with its direction-only version."""
    _fig_factors_group(results, ARMS_BDOR, "fig_factors_bdor_v1.png",
                       match_map=PARTNER)


def fig_baselines_t1(results):
    """Figure 3: the proposed estimators against SMAA-DOR and DOR after the first
    session only (T = 1), every arm on all its replications (20 per cell)."""
    _fig_factors_group(results, ARMS_BASELINES, "fig_baselines_t1_v1.png",
                       T_fix=1)


def fig_factors(results):
    """Figures 2 and 3."""
    fig_factors_bdor(results)
    fig_baselines_t1(results)


# ---------------------------------------------------------------- Figure 4
def fig_sets_tau():
    """Figure 4: ranking metrics against the merge threshold tau (ties study).

    Reads ``results/sets_tau_comparison.csv`` (scripts/simulation/run_ties_study.py:
    |A| = 10, m = 3, k = 3, arms FTRL_DOR_SETJ1 and BAYES_DOR_SETJ1) and pools, for each
    tau, all Finc levels and the horizons T in {1, 3, 5, 10}; NaN values (a session
    collapsed to one level gives no gap, so no BAYES-DOR sample at T = 1) are dropped.
    The top axis shows the mean number of levels per session. Prints, per tau, the mean
    number of levels per session, the share of levels with more than one member and
    the ASR of each arm.
    """
    rows = []
    with open(os.path.join(RESULTS_DIR, "sets_tau_comparison.csv")) as f:
        for r in csv.DictReader(f):
            rows.append((float(r['Tau']), r['Method'], r['Metric'], int(r['Step']),
                         float(r['Value']), float(r['MeanLevelsPerSubrank']),
                         float(r['MultiLevelShare'])))
    taus = sorted({r[0] for r in rows})
    arms = [('FTRL_DOR_SETJ1', STYLE['FTRL_DOR_ADJJ1']),
            ('BAYES_DOR_SETJ1', STYLE['BAYES_DOR_ADJJ1'])]

    print("tau summary (pooled over Inc, T in {1,3,5,10}):")
    slev = {}
    for tau in taus:
        lv = [r[5] for r in rows if r[0] == tau]
        ml = [r[6] for r in rows if r[0] == tau]
        slev[tau] = np.mean(lv)
        print(f"  tau={tau:g}: mean levels/session {np.mean(lv):.2f}, "
              f"multi-level share {np.mean(ml) * 100:.1f}%")

    fig, axes = plt.subplots(1, len(METRICS), figsize=(16.5, 5.4))
    for ax, metric in zip(axes, METRICS):
        for arm, st in arms:
            means, cis = [], []
            for tau in taus:
                v = np.array([r[4] for r in rows
                              if r[0] == tau and r[1] == arm and r[2] == metric
                              and r[3] in T_LEVELS])
                m, c = mean_ci(v)
                means.append(m)
                cis.append(c)
                if metric == 'asrs':
                    print(f"    {arm} tau={tau:g}: ASR {m:.3f}+-{c:.3f} "
                          f"(n={np.sum(~np.isnan(v))}/{len(v)})")
            means = np.array(means); cis = np.array(cis)
            ax.plot(taus, means, color=st['color'], ls=st['ls'], marker=st['marker'],
                    label=st['label'])
            ax.fill_between(taus, means - cis, means + cis, color=st['color'], alpha=0.15)
        ax.set_xticks(taus)
        ax.set_xlabel(r"Merge threshold $\tau$")
        ax.set_title(MET_LABEL[metric], fontsize=18, pad=52)
        ax.grid(True)
        ax.set_axisbelow(True)
        sec = ax.secondary_xaxis('top')
        sec.set_xticks(taus)
        sec.set_xticklabels([f"{slev[t]:.1f}" for t in taus], fontsize=12)
        sec.set_xlabel(r"Mean levels per session $\bar{l}$", fontsize=13, labelpad=7)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0),
               ncol=2, fontsize=16, frameon=True, columnspacing=2.5)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(os.path.join(OUT, "fig_sets_tau_v1.png"), dpi=DPI)
    plt.close(fig)
    print(f"Wrote {os.path.join(OUT, 'fig_sets_tau_v1.png')}")


# ---------------------------------------------------------------- Figure 5
def fig_logloss(results):
    """Figure 5: regret against n_T on a log axis, and the Sec. 5.7 numbers.

    Configuration |A| = 10, m = 3, k = 5 (AR = K = 5), Finc = 0% and 50%, replications
    0-4. The regret after session t is the arm's cumulative prequential log-loss
    (``'log_loss'`` in the grid results) minus the cumulative loss of the
    best-fixed-in-hindsight fit (``bdor.regret.hindsight_loss``, fitted on the dataset
    folder); n_T is the cumulative number of gap observations, so logarithmic growth is
    a straight line. Bands are mean +- 1.96 standard errors over the replications; the
    grey references are a log-n_T chord through the FTRL-DOR end points and linear
    growth from the first FTRL-DOR point. Prints, per Finc level, each arm's final
    cumulative loss, the hindsight loss and their difference (the regret).
    """
    from bdor.regret import hindsight_loss  # lazy: the hindsight fit imports JAX
    F1, F2, K = 10, 3, 5
    incs = [0, 50]
    n_tab = 5  # replications 0-4 (Figure 5 caption: mean over five replications)
    fig, axes = plt.subplots(1, len(incs), figsize=(14, 5.6))
    gaps = {}
    for ax, inc in zip(axes, incs):
        cs = f"F1={F1}_F2={F2}_S=10_AR={K}_K={K}_Inc={inc}"
        arms = results.get(cs, {})
        hs_curves = hindsight_loss(f"grid_{cs}", F1, F2, 10, n_tab, 'adjacent',
                                   return_curves=True)   # n_tab x steps
        gaps[(inc, 'hindsight', 'final')] = hs_curves.mean(axis=0)[-1]
        n_t, reg_top, ftrl_rm = None, 0.0, None
        for arm in ['FTRL_DOR_ADJJ1', 'BAYES_DOR_ADJJ1']:
            st = STYLE[arm]
            # steps x tab x [prequential loss, number of gaps]
            loss = np.array(arms[arm]['log_loss'], dtype=float)[:, :n_tab]
            cum = np.cumsum(loss[:, :, 0], axis=0)
            n_t = np.cumsum(loss[:, :, 1], axis=0).mean(axis=1)
            gaps[(inc, arm, 'final')] = cum.mean(axis=1)[-1]
            reg = cum - hs_curves.T                       # steps x tab
            rm = reg.mean(axis=1)
            rse = reg.std(axis=1) / np.sqrt(n_tab)
            ax.plot(n_t, rm, color=st['color'], ls=st['ls'], marker=st['marker'],
                    label=st['label'])
            ax.fill_between(n_t, rm - 1.96 * rse, rm + 1.96 * rse,
                            color=st['color'], alpha=0.15)
            reg_top = max(reg_top, np.max(rm + 1.96 * rse))
            if arm == 'FTRL_DOR_ADJJ1':
                ftrl_rm = rm
        # references: straight-in-log-x chord through the FTRL endpoints, and
        # linear growth anchored at the first FTRL point (exits the panel top)
        guide = ftrl_rm[0] + (ftrl_rm[-1] - ftrl_rm[0]) \
            * (np.log(n_t) - np.log(n_t[0])) / (np.log(n_t[-1]) - np.log(n_t[0]))
        ax.plot(n_t, guide, color='0.45', ls='--', lw=1.8,
                label=r'$\propto \log n_T$ reference')
        lin = ftrl_rm[0] * n_t / n_t[0]
        ax.plot(n_t, lin, color='0.45', ls=':', lw=2.2,
                label=r'$\propto n_T$ reference')
        ax.set_ylim(top=reg_top * 1.18)
        ax.set_title(f"Inconsistency = {inc}%", fontsize=17, pad=8)
        ax.set_xscale('log')
        ax.set_xticks([4, 8, 16, 40])
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.minorticks_off()
        ax.set_xlabel(r"$n_T$ (cumulative gap observations, log scale)")
        ax.grid(True)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Regret")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0),
               ncol=4, fontsize=14, frameon=True, columnspacing=1.6)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(os.path.join(OUT, "fig_logloss_v1.png"), dpi=DPI)
    plt.close(fig)
    print(f"Wrote {os.path.join(OUT, 'fig_logloss_v1.png')}")
    for inc in incs:
        h = gaps[(inc, 'hindsight', 'final')]
        for arm in ['FTRL_DOR_ADJJ1', 'BAYES_DOR_ADJJ1']:
            f = gaps[(inc, arm, 'final')]
            rel = (f" ({(f - h) / h * 100:.1f}% of hindsight)" if h > 1
                   else " (hindsight ~ 0: separable)")
            print(f"  Inc={inc}: {arm} final cum loss {f:.2f}, hindsight {h:.2f}, "
                  f"regret {f - h:.2f}{rel}")


# ---------------------------------------------------------- printed checks
def diagnostics(results):
    """Print the HMC health of BAYES-DOR over every fit stored in the grid results.

    Pools ``'hmc_diag'`` (per session and replication: number of divergent transitions
    and the largest split-Rhat over the parameters, one chain split in two halves) of
    BAYES_DOR_ADJJ1 over all configurations in the npz (all 576, not only AR == K),
    and prints the number of fits, the total number of divergent transitions, the share
    of fits with at least one, the largest split-Rhat and the shares of fits whose
    split-Rhat exceeds 1.05 and 1.1.
    """
    div, rhat, fits = [], [], 0
    for c, arms in results.items():
        md = arms.get('BAYES_DOR_ADJJ1')
        if md is None or 'hmc_diag' not in md:
            continue
        d = np.array(md['hmc_diag'], dtype=float)  # steps x tables x 2
        div.extend(d[:, :, 0].ravel())
        rhat.extend(d[:, :, 1].ravel())
        fits += d.shape[0] * d.shape[1]
    div = np.array(div); rhat = np.array(rhat)
    print(f"BAYES_DOR_ADJJ1 diagnostics: {fits} HMC fits, "
          f"{int(np.nansum(div))} divergent transitions total, "
          f"{np.mean(div > 0) * 100:.2f}% of fits with >=1 divergence, "
          f"max split-Rhat {np.nanmax(rhat):.3f}, "
          f"share Rhat>1.05: {np.nanmean(rhat > 1.05) * 100:.2f}%, "
          f">1.1: {np.nanmean(rhat > 1.1) * 100:.2f}%")


def l1_check(n_files=400):
    """Print the l1 norm of a random subsample of FTRL-DOR final-session fits.

    Globs the stored session-10 states of FTRL_DOR_ADJJ1 (the MAP vector
    ``[w, Delta - rho]``) under data/samples, loads ``n_files`` of them chosen with a
    fixed seed, and prints the largest and the median l1 norm (cf. the radius M of the
    comparator class Phi_{rho,M}, eq. (18)). Prints a 'skipped' line when no sample
    file is present, e.g. in a fresh clone.

    Parameters
    ----------
    n_files : int
        Number of files to load (all of them if fewer are found).
    """
    files = glob.glob(os.path.join(SAMPLES_DIR, "grid_*", "f1_*_f2_*_nsteps_10",
                                   "FTRL_DOR_ADJJ1", "table_*", "10.npy"))
    if not files:
        print(f"FTRL final-step fits: l1 check skipped (no FTRL_DOR_ADJJ1 samples "
              f"under {SAMPLES_DIR})")
        return
    rng = np.random.default_rng(0)
    sel = rng.choice(len(files), size=min(n_files, len(files)), replace=False)
    norms = [np.abs(np.load(files[i])).sum() for i in sel]
    print(f"FTRL final-step fits: max l1 norm {np.max(norms):.2f} "
          f"(median {np.median(norms):.2f}, n={len(norms)} of {len(files)} files)")


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Build Figures 2-5 of the paper and print the numbers quoted with "
                    "them.")
    parser.add_argument(
        '--only', nargs='+', choices=STEPS, default=list(STEPS), metavar='STEP',
        help="run only these steps: fig2, fig3, fig4, diagnostics (HMC diagnostics "
             "and l1 check), fig5; they always run in this order (default: all)")
    return parser.parse_args(argv)


def main(argv=None):
    """Run the selected steps (all by default) in the fixed order of ``STEPS``.

    Parameters
    ----------
    argv : list of str, optional
        Command-line options (see ``--help``). ``None`` means no options, so that
        ``main()`` called from Python builds everything; the ``__main__`` guard passes
        ``sys.argv[1:]``.
    """
    steps = set(_parse_args([] if argv is None else argv).only)
    os.makedirs(OUT, exist_ok=True)
    results = load_results()
    print(f"Configs: {len(results)}")
    if 'fig2' in steps:
        fig_factors_bdor(results)
    if 'fig3' in steps:
        fig_baselines_t1(results)
    if 'fig4' in steps:
        fig_sets_tau()
    if 'diagnostics' in steps:
        diagnostics(results)
        l1_check()
    if 'fig5' in steps:
        fig_logloss(results)   # last: pulls in JAX for the hindsight fits


if __name__ == "__main__":
    main(sys.argv[1:])
