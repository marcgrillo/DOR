"""
Paper-grade figures for the rewritten experimental section (main_new_base.tex),
styled to match the June paper figures (fig_overall.png / fig_factors.png):
large fonts, framed legends outside on top, descriptive column titles, bold
row labels, dashed grid.

Writes into paper_figs/:
  fig_overall_v1.png       - overall mean-metric bars, all six compared arms
  fig_factors_v1.png       - 3x5 marginal-effect grid over the five factors,
                             all six arms (no CI bands for the dir ablations)
  fig_logloss_v1.png       - regret vs n_T (log axis) with log/linear references
  fig_sets_tau_v1.png      - set-valued sessions: metrics vs merge threshold tau

Also prints the aggregate numbers quoted in the text: HMC diagnostics for the
primary BAYES arm, max l1 norm of the FTRL fits (Phi_{rho,M} statement), the
final-step regret gaps, and the tau-experiment summary.

Run from DOR/:  C:\\Users\\marco\\anaconda3\\python.exe make_paper_figs_v1.py
"""
import os
import re
import csv
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NPZ = "grid_results_joint_v1.npz"
OUT = "paper_figs"
METRICS = ['asrs', 'asps', 'aios']
MET_LABEL = {'asrs': 'ASR', 'asps': 'ASP', 'aios': 'AIO'}
T_LEVELS = [1, 3, 5, 10]
DPI = 200

# Okabe-Ito colour-blind-safe palette: blue / sky blue for the FTRL family,
# orange / light orange for the BAYES family, bluish green and reddish purple
# for the two competing baselines.
STYLE = {
    'FTRL_DOR_ADJJ1':  dict(color='#0072B2', ls='-',  marker='o', label='FTRL-DOR'),
    'BAYES_DOR_ADJJ1': dict(color='#E69F00', ls='--', marker='s', label='BAYES-DOR'),
    'SMAA_DOR_ADJJ1':  dict(color='#009E73', ls='-.', marker='^', label='SMAA-DOR'),
    'DORLP_DOR_ADJJ1': dict(color='#CC79A7', ls=':',  marker='D', label='DOR (LP)'),
    'FTRLDIR_DOR_ADJJ1':  dict(color='#7FC4EC', ls='-',  marker='o',
                               label=r'FTRL$_{\mathrm{dir}}$'),
    'BAYESDIR_DOR_ADJJ1': dict(color='#F3C673', ls='--', marker='s',
                               label=r'BAYES$_{\mathrm{dir}}$'),
}
ARMS_ALL = ['FTRL_DOR_ADJJ1', 'BAYES_DOR_ADJJ1', 'SMAA_DOR_ADJJ1',
            'DORLP_DOR_ADJJ1', 'FTRLDIR_DOR_ADJJ1', 'BAYESDIR_DOR_ADJJ1']
DIR_ARMS = {'FTRLDIR_DOR_ADJJ1', 'BAYESDIR_DOR_ADJJ1'}
# The paper's two comparisons: the four B-DOR arms over the full grid, and the
# proposed estimators against the (one-shot) existing DOR methods at T = 1.
ARMS_BDOR = ['FTRL_DOR_ADJJ1', 'BAYES_DOR_ADJJ1',
             'FTRLDIR_DOR_ADJJ1', 'BAYESDIR_DOR_ADJJ1']
ARMS_BASELINES = ['FTRL_DOR_ADJJ1', 'BAYES_DOR_ADJJ1',
                  'SMAA_DOR_ADJJ1', 'DORLP_DOR_ADJJ1']

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


def parse_config(c):
    m = re.match(r"F1=(\d+)_F2=(\d+)_S=(\d+)_AR=(\d+)_K=(\d+)_Inc=(\d+)", c)
    return {k: int(v) for k, v in zip(['F1', 'F2', 'S', 'AR', 'K', 'Inc'], m.groups())}


def pool(results, arm, metric, T=None, **filters):
    """Pooled per-table values under AR==K, T in T_LEVELS, given filters."""
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


def mean_ci(v):
    # NaN-aware: a session that collapses to a single level yields no gap
    # observation, so at t=1 BAYES has no posterior sample (tau experiment).
    v = v[~np.isnan(v)]
    return v.mean(), 1.96 * v.std() / np.sqrt(len(v))


# The former fig_overall bar chart is replaced in the paper by the summary
# table with Wilcoxon tests (make_summary_table_v1.py, tab:overall_summary).


# ---------------------------------------------------------------- fig_factors
def _fig_factors_group(results, arms, fname, T_fix=None):
    """Factors as rows, metrics as columns (portrait layout); grouped bars
    sorted best-to-worst within each level group; per-panel vertical scale.
    With T_fix, every panel is restricted to that horizon and the T row is
    dropped (used for the one-shot baseline comparison at T = 1)."""
    from matplotlib.patches import Patch
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
                stats = {arm: mean_ci(pool(results, arm, metric, **filt))
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
    print(f"Wrote paper_figs/{fname}")


def fig_factors(results):
    # single figure with the four B-DOR arms over the full grid
    _fig_factors_group(results, ARMS_BDOR, "fig_factors_bdor_v1.png")
    # proposed estimators vs the one-shot existing DOR methods, at T = 1 only
    _fig_factors_group(results, ARMS_BASELINES, "fig_baselines_t1_v1.png",
                       T_fix=1)


# -------------------------------------------------------------- fig_logloss
def fig_logloss(results):
    """Regret (cumulative prequential loss minus best-fixed-in-hindsight loss)
    on a log n_T axis -- logarithmic growth = straight line."""
    from matplotlib.ticker import ScalarFormatter
    from analyze_joint_v1 import hindsight_loss
    F1, F2, K = 10, 3, 5
    incs = [0, 50]
    n_tab = 5  # tables common to the BAYES and FTRL arms
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
            loss = np.array(arms[arm]['log_loss'], dtype=float)[:, :n_tab]  # steps x tab x 2
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
    print("Wrote paper_figs/fig_logloss_v1.png")
    for inc in incs:
        h = gaps[(inc, 'hindsight', 'final')]
        for arm in ['FTRL_DOR_ADJJ1', 'BAYES_DOR_ADJJ1']:
            f = gaps[(inc, arm, 'final')]
            rel = f" ({(f - h) / h * 100:.1f}% of hindsight)" if h > 1 else " (hindsight ~ 0: separable)"
            print(f"  Inc={inc}: {arm} final cum loss {f:.2f}, hindsight {h:.2f}, "
                  f"regret {f - h:.2f}{rel}")


# -------------------------------------------------------------- fig_sets_tau
def fig_sets_tau():
    rows = []
    with open("sets_tau_comparison.csv") as f:
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
    print("Wrote paper_figs/fig_sets_tau_v1.png")


# ------------------------------------------------------------- text numbers
def diagnostics(results):
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
    files = glob.glob(os.path.join("samples", "grid_*", "f1_*_f2_*_nsteps_10",
                                   "FTRL_DOR_ADJJ1", "table_*", "10.npy"))
    rng = np.random.default_rng(0)
    sel = rng.choice(len(files), size=min(n_files, len(files)), replace=False)
    norms = [np.abs(np.load(files[i])).sum() for i in sel]
    print(f"FTRL final-step fits: max l1 norm {np.max(norms):.2f} "
          f"(median {np.median(norms):.2f}, n={len(norms)} of {len(files)} files)")


def main():
    os.makedirs(OUT, exist_ok=True)
    results = np.load(NPZ, allow_pickle=True)['results'].item()
    print(f"Configs: {len(results)}")
    fig_factors(results)
    fig_sets_tau()
    diagnostics(results)
    l1_check()
    fig_logloss(results)   # last: pulls in jax for the hindsight fits


if __name__ == "__main__":
    main()
