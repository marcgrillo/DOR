"""
Paper-grade figures for the rewritten experimental section (main_new_base.tex),
styled to match the June paper figures (fig_overall.png / fig_factors.png):
large fonts, framed legends outside on top, descriptive column titles, bold
row labels, dashed grid.

Writes into paper_figs/:
  fig_overall_v1.png       - overall mean-metric bars, FTRL vs BAYES (adjacent)
  fig_factors_v1.png       - 3x5 marginal-effect grid over the five factors
  fig_baselines_inc_v1.png - ASR/ASP/AIOS vs inconsistency for FTRL/BAYES/SMAA/LP
  fig_logloss_v1.png       - cumulative prequential log-loss vs n_T with the
                             best-fixed-in-hindsight comparator (Inc = 0 / 50)
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
MET_LABEL = {'asrs': 'ASR', 'asps': 'ASP', 'aios': 'AIOS'}
T_LEVELS = [1, 3, 5, 10]
DPI = 200

STYLE = {
    'FTRL_DOR_ADJJ1':  dict(color='#1f4fd8', ls='-',  marker='o', label='FTRL-DOR'),
    'BAYES_DOR_ADJJ1': dict(color='#e8810c', ls='--', marker='s', label='BAYES-DOR'),
    'SMAA_DOR_ADJJ1':  dict(color='#2e8b57', ls='-.', marker='^', label='SMAA-DOR'),
    'DORLP_DOR_ADJJ1': dict(color='#777777', ls=':',  marker='D', label='DOR (LP)'),
}

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


# ---------------------------------------------------------------- fig_overall
def fig_overall(results):
    fig, ax = plt.subplots(figsize=(11, 6.2))
    xs = np.arange(len(METRICS))
    for off, arm in [(-0.18, 'FTRL_DOR_ADJJ1'), (0.18, 'BAYES_DOR_ADJJ1')]:
        st = STYLE[arm]
        means, cis = zip(*[mean_ci(pool(results, arm, m)) for m in METRICS])
        ax.bar(xs + off, means, width=0.34, yerr=cis, capsize=5,
               color=st['color'], label=st['label'],
               error_kw=dict(lw=1.8, capthick=1.8))
        for x, v in zip(xs + off, means):
            ax.text(x, v + 0.03, f"{v:.3f}", ha='center', fontsize=17,
                    fontweight='bold', color=st['color'])
    ax.set_xticks(xs)
    ax.set_xticklabels([MET_LABEL[m] for m in METRICS], fontsize=18)
    ax.tick_params(axis='y', labelsize=17)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean metric value", fontsize=19)
    ax.legend(loc='upper left', fontsize=18, framealpha=0.95)
    ax.grid(axis='y')
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_overall_v1.png"), dpi=DPI)
    plt.close(fig)
    print("Wrote paper_figs/fig_overall_v1.png")


# ---------------------------------------------------------------- fig_factors
def fig_factors(results):
    factors = [('F1', [10, 20, 35, 50], r"Number of alternatives $|A|$"),
               ('F2', [3, 5, 7, 9], r"Number of criteria $m$"),
               ('T', T_LEVELS, r"Elicitation steps $T$"),
               ('K', [3, 4, 5], r"Subset size $k$"),
               ('Inc', [0, 15, 35, 50], "Inconsistency (%)")]
    fig, axes = plt.subplots(len(METRICS), len(factors), figsize=(19.9, 10),
                             sharey='row')
    for r, metric in enumerate(METRICS):
        for c, (fac, levels, title) in enumerate(factors):
            ax = axes[r, c]
            for arm in ['FTRL_DOR_ADJJ1', 'BAYES_DOR_ADJJ1']:
                st = STYLE[arm]
                if fac == 'T':
                    stats = [mean_ci(pool(results, arm, metric, T=lv)) for lv in levels]
                else:
                    stats = [mean_ci(pool(results, arm, metric, **{fac: lv})) for lv in levels]
                means = np.array([s[0] for s in stats])
                cis = np.array([s[1] for s in stats])
                ax.plot(levels, means, color=st['color'], ls=st['ls'],
                        marker=st['marker'], label=st['label'])
                ax.fill_between(levels, means - cis, means + cis,
                                color=st['color'], alpha=0.15)
            ax.set_xticks(levels)
            ax.tick_params(labelleft=True)
            if r == 0:
                ax.set_title(title, fontsize=16, pad=8)
            if c == 0:
                ax.set_ylabel(MET_LABEL[metric], fontsize=18, fontweight='bold')
            ax.grid(True)
            ax.set_axisbelow(True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0),
               ncol=2, fontsize=17, frameon=True, columnspacing=2.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT, "fig_factors_v1.png"), dpi=DPI)
    plt.close(fig)
    print("Wrote paper_figs/fig_factors_v1.png")


# ---------------------------------------------------------- fig_baselines_inc
def fig_baselines(results):
    incs = [0, 15, 35, 50]
    arms = ['FTRL_DOR_ADJJ1', 'BAYES_DOR_ADJJ1', 'SMAA_DOR_ADJJ1', 'DORLP_DOR_ADJJ1']
    fig, axes = plt.subplots(1, len(METRICS), figsize=(16.5, 5.0))
    for ax, metric in zip(axes, METRICS):
        for arm in arms:
            st = STYLE[arm]
            stats = [mean_ci(pool(results, arm, metric, Inc=inc)) for inc in incs]
            means = np.array([s[0] for s in stats])
            cis = np.array([s[1] for s in stats])
            ax.plot(incs, means, color=st['color'], ls=st['ls'],
                    marker=st['marker'], label=st['label'])
            ax.fill_between(incs, means - cis, means + cis, color=st['color'], alpha=0.15)
        ax.set_xticks(incs)
        ax.set_xlabel("Inconsistency (%)")
        ax.set_title(MET_LABEL[metric], fontsize=18, pad=8)
        ax.grid(True)
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0),
               ncol=4, fontsize=16, frameon=True, columnspacing=2.0)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(os.path.join(OUT, "fig_baselines_inc_v1.png"), dpi=DPI)
    plt.close(fig)
    print("Wrote paper_figs/fig_baselines_inc_v1.png")


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
    fig_overall(results)
    fig_factors(results)
    fig_baselines(results)
    fig_sets_tau()
    diagnostics(results)
    l1_check()
    fig_logloss(results)   # last: pulls in jax for the hindsight fits


if __name__ == "__main__":
    main()
