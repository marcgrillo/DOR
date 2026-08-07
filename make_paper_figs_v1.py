"""
Paper-grade figures for the rewritten experimental section (main_new_base.tex).

Writes into paper_figs/:
  fig_overall_v1.png       - overall mean accuracy bars, FTRL vs BAYES (adjacent)
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
import itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NPZ = "grid_results_joint_v1.npz"
OUT = "paper_figs"
METRICS = ['asrs', 'asps', 'aios']
MET_LABEL = {'asrs': 'ASR', 'asps': 'ASP', 'aios': 'AIOS'}
T_LEVELS = [1, 3, 5, 10]

STYLE = {
    'FTRL_DOR_ADJJ1':  dict(color='#1f4fd8', ls='-',  marker='o', label='FTRL-DOR'),
    'BAYES_DOR_ADJJ1': dict(color='#e8810c', ls='--', marker='s', label='BAYES-DOR'),
    'SMAA_DOR_ADJJ1':  dict(color='#2e8b57', ls='-.', marker='^', label='SMAA-DOR'),
    'DORLP_DOR_ADJJ1': dict(color='#777777', ls=':',  marker='D', label='DOR (LP)'),
}

plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'legend.fontsize': 9, 'figure.dpi': 100,
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
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    xs = np.arange(len(METRICS))
    for off, arm in [(-0.18, 'FTRL_DOR_ADJJ1'), (0.18, 'BAYES_DOR_ADJJ1')]:
        st = STYLE[arm]
        means, cis = zip(*[mean_ci(pool(results, arm, m)) for m in METRICS])
        ax.bar(xs + off, means, width=0.34, yerr=cis, capsize=3,
               color=st['color'], label=st['label'])
        for x, v in zip(xs + off, means):
            ax.text(x, v + 0.025, f"{v:.3f}", ha='center', fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels([MET_LABEL[m] for m in METRICS])
    ax.set_ylim(0, 1)
    ax.set_ylabel("mean accuracy")
    ax.legend(frameon=False)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_overall_v1.png"), dpi=300)
    plt.close(fig)
    print("Wrote paper_figs/fig_overall_v1.png")


# ---------------------------------------------------------------- fig_factors
def fig_factors(results):
    factors = [('F1', [10, 20, 35, 50], r"$|A|$"),
               ('F2', [3, 5, 7, 9], r"$m$"),
               ('T', T_LEVELS, r"$T$"),
               ('K', [3, 4, 5], r"$k$"),
               ('Inc', [0, 15, 35, 50], r"$|F_{\mathrm{inc}}|$ (%)")]
    fig, axes = plt.subplots(len(METRICS), len(factors), figsize=(13.5, 7.2), sharey='row')
    for r, metric in enumerate(METRICS):
        for c, (fac, levels, lab) in enumerate(factors):
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
                        marker=st['marker'], ms=4, label=st['label'])
                ax.fill_between(levels, means - cis, means + cis,
                                color=st['color'], alpha=0.12)
            ax.set_xticks(levels)
            if r == len(METRICS) - 1:
                ax.set_xlabel(lab)
            if c == 0:
                ax.set_ylabel(MET_LABEL[metric])
            ax.grid(alpha=0.3)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_factors_v1.png"), dpi=300)
    plt.close(fig)
    print("Wrote paper_figs/fig_factors_v1.png")


# ---------------------------------------------------------- fig_baselines_inc
def fig_baselines(results):
    incs = [0, 15, 35, 50]
    arms = ['FTRL_DOR_ADJJ1', 'BAYES_DOR_ADJJ1', 'SMAA_DOR_ADJJ1', 'DORLP_DOR_ADJJ1']
    fig, axes = plt.subplots(1, len(METRICS), figsize=(11.5, 3.3))
    for ax, metric in zip(axes, METRICS):
        for arm in arms:
            st = STYLE[arm]
            stats = [mean_ci(pool(results, arm, metric, Inc=inc)) for inc in incs]
            means = np.array([s[0] for s in stats])
            cis = np.array([s[1] for s in stats])
            ax.plot(incs, means, color=st['color'], ls=st['ls'],
                    marker=st['marker'], ms=4.5, label=st['label'])
            ax.fill_between(incs, means - cis, means + cis, color=st['color'], alpha=0.12)
        ax.set_xticks(incs)
        ax.set_xlabel(r"$|F_{\mathrm{inc}}|$ (%)")
        ax.set_title(MET_LABEL[metric])
        ax.grid(alpha=0.3)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_baselines_inc_v1.png"), dpi=300)
    plt.close(fig)
    print("Wrote paper_figs/fig_baselines_inc_v1.png")


# -------------------------------------------------------------- fig_logloss
def fig_logloss(results):
    from analyze_joint_v1 import hindsight_loss
    F1, F2, K = 10, 3, 5
    incs = [0, 50]
    n_tab = 5  # tables common to the BAYES and FTRL arms
    fig, axes = plt.subplots(1, len(incs), figsize=(9.2, 3.4), sharex=True)
    gaps = {}
    for ax, inc in zip(axes, incs):
        cs = f"F1={F1}_F2={F2}_S=10_AR={K}_K={K}_Inc={inc}"
        arms = results.get(cs, {})
        n_t = None
        for arm in ['FTRL_DOR_ADJJ1', 'BAYES_DOR_ADJJ1']:
            st = STYLE[arm]
            loss = np.array(arms[arm]['log_loss'], dtype=float)[:, :n_tab]  # steps x tab x 2
            cum = np.cumsum(loss[:, :, 0], axis=0)
            n_t = np.cumsum(loss[:, :, 1], axis=0).mean(axis=1)
            m = cum.mean(axis=1)
            se = cum.std(axis=1) / np.sqrt(n_tab)
            ax.plot(n_t, m, color=st['color'], ls=st['ls'], marker=st['marker'],
                    ms=4, label=st['label'])
            ax.fill_between(n_t, m - 1.96 * se, m + 1.96 * se, color=st['color'], alpha=0.12)
            gaps[(inc, arm, 'final')] = m[-1]
        hs = hindsight_loss(f"grid_{cs}", F1, F2, 10, n_tab, 'adjacent')
        ax.plot(n_t, hs, color='k', ls=':', lw=1.8, label='best fixed in hindsight')
        gaps[(inc, 'hindsight', 'final')] = hs[-1]
        ax.set_title(rf"$|F_{{\mathrm{{inc}}}}| = {inc}\%$")
        ax.set_xlabel(r"$n_T$ (cumulative gap observations)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("cumulative log-loss")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_logloss_v1.png"), dpi=300)
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

    fig, axes = plt.subplots(1, len(METRICS), figsize=(11.5, 3.5))
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
                    ms=4.5, label=st['label'])
            ax.fill_between(taus, means - cis, means + cis, color=st['color'], alpha=0.12)
        ax.set_xticks(taus)
        ax.set_xlabel(r"merge threshold $\tau$")
        ax.set_title(MET_LABEL[metric])
        ax.grid(alpha=0.3)
        sec = ax.secondary_xaxis('top')
        sec.set_xticks(taus)
        sec.set_xticklabels([f"{slev[t]:.1f}" for t in taus], fontsize=8)
        if metric == 'asrs':
            sec.set_xlabel(r"mean levels per session $\bar{s}$", fontsize=8)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_sets_tau_v1.png"), dpi=300)
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
