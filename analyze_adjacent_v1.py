"""
Analysis / export for the adjacent-v1 campaign (grid_results_adjacent_v1.npz).

Produces (into figs/adjacent_v1/ and CSV at repo root):
  1. grid_results_adjacent_v1.csv  - long-format export incl. per-step log-loss and runtimes
  2. evolution_<metric>.png        - 4-arm step evolution per Inc level (pilot slice F1=10, F2=3, AR=K=3)
  3. logloss_cumulative.png        - cumulative prequential log-loss vs n_T (plan C14), with
                                     best-fixed-in-hindsight benchmark on the pilot cells
  4. fig_overall_v1.png / fig_factors_v1.png - paper-style figures for the ADJACENT arms
                                     (AR==K, T in {1,3,5,10}) over whatever configs are present

Usage:  C:\\Users\\marco\\anaconda3\\python.exe analyze_adjacent_v1.py [--no-hindsight]
"""
import os
import re
import sys
import csv
import itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NPZ = "grid_results_adjacent_v1.npz"
OUT_DIR = os.path.join("figs", "adjacent_v1")
CSV_PATH = "grid_results_adjacent_v1.csv"
ARMS = ['FTRL_DOR_ADJV1', 'BAYES_DOR_ADJV1', 'FTRL_DOR_ALLV1', 'BAYES_DOR_ALLV1']
ARM_LABEL = {
    'FTRL_DOR_ADJV1': 'FTRL-DOR (adjacent)',
    'BAYES_DOR_ADJV1': 'BAYES-DOR (adjacent)',
    'FTRL_DOR_ALLV1': 'FTRL-DOR (all-pairs)',
    'BAYES_DOR_ALLV1': 'BAYES-DOR (all-pairs)',
}
ARM_STYLE = {
    'FTRL_DOR_ADJV1': dict(color='#1f4fd8', ls='-', marker='o'),
    'BAYES_DOR_ADJV1': dict(color='#e8810c', ls='-', marker='s'),
    'FTRL_DOR_ALLV1': dict(color='#1f4fd8', ls='--', marker='o', alpha=0.55),
    'BAYES_DOR_ALLV1': dict(color='#e8810c', ls='--', marker='s', alpha=0.55),
}
METRICS = ['asrs', 'asps', 'aios']
PILOT = dict(F1=10, F2=3, AR=3, K=3)
ETA_MAX = 5


def parse_config(config_str):
    m = re.match(r"F1=(\d+)_F2=(\d+)_S=(\d+)_AR=(\d+)_K=(\d+)_Inc=(\d+)", config_str)
    return {k: int(v) for k, v in zip(['F1', 'F2', 'S', 'AR', 'K', 'Inc'], m.groups())}


def load_results():
    if not os.path.exists(NPZ):
        sys.exit(f"{NPZ} not found - run run_grid_adjacent_v1.py first")
    return np.load(NPZ, allow_pickle=True)['results'].item()


def export_csv(results):
    with open(CSV_PATH, 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['Method', 'F1', 'F2', 'S', 'AR', 'K', 'Inc', 'Step', 'Table',
                     'ASRS', 'ASPS', 'AIOS', 'BatchLogLoss', 'BatchSize',
                     'Divergences', 'MaxRhat', 'ArmRuntime_s'])
        n_rows = 0
        for config_str, arms in results.items():
            cfg = parse_config(config_str)
            for sub_fold, md in arms.items():
                asrs = np.array(md['asrs']); asps = np.array(md['asps'])
                aios = np.array(md['aios'])
                loss = np.array(md.get('log_loss', []), dtype=float)
                diag = np.array(md.get('hmc_diag', []), dtype=float)
                steps, tables = asrs.shape
                for j, i in itertools.product(range(steps), range(tables)):
                    lo = loss[j, i] if loss.size else (np.nan, np.nan)
                    dg = diag[j, i] if diag.size else (np.nan, np.nan)
                    wr.writerow([sub_fold, cfg['F1'], cfg['F2'], cfg['S'], cfg['AR'], cfg['K'],
                                 cfg['Inc'], j + 1, i, asrs[j, i], asps[j, i], aios[j, i],
                                 lo[0], lo[1], dg[0], dg[1],
                                 md.get('runtime_s', np.nan)])
                    n_rows += 1
    print(f"Wrote {CSV_PATH} ({n_rows} rows)")


def plot_evolution(results):
    """4-arm step evolution per metric, pilot slice, one column per Inc level."""
    incs = [0, 15, 35, 50]
    for metric in METRICS:
        fig, axes = plt.subplots(1, len(incs), figsize=(4 * len(incs), 3.2), sharey=True)
        found = False
        for ax, inc in zip(axes, incs):
            config_str = (f"F1={PILOT['F1']}_F2={PILOT['F2']}_S=10_"
                          f"AR={PILOT['AR']}_K={PILOT['K']}_Inc={inc}")
            arms = results.get(config_str, {})
            for sub_fold in ARMS:
                if sub_fold not in arms:
                    continue
                found = True
                data = np.array(arms[sub_fold][metric])  # steps x tables
                mean = data.mean(axis=1)
                se = data.std(axis=1) / np.sqrt(data.shape[1])
                x = np.arange(1, data.shape[0] + 1)
                st = ARM_STYLE[sub_fold]
                ax.plot(x, mean, label=ARM_LABEL[sub_fold], **st)
                ax.fill_between(x, mean - 1.96 * se, mean + 1.96 * se,
                                color=st['color'], alpha=0.12)
            ax.set_title(f"Inc = {inc}%")
            ax.set_xlabel("step $t$")
            ax.grid(alpha=0.3)
        if not found:
            plt.close(fig)
            continue
        axes[0].set_ylabel(metric.upper())
        axes[0].legend(fontsize=8)
        fig.suptitle(f"adjacent-v1 pilot slice (F1={PILOT['F1']}, F2={PILOT['F2']}, "
                     f"AR=K={PILOT['K']}): {metric.upper()}")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f"evolution_{metric}.png"), dpi=150)
        plt.close(fig)
        print(f"Wrote {OUT_DIR}/evolution_{metric}.png")


def hindsight_loss(dataset_fold, f1, f2, steps, hm, expansion):
    """Best-fixed-in-hindsight cumulative loss per step (mean over tables):
    ML fit (no prior) on ALL steps' data, then its cumulative batch losses."""
    from scipy.optimize import minimize as sp_minimize
    import jax, jax.numpy as jnp
    from common import utils
    from mcda.models import PiecewiseLinearTransformer
    from inference.engine import _jax_log_prob_per_obs, RHO
    from experiments.simulation import expand_subrank

    tables, rankings, dm_prefs, _ = utils.read_dataset(
        os.path.join("datasets", dataset_fold), f1, f2, steps)
    curves = []
    for i in range(min(hm, len(tables))):
        transformer = PiecewiseLinearTransformer.from_equal_intervals(tables[i], num_intervals=3)
        Xf = transformer.transform(tables[i])
        n_params = transformer.total_params
        batches = []
        for j in range(steps):
            pairs = expand_subrank(dm_prefs[i][j]['items'], dm_prefs[i][j]['cards'], expansion)
            X = Xf[[p[0] for p in pairs]] - Xf[[p[1] for p in pairs]]
            batches.append((jnp.array(X), jnp.array([p[2] for p in pairs])))
        X_all = jnp.concatenate([b[0] for b in batches])
        e_all = jnp.concatenate([b[1] for b in batches])
        ndim = n_params + ETA_MAX

        def neg_ll(om):
            return -jnp.sum(_jax_log_prob_per_obs(jnp.array(om), X_all, e_all,
                                                  n_params, ETA_MAX, RHO))
        val_grad = jax.value_and_grad(neg_ll)

        def neg_ll_np(om):
            v, g = val_grad(jnp.array(om))
            return float(v), np.array(g, dtype=np.float64)

        res = sp_minimize(neg_ll_np, np.ones(ndim) * 0.5, method='L-BFGS-B', jac=True,
                          bounds=[(1e-6, None)] * ndim)
        om_star = jnp.array(res.x)
        per_step = [float(-jnp.sum(_jax_log_prob_per_obs(om_star, X, e, n_params, ETA_MAX, RHO)))
                    for X, e in batches]
        curves.append(np.cumsum(per_step))
    return np.mean(curves, axis=0)


def plot_logloss(results, with_hindsight=True):
    incs = [0, 15, 35, 50]
    fig, axes = plt.subplots(1, len(incs), figsize=(4 * len(incs), 3.4), sharey=False)
    found = False
    for ax, inc in zip(axes, incs):
        config_str = (f"F1={PILOT['F1']}_F2={PILOT['F2']}_S=10_"
                      f"AR={PILOT['AR']}_K={PILOT['K']}_Inc={inc}")
        dataset_fold = f"grid_{config_str}"
        arms = results.get(config_str, {})
        for sub_fold in ARMS:
            md = arms.get(sub_fold, {})
            loss = np.array(md.get('log_loss', []), dtype=float)  # steps x tables x 2
            if loss.size == 0 or np.all(np.isnan(loss[:, :, 0])):
                continue
            found = True
            cum = np.nancumsum(loss[:, :, 0], axis=0)          # cumulative loss per table
            n_t = np.nancumsum(loss[:, :, 1], axis=0)          # cumulative #observations
            st = ARM_STYLE[sub_fold]
            ax.plot(np.nanmean(n_t, axis=1), np.nanmean(cum, axis=1),
                    label=ARM_LABEL[sub_fold], **st)
            if with_hindsight and sub_fold.endswith('ADJV1') and sub_fold.startswith('FTRL'):
                try:
                    hs = hindsight_loss(dataset_fold, PILOT['F1'], PILOT['F2'], 10,
                                        int(loss.shape[1]), 'adjacent')
                    ax.plot(np.nanmean(n_t, axis=1), hs, color='k', ls=':',
                            label='best fixed in hindsight (adjacent)')
                except Exception as e:
                    print(f"hindsight failed for Inc={inc}: {e}")
        ax.set_title(f"Inc = {inc}%")
        ax.set_xlabel("$n_T$ (cumulative observations)")
        ax.grid(alpha=0.3)
    if not found:
        plt.close(fig)
        print("No log-loss data yet - skipping logloss figure")
        return
    axes[0].set_ylabel("cumulative predictive log-loss")
    axes[0].legend(fontsize=7)
    fig.suptitle("Prequential cumulative log-loss (plan C14), pilot slice")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "logloss_cumulative.png"), dpi=150)
    plt.close(fig)
    print(f"Wrote {OUT_DIR}/logloss_cumulative.png")


def paper_figs(results):
    """fig_overall / fig_factors style for the ADJACENT arms (AR==K, T in {1,3,5,10})."""
    rows = []
    for config_str, arms in results.items():
        cfg = parse_config(config_str)
        if cfg['AR'] != cfg['K']:
            continue
        for sub_fold in ['FTRL_DOR_ADJV1', 'BAYES_DOR_ADJV1']:
            if sub_fold not in arms:
                continue
            md = arms[sub_fold]
            for metric in METRICS:
                data = np.array(md[metric])
                for T in [1, 3, 5, 10]:
                    if T <= data.shape[0]:
                        for v in data[T - 1]:
                            rows.append((sub_fold, metric, cfg['F1'], cfg['F2'], T,
                                         cfg['K'], cfg['Inc'], v))
    if not rows:
        print("No AR==K adjacent data yet - skipping paper figs")
        return
    arr = np.array(rows, dtype=object)

    def sel(method, metric, **kv):
        m = (arr[:, 0] == method) & (arr[:, 1] == metric)
        col = {'F1': 2, 'F2': 3, 'T': 4, 'K': 5, 'Inc': 6}
        for k, v in kv.items():
            m &= (arr[:, col[k]] == v)
        return arr[m, 7].astype(float)

    # Overall bars
    fig, ax = plt.subplots(figsize=(6, 3.6))
    xs = np.arange(len(METRICS))
    for off, method, color in [(-0.18, 'FTRL_DOR_ADJV1', '#1f4fd8'),
                               (0.18, 'BAYES_DOR_ADJV1', '#e8810c')]:
        means = [sel(method, m).mean() for m in METRICS]
        cis = [1.96 * sel(method, m).std() / np.sqrt(len(sel(method, m))) for m in METRICS]
        ax.bar(xs + off, means, width=0.34, yerr=cis, capsize=3,
               color=color, label=ARM_LABEL[method])
        for x, v in zip(xs + off, means):
            ax.text(x, v + 0.02, f"{v:.3f}", ha='center', fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels([m.upper() for m in METRICS])
    ax.set_ylim(0, 1); ax.legend(); ax.grid(axis='y', alpha=0.3)
    ax.set_title("adjacent-v1: overall mean accuracy (AR=K, T in {1,3,5,10})")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_overall_v1.png"), dpi=150)
    plt.close(fig)
    print(f"Wrote {OUT_DIR}/fig_overall_v1.png")

    # Factor grid
    factors = [('F1', [10, 20, 35, 50]), ('F2', [3, 5, 7, 9]), ('T', [1, 3, 5, 10]),
               ('K', [3, 4, 5]), ('Inc', [0, 15, 35, 50])]
    fig, axes = plt.subplots(len(METRICS), len(factors), figsize=(16, 8), sharey='row')
    for r, metric in enumerate(METRICS):
        for c, (fac, levels) in enumerate(factors):
            ax = axes[r, c]
            for method, st in [('FTRL_DOR_ADJV1', ARM_STYLE['FTRL_DOR_ADJV1']),
                               ('BAYES_DOR_ADJV1', ARM_STYLE['BAYES_DOR_ADJV1'])]:
                means, cis, lv_present = [], [], []
                for lv in levels:
                    vals = sel(method, metric, **{fac: lv})
                    if len(vals):
                        lv_present.append(lv)
                        means.append(vals.mean())
                        cis.append(1.96 * vals.std() / np.sqrt(len(vals)))
                if means:
                    means = np.array(means); cis = np.array(cis)
                    ax.plot(lv_present, means, **st, label=ARM_LABEL[method])
                    ax.fill_between(lv_present, means - cis, means + cis,
                                    color=st['color'], alpha=0.12)
            if r == 0:
                ax.set_title(fac)
            if r == len(METRICS) - 1:
                ax.set_xlabel(fac)
            if c == 0:
                ax.set_ylabel(metric.upper())
            ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("adjacent-v1: marginal effects (AR=K)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_factors_v1.png"), dpi=150)
    plt.close(fig)
    print(f"Wrote {OUT_DIR}/fig_factors_v1.png")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = load_results()
    print(f"Configs in npz: {len(results)}")
    export_csv(results)
    plot_evolution(results)
    plot_logloss(results, with_hindsight='--no-hindsight' not in sys.argv)
    paper_figs(results)


if __name__ == "__main__":
    main()
