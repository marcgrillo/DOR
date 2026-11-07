"""
Regret figure (fig_logloss) redone for the cell F1=50, F2=9, AR=K=3, T=10
(hence C_t = k-1 = 2 gaps/session, n_T = 2T = 20 gap observations).

Same construction as make_paper_figs_v1.fig_logloss: cumulative prequential
log-loss of the two adjacent arms minus the best-fixed-in-hindsight cumulative
loss (unpenalized ML fit on all pooled gap observations, analyze_joint_v1.
hindsight_loss), on a log n_T axis. Uses all 20 replications (both arms hold
20 tables for this cell since the 2026-08-10 promotion).

Run from DOR/:  C:\\Users\\marco\\anaconda3\\python.exe redo_logloss_f50_f9_k3.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from analyze_joint_v1 import hindsight_loss
from make_paper_figs_v1 import NPZ, OUT, DPI, STYLE

F1, F2, K = 50, 9, 3
STEPS = 10
N_TAB = 20
INCS = [0, 50]
FNAME = "fig_logloss_f50_f9_k3_v1.png"


def main():
    results = np.load(NPZ, allow_pickle=True)['results'].item()
    os.makedirs(OUT, exist_ok=True)
    fig, axes = plt.subplots(1, len(INCS), figsize=(14, 5.6))
    gaps = {}
    for ax, inc in zip(axes, INCS):
        cs = f"F1={F1}_F2={F2}_S={STEPS}_AR={K}_K={K}_Inc={inc}"
        arms = results[cs]
        hs_curves = hindsight_loss(f"grid_{cs}", F1, F2, STEPS, N_TAB, 'adjacent',
                                   return_curves=True)   # n_tab x steps
        gaps[(inc, 'hindsight', 'final')] = hs_curves.mean(axis=0)[-1]
        n_t, reg_top, ftrl_rm = None, 0.0, None
        for arm in ['FTRL_DOR_ADJJ1', 'BAYES_DOR_ADJJ1']:
            st = STYLE[arm]
            loss = np.array(arms[arm]['log_loss'], dtype=float)[:, :N_TAB]
            cum = np.cumsum(loss[:, :, 0], axis=0)
            n_t = np.cumsum(loss[:, :, 1], axis=0).mean(axis=1)
            gaps[(inc, arm, 'final')] = cum.mean(axis=1)[-1]
            reg = cum - hs_curves.T                       # steps x tab
            rm = reg.mean(axis=1)
            rse = reg.std(axis=1) / np.sqrt(N_TAB)
            ax.plot(n_t, rm, color=st['color'], ls=st['ls'], marker=st['marker'],
                    label=st['label'])
            ax.fill_between(n_t, rm - 1.96 * rse, rm + 1.96 * rse,
                            color=st['color'], alpha=0.15)
            reg_top = max(reg_top, np.max(rm + 1.96 * rse))
            if arm == 'FTRL_DOR_ADJJ1':
                ftrl_rm = rm
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
        ax.set_xticks([2, 4, 8, 20])
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
    fig.savefig(os.path.join(OUT, FNAME), dpi=DPI)
    plt.close(fig)
    print(f"Wrote paper_figs/{FNAME}")
    for inc in INCS:
        h = gaps[(inc, 'hindsight', 'final')]
        for arm in ['FTRL_DOR_ADJJ1', 'BAYES_DOR_ADJJ1']:
            f = gaps[(inc, arm, 'final')]
            rel = f" ({(f - h) / h * 100:.1f}% of hindsight)" if h > 1 else " (hindsight ~ 0: separable)"
            print(f"  Inc={inc}: {arm} final cum loss {f:.2f}, hindsight {h:.2f}, "
                  f"regret {f - h:.2f}{rel}")


if __name__ == "__main__":
    main()
