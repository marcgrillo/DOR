"""
Descriptive picture of the synthetic DGP: how the ground-truth utilities and the
utility DIFFERENCES that drive the elicited answers are distributed.

Reads the grid_* datasets (exponential utilities, Dirichlet weights, n_max = 5)
and produces paper_figs/fig_utility_distributions.png with four panels:

  (a) marginal distribution of the ground-truth utilities U, by number of
      criteria f2 (U does not depend on f1);
  (b) all-pairs |U(a) - U(b)| within a decision problem, by f2;
  (c) gap between consecutive alternatives in the TRUE full ranking, by f1
      (this is what shrinks as the problem grows);
  (d) the elicited scale: n_max * gap between consecutive alternatives of a
      presented subset (what the card number eta encodes), with the n_max
      ceiling marked, next to the empirical distribution of the declared cards.

Run from DOR/:  C:\\Users\\marco\\anaconda3\\python.exe analyze_utility_distributions.py
"""
import os
import re
import json
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = "paper_figs"
DPI = 200
N_MAX = 5

F2_LEVELS = [3, 5, 7, 9]
F1_LEVELS = [10, 20, 35, 50]
INC_LEVELS = [0, 15, 35, 50]

# Okabe-Ito, same family as the other paper figures.
SEQ = ['#0072B2', '#E69F00', '#009E73', '#CC79A7']

plt.rcParams.update({
    'font.size': 13,
    'axes.titlesize': 15,
    'axes.labelsize': 13,
    'legend.fontsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
})

PAT = re.compile(r"grid_F1=(\d+)_F2=(\d+)_S=(\d+)_AR=(\d+)_K=(\d+)_Inc=(\d+)$")


def dataset_index(num_steps=10):
    """Maps (f1, f2, ar, k, inc) -> json path for the grid folds with S=num_steps.

    Legacy folds with S in {1, 3, 5} exist alongside the campaign ones; they must
    not collide with (or silently replace) the S=10 folds actually used.
    """
    idx = {}
    for d in glob.glob(os.path.join("datasets", "grid_*")):
        m = PAT.search(os.path.basename(d))
        if not m:
            continue
        f1, f2, s, ar, k, inc = (int(x) for x in m.groups())
        if s != num_steps:
            continue
        js = os.path.join(d, f"f1_{f1}__f2_{f2}__nsteps_{s}.json")
        if os.path.exists(js):
            idx[(f1, f2, ar, k, inc)] = js
    return idx


def collect():
    """Pools every S=10 grid fold.

    The utilities are drawn independently in each fold, and the inconsistency
    level only perturbs the DECLARED subranks - never the ground truth - so all
    (AR, K, Inc) folds are exchangeable draws of the same utility DGP and can be
    pooled for the truth-side panels.  Only the declared cards are stratified by
    inconsistency level.
    """
    idx = dataset_index()
    utils_by_f2 = {f2: [] for f2 in F2_LEVELS}
    allpairs_by_f2 = {f2: [] for f2 in F2_LEVELS}
    rankgap_by_f1 = {f1: [] for f1 in F1_LEVELS}
    subgap_by_f1 = {f1: [] for f1 in F1_LEVELS}      # scaled by n_max
    cards_by_inc = {inc: [] for inc in INC_LEVELS}
    n_tables = 0

    for (f1, f2, ar, k, inc), path in sorted(idx.items()):
        if f1 not in F1_LEVELS or f2 not in F2_LEVELS or inc not in INC_LEVELS:
            continue    # legacy folds outside the campaign grid (e.g. f2 = 4)
        with open(path) as fh:
            data = json.load(fh)
        for run in data['runs']:
            U = np.asarray(run['true_utilities'], dtype=float)
            n_tables += 1
            utils_by_f2[f2].append(U)
            d = np.abs(U[:, None] - U[None, :])
            allpairs_by_f2[f2].append(d[np.triu_indices(len(U), 1)])
            Us = np.sort(U)[::-1]
            rankgap_by_f1[f1].append(Us[:-1] - Us[1:])
            for sr in run['subranks']:
                items = np.asarray(sr['items'], dtype=int)
                # true gaps between alternatives shown together, in true-utility
                # order (independent of what the noisy DM declared)
                u = np.sort(U[items])[::-1]
                subgap_by_f1[f1].append((u[:-1] - u[1:]) * N_MAX)
                cards_by_inc[inc].append(np.asarray(sr['cards'], dtype=float))

    cat = lambda d: {k: (np.concatenate(v) if v else np.array([])) for k, v in d.items()}
    return (cat(utils_by_f2), cat(allpairs_by_f2), cat(rankgap_by_f1),
            cat(subgap_by_f1), cat(cards_by_inc), n_tables)


def stepped_hist(ax, x, bins, color, label, ls='-'):
    h, e = np.histogram(x, bins=bins, density=True)
    ax.step(np.r_[e[0], e[:-1], e[-1]], np.r_[0, h, 0], where='post',
            color=color, lw=2, ls=ls, label=label)


def describe(name, x):
    q = np.percentile(x, [5, 25, 50, 75, 95])
    print(f"  {name:>10s}  n={len(x):>8d}  mean={x.mean():.4f}  sd={x.std():.4f}  "
          f"q05={q[0]:.4f} q25={q[1]:.4f} med={q[2]:.4f} q75={q[3]:.4f} q95={q[4]:.4f}")


def main():
    os.makedirs(OUT, exist_ok=True)
    U2, AP2, RG1, SG1, CD, n_tables = collect()

    print(f"decision problems read: {n_tables}\n")
    print("ground-truth utilities U, by number of criteria f2:")
    for f2 in F2_LEVELS:
        describe(f"f2={f2}", U2[f2])
    print("\nall-pairs |U(a)-U(b)| within a problem, by f2:")
    for f2 in F2_LEVELS:
        describe(f"f2={f2}", AP2[f2])
    print("\ngap between consecutive alternatives in the true ranking, by f1:")
    for f1 in F1_LEVELS:
        describe(f"f1={f1}", RG1[f1])
    print(f"\nscaled gap n_max*(u_a - u_b) between consecutive alternatives of a\n"
          f"presented subset (n_max = {N_MAX}), by f1:")
    for f1 in F1_LEVELS:
        x = SG1[f1]
        describe(f"f1={f1}", x)
        print(f"              P(gap > n_max) = {np.mean(x > N_MAX):.4f}   "
              f"P(rounds to 0) = {np.mean(x < 0.5):.4f}")
    print("\ndeclared cards eta, by inconsistency level:")
    for inc in INC_LEVELS:
        x = CD[inc]
        describe(f"Inc={inc}%", x)
        print(f"              P(eta = 0) = {np.mean(x == 0):.4f}   "
              f"P(eta = n_max) = {np.mean(x == N_MAX):.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))

    # (a) utilities
    ax = axes[0, 0]
    bins = np.linspace(0, 1, 41)
    for c, f2 in zip(SEQ, F2_LEVELS):
        stepped_hist(ax, U2[f2], bins, c, f"$f_2$ = {f2}")
    ax.set_xlabel("ground-truth utility $u(a)$")
    ax.set_ylabel("density")
    ax.set_title("(a)  Utilities, by number of criteria")
    ax.legend(frameon=True, framealpha=0.95)

    # (b) all-pairs differences
    ax = axes[0, 1]
    bins = np.linspace(0, 1, 61)
    for c, f2 in zip(SEQ, F2_LEVELS):
        stepped_hist(ax, AP2[f2], bins, c, f"$f_2$ = {f2}")
    ax.set_xlabel(r"$|u(a) - u(b)|$, all pairs in a problem")
    ax.set_ylabel("density")
    ax.set_title("(b)  Utility differences, by number of criteria")
    ax.legend(frameon=True, framealpha=0.95)

    # (c) consecutive gaps in the full true ranking - the mass piles up at zero,
    # so a log axis is the only way to see the four sizes separate.
    ax = axes[1, 0]
    bins = np.logspace(-5, 0, 61)
    for c, f1 in zip(SEQ, F1_LEVELS):
        x = RG1[f1]
        h, e = np.histogram(np.clip(x, bins[0], None), bins=bins)
        ax.step(np.r_[e[0], e[:-1], e[-1]], np.r_[0, h / h.sum(), 0],
                where='post', color=c, lw=2, label=f"$f_1$ = {f1}")
    ax.set_xscale('log')
    ax.set_xlabel("gap between consecutive alternatives in the true ranking")
    ax.set_ylabel("fraction of neighbouring pairs")
    ax.set_title("(c)  How close the neighbours are, by problem size")
    ax.legend(frameon=True, framealpha=0.95)

    # (d) the elicited scale
    ax = axes[1, 1]
    bins = np.linspace(0, 6, 61)
    for c, f1 in zip(SEQ, F1_LEVELS):
        stepped_hist(ax, SG1[f1], bins, c, f"$f_1$ = {f1}")
    top = ax.get_ylim()[1]
    ax.axvspan(0, 0.5, color='0.5', alpha=0.12, lw=0)
    ax.axvline(N_MAX, color='0.35', ls='--', lw=2)
    p0 = np.mean(np.concatenate([SG1[f1] for f1 in F1_LEVELS]) < 0.5)
    ax.annotate(f"rounds to\n" + r"$\eta = 0$" + f"  ({p0:.0%})", xy=(0.62, top * 0.72),
                color='0.3', fontsize=12, va='center')
    ax.annotate(r"$n_{\max} = 5$", xy=(N_MAX - 0.15, top * 0.55), color='0.35',
                fontsize=12, va='center', ha='right')
    ax.set_ylim(0, top)
    ax.set_xlabel(r"$n_{\max}\,[\,u(a) - u(b)\,]$, consecutive pair of a shown subset")
    ax.set_ylabel("density")
    ax.set_title("(d)  What the card number has to encode")
    ax.legend(frameon=True, framealpha=0.95)

    for ax in axes.ravel():
        ax.grid(True, ls='--', alpha=0.3)
        ax.set_axisbelow(True)

    fig.tight_layout()
    p = os.path.join(OUT, "fig_utility_distributions.png")
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    print(f"\nwrote {p}")

    # Companion: declared cards by inconsistency level.
    fig, ax = plt.subplots(figsize=(7, 4.6))
    width = 0.2
    centres = np.arange(N_MAX + 1)
    for i, (c, inc) in enumerate(zip(SEQ, INC_LEVELS)):
        x = CD[inc]
        freq = np.array([np.mean(x == v) for v in centres])
        ax.bar(centres + (i - 1.5) * width, freq, width * 0.9, color=c,
               label=f"{inc}% inconsistency")
    ax.set_xticks(centres)
    ax.set_xlabel(r"declared card number $\eta$")
    ax.set_ylabel("relative frequency")
    ax.set_title("Declared cards, by inconsistency level")
    ax.legend(frameon=True, framealpha=0.95)
    ax.grid(True, axis='y', ls='--', alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    p = os.path.join(OUT, "fig_card_distribution.png")
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
