"""
Case study for the B-DOR paper revision (Section 6): ranking the 21 Italian
regions on healthcare performance (LEA criteria) under the ADJACENT-ONLY
likelihood and the new inference engine.

Structure (2026-08-14 revision, Marco's restructure): BOTH illustration cases
run the same three elicitation sessions (T = 3, cumulative) and differ only in
the marginal utility functions -- case LIN uses gamma_j = 1 (linear), case PW
uses gamma_j = 3 (piecewise linear).  Outputs per case: a corner plot of the
RAW per-criterion totals U_j(100) after each session with the FTRL estimates
overlaid (croner_steps_{lin,pw}.png), FTRL parameter and ranking tables with
one row/column group per session, full RAI/PWI supplementary tables for the
final posterior (table_S1-S4), and RAI/PWI EVOLUTION tables across sessions
for the four top-ranked regions (table_{rai,pwi}_evolution_{lin,pw}.tex).

Original migration-plan items E1-E6 (Decision 1: e_max = 30):
  E1  Data and elicitation sessions hard-coded from main_EJOR_1.tex
      (tab:LEA_data, fig:example_t1, fig:example_t3).
  E2  Responses are RAW card counts on consecutive gaps only: a subrank
      A -e1- B -e2- C contributes (A,B,e1), (B,C,e2).  No nu(a), no +1,
      no summing across gaps (expand_subrank(..., expansion='adjacent')).
  E3  Protocol cap: e_max = n_max = 30 for BOTH the T=1 and the T=3 case,
      so the threshold vector Delta has 30 components in each case and the
      two illustrations share the same response space.  The largest declared
      card count is 28 <= 30 (asserted below).
  E4  After every session (cumulatively) we compute the BAYES-DOR posterior
      via HMC (2000 draws / 1000 warmup, Gamma(1,1) prior) and the FTRL-DOR
      constrained MAP (Gamma(2,1) prior), as implemented in inference/engine.py.
  E5  Outputs (paper_figs/case_study/): marginal-utility figures, corner plot,
      full RAI / PWI supplementary tables (S.1-S.4), FTRL parameter tables,
      ranking tables with DISPLAY scaling U(a)/U_max*100 (U_max = utility of
      the ideal alternative scoring 100 on every criterion, i.e. the sum of
      all piece weights), and a long-format CSV dump.
  E6  RAI / PWI mirror experiments/metrics.py calc_rai / calc_poi exactly
      (only the first n_params components of each posterior draw are used;
      PWI[a,b] = fraction of draws with U(a) >= U(b), hence diagonal = 100).

Feature construction: the paper reports FTRL estimates as U_j evaluated at 0
and 100 (tab:ftrl_ex_t1 / tab:ftrl_ex_t3), i.e. the breakpoints live on the
common [0,100] measurement scale, NOT on each column's [min,max] range.  Since
PiecewiseLinearTransformer.from_equal_intervals uses column min/max, we build
the transformer through its base constructor with explicit per-criterion
breakpoints np.linspace(0, 100, gamma_j + 1)  (gamma_j = 1 for T=1, 3 for T=3).

Run from the DOR root:
  C:\\Users\\marco\\anaconda3\\python.exe case_study_lea.py
"""
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mcda.models import PiecewiseLinearTransformer
from inference.engine import PreferenceSampler
from experiments.simulation import expand_subrank

try:
    import corner as corner_pkg
    HAVE_CORNER = True
except ImportError:
    HAVE_CORNER = False

# ----------------------------------------------------------------------------
# Data (main_EJOR_1.tex, tab:LEA_data) - values in [0, 100], all criteria (up)
# ----------------------------------------------------------------------------
REGIONS = [
    "Piedmont",                          # R1
    "Aosta Valley",                      # R2
    "Lombardy",                          # R3
    "Autonomous Province of Bolzano",    # R4
    "Autonomous Province of Trento",     # R5
    "Veneto",                            # R6
    "Friuli Venezia Giulia",             # R7
    "Liguria",                           # R8
    "Emilia Romagna",                    # R9
    "Tuscany",                           # R10
    "Umbria",                            # R11
    "Marche",                            # R12
    "Lazio",                             # R13
    "Abruzzo",                           # R14
    "Molise",                            # R15
    "Campania",                          # R16
    "Apulia",                            # R17
    "Basilicata",                        # R18
    "Calabria",                          # R19
    "Sicily",                            # R20
    "Sardinia",                          # R21
]
#                          Prevention  Primary  Hospital
LEA_TABLE = np.array([
    [93, 90, 87],   # Piedmont
    [77, 35, 53],   # Aosta Valley
    [95, 76, 86],   # Lombardy
    [58, 82, 62],   # A. P. of Bolzano
    [98, 83, 97],   # A. P. of Trento
    [98, 96, 94],   # Veneto
    [81, 81, 73],   # Friuli Venezia Giulia
    [54, 85, 80],   # Liguria
    [97, 89, 92],   # Emilia Romagna
    [95, 95, 96],   # Tuscany
    [93, 80, 84],   # Umbria
    [74, 83, 91],   # Marche
    [63, 68, 85],   # Lazio
    [54, 45, 83],   # Abruzzo
    [58, 73, 62],   # Molise
    [62, 72, 72],   # Campania
    [74, 69, 85],   # Apulia
    [68, 52, 69],   # Basilicata
    [68, 40, 69],   # Calabria
    [49, 44, 80],   # Sicily
    [65, 67, 60],   # Sardinia
], dtype=float)

IDX = {name: i for i, name in enumerate(REGIONS)}
NUM_REGIONS = len(REGIONS)
CRITERIA = ["Prevention care ($g_1$)", "Primary care ($g_2$)", "Hospital care ($g_3$)"]
# Row labels of the marginal-utility grid: the panels show the NORMALIZED
# marginal utility hat{u}_j, so the criterion is named after its function.
CRITERIA_HAT = [r"$\hat{u}_1(\cdot)$ (Prevention care)",
                r"$\hat{u}_2(\cdot)$ (Primary care)",
                r"$\hat{u}_3(\cdot)$ (Hospital care)"]

# Short display names for the three-column T=3 ranking table (as in the paper).
SHORT = {
    "Autonomous Province of Bolzano": "A. P. of Bolzano",
    "Autonomous Province of Trento": "A. P. of Trento",
}

# ----------------------------------------------------------------------------
# Elicitation sessions (fig:example_t1, fig:example_t3): best -> worst, raw
# card counts on the consecutive gaps (E2).
# ----------------------------------------------------------------------------
STEP_1 = {"items": ["Lombardy", "Marche", "Campania", "Basilicata"],
          "cards": [13, 28, 9]}
STEP_2 = {"items": ["Emilia Romagna", "Friuli Venezia Giulia", "Campania", "Abruzzo"],
          "cards": [21, 17, 13]}
STEP_3 = {"items": ["Tuscany", "Lombardy", "Lazio", "Sicily"],
          "cards": [10, 21, 18]}

# Both illustration cases run the SAME three sessions (T = 3, cumulative);
# they differ only in the marginal utility functions: LIN uses gamma_j = 1
# (linear), PW uses gamma_j = 3 (piecewise linear, equally spaced breakpoints).
SESSIONS = [STEP_1, STEP_2, STEP_3]

# Protocol cap (Decision 1): e_max = n_max = 30 for both cases (E3).
E_MAX = 30
_all_cards = [c for s in SESSIONS for c in s["cards"]]
assert max(_all_cards) <= E_MAX, \
    f"declared card count {max(_all_cards)} exceeds the protocol cap e_max={E_MAX}"

SEED_BASE = 20260806
NUM_SAMPLES, NUM_WARMUP = 2000, 1000

OUT_DIR = os.path.join("paper_figs", "case_study")
os.makedirs(OUT_DIR, exist_ok=True)

# Figure colors: paper convention (captions of fig:example_t1_posterior_sample):
# blue = BAYES posterior band, red = FTRL (MAP) estimate.
C_BAYES_FILL = "#1f77b4"
C_BAYES_LINE = "#1f4fd8"
C_FTRL = "#d62728"


# ----------------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------------
def build_transformer(gamma):
    """
    Paper framework: the evaluation table is augmented with two ANCHOR
    alternatives -- an ideal one scoring 100 on every criterion and an
    anti-ideal one scoring 0 -- so that the equal-interval breakpoints span
    the full [0,100] measurement scale regardless of the observed data range.
    The anchors are used ONLY here (normalization); they never enter the
    elicitation data, the rankings, or the RAI/PWI tables.

    Since from_equal_intervals takes linspace(min, max, gamma+1) per column,
    the augmented table yields breakpoints linspace(0, 100, gamma+1) -- i.e.
    exactly the explicit [0,100] breakpoints this script used before; the two
    constructions are mathematically identical (asserted below).
    """
    anchors = np.array([[100.0] * LEA_TABLE.shape[1],
                        [0.0] * LEA_TABLE.shape[1]])
    augmented = np.vstack([LEA_TABLE, anchors])
    transformer = PiecewiseLinearTransformer.from_equal_intervals(augmented, gamma)
    explicit = [np.linspace(0.0, 100.0, gamma + 1) for _ in range(LEA_TABLE.shape[1])]
    for got, exp in zip(transformer.ch_p, explicit):
        assert np.allclose(got, exp), (got, exp)
    return transformer


def run_case(case_name, sessions, gamma):
    """
    Cumulative inference: after each session, FTRL MAP + BAYES HMC posterior.
    Returns (transformer, feature_matrix, list of per-step result dicts).
    """
    transformer = build_transformer(gamma)
    X = transformer.transform(LEA_TABLE)
    n_params = transformer.total_params
    sampler = PreferenceSampler(X, [], n_params, E_MAX)

    results = []
    for step, sess in enumerate(sessions, start=1):
        items = [IDX[name] for name in sess["items"]]
        gaps = expand_subrank(items, sess["cards"], expansion="adjacent")
        for upper, lower, eta in gaps:
            sampler.add_gap(upper, lower, eta)

        t0 = time.perf_counter()
        w_map = sampler.find_map()
        t_map = time.perf_counter() - t0

        t0 = time.perf_counter()
        samples, diag = sampler.run_hmc(num_samples=NUM_SAMPLES, num_warmup=NUM_WARMUP,
                                        seed=SEED_BASE + step, return_diagnostics=True)
        t_hmc = time.perf_counter() - t0

        print(f"[{case_name} step {step}] gaps added: {gaps}")
        print(f"[{case_name} step {step}] FTRL MAP in {t_map:.1f}s | "
              f"HMC {NUM_SAMPLES}/{NUM_WARMUP} in {t_hmc:.1f}s | "
              f"divergences={diag['divergences']}, max split-Rhat={diag['max_rhat']:.4f}")
        results.append({"step": step, "w_map": w_map, "samples": samples,
                        "diag": diag, "t_hmc": t_hmc})
    return transformer, X, results


# ----------------------------------------------------------------------------
# Metrics: exact mirrors of experiments/metrics.py calc_rai / calc_poi (E6)
# ----------------------------------------------------------------------------
def calc_rai(f1, t, sam):
    """RAI[a, r] = fraction of draws in which alternative a occupies rank r (0-based, 0 = best)."""
    rank_count = np.zeros((f1, f1))
    sam = np.atleast_2d(sam)
    w = sam[:, :t.shape[1]]
    vals = t @ w.T
    for col in range(vals.shape[1]):
        scores = vals[:, col]
        ranks = np.argsort(np.argsort(-scores))
        rank_count[np.arange(f1), ranks] += 1
    return rank_count / vals.shape[1]


def calc_pwi(f1, t, sam):
    """PWI[a, b] = fraction of draws with U(a) >= U(b) (mirror of calc_poi; diagonal = 1)."""
    rank_count = np.zeros((f1, f1))
    sam = np.atleast_2d(sam)
    w = sam[:, :t.shape[1]]
    vals = t @ w.T
    for col in range(vals.shape[1]):
        scores = vals[:, col]
        rank_count += (scores[:, None] >= scores[None, :])
    return rank_count / vals.shape[1]


def scaled_utilities(X, w):
    """
    Display scaling (plan E5): U_disp(a) = U(a) / U_max * 100, where
    U_max = sum of all piece weights = utility of the ideal alternative that
    scores 100 on every criterion (all features saturate at 1).
    """
    n_params = X.shape[1]
    w = np.asarray(w)[:n_params]
    return (X @ w) / np.sum(w) * 100.0


def bayes_mean_scaled_utilities(X, samples):
    """Posterior mean of the per-draw display-scaled utilities."""
    n_params = X.shape[1]
    w = samples[:, :n_params]
    vals = (X @ w.T) / np.sum(w, axis=1)[None, :] * 100.0
    return vals.mean(axis=1)


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def criterion_features(bp, x_grid):
    """Feature block of one criterion on a grid (same clipping as the transformer)."""
    n_seg = len(bp) - 1
    return np.stack([np.clip((x_grid - bp[k]) / (bp[k + 1] - bp[k]), 0.0, 1.0)
                     for k in range(n_seg)], axis=1)


def plot_utilities(transformer, w_map, samples, out_path):
    """
    Three panels (one per criterion): BAYES posterior mean and 90% pointwise
    credible band (blue) + FTRL MAP estimate (red), x in [0, 100].

    Utilities are NORMALIZED PER DRAW by the total weight sum, i.e. the curves
    show U_j(x) / U_max with U_max = U(ideal alternative scoring 100 on every
    criterion) = sum of all piece weights.  The value at x = 100 is therefore
    the criterion importance share hat{w}_j of the corner plots, and the three
    panel endpoints sum to 1 in every draw.  Rationale: the raw scale of w is
    shared with the thresholds Delta and is only weakly identified by a few
    sessions, so raw-scale bands conflate irrelevant scale uncertainty with
    the shape/share uncertainty that actually drives the rankings and the
    RAI/PWI tables (the raw-scale band was ~[0, 4] around a mean of ~1.3).
    """
    m = transformer.num_criteria
    n_params = transformer.total_params
    col_end = np.cumsum(transformer.n_w)
    col_start = np.insert(col_end[:-1], 0, 0)
    x_grid = np.linspace(0.0, 100.0, 201)

    w_draws = samples[:, :n_params]
    tot_draws = w_draws.sum(axis=1)                       # (n_draws,)
    w_map = np.asarray(w_map)[:n_params]
    tot_map = np.sum(w_map)

    plt.rcParams.update({"font.size": 13})
    fig, axes = plt.subplots(1, m, figsize=(4.6 * m, 4.2), sharey=True)
    for j in range(m):
        ax = axes[j]
        bp = np.asarray(transformer.ch_p[j], dtype=float)
        F = criterion_features(bp, x_grid)
        s, e = int(col_start[j]), int(col_end[j])

        curves = (F @ w_draws[:, s:e].T) / tot_draws[None, :]   # (n_grid, n_draws)
        y_lo = np.percentile(curves, 5, axis=1)
        y_hi = np.percentile(curves, 95, axis=1)
        ax.fill_between(x_grid, y_lo, y_hi, color=C_BAYES_FILL, alpha=0.25,
                        linewidth=0, label="BAYES 90% credible band")
        ax.plot(x_grid, curves.mean(axis=1), color=C_BAYES_LINE, lw=2,
                label="BAYES posterior mean")
        ax.plot(x_grid, F @ w_map[s:e] / tot_map, color=C_FTRL, lw=2,
                label="FTRL (MAP)")

        ax.set_title(CRITERIA[j])
        ax.set_xlabel("Criterion value")
        if j == 0:
            ax.set_ylabel("Normalized marginal utility")
            ax.legend(fontsize=10, loc="upper left", framealpha=0.95)
        ax.set_xlim(0, 100)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def criterion_totals(samples, n_params, group_sizes):
    """Per-draw RAW per-criterion totals U_j(100) = sum of that criterion's
    piece weights.  No normalization: this is the same scale as the FTRL
    parameter tables (for gamma_j = 1, U_j(100) is the weight w_j itself)."""
    w = np.atleast_2d(samples)[:, :n_params]
    idx = np.cumsum([0] + list(group_sizes))
    return np.stack([w[:, idx[j]:idx[j + 1]].sum(axis=1)
                     for j in range(len(group_sizes))], axis=1)


def plot_corner_steps(results, transformer, out_path, axis_labels):
    """
    Single corner plot of the per-criterion utilities U_j(100) (the sum of
    that criterion's piece weights) NORMALIZED per draw by
    U_max = sum_l U_l(100) = the utility of the ideal alternative scoring 100
    on every criterion.  With this normalization U(ideal) = 1, the plotted
    U_j(100) are the criterion importance shares and sum to 1 in every draw;
    they are the ranking-relevant quantity.  One color per
    session (matplotlib default cycle: blue T=1, orange T=2, green T=3),
    with the FTRL MAP estimate of each session overlaid (dashed vertical
    line on the diagonal panels, X marker on the bivariate panels).  2-D
    panels show the 90% credible contour only; every panel has a grid.
    """
    n_params, sizes = transformer.total_params, transformer.n_w
    totals = [criterion_totals(r["samples"], n_params, sizes) for r in results]
    totals = [t / t.sum(axis=1, keepdims=True) for t in totals]
    maps = [criterion_totals(r["w_map"], n_params, sizes)[0] for r in results]
    maps = [m / m.sum() for m in maps]
    # Matplotlib default color cycle (C0/C1/C2): with only three sessions,
    # blue / orange / green are far easier to tell apart than colormap shades.
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"][:len(totals)]
    step_labels = [f"$T={r['step']}$" for r in results]
    k = totals[0].shape[1]
    rng = [(0.0, 1.0)] * k

    def add_grid(ax):
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

    if HAVE_CORNER:
        fig = None
        for w, c in zip(totals, colors):
            fig = corner_pkg.corner(
                w, labels=axis_labels, range=rng,
                color=c, fig=fig, plot_datapoints=False, plot_density=False,
                no_fill_contours=True, levels=(0.9,), smooth=1.0,
                hist_kwargs={"density": True, "lw": 2},
                contour_kwargs={"linewidths": 1.8},
                label_kwargs={"fontsize": 18})
        axes = np.array(fig.axes).reshape(k, k)
        for m, c in zip(maps, colors):
            for i in range(k):
                axes[i, i].axvline(m[i], color=c, ls="--", lw=1.6, zorder=5)
                for j in range(i):
                    axes[i, j].plot(m[j], m[i], marker="X", ms=10, color=c,
                                    mec="white", mew=0.8, ls="none", zorder=6)
        # Styling for half-page use: larger tick labels, axis labels pulled
        # closer to the ticks, legend centered in the empty upper triangle.
        for i in range(k):
            for j in range(i + 1):
                add_grid(axes[i, j])
                axes[i, j].tick_params(axis="both", labelsize=14)
        for j in range(k):
            axes[k - 1, j].xaxis.set_label_coords(0.5, -0.25)
        for i in range(1, k):
            axes[i, 0].yaxis.set_label_coords(-0.25, 0.5)
        handles = ([plt.Line2D([], [], color=c, lw=2) for c in colors]
                   + [plt.Line2D([], [], color="#444444", marker="X", ms=9,
                                 ls="none", mec="white", mew=0.8)])
        fig.legend(handles, step_labels + ["FTRL estimate"],
                   loc="center", bbox_to_anchor=(0.74, 0.8),
                   frameon=True, fontsize=16)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return
    # Pure-matplotlib fallback: overlaid density hists / scatter, same overlays.
    fig, axes = plt.subplots(k, k, figsize=(3.0 * k, 3.0 * k))
    for i in range(k):
        for j in range(k):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue
            for w, m, c, lab in zip(totals, maps, colors, step_labels):
                if i == j:
                    ax.hist(w[:, i], bins=40, density=True, histtype="step",
                            color=c, lw=2, label=lab)
                    ax.axvline(m[i], color=c, ls="--", lw=1.6)
                else:
                    ax.scatter(w[:, j], w[:, i], s=3, alpha=0.15, color=c)
                    ax.plot(m[j], m[i], marker="X", ms=10, color=c,
                            mec="white", mew=0.8, ls="none")
            if i == j:
                ax.set_yticks([])
            ax.set_xlim(*rng[j])
            add_grid(ax)
            if i == k - 1:
                ax.set_xlabel(axis_labels[j])
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(axis_labels[i])
    axes[0, 0].legend(fontsize=10, loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_marginals_steps(results, transformer, out_path):
    """
    Grid of NORMALIZED marginal utility functions: rows = criteria, columns =
    sessions (T = 1..3).  Each panel shows the BAYES posterior mean with its
    90% pointwise credible band (blue) and the FTRL MAP estimate (red) of
    u_j(x)/U_max, with U_max = utility of the ideal alternative (per-draw
    normalization, as in the corner plots; the value at x = 100 is the
    criterion importance share).  Panels in a row SHARE the y-axis, so
    reading a row left to right shows how that marginal utility evolves and
    its band narrows as the sessions accumulate.
    """
    m = transformer.num_criteria
    n_params = transformer.total_params
    col_end = np.cumsum(transformer.n_w)
    col_start = np.insert(col_end[:-1], 0, 0)
    x_grid = np.linspace(0.0, 100.0, 201)

    plt.rcParams.update({"font.size": 17})
    fig, axes = plt.subplots(m, len(results),
                             figsize=(4.3 * len(results), 3.2 * m),
                             sharex=True, sharey="row")
    for t, res in enumerate(results):
        w = res["samples"][:, :n_params]
        w = w / w.sum(axis=1, keepdims=True)
        wm = np.asarray(res["w_map"])[:n_params]
        wm = wm / wm.sum()
        for j in range(m):
            ax = axes[j, t]
            bp = np.asarray(transformer.ch_p[j], dtype=float)
            F = criterion_features(bp, x_grid)
            s, e = int(col_start[j]), int(col_end[j])
            curves = F @ w[:, s:e].T                     # (n_grid, n_draws)
            ax.fill_between(x_grid, np.percentile(curves, 5, axis=1),
                            np.percentile(curves, 95, axis=1),
                            color=C_BAYES_FILL, alpha=0.25, linewidth=0,
                            label="BAYES-DOR 90% band")
            ax.plot(x_grid, curves.mean(axis=1), color=C_BAYES_LINE, lw=1.8,
                    label="BAYES-DOR mean")
            ax.plot(x_grid, F @ wm[s:e], color=C_FTRL, lw=1.8,
                    label="FTRL-DOR")
            ax.set_xlim(0, 100)
            ax.grid(True, alpha=0.3)
            ax.set_axisbelow(True)
            ax.tick_params(axis="both", labelsize=16)
            if j == 0:
                ax.set_title(f"$T={res['step']}$", fontsize=21)
            if t == 0:
                ax.set_ylabel(CRITERIA_HAT[j], fontsize=17)
                ax.set_ylim(bottom=0)
            if j == m - 1:
                ax.set_xlabel("Criterion value", fontsize=17)
    axes[0, 0].legend(fontsize=14, loc="upper left", framealpha=0.95)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_corner(samples, n_params, out_path, group_sizes=None):
    """
    Corner plot of the posterior weights, NORMALIZED per draw to shares
    hat{w}_i = w_i / sum_j w_j (the convention of the paper's Figure 5: axes in
    [0,1], bivariate support inside the simplex triangle).  The overall scale
    of w is shared with the thresholds and is not what the figure is meant to
    show; the shares are the ranking-relevant quantity.

    group_sizes: piece counts per criterion (transformer.n_w).  When given,
    piece weights are summed within each criterion first, so the plot shows
    the CRITERION importance shares hat{w}_j = U_j(100)/sum_l U_l(100) --
    the gamma_j > 1 counterpart of the T=1 plot (used for the T=3 case).
    """
    w = samples[:, :n_params]
    if group_sizes is not None:
        idx = np.cumsum([0] + list(group_sizes))
        w = np.stack([w[:, idx[j]:idx[j + 1]].sum(axis=1)
                      for j in range(len(group_sizes))], axis=1)
    w = w / w.sum(axis=1, keepdims=True)
    labels = [r"$\hat{w}_1$", r"$\hat{w}_2$", r"$\hat{w}_3$"]
    if HAVE_CORNER:
        fig = corner_pkg.corner(w, labels=labels, show_titles=True,
                                quantiles=[0.05, 0.5, 0.95],
                                range=[(0, 1)] * w.shape[1],
                                title_fmt=".3f", color=C_BAYES_FILL)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return
    # Pure-matplotlib fallback: marginal hists on the diagonal, scatter below.
    k = w.shape[1]
    fig, axes = plt.subplots(k, k, figsize=(3.0 * k, 3.0 * k))
    for i in range(k):
        for j in range(k):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
            elif i == j:
                ax.hist(w[:, i], bins=40, color=C_BAYES_FILL, alpha=0.8)
                ax.set_yticks([])
            else:
                ax.scatter(w[:, j], w[:, i], s=3, alpha=0.15, color=C_BAYES_FILL)
            if i == k - 1:
                ax.set_xlabel(labels[j])
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(labels[i])
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ----------------------------------------------------------------------------
# LaTeX tables (booktabs style, percentages with 1 decimal like the paper)
# ----------------------------------------------------------------------------
def fmt_pct(v):
    """Fraction -> percentage string: '0' and '100' plain, otherwise 1 decimal."""
    p = 100.0 * v
    if p < 0.05:
        return "0"
    if p > 99.95:
        return "100"
    return f"{p:.1f}"


def write_rai_table(rai, path, caption, label):
    cols = "X|" + "C" * NUM_REGIONS
    lines = [
        "\\begin{table}[h!]",
        "\\centering",
        "\\footnotesize",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabularx}}{{\\textwidth}}{{{cols}}}",
        "\\toprule",
        f"\\multirow{{2}}{{*}}{{ID}}&\\multicolumn{{{NUM_REGIONS}}}{{c}}{{Rank}} \\\\",
        f"\\cmidrule(lr){{2-{NUM_REGIONS + 1}}}",
        "&" + "&".join(str(r) for r in range(1, NUM_REGIONS + 1)) + "\\\\",
        "\\midrule",
    ]
    for a in range(NUM_REGIONS):
        row = " & ".join(fmt_pct(rai[a, r]) for r in range(NUM_REGIONS))
        lines.append(f"$R_{{{a + 1}}}$ & {row} \\\\")
    lines += ["\\bottomrule", "\\end{tabularx}", "\\end{table}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_pwi_table(pwi, path, caption, label):
    # Diagonal is 100 by the U(a) >= U(b) convention of calc_poi (E6).
    cols = "X|" + "C" * NUM_REGIONS
    lines = [
        "% PWI[a,b] = fraction of posterior draws with U(a) >= U(b);",
        "% the diagonal equals 100 by this convention (mirror of calc_poi).",
        "\\begin{table}[h!]",
        "\\centering",
        "\\footnotesize",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabularx}}{{\\textwidth}}{{{cols}}}",
        "\\toprule",
        "ID&" + "&".join(f"$R_{{{b + 1}}}$" for b in range(NUM_REGIONS)) + "\\\\",
        "\\midrule",
    ]
    for a in range(NUM_REGIONS):
        row = " & ".join(fmt_pct(pwi[a, b]) for b in range(NUM_REGIONS))
        lines.append(f"$R_{{{a + 1}}}$ & {row} \\\\")
    lines += ["\\bottomrule", "\\end{tabularx}", "\\end{table}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def utilities_at_breakpoints(transformer, w_map):
    """
    Per criterion, U_j at each breakpoint = cumulative sum of the piece weights
    (U_j(bp_0) = 0), as reported in the paper's tab:ftrl_ex_t3.
    """
    col_end = np.cumsum(transformer.n_w)
    col_start = np.insert(col_end[:-1], 0, 0)
    out = []
    for j in range(transformer.num_criteria):
        w_j = np.asarray(w_map)[int(col_start[j]):int(col_end[j])]
        out.append(np.concatenate([[0.0], np.cumsum(w_j)]))
    return out


def fmt_bp(x):
    return f"{x:.0f}" if abs(x - round(x)) < 1e-9 else f"{x:.1f}"


def write_ftrl_tables(lin, pw, path):
    """
    LIN table: one row per session, U_j(0) and U_j(100) (linear utilities).
    PW table: one row per session, U_j at every breakpoint (gamma_j = 3).
    Raw parameter scale in both (same scale as the corner plots).
    """
    tr_lin, res_lin = lin
    tr_pw, res_pw = pw
    lines = [
        "% FTRL-DOR parameter estimates for the LEA case study",
        "% (adjacent-only likelihood, e_max = n_max = 30).",
        "",
        "% --- Linear utilities: gamma_j = 1, one row per session ---",
        "\\begin{table}[h!]",
        "\\centering",
        "\\caption{Parameters estimated through FTRL-DOR after each of the "
        "$T=3$ sessions in the case of linear marginal utility functions "
        "(one breakpoint per criterion)}",
        "\\label{tab:ftrl_lin}",
        "\\begin{tabular}{l|cc|cc|cc}",
        "\\toprule",
        "&$u_1(0)$&$u_1(100)$&$u_2(0)$&$u_2(100)$&$u_3(0)$&$u_3(100)$\\\\",
        "\\midrule",
    ]
    for res in res_lin:
        u = utilities_at_breakpoints(tr_lin, res["w_map"])
        vals = "&".join("0" if u[j][k] == 0 else f"{u[j][k]:.4f}"
                        for j in range(3) for k in range(2))
        lines.append(f"$T={res['step']}$&" + vals + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]

    bps = tr_pw.ch_p[0]
    header = "&".join(f"$u_{{{j + 1}}}({fmt_bp(bp)})$"
                      for j in range(3) for bp in bps)
    cols = "X|" + "|".join("C" * len(bps) for _ in range(3))
    lines += [
        "% --- Piecewise linear utilities: gamma_j = 3 equally spaced breakpoints ---",
        "\\begin{table}[H]",
        "\\centering",
        "\\footnotesize",
        "\\caption{Parameters estimated through FTRL-DOR after each of the "
        "$T=3$ sessions in the case of piecewise linear marginal utility "
        "functions with $\\gamma_j=3$ equally spaced breakpoints per criterion}",
        "\\label{tab:ftrl_pw}",
        f"\\begin{{tabularx}}{{\\textwidth}}{{{cols}}}",
        "\\toprule",
        "&" + header + "\\\\",
        "\\midrule",
    ]
    for res in res_pw:
        u = utilities_at_breakpoints(tr_pw, res["w_map"])
        vals = "&".join("0" if u[j][k] == 0 else f"{u[j][k]:.4f}"
                        for j in range(3) for k in range(len(bps)))
        lines.append(f"$T={res['step']}$&" + vals + "\\\\")
    lines += ["\\bottomrule", "\\end{tabularx}", "\\end{table}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def ranking_from_utilities(u_disp):
    """Regions sorted by decreasing display utility; returns list of (region_idx, value)."""
    order = np.argsort(-u_disp)
    return [(int(i), float(u_disp[i])) for i in order]


def cumulative_reference_sets():
    """Reference alternatives seen up to and including each session."""
    refs, acc = [], set()
    for sess in SESSIONS:
        acc |= {IDX[n] for n in sess["items"]}
        refs.append(set(acc))
    return refs


def ranking_table_lines(X, res_list, caption, label, top_n=None):
    """Three-column-group FTRL ranking table, one group per session.
    top_n limits the table to the first top_n positions of each session."""
    refs_steps = cumulative_reference_sets()
    rankings = [ranking_from_utilities(scaled_utilities(X, res["w_map"]))
                for res in res_list]
    lines = [
        "\\begin{table}[h!]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabularx}{\\textwidth}{c|Cc|Cc|Cc}",
        "\\toprule",
        "\\multirow{2}{*}{Rank}&\\multicolumn{2}{c}{$T=1$}&"
        "\\multicolumn{2}{c}{$T=2$}&\\multicolumn{2}{c}{$T=3$}\\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
        "&Region&$U\\left(a\\right)$&Region&$U\\left(a\\right)$&"
        "Region&$U\\left(a\\right)$\\\\",
        "\\midrule",
    ]
    for rank in range(top_n if top_n is not None else NUM_REGIONS):
        cells = []
        for s in range(len(res_list)):
            idx, val = rankings[s][rank]
            name = SHORT.get(REGIONS[idx], REGIONS[idx])
            sval = f"{val:.2f}"
            if idx in refs_steps[s]:
                name, sval = f"\\textbf{{{name}}}", f"\\textbf{{{sval}}}"
            cells += [name, sval]
        lines.append(f"{rank + 1} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabularx}", "\\end{table}"]
    return lines


def write_ranking_tables(lin, pw, path):
    """
    FTRL ranking after each session for both cases (linear / piecewise).
    Reference alternatives in bold (cumulative).  Display scaling (plan E5):
    U_disp(a) = U(a)/U_max*100 with U_max the utility of the ideal alternative
    scoring 100 on every criterion.
    """
    _, X_lin, res_lin = lin
    _, X_pw, res_pw = pw
    lines = [
        "% FTRL-DOR rankings for the LEA case study (adjacent-only likelihood,",
        "% e_max = n_max = 30).  Display scaling (plan E5):",
        "% U_disp(a) = U(a) / U_max * 100, U_max = sum of all piece weights",
        "% (utility of the ideal alternative scoring 100 on every criterion).",
        "",
    ]
    lines += ranking_table_lines(
        X_lin, res_lin,
        "Ranking of the alternatives considering the parameters estimated "
        "through FTRL-DOR after each of the $T=3$ sessions, in the case of "
        "linear marginal utility functions. The utilities are expressed as "
        "percentages of the utility of the ideal alternative, which scores "
        "100 on every criterion. The reference alternatives are written in "
        "bold.", "tab:RANK_lin")
    lines.append("")
    lines += ranking_table_lines(
        X_pw, res_pw,
        "Ranking of the alternatives considering the parameters estimated "
        "through FTRL-DOR after each of the $T=3$ sessions, in the case of "
        "piecewise linear marginal utility functions ($\\gamma_j=3$). The "
        "utilities are expressed as percentages of the utility of the ideal "
        "alternative, which scores 100 on every criterion. The reference "
        "alternatives are written in bold.", "tab:RANK_pw")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_rai_evolution_table(rai_steps, top_idx, path, caption, label,
                              max_rank=8):
    """
    RAI evolution for the top-ranked regions: one block of three rows
    (T = 1, 2, 3) per region, columns = ranks 1..max_rank plus the aggregated
    tail; shows how the rank distributions sharpen across sessions.
    """
    cols = "l l|" + "C" * max_rank + "C"
    lines = [
        f"\\begin{{table}}[h!]",
        "\\centering",
        "\\footnotesize",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabularx}}{{\\textwidth}}{{{cols}}}",
        "\\toprule",
        f"\\multirow{{2}}{{*}}{{Region}}&\\multirow{{2}}{{*}}{{Sessions}}&"
        f"\\multicolumn{{{max_rank + 1}}}{{c}}{{Rank}} \\\\",
        f"\\cmidrule(lr){{3-{max_rank + 3}}}",
        "&&" + "&".join(str(r) for r in range(1, max_rank + 1))
        + f"&$>{max_rank}$\\\\",
        "\\midrule",
    ]
    for n, idx in enumerate(top_idx):
        name = SHORT.get(REGIONS[idx], REGIONS[idx])
        for t, rai in enumerate(rai_steps, start=1):
            head = f"\\multirow{{3}}{{*}}{{{name} ($R_{{{idx + 1}}}$)}}" \
                if t == 1 else ""
            row = "&".join(fmt_pct(rai[idx, r]) for r in range(max_rank))
            tail = fmt_pct(rai[idx, max_rank:].sum())
            lines.append(f"{head}&$T={t}$&{row}&{tail}\\\\")
        if n < len(top_idx) - 1:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabularx}", "\\end{table}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_pwi_evolution_table(pwi_steps, top_idx, path, caption, label):
    """
    PWI among the top-ranked regions after each session: rows = regions,
    one column group per session with the PWI against each of the same
    regions ('--' on the diagonal).
    """
    k = len(top_idx)
    cols = "l|" + "|".join("C" * k for _ in pwi_steps)
    ids = "&".join(f"$R_{{{b + 1}}}$" for b in top_idx)
    lines = [
        "% PWI[a,b] = fraction of posterior draws with U(a) >= U(b).",
        "\\begin{table}[h!]",
        "\\centering",
        "\\footnotesize",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabularx}}{{\\textwidth}}{{{cols}}}",
        "\\toprule",
        "&" + "&".join(f"\\multicolumn{{{k}}}{{c}}{{$T={t}$}}"
                       for t in range(1, len(pwi_steps) + 1)) + "\\\\",
        "".join(f"\\cmidrule(lr){{{2 + s * k}-{1 + (s + 1) * k}}}"
                for s in range(len(pwi_steps))),
        "Region&" + "&".join([ids] * len(pwi_steps)) + "\\\\",
        "\\midrule",
    ]
    for a in top_idx:
        name = SHORT.get(REGIONS[a], REGIONS[a])
        cells = []
        for pwi in pwi_steps:
            for b in top_idx:
                cells.append("--" if a == b else fmt_pct(pwi[a, b]))
        lines.append(f"{name} ($R_{{{a + 1}}}$)&" + "&".join(cells) + "\\\\")
    lines += ["\\bottomrule", "\\end{tabularx}", "\\end{table}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    t_start = time.perf_counter()

    print("=" * 72)
    print(f"LEA case study - adjacent-only likelihood, e_max = n_max = {E_MAX}")
    print("=" * 72)

    # --- inference -----------------------------------------------------------
    print("\n--- Case LIN (gamma_j = 1, T=3 cumulative) ---")
    tr_lin, X_lin, res_lin = run_case("LIN", SESSIONS, gamma=1)
    print("\n--- Case PW (gamma_j = 3, T=3 cumulative) ---")
    tr_pw, X_pw, res_pw = run_case("PW", SESSIONS, gamma=3)

    # --- corner figures (raw scale, FTRL overlays, one per case) -------------
    # Paper notation (Section 2): lowercase u_j for the MARGINAL value
    # functions; capital U(a) is reserved for the global utility.  The hat
    # marks the per-draw normalization by the ideal alternative's utility.
    labels_u = [r"$\hat{u}_1(100)$", r"$\hat{u}_2(100)$", r"$\hat{u}_3(100)$"]
    plot_corner_steps(res_lin, tr_lin,
                      os.path.join(OUT_DIR, "croner_steps_lin.png"), labels_u)
    plot_corner_steps(res_pw, tr_pw,
                      os.path.join(OUT_DIR, "croner_steps_pw.png"), labels_u)
    plot_marginals_steps(res_pw, tr_pw,
                         os.path.join(OUT_DIR, "marginals_steps_pw.png"))

    # --- RAI / PWI after every session ---------------------------------------
    rai_lin = [calc_rai(NUM_REGIONS, X_lin, r["samples"]) for r in res_lin]
    pwi_lin = [calc_pwi(NUM_REGIONS, X_lin, r["samples"]) for r in res_lin]
    rai_pw = [calc_rai(NUM_REGIONS, X_pw, r["samples"]) for r in res_pw]
    pwi_pw = [calc_pwi(NUM_REGIONS, X_pw, r["samples"]) for r in res_pw]

    # Supplementary tables: full 21-region RAI/PWI after the FIRST and after
    # the THIRD session of each case -- the section compares T=1 with T=3, so
    # both posteriors need to be documented (not only the final one).
    for tag, desc, sec_ref, rai_steps, pwi_steps in [
            ("lin", "linear marginal utility functions",
             "sec:example_linear", rai_lin, pwi_lin),
            ("pw", "piecewise linear marginal utility functions "
             "($\\gamma_j=3$)", "sec:example_pw", rai_pw, pwi_pw)]:
        for t in (1, 3):
            when = ("the first session ($T=1$)" if t == 1
                    else f"the $T={t}$ sessions")
            write_rai_table(
                rai_steps[t - 1],
                os.path.join(OUT_DIR, f"table_rai_{tag}_t{t}.tex"),
                "Rank Acceptability Indices (\\%) of all regions computed from "
                f"the posterior distribution after {when}, with {desc} "
                f"(Section \\ref{{{sec_ref}}})", f"tab:RAI_{tag}_t{t}")
            write_pwi_table(
                pwi_steps[t - 1],
                os.path.join(OUT_DIR, f"table_pwi_{tag}_t{t}.tex"),
                "Pairwise Winning Indices (\\%) of all regions computed from "
                f"the posterior distribution after {when}, with {desc} "
                f"(Section \\ref{{{sec_ref}}})", f"tab:PWI_{tag}_t{t}")

    # --- RAI / PWI evolution across sessions for the top-4 regions -----------
    rank_final_lin = ranking_from_utilities(
        scaled_utilities(X_lin, res_lin[-1]["w_map"]))
    top4_lin = [idx for idx, _ in rank_final_lin[:4]]
    rank_final_pw = ranking_from_utilities(
        scaled_utilities(X_pw, res_pw[-1]["w_map"]))
    top4_pw = [idx for idx, _ in rank_final_pw[:4]]

    write_rai_evolution_table(
        rai_lin, top4_lin,
        os.path.join(OUT_DIR, "table_rai_evolution_lin.tex"),
        "Evolution across the $T=3$ sessions of the Rank Acceptability "
        "Indices (\\%) of the four top-ranked regions, in the case of linear "
        "marginal utility functions", "tab:rai_evo_lin")
    write_pwi_evolution_table(
        pwi_lin, top4_lin,
        os.path.join(OUT_DIR, "table_pwi_evolution_lin.tex"),
        "Evolution across the $T=3$ sessions of the Pairwise Winning Indices "
        "(\\%) among the four top-ranked regions, in the case of linear "
        "marginal utility functions", "tab:pwi_evo_lin")
    write_rai_evolution_table(
        rai_pw, top4_pw,
        os.path.join(OUT_DIR, "table_rai_evolution_pw.tex"),
        "Evolution across the $T=3$ sessions of the Rank Acceptability "
        "Indices (\\%) of the four top-ranked regions, in the case of "
        "piecewise linear marginal utility functions ($\\gamma_j=3$)",
        "tab:rai_evo_pw")
    write_pwi_evolution_table(
        pwi_pw, top4_pw,
        os.path.join(OUT_DIR, "table_pwi_evolution_pw.tex"),
        "Evolution across the $T=3$ sessions of the Pairwise Winning Indices "
        "(\\%) among the four top-ranked regions, in the case of piecewise "
        "linear marginal utility functions ($\\gamma_j=3$)", "tab:pwi_evo_pw")

    # --- FTRL parameter and ranking tables -----------------------------------
    write_ftrl_tables((tr_lin, res_lin), (tr_pw, res_pw),
                      os.path.join(OUT_DIR, "ftrl_tables.tex"))
    write_ranking_tables((tr_lin, X_lin, res_lin), (tr_pw, X_pw, res_pw),
                         os.path.join(OUT_DIR, "ranking_tables.tex"))
    # Compact top-5 FTRL ranking for the linear subsection (the five regions
    # its RAI/PWI discussion refers to), one column group per session.
    top5 = ranking_table_lines(
        X_lin, res_lin,
        "The five highest-scoring regions after each of the $T=3$ sessions, "
        "with the utilities computed from the FTRL-DOR parameter estimates "
        "of the corresponding session, in the case of linear marginal "
        "utility functions. The utilities are expressed as percentages of "
        "the utility of the ideal alternative, which scores 100 on every "
        "criterion. The reference alternatives are written in bold.",
        "tab:rank_lin_top5", top_n=5)
    with open(os.path.join(OUT_DIR, "table_rank_top5_lin.tex"), "w") as f:
        f.write("\n".join(top5) + "\n")
    top5_pw = ranking_table_lines(
        X_pw, res_pw,
        "The five highest-scoring regions after each of the $T=3$ sessions, "
        "with the utilities computed from the FTRL-DOR parameter estimates "
        "of the corresponding session, in the case of piecewise linear "
        "marginal utility functions ($\\gamma_j=3$). The utilities are "
        "expressed as percentages of the utility of the ideal alternative, "
        "which scores 100 on every criterion. The reference alternatives "
        "are written in bold.",
        "tab:rank_pw_top5", top_n=5)
    with open(os.path.join(OUT_DIR, "table_rank_top5_pw.tex"), "w") as f:
        f.write("\n".join(top5_pw) + "\n")

    # --- long-format CSV dump -------------------------------------------------
    rows = []
    for case_name, X, res_list in [("LIN", X_lin, res_lin),
                                   ("PW", X_pw, res_pw)]:
        for res in res_list:
            u_ftrl = scaled_utilities(X, res["w_map"])
            u_bayes = bayes_mean_scaled_utilities(X, res["samples"])
            for method, u in [("FTRL", u_ftrl), ("BAYES", u_bayes)]:
                for rank, (idx, val) in enumerate(ranking_from_utilities(u), start=1):
                    rows.append({"case": case_name, "step": res["step"],
                                 "method": method, "region": REGIONS[idx],
                                 "rank": rank, "utility": round(val, 4)})
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "case_study_summary.csv"),
                              index=False)

    # --- console report -------------------------------------------------------
    print("\n" + "=" * 72)
    for case_name, X, res_list in [("LIN", X_lin, res_lin),
                                   ("PW", X_pw, res_pw)]:
        print(f"\n{case_name} FTRL rankings (display-scaled utilities), "
              f"T=1 | T=2 | T=3:")
        ranks = [ranking_from_utilities(scaled_utilities(X, res["w_map"]))
                 for res in res_list]
        for r in range(NUM_REGIONS):
            cells = " | ".join(f"{REGIONS[ranks[s][r][0]]:<20.20s}{ranks[s][r][1]:7.2f}"
                               for s in range(len(ranks)))
            print(f"  {r + 1:2d}. {cells}")

    # Sanity check (E1/E2): the session-1 reference order must be respected.
    rank_s1 = ranking_from_utilities(scaled_utilities(X_lin, res_lin[0]["w_map"]))
    pos = {idx: r for r, (idx, _) in enumerate(rank_s1)}
    ref_order = [IDX[n] for n in STEP_1["items"]]
    assert all(pos[ref_order[i]] < pos[ref_order[i + 1]] for i in range(3)), \
        "FTRL LIN T=1 ranking violates the reference order Lombardy > Marche > Campania > Basilicata"
    print("\n  [check] reference order Lombardy > Marche > Campania > Basilicata respected")

    print("\nRaw totals U_j(100): FTRL / BAYES median [5%, 95%]:")
    for case_name, tr, res_list in [("LIN", tr_lin, res_lin),
                                    ("PW", tr_pw, res_pw)]:
        for res in res_list:
            m = criterion_totals(res["w_map"], tr.total_params, tr.n_w)[0]
            tot = criterion_totals(res["samples"], tr.total_params, tr.n_w)
            qs = np.percentile(tot, [5, 50, 95], axis=0)
            print(f"  {case_name} T={res['step']}: " + " | ".join(
                f"U{j + 1}: {m[j]:.3f} / {qs[1][j]:.3f} [{qs[0][j]:.3f}, {qs[2][j]:.3f}]"
                for j in range(len(m))))

    print("\nNormalized U_j(100) (importance shares): FTRL / BAYES median [5%, 95%]:")
    for case_name, tr, res_list in [("LIN", tr_lin, res_lin),
                                    ("PW", tr_pw, res_pw)]:
        for res in res_list:
            m = criterion_totals(res["w_map"], tr.total_params, tr.n_w)[0]
            m = m / m.sum()
            tot = criterion_totals(res["samples"], tr.total_params, tr.n_w)
            tot = tot / tot.sum(axis=1, keepdims=True)
            qs = np.percentile(tot, [5, 50, 95], axis=0)
            print(f"  {case_name} T={res['step']}: " + " | ".join(
                f"U{j + 1}(100): {m[j]:.3f} / {qs[1][j]:.3f} "
                f"[{qs[0][j]:.3f}, {qs[2][j]:.3f}]"
                for j in range(len(m))))

    # Normalized u_j at each interior breakpoint and at 100 (PW case), to
    # quantify where along the scale the posterior is tight vs. wide.
    print("\nPW normalized u_j at breakpoints (33.3/66.7/100): "
          "FTRL / median [5%, 95%]:")
    idx = np.cumsum([0] + list(tr_pw.n_w))
    for res in res_pw:
        w = res["samples"][:, :tr_pw.total_params]
        w = w / w.sum(axis=1, keepdims=True)
        wm = np.asarray(res["w_map"])[:tr_pw.total_params]
        wm = wm / wm.sum()
        for j in range(tr_pw.num_criteria):
            cum = np.cumsum(w[:, idx[j]:idx[j + 1]], axis=1)
            cm = np.cumsum(wm[idx[j]:idx[j + 1]])
            q = np.percentile(cum, [5, 50, 95], axis=0)
            cells = " ".join(
                f"{cm[k]:.3f}/{q[1][k]:.3f}[{q[0][k]:.3f},{q[2][k]:.3f}]"
                for k in range(cum.shape[1]))
            print(f"  T={res['step']} u{j + 1}: {cells}")

    # Utility of the IDEAL alternative (score 100 on every criterion):
    # U_max = sum of all piece weights = sum_j U_j(100).  The figures and the
    # ranking tables are expressed relative to it (=1 and =100 respectively).
    print("\nIdeal alternative (100 on every criterion): raw U_max = sum_j U_j(100)")
    print("  case  T   FTRL U_max   BAYES median [5%, 95%]   normalized   display")
    for case_name, tr, X, res_list in [("LIN", tr_lin, X_lin, res_lin),
                                       ("PW", tr_pw, X_pw, res_pw)]:
        for res in res_list:
            w = np.asarray(res["w_map"])[:tr.total_params]
            u_max_ftrl = w.sum()
            u_max_post = res["samples"][:, :tr.total_params].sum(axis=1)
            q = np.percentile(u_max_post, [5, 50, 95])
            # The ideal alternative saturates every feature, so its feature
            # row is all ones: check against the display scaling used in the
            # ranking tables (U(a)/U_max*100) and the figure normalization.
            x_ideal = np.ones(tr.total_params)
            disp = float(x_ideal @ w) / u_max_ftrl * 100.0
            norm = float(x_ideal @ w) / u_max_ftrl
            assert np.isclose(disp, 100.0) and np.isclose(norm, 1.0)
            # No real region can exceed it.
            assert scaled_utilities(X, res["w_map"]).max() <= 100.0 + 1e-9
            print(f"  {case_name:<4s} {res['step']}   {u_max_ftrl:9.4f}   "
                  f"{q[1]:7.3f} [{q[0]:6.3f}, {q[2]:7.3f}]   "
                  f"{norm:10.1f}   {disp:7.1f}")
    print("  (every real region scores below the ideal: max display utility "
          f"{max(scaled_utilities(X_lin, res_lin[-1]['w_map']).max(), scaled_utilities(X_pw, res_pw[-1]['w_map']).max()):.2f})")

    for case_name, rai_steps, pwi_steps, top4 in [
            ("LIN", rai_lin, pwi_lin, top4_lin),
            ("PW", rai_pw, pwi_pw, top4_pw)]:
        print(f"\n{case_name} RAI evolution for the top-4 regions "
              f"(RAI(1) / P(top 4)):")
        for idx in top4:
            row = " | ".join(f"T={t}: {100 * rai[idx, 0]:5.1f} / "
                             f"{100 * rai[idx, :4].sum():5.1f}"
                             for t, rai in enumerate(rai_steps, start=1))
            print(f"  {REGIONS[idx]:<32.32s} {row}")
        print(f"{case_name} PWI evolution among the top-4 (row beats column):")
        for i, a in enumerate(top4):
            for b in top4[i + 1:]:
                row = " | ".join(f"T={t}: {100 * pwi[a, b]:5.1f}"
                                 for t, pwi in enumerate(pwi_steps, start=1))
                print(f"  {REGIONS[a][:14]:<14s} vs {REGIONS[b][:14]:<14s} {row}")

    print("\nHMC diagnostics:")
    for case_name, res_list in [("LIN", res_lin), ("PW", res_pw)]:
        for res in res_list:
            d = res["diag"]
            print(f"  {case_name} T={res['step']}: divergences={d['divergences']}, "
                  f"max split-Rhat={d['max_rhat']:.4f} ({res['t_hmc']:.1f}s)")

    print(f"\nOutputs written to {os.path.abspath(OUT_DIR)}")
    print(f"Total runtime: {time.perf_counter() - t_start:.1f}s")


if __name__ == "__main__":
    main()
