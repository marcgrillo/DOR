"""Illustrative example of Section 6: ranking 21 Italian regions on the LEA criteria.

Runs FTRL-DOR and BAYES-DOR on the paper's illustrative example and writes all of its figures
and tables. The data are embedded below: Table 4 (21 regions scored in [0, 100] on the three LEA
macro-areas Prevention care g1, Primary care g2 and Hospital care g3, all increasing) and the
three elicitation sessions of Figure 6. The script is self-contained: it reads nothing from
data/ or results/, and it is the only producer of outputs/case_study/.

Paper:   Sec. 6: Figures 7-8, Tables 5-10 and Supplementary Tables C.1-C.8; the posterior
         medians, 90% intervals and FTRL-DOR shares quoted in Secs. 6.1-6.2 (stdout)
Inputs:  none (Table 4 and Figure 6 are embedded constants)
Outputs: outputs/case_study/, file -> paper item:
           croner_steps_lin.png          Figure 7     marginals_steps_pw.png        Figure 8
           table_rank_top5_lin.tex       Table 5      table_rank_top5_pw.tex        Table 8
           table_rai_evolution_lin.tex   Table 6      table_rai_evolution_pw.tex    Table 9
           table_pwi_evolution_lin.tex   Table 7      table_pwi_evolution_pw.tex    Table 10
           table_rai_lin_t1.tex          Table C.1    table_rai_pw_t1.tex           Table C.5
           table_pwi_lin_t1.tex          Table C.2    table_pwi_pw_t1.tex           Table C.6
           table_rai_lin_t3.tex          Table C.3    table_rai_pw_t3.tex           Table C.7
           table_pwi_lin_t3.tex          Table C.4    table_pwi_pw_t3.tex           Table C.8
         plus four extra outputs that are not in the paper: croner_steps_pw.png (the Figure 7
         plot for the PW model), ftrl_tables.tex (FTRL-DOR u_j at every breakpoint after each
         session, raw scale), ranking_tables.tex (full 21-region FTRL-DOR rankings) and
         case_study_summary.csv (FTRL-DOR and posterior-mean rankings, long format). The
         'croner' typo is kept on purpose: the manuscript LaTeX includes that file name.
         stdout: FTRL-DOR rankings, importance shares and PW breakpoint values (FTRL-DOR and
         posterior median with 90% interval), RAI/PWI summaries of the top-4 regions and the
         NUTS diagnostics (divergences, max split-Rhat) of every session.
Runtime: about 1 minute on one CPU core (six NUTS runs of 3000 iterations each)

Usage:
    python scripts/case_study/case_study_lea.py

Setup (Sec. 6):
  - e_max = 30 cards (n_max in the code; the largest declared count is 28) and T = 3 sessions,
    processed as in Algorithm 1 (Sec. 4): after session t both estimators use the 3t
    adjacent-gap observations (Sec. 3.1) of sessions 1..t, i.e. raw card counts between
    consecutive regions (expand_subrank(..., expansion='adjacent')), under the likelihood of
    eq. (11) and the Gamma priors of eq. (14).
  - Two value-function models on the common [0, 100] scale: LIN, gamma_j = 1 (one weight per
    criterion), and PW, gamma_j = 3 (breakpoints 0, 33.3, 66.7, 100). The breakpoints come
    from PiecewiseLinearTransformer.from_equal_intervals applied to Table 4 plus an ideal
    (100, 100, 100) and an anti-ideal (0, 0, 0) anchor row; the anchors only fix the
    breakpoints and never enter the observations, the rankings or the indices.
  - FTRL-DOR = PreferenceSampler.find_map, Gamma(shape 2, scale 1) prior. BAYES-DOR =
    PreferenceSampler.run_hmc, Gamma(1, 1) prior: 2000 NUTS draws after 1000 warm-up
    iterations, seed SEED_BASE + t = 20260806 + t for session t (same seeds for LIN and PW).
  - Normalisation: every draw (and the MAP) is divided by U* = sum_j u_j(100), the value of
    the ideal alternative (= the sum of all piece weights), so u_j(100) / U* is the importance
    share of criterion j (Figures 7-8); the ranking tables report U(a) / U* * 100.
  - RAI[a, r] = share of draws in which region a has rank r (eq. (4)); PWI[a, b] = share of
    draws with U(a) >= U(b) (eq. (5)), so the diagonal is 100. Same computation as calc_rai /
    calc_poi of bdor.metrics.BenchmarkRunner. Tables 6, 7, 9 and 10 follow the four regions
    ranked first by FTRL-DOR after the last session.
  - The corner plots need the optional 'corner' package (pip install corner). Without it the
    script still runs, but croner_steps_*.png fall back to a plain-matplotlib version
    (histograms and scatter clouds, no 90% contours) that does not reproduce Figure 7; no
    other output depends on it.

(formerly case_study_lea.py at the repository root)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, so `import bdor` works without installing
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bdor.features import PiecewiseLinearTransformer
from bdor.engine import PreferenceSampler
from bdor.simulation import expand_subrank
from bdor.paths import CASE_STUDY_DIR

try:
    import corner as corner_pkg
    HAVE_CORNER = True
except ImportError:
    HAVE_CORNER = False

# ----------------------------------------------------------------------------
# Data: Table 4 (LEA scores 2024, in [0, 100], all criteria increasing)
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
# Row labels of Figure 8: each row shows the normalised marginal value function
# hat{u}_j of one criterion.
CRITERIA_HAT = [r"$\hat{u}_1(\cdot)$ (Prevention care)",
                r"$\hat{u}_2(\cdot)$ (Primary care)",
                r"$\hat{u}_3(\cdot)$ (Hospital care)"]

# Short region names used in the LaTeX tables (as in the paper).
SHORT = {
    "Autonomous Province of Bolzano": "A. P. of Bolzano",
    "Autonomous Province of Trento": "A. P. of Trento",
}

# ----------------------------------------------------------------------------
# Elicitation sessions (Figure 6): regions from best to worst; cards[m] is the
# raw number of blank cards between items[m] and items[m + 1], and each
# consecutive pair is one gap observation.
# ----------------------------------------------------------------------------
STEP_1 = {"items": ["Lombardy", "Marche", "Campania", "Basilicata"],
          "cards": [13, 28, 9]}
STEP_2 = {"items": ["Emilia Romagna", "Friuli Venezia Giulia", "Campania", "Abruzzo"],
          "cards": [21, 17, 13]}
STEP_3 = {"items": ["Tuscany", "Lombardy", "Lazio", "Sicily"],
          "cards": [10, 21, 18]}

# Both models (LIN: gamma_j = 1, PW: gamma_j = 3) use the same three sessions,
# cumulatively; they differ only in the marginal value functions.
SESSIONS = [STEP_1, STEP_2, STEP_3]

# Maximum number of blank cards between two consecutive levels (e_max in the
# paper, n_max in the engine). The same cap for both models, so both have a
# threshold vector Delta with 30 components.
E_MAX = 30
_all_cards = [c for s in SESSIONS for c in s["cards"]]
assert max(_all_cards) <= E_MAX, \
    f"declared card count {max(_all_cards)} exceeds the protocol cap e_max={E_MAX}"

# BAYES-DOR: NUTS draws and warm-up iterations per session; the chain after
# session t uses seed SEED_BASE + t.
SEED_BASE = 20260806
NUM_SAMPLES, NUM_WARMUP = 2000, 1000

# Module-level so that a caller can redirect the outputs before calling main().
OUT_DIR = str(CASE_STUDY_DIR)

# Figure 8 colours: blue = BAYES-DOR posterior mean and 90% band, red = FTRL-DOR.
C_BAYES_FILL = "#1f77b4"
C_BAYES_LINE = "#1f4fd8"
C_FTRL = "#d62728"


# ----------------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------------
def build_transformer(gamma):
    """Piecewise-linear feature map with equal-width breakpoints on [0, 100].

    The evaluation table is augmented with two anchor alternatives, an ideal one
    scoring 100 on every criterion and an anti-ideal one scoring 0, so that the
    equal-interval breakpoints of ``from_equal_intervals`` (from each column's
    min to its max) span the full [0, 100] measurement scale whatever the
    observed range. The anchors are used only here; they never enter the
    observations, the rankings or the RAI/PWI tables. The resulting breakpoints
    are asserted to equal ``np.linspace(0, 100, gamma + 1)`` for every criterion.

    Parameters
    ----------
    gamma : int
        Number of linear pieces per criterion (gamma_j, the same for all j).

    Returns
    -------
    PiecewiseLinearTransformer
        Transformer with 3 * gamma features.
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
    """Run the T sessions cumulatively, with FTRL-DOR and BAYES-DOR after each one.

    After session t the sampler holds the adjacent-gap observations of sessions
    1..t; FTRL-DOR (``find_map``) and BAYES-DOR (``run_hmc``, NUM_SAMPLES draws
    after NUM_WARMUP warm-up iterations, seed SEED_BASE + t) are fitted on all of
    them. Prints the gaps, timings and HMC diagnostics of every session.

    Parameters
    ----------
    case_name : str
        Label used in the console output ("LIN" or "PW").
    sessions : list of dict
        Sessions with keys "items" (region names, best to worst) and "cards"
        (blank cards between consecutive items).
    gamma : int
        Number of linear pieces per criterion.

    Returns
    -------
    transformer : PiecewiseLinearTransformer
    X : ndarray, shape (21, n_params)
        Feature matrix of the regions (n_params = 3 * gamma).
    results : list of dict
        One dict per session with keys "step" (1-based), "w_map" (FTRL-DOR
        estimate, shape (n_params + E_MAX,): weights then threshold increments),
        "samples" (posterior draws, shape (NUM_SAMPLES, n_params + E_MAX)),
        "diag" (dict with "divergences" and "max_rhat") and "t_hmc" (seconds).
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
# Metrics: same computation as calc_rai / calc_poi of
# bdor.metrics.BenchmarkRunner (RAI eq. (4), PWI eq. (5))
# ----------------------------------------------------------------------------
def calc_rai(f1, t, sam):
    """Rank Acceptability Indices of all alternatives.

    Parameters
    ----------
    f1 : int
        Number of alternatives.
    t : ndarray, shape (f1, n_params)
        Feature matrix.
    sam : ndarray, shape (n_draws, n_params + n_thresholds)
        Posterior draws; only the first n_params columns (the weights) are used.

    Returns
    -------
    ndarray, shape (f1, f1)
        RAI[a, r] = fraction of draws in which alternative a has rank r
        (0-based, 0 = best).
    """
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
    """Pairwise Winning Indices of all alternatives (calc_poi in bdor.metrics).

    Same arguments as :func:`calc_rai`. Returns an (f1, f1) array with
    PWI[a, b] = fraction of draws with U(a) >= U(b); the diagonal is 1.
    """
    rank_count = np.zeros((f1, f1))
    sam = np.atleast_2d(sam)
    w = sam[:, :t.shape[1]]
    vals = t @ w.T
    for col in range(vals.shape[1]):
        scores = vals[:, col]
        rank_count += (scores[:, None] >= scores[None, :])
    return rank_count / vals.shape[1]


def scaled_utilities(X, w):
    """Display-scaled utilities U(a) / U_max * 100 of all alternatives.

    U_max is the sum of all piece weights, i.e. the utility of the ideal
    alternative that scores 100 on every criterion (all its features equal 1).

    Parameters
    ----------
    X : ndarray, shape (n_alternatives, n_params)
        Feature matrix.
    w : array_like, shape (>= n_params,)
        Parameter vector; only the first n_params entries (the weights) are used.

    Returns
    -------
    ndarray, shape (n_alternatives,)
    """
    n_params = X.shape[1]
    w = np.asarray(w)[:n_params]
    return (X @ w) / np.sum(w) * 100.0


def bayes_mean_scaled_utilities(X, samples):
    """Posterior mean of the per-draw display-scaled utilities.

    ``samples`` has shape (n_draws, >= n_params); each draw is scaled as in
    :func:`scaled_utilities` before averaging. Returns shape (n_alternatives,).
    """
    n_params = X.shape[1]
    w = samples[:, :n_params]
    vals = (X @ w.T) / np.sum(w, axis=1)[None, :] * 100.0
    return vals.mean(axis=1)


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def criterion_features(bp, x_grid):
    """Feature block of one criterion on a grid (same clipping as the transformer).

    Parameters
    ----------
    bp : ndarray, shape (n_pieces + 1,)
        Breakpoints of the criterion.
    x_grid : ndarray, shape (n_grid,)
        Criterion values.

    Returns
    -------
    ndarray, shape (n_grid, n_pieces)
        Piece activations clip((x - bp[k]) / (bp[k + 1] - bp[k]), 0, 1).
    """
    n_seg = len(bp) - 1
    return np.stack([np.clip((x_grid - bp[k]) / (bp[k + 1] - bp[k]), 0.0, 1.0)
                     for k in range(n_seg)], axis=1)


def criterion_totals(samples, n_params, group_sizes):
    """Raw per-criterion totals u_j(100) = sum of that criterion's piece weights.

    No normalisation (callers divide by the row sum where needed); for
    gamma_j = 1, u_j(100) is the weight w_j itself.

    Parameters
    ----------
    samples : ndarray, shape (n_draws, >= n_params) or (>= n_params,)
        Posterior draws or a single parameter vector (treated as one draw).
    n_params : int
        Number of weights (leading entries of each draw).
    group_sizes : sequence of int
        Number of pieces per criterion (``transformer.n_w``).

    Returns
    -------
    ndarray, shape (n_draws, n_criteria)
    """
    w = np.atleast_2d(samples)[:, :n_params]
    idx = np.cumsum([0] + list(group_sizes))
    return np.stack([w[:, idx[j]:idx[j + 1]].sum(axis=1)
                     for j in range(len(group_sizes))], axis=1)


def plot_corner_steps(results, transformer, out_path, axis_labels):
    """Corner plot of the importance shares after every session (Figure 7).

    Plots the per-criterion totals u_j(100) (sum of that criterion's piece
    weights) normalised per draw by U_max = sum_l u_l(100), the utility of the
    ideal alternative scoring 100 on every criterion. With this normalisation
    U(ideal) = 1 and the plotted u_j(100) are the criterion importance shares,
    which sum to 1 in every draw. One colour per session (blue T=1, orange
    T=2, green T=3), with the FTRL-DOR estimate of each session overlaid
    (dashed vertical line on the diagonal panels, X marker on the bivariate
    panels). The 2-D panels show the 90% credible contour only.

    Parameters
    ----------
    results : list of dict
        Output of :func:`run_case` (uses "step", "samples" and "w_map").
    transformer : PiecewiseLinearTransformer
    out_path : str
        PNG file to write (200 dpi).
    axis_labels : list of str
        One axis label per criterion.

    Notes
    -----
    Uses ``corner.corner`` when the optional ``corner`` package is installed,
    otherwise a plain-matplotlib fallback (step histograms and scatter clouds)
    that does not reproduce the paper figure.
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
    # Fallback without the 'corner' package: overlaid density histograms and
    # scatter clouds with the same FTRL-DOR overlays (not the paper figure).
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
    """Grid of normalised marginal value functions (Figure 8).

    Rows = criteria, columns = sessions (T = 1..3). Each panel shows the
    BAYES-DOR posterior mean with its 90% pointwise credible band (blue) and
    the FTRL-DOR estimate (red) of u_j(x) / U_max on x in [0, 100], with U_max
    the utility of the ideal alternative (per-draw normalisation, as in
    :func:`plot_corner_steps`; the value at x = 100 is the criterion importance
    share). Panels in a row share the y-axis, so a row read left to right shows
    how that marginal value function evolves as the sessions accumulate.

    Parameters
    ----------
    results : list of dict
        Output of :func:`run_case` (uses "step", "samples" and "w_map").
    transformer : PiecewiseLinearTransformer
    out_path : str
        PNG file to write (200 dpi).

    Notes
    -----
    Sets the global ``font.size`` rcParam to 17, which also applies to every
    figure drawn afterwards; main() therefore draws this figure last.
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
    """Write the full 21 x 21 RAI table (Tables C.1, C.3, C.5, C.7).

    Parameters
    ----------
    rai : ndarray, shape (21, 21)
        Output of :func:`calc_rai` (rows = regions R1..R21, columns = ranks).
    path : str
        .tex file to write.
    caption, label : str
        LaTeX caption and label.
    """
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
    with open(path, "w", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def write_pwi_table(pwi, path, caption, label):
    """Write the full 21 x 21 PWI table (Tables C.2, C.4, C.6, C.8).

    ``pwi`` is the output of :func:`calc_pwi` (entry [a, b] = share of draws
    with U(a) >= U(b), so the diagonal is 100); other arguments as in
    :func:`write_rai_table`.
    """
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
    with open(path, "w", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def utilities_at_breakpoints(transformer, w_map):
    """Raw marginal values u_j at every breakpoint, for each criterion.

    u_j at the breakpoints is the cumulative sum of the piece weights, with
    u_j(bp_0) = 0. Returns a list with one array of length n_pieces_j + 1 per
    criterion (used for ftrl_tables.tex).
    """
    col_end = np.cumsum(transformer.n_w)
    col_start = np.insert(col_end[:-1], 0, 0)
    out = []
    for j in range(transformer.num_criteria):
        w_j = np.asarray(w_map)[int(col_start[j]):int(col_end[j])]
        out.append(np.concatenate([[0.0], np.cumsum(w_j)]))
    return out


def fmt_bp(x):
    """Breakpoint label: integer if whole, else one decimal (33.3, 66.7)."""
    return f"{x:.0f}" if abs(x - round(x)) < 1e-9 else f"{x:.1f}"


def write_ftrl_tables(lin, pw, path):
    """Write the FTRL-DOR parameter tables of both models (ftrl_tables.tex).

    LIN table: one row per session, u_j(0) and u_j(100). PW table: one row per
    session, u_j at every breakpoint (gamma_j = 3). Both on the raw parameter
    scale, i.e. not normalised by U_max as in Figures 7-8. Not in the paper.

    Parameters
    ----------
    lin, pw : tuple
        (transformer, results) of the LIN and PW models, ``results`` as
        returned by :func:`run_case`.
    path : str
        .tex file to write.
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
    with open(path, "w", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def ranking_from_utilities(u_disp):
    """Regions sorted by decreasing display utility; returns list of (region_idx, value)."""
    order = np.argsort(-u_disp)
    return [(int(i), float(u_disp[i])) for i in order]


def cumulative_reference_sets():
    """Reference alternatives seen up to and including each session.

    Returns a list with one set of region indices per session of SESSIONS.
    """
    refs, acc = [], set()
    for sess in SESSIONS:
        acc |= {IDX[n] for n in sess["items"]}
        refs.append(set(acc))
    return refs


def ranking_table_lines(X, res_list, caption, label, top_n=None):
    """LaTeX lines of an FTRL-DOR ranking table, one column group per session.

    Each group lists the regions by decreasing display-scaled utility
    (:func:`scaled_utilities`); the reference alternatives seen up to that
    session are in bold.

    Parameters
    ----------
    X : ndarray, shape (21, n_params)
        Feature matrix.
    res_list : list of dict
        Output of :func:`run_case` (uses "w_map").
    caption, label : str
        LaTeX caption and label.
    top_n : int, optional
        Keep only the first top_n positions (Tables 5 and 8 use 5); all 21
        regions by default.

    Returns
    -------
    list of str
    """
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
    """Write the full FTRL-DOR rankings of both models (ranking_tables.tex).

    One table per model, one column group per session, reference alternatives
    in bold (cumulative); utilities are U(a) / U_max * 100 with U_max the
    utility of the ideal alternative scoring 100 on every criterion. Not in the
    paper (Tables 5 and 8 are the top-5 versions).

    Parameters
    ----------
    lin, pw : tuple
        (transformer, X, results) of the LIN and PW models.
    path : str
        .tex file to write.
    """
    _, X_lin, res_lin = lin
    _, X_pw, res_pw = pw
    lines = [
        "% FTRL-DOR rankings for the LEA case study (adjacent-only likelihood,",
        "% e_max = n_max = 30).  Display scaling:",
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
    with open(path, "w", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def write_rai_evolution_table(rai_steps, top_idx, path, caption, label,
                              max_rank=8):
    """Write the RAI evolution of the top-ranked regions (Tables 6 and 9).

    One block of three rows (T = 1, 2, 3) per region; columns = ranks
    1..max_rank plus the aggregated tail ">max_rank".

    Parameters
    ----------
    rai_steps : list of ndarray, shape (21, 21)
        RAI matrix after each session.
    top_idx : list of int
        Region indices, in display order.
    path : str
        .tex file to write.
    caption, label : str
        LaTeX caption and label.
    max_rank : int, default 8
        Last rank shown individually.
    """
    cols = "l l|" + "C" * max_rank + "C"
    lines = [
        "\\begin{table}[h!]",
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
    with open(path, "w", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def write_pwi_evolution_table(pwi_steps, top_idx, path, caption, label):
    """Write the PWI among the top-ranked regions after each session (Tables 7 and 10).

    Rows = regions, one column group per session with the PWI against each of
    the same regions ('--' on the diagonal). ``pwi_steps`` is a list of
    (21, 21) PWI matrices, one per session; other arguments as in
    :func:`write_rai_evolution_table`.
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
    with open(path, "w", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    """Fit both models, write every output to OUT_DIR and print the console report."""
    t_start = time.perf_counter()
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 72)
    print(f"LEA case study - adjacent-only likelihood, e_max = n_max = {E_MAX}")
    print("=" * 72)

    # --- inference -----------------------------------------------------------
    print("\n--- Case LIN (gamma_j = 1, T=3 cumulative) ---")
    tr_lin, X_lin, res_lin = run_case("LIN", SESSIONS, gamma=1)
    print("\n--- Case PW (gamma_j = 3, T=3 cumulative) ---")
    tr_pw, X_pw, res_pw = run_case("PW", SESSIONS, gamma=3)

    # --- figures: corner plots (Figure 7 = LIN; PW extra) and Figure 8 -------
    # Paper notation (Section 2): lowercase u_j for the marginal value
    # functions, capital U(a) for the global value; the hat marks the per-draw
    # normalisation by the ideal alternative's utility.
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

    # Supplementary Tables C.1-C.8: full 21-region RAI/PWI after the first and
    # after the third session of each model (Section 6 compares T=1 with T=3).
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

    # --- Tables 6, 7, 9, 10: RAI / PWI evolution for the top-4 regions -------
    # The four regions ranked first by FTRL-DOR after the last session.
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

    # --- FTRL-DOR parameter and full ranking tables (not in the paper) -------
    write_ftrl_tables((tr_lin, res_lin), (tr_pw, res_pw),
                      os.path.join(OUT_DIR, "ftrl_tables.tex"))
    write_ranking_tables((tr_lin, X_lin, res_lin), (tr_pw, X_pw, res_pw),
                         os.path.join(OUT_DIR, "ranking_tables.tex"))
    # Tables 5 (LIN) and 8 (PW): top-5 FTRL-DOR ranking, one column group per
    # session.
    top5 = ranking_table_lines(
        X_lin, res_lin,
        "The five highest-scoring regions after each of the $T=3$ sessions, "
        "with the utilities computed from the FTRL-DOR parameter estimates "
        "of the corresponding session, in the case of linear marginal "
        "utility functions. The utilities are expressed as percentages of "
        "the utility of the ideal alternative, which scores 100 on every "
        "criterion. The reference alternatives are written in bold.",
        "tab:rank_lin_top5", top_n=5)
    with open(os.path.join(OUT_DIR, "table_rank_top5_lin.tex"), "w", newline="\n") as f:
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
    with open(os.path.join(OUT_DIR, "table_rank_top5_pw.tex"), "w", newline="\n") as f:
        f.write("\n".join(top5_pw) + "\n")

    # --- case_study_summary.csv: FTRL-DOR and posterior-mean rankings -------
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
                              index=False, lineterminator="\n")

    # --- console report (source of the numbers quoted in Secs. 6.1-6.2) ------
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

    # Sanity check: the FTRL-DOR LIN ranking after session 1 respects the order
    # declared in session 1.
    rank_s1 = ranking_from_utilities(scaled_utilities(X_lin, res_lin[0]["w_map"]))
    pos = {idx: r for r, (idx, _) in enumerate(rank_s1)}
    ref_order = [IDX[n] for n in STEP_1["items"]]
    assert all(pos[ref_order[i]] < pos[ref_order[i + 1]] for i in range(3)), \
        ("FTRL LIN T=1 ranking violates the reference order "
         "Lombardy > Marche > Campania > Basilicata")
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
    max_disp = max(scaled_utilities(X_lin, res_lin[-1]['w_map']).max(),
                   scaled_utilities(X_pw, res_pw[-1]['w_map']).max())
    print("  (every real region scores below the ideal: max display utility "
          f"{max_disp:.2f})")

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
