"""
Case study for the B-DOR paper revision (Section 6): ranking the 21 Italian
regions on healthcare performance (LEA criteria) under the ADJACENT-ONLY
likelihood and the new inference engine.

Migration-plan items E1-E6 (Decision 1: e_max = 30):
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

SESSIONS_T1 = [STEP_1]
SESSIONS_T3 = [STEP_1, STEP_2, STEP_3]

# Protocol cap (Decision 1): e_max = n_max = 30 for both cases (E3).
E_MAX = 30
_all_cards = [c for s in SESSIONS_T3 for c in s["cards"]]
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
    """Explicit breakpoints on the common [0,100] scale (see module docstring)."""
    ch_p = [np.linspace(0.0, 100.0, gamma + 1) for _ in range(LEA_TABLE.shape[1])]
    return PiecewiseLinearTransformer(ch_p)


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
        pairs = expand_subrank(items, sess["cards"], expansion="adjacent")
        for up_idx, down_idx, eta in pairs:
            sampler.add_preference(up_idx, down_idx, eta)

        t0 = time.perf_counter()
        w_map = sampler.find_map()
        t_map = time.perf_counter() - t0

        t0 = time.perf_counter()
        samples, diag = sampler.run_hmc(num_samples=NUM_SAMPLES, num_warmup=NUM_WARMUP,
                                        seed=SEED_BASE + step, return_diagnostics=True)
        t_hmc = time.perf_counter() - t0

        print(f"[{case_name} step {step}] pairs added: {pairs}")
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
    Utilities are on the raw parameter scale, as in the paper's FTRL tables.
    """
    m = transformer.num_criteria
    col_end = np.cumsum(transformer.n_w)
    col_start = np.insert(col_end[:-1], 0, 0)
    x_grid = np.linspace(0.0, 100.0, 201)

    plt.rcParams.update({"font.size": 13})
    fig, axes = plt.subplots(1, m, figsize=(4.6 * m, 4.2), sharey=True)
    for j in range(m):
        ax = axes[j]
        bp = np.asarray(transformer.ch_p[j], dtype=float)
        F = criterion_features(bp, x_grid)
        s, e = int(col_start[j]), int(col_end[j])

        curves = F @ samples[:, s:e].T           # (n_grid, n_draws)
        y_lo = np.percentile(curves, 5, axis=1)
        y_hi = np.percentile(curves, 95, axis=1)
        ax.fill_between(x_grid, y_lo, y_hi, color=C_BAYES_FILL, alpha=0.25,
                        linewidth=0, label="BAYES 90% credible band")
        ax.plot(x_grid, curves.mean(axis=1), color=C_BAYES_LINE, lw=2,
                label="BAYES posterior mean")
        ax.plot(x_grid, F @ w_map[s:e], color=C_FTRL, lw=2, label="FTRL (MAP)")

        ax.set_title(CRITERIA[j])
        ax.set_xlabel("Criterion value")
        if j == 0:
            ax.set_ylabel("Marginal utility")
            ax.legend(fontsize=10, loc="upper left", framealpha=0.95)
        ax.set_xlim(0, 100)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_corner(samples, n_params, out_path):
    """Corner plot of the T=1 posterior weights (w1, w2, w3)."""
    w = samples[:, :n_params]
    labels = [r"$w_1$", r"$w_2$", r"$w_3$"]
    if HAVE_CORNER:
        fig = corner_pkg.corner(w, labels=labels, show_titles=True,
                                quantiles=[0.05, 0.5, 0.95],
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


def write_ftrl_tables(t1, t3, path):
    """
    T=1 table: U_j(0), U_j(100) (cf. paper tab:ftrl_ex_t1).
    T=3 table: one row per step, U_j at every breakpoint (cf. tab:ftrl_ex_t3).
    """
    tr1, res1 = t1
    tr3, res3 = t3
    lines = [
        "% FTRL-DOR parameter estimates for the LEA case study",
        "% (adjacent-only likelihood, e_max = n_max = 30).",
        "",
        "% --- T = 1, one linear piece per criterion (gamma_j = 1) ---",
        "\\begin{table}[h!]",
        "\\centering",
        "\\caption{Parameters estimated through FTRL-DOR in the case of a "
        "single-session procedure ($T=1$) with one breakpoint per criterion}",
        "\\label{tab:ftrl_ex_t1}",
        "\\begin{tabular}{cc|cc|cc}",
        "\\toprule",
        "$U_1(0)$&$U_1(100)$&$U_2(0)$&$U_2(100)$&$U_3(0)$&$U_3(100)$\\\\",
        "\\midrule",
    ]
    u1 = utilities_at_breakpoints(tr1, res1[0]["w_map"])
    lines.append("&".join(f"{u1[j][k]:.4f}".rstrip("0").rstrip(".") if u1[j][k] == 0
                          else f"{u1[j][k]:.4f}"
                          for j in range(3) for k in range(2)) + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]

    bps = tr3.ch_p[0]
    header = "&".join(f"$U_{{{j + 1}}}({fmt_bp(bp)})$"
                      for j in range(3) for bp in bps)
    cols = "X|" + "|".join("C" * len(bps) for _ in range(3))
    lines += [
        "% --- T = 3, gamma_j = 3 equally spaced breakpoints per criterion ---",
        "\\begin{table}[H]",
        "\\centering",
        "\\footnotesize",
        "\\caption{Parameters estimated through FTRL-DOR in the case of a "
        "three-session procedure ($T=3$) with $\\gamma_j=3$ equally spaced "
        "breakpoints per criterion}",
        "\\label{tab:ftrl_ex_t3}",
        f"\\begin{{tabularx}}{{\\textwidth}}{{{cols}}}",
        "\\toprule",
        "&" + header + "\\\\",
        "\\midrule",
    ]
    for res in res3:
        u3 = utilities_at_breakpoints(tr3, res["w_map"])
        vals = "&".join("0" if u3[j][k] == 0 else f"{u3[j][k]:.4f}"
                        for j in range(3) for k in range(len(bps)))
        lines.append(f"Step {res['step']}&" + vals + "\\\\")
    lines += ["\\bottomrule", "\\end{tabularx}", "\\end{table}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def ranking_from_utilities(u_disp):
    """Regions sorted by decreasing display utility; returns list of (region_idx, value)."""
    order = np.argsort(-u_disp)
    return [(int(i), float(u_disp[i])) for i in order]


def write_ranking_tables(t1, t3, path):
    """
    T=1: FTRL ranking (cf. paper tab:RANK_Ex1); T=3: FTRL ranking after each
    step (cf. tab:RANK_Ex2).  Reference alternatives in bold (cumulative for
    T=3).  Display scaling (plan E5): U_disp(a) = U(a)/U_max*100 with U_max the
    utility of the ideal alternative scoring 100 on every criterion.
    """
    tr1, X1, res1 = t1
    tr3, X3, res3 = t3

    refs_t1 = {IDX[n] for n in STEP_1["items"]}
    refs_t3_steps = []
    acc = set()
    for sess in SESSIONS_T3:
        acc |= {IDX[n] for n in sess["items"]}
        refs_t3_steps.append(set(acc))

    lines = [
        "% FTRL-DOR rankings for the LEA case study (adjacent-only likelihood,",
        "% e_max = n_max = 30).  Display scaling (plan E5):",
        "% U_disp(a) = U(a) / U_max * 100, U_max = sum of all piece weights",
        "% (utility of the ideal alternative scoring 100 on every criterion).",
        "",
        "\\begin{table}[h!]",
        "\\centering",
        "\\caption{Ranking of the alternatives considering the parameters "
        "estimated through FTRL-DOR ($T=1$). The reference alternatives are "
        "written in bold.}",
        "\\label{tab:RANK_Ex1}",
        "\\begin{tabular}{c|lc}",
        "\\toprule",
        "Rank&Region & $U\\left(a\\right)$ \\\\",
        "\\midrule",
    ]
    u1 = scaled_utilities(X1, res1[0]["w_map"])
    for rank, (idx, val) in enumerate(ranking_from_utilities(u1), start=1):
        name, sval = REGIONS[idx], f"{val:.4f}"
        if idx in refs_t1:
            name, sval = f"\\textbf{{{name}}}", f"\\textbf{{{sval}}}"
        lines.append(f"{rank} & {name} & {sval} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]

    rankings = [ranking_from_utilities(scaled_utilities(X3, res["w_map"]))
                for res in res3]
    lines += [
        "\\begin{table}[h!]",
        "\\centering",
        "\\caption{Ranking of the alternatives considering the parameters "
        "estimated through FTRL-DOR after each of the $T=3$ sessions. The "
        "reference alternatives are written in bold}",
        "\\label{tab:RANK_Ex2}",
        "\\begin{tabularx}{\\textwidth}{c|Cc|Cc|Cc}",
        "\\toprule",
        "\\multirow{2}{*}{Rank}&\\multicolumn{2}{c}{Step 1}&"
        "\\multicolumn{2}{c}{Step 2}&\\multicolumn{2}{c}{Step 3}\\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
        "&Region&$U\\left(a\\right)$&Region&$U\\left(a\\right)$&"
        "Region&$U\\left(a\\right)$\\\\",
        "\\midrule",
    ]
    for rank in range(NUM_REGIONS):
        cells = []
        for s in range(3):
            idx, val = rankings[s][rank]
            name = SHORT.get(REGIONS[idx], REGIONS[idx])
            sval = f"{val:.2f}"
            if idx in refs_t3_steps[s]:
                name, sval = f"\\textbf{{{name}}}", f"\\textbf{{{sval}}}"
            cells += [name, sval]
        lines.append(f"{rank + 1} & " + " & ".join(cells) + " \\\\")
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
    print("\n--- Case T=1 (gamma_j = 1) ---")
    tr1, X1, res1 = run_case("T1", SESSIONS_T1, gamma=1)
    print("\n--- Case T=3 (gamma_j = 3, cumulative) ---")
    tr3, X3, res3 = run_case("T3", SESSIONS_T3, gamma=3)

    # --- figures -------------------------------------------------------------
    plot_utilities(tr1, res1[0]["w_map"], res1[0]["samples"],
                   os.path.join(OUT_DIR, "utilities_t1_plot.png"))
    plot_corner(res1[0]["samples"], tr1.total_params,
                os.path.join(OUT_DIR, "croner.png"))
    for res in res3:
        plot_utilities(tr3, res["w_map"], res["samples"],
                       os.path.join(OUT_DIR, f"t_3_{res['step']}.png"))

    # --- RAI / PWI supplementary tables -------------------------------------
    rai_t1 = calc_rai(NUM_REGIONS, X1, res1[0]["samples"])
    pwi_t1 = calc_pwi(NUM_REGIONS, X1, res1[0]["samples"])
    rai_t3 = calc_rai(NUM_REGIONS, X3, res3[-1]["samples"])
    pwi_t3 = calc_pwi(NUM_REGIONS, X3, res3[-1]["samples"])

    write_rai_table(rai_t1, os.path.join(OUT_DIR, "table_S1_rai_t1.tex"),
                    "Rank Acceptability Indices (\\%) of all regions computed "
                    "from the posterior distribution estimated in the "
                    "single-session case ($T=1$)", "tab:S1_rai_t1")
    write_pwi_table(pwi_t1, os.path.join(OUT_DIR, "table_S2_pwi_t1.tex"),
                    "Pairwise Winning Indices (\\%) of all regions computed "
                    "from the posterior distribution estimated in the "
                    "single-session case ($T=1$)", "tab:S2_pwi_t1")
    write_rai_table(rai_t3, os.path.join(OUT_DIR, "table_S3_rai_t3.tex"),
                    "Rank Acceptability Indices (\\%) of all regions computed "
                    "from the final posterior distribution of the "
                    "three-session case ($T=3$)", "tab:S3_rai_t3")
    write_pwi_table(pwi_t3, os.path.join(OUT_DIR, "table_S4_pwi_t3.tex"),
                    "Pairwise Winning Indices (\\%) of all regions computed "
                    "from the final posterior distribution of the "
                    "three-session case ($T=3$)", "tab:S4_pwi_t3")

    # --- FTRL parameter and ranking tables -----------------------------------
    write_ftrl_tables((tr1, res1), (tr3, res3),
                      os.path.join(OUT_DIR, "ftrl_tables.tex"))
    write_ranking_tables((tr1, X1, res1), (tr3, X3, res3),
                         os.path.join(OUT_DIR, "ranking_tables.tex"))

    # --- long-format CSV dump -------------------------------------------------
    rows = []
    for case_name, X, res_list in [("T1", X1, res1), ("T3", X3, res3)]:
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
    print("T=1 FTRL ranking (display-scaled utilities):")
    rank_t1 = ranking_from_utilities(scaled_utilities(X1, res1[0]["w_map"]))
    for rank, (idx, val) in enumerate(rank_t1, start=1):
        print(f"  {rank:2d}. {REGIONS[idx]:<32s} {val:7.2f}")

    # Sanity check (E1/E2): the reference-set order must be respected.
    pos = {idx: r for r, (idx, _) in enumerate(rank_t1)}
    ref_order = [IDX[n] for n in STEP_1["items"]]
    assert all(pos[ref_order[i]] < pos[ref_order[i + 1]] for i in range(3)), \
        "FTRL T=1 ranking violates the reference order Lombardy > Marche > Campania > Basilicata"
    print("  [check] reference order Lombardy > Marche > Campania > Basilicata respected")

    top_rai = np.argsort(-rai_t1[:, 0])[:3]
    print("\nT=1 BAYES top-3 by RAI rank-1:")
    for idx in top_rai:
        print(f"  {REGIONS[idx]:<32s} RAI(1) = {100 * rai_t1[idx, 0]:.1f}%")

    print("\nHMC diagnostics:")
    for case_name, res_list in [("T1", res1), ("T3", res3)]:
        for res in res_list:
            d = res["diag"]
            print(f"  {case_name} step {res['step']}: divergences={d['divergences']}, "
                  f"max split-Rhat={d['max_rhat']:.4f} ({res['t_hmc']:.1f}s)")

    print(f"\nOutputs written to {os.path.abspath(OUT_DIR)}")
    print(f"Total runtime: {time.perf_counter() - t_start:.1f}s")


if __name__ == "__main__":
    main()
