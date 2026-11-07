"""Subset-size and noise statistics of the grid data behind statements of Sec. 5.1 and 5.3.

Expands every declared ranking of the 192 S = 10, AR == K grid datasets into its adjacent
gap observations (bdor.simulation.expand_subrank, e_max = 5) and compares them with the
true utilities. It prints:
  - per subset size k: the mean true-utility distance between adjacent declared levels,
    the share of gaps with zero blank cards, the share of flipped adjacent judgements
    (declared order contradicting the true utilities; Finc > 0 only) and the mean
    true-utility gap between the best and second-best alternative of the problem;
  - the flipped and the zero-card shares by k and Finc level;
  - the flipped share by |A| and by m (Finc > 0 only);
  - from the grid results, the AIO of the four B-DOR arms by k at each Finc level (pooled
    over |A|, m and T in {1, 3, 5, 10}), with a two-sided Mann-Whitney test of k = 4
    against k = 5. Splitting AIO by Finc separates the two candidate causes of the AIO
    peak at k = 4: the zero-card share grows with k at every Finc level, while flipped
    judgements occur only when Finc > 0.

The first AIO block pools each arm over all its stored replications (bdor.results.pool
without match): 20 per cell for FTRL-DOR, FTRLdir and BAYES-DOR, 5 for BAYESdir. The last
block repeats the means with each family read on the replications it shares with its
direction-only partner, the convention of Figure 2 and Tables 3 and B.1-B.5 (BAYES-DOR on
5); this is the reading of the AIO-by-k values quoted in Sec. 5.3.

Paper:   Sec. 5.1, flip shares after Table 1 (21.0% at m = 3 to 23.9% at m = 9, 18.7% at
         k = 3 to 24.5% at k = 5: adjacent judgements, Finc > 0); Sec. 5.3, subset-size
         paragraph (mean distance 0.120 at k = 3 to 0.090 at k = 5; AIO by k at Finc = 0%)
Inputs:  data/datasets/grid_F1=*_F2=*_S=10_AR=<k>_K=<k>_Inc=*/ (the full local datasets
         folder; the script stops if any of the 192 folders is missing),
         results/grid_results_joint_v1.npz
Outputs: stdout only
Runtime: a few seconds

Usage:
    python scripts/paper/subset_density.py

(formerly analyze_subset_density_v1.py)
"""
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, so `import bdor` works without installing

import numpy as np
from scipy.stats import mannwhitneyu

from bdor import utils
from bdor.paths import DATASETS_DIR
from bdor.results import PARTNER, load_results, mean_ci, pool
from bdor.simulation import expand_subrank

F1_LEVELS = [10, 20, 35, 50]
F2_LEVELS = [3, 5, 7, 9]
INC_LEVELS = [0, 15, 35, 50]
K_LEVELS = [3, 4, 5]
STEPS = 10
ETA_MAX = 5


def main():
    """Compute the dataset statistics and the AIO-by-k block, and print them."""
    dU = defaultdict(list)                      # k -> adjacent |delta U|
    top_gap = defaultdict(list)                 # k -> U(best) - U(second)
    zero = defaultdict(lambda: [0, 0])          # k -> [eta==0, total]
    zero_ki = defaultdict(lambda: [0, 0])       # (k, inc)
    flip_k = defaultdict(lambda: [0, 0])        # k -> [flipped, total] (inc>0)
    flip_ki = defaultdict(lambda: [0, 0])       # (k, inc)
    flip_a = defaultdict(lambda: [0, 0])        # |A| (inc>0)
    flip_m = defaultdict(lambda: [0, 0])        # m   (inc>0)
    missing = 0

    for f1 in F1_LEVELS:
        for f2 in F2_LEVELS:
            for inc in INC_LEVELS:
                for k in K_LEVELS:
                    fold = (f"grid_F1={f1}_F2={f2}_S={STEPS}"
                            f"_AR={k}_K={k}_Inc={inc}")
                    try:
                        _, _, prefs_all, Us = utils.read_dataset(
                            os.path.join(DATASETS_DIR, fold), f1, f2, STEPS)
                    except FileNotFoundError:
                        missing += 1
                        continue
                    for u, prefs in zip(Us, prefs_all):
                        u = np.asarray(u, dtype=float).ravel()
                        srt = np.sort(u)[::-1]
                        top_gap[k].append(srt[0] - srt[1])
                        for sub in prefs:
                            items = sub.get('levels', sub['items'])
                            gaps = expand_subrank(items, sub['cards'],
                                                  'adjacent', eta_cap=ETA_MAX)
                            for up, lo, e in gaps:
                                ui = u[np.array(up)].mean()
                                uj = u[np.array(lo)].mean()
                                dU[k].append(abs(ui - uj))
                                zero[k][0] += int(e == 0)
                                zero[k][1] += 1
                                zero_ki[(k, inc)][0] += int(e == 0)
                                zero_ki[(k, inc)][1] += 1
                                bad = int(ui < uj)
                                flip_ki[(k, inc)][0] += bad
                                flip_ki[(k, inc)][1] += 1
                                if inc > 0:
                                    for d, key in ((flip_k, k), (flip_a, f1),
                                                   (flip_m, f2)):
                                        d[key][0] += bad
                                        d[key][1] += 1

    if missing:
        total = len(F1_LEVELS) * len(F2_LEVELS) * len(INC_LEVELS) * len(K_LEVELS)
        sys.exit(f"subset_density.py needs all {total} grid dataset folders with AR == K under "
                 f"{DATASETS_DIR} ({missing} missing). They are not part of the public "
                 "repository; see data/README.md.")

    print("=== per subset size k (whole grid) ===")
    for k in K_LEVELS:
        f, n = flip_k[k]
        z, zn = zero[k]
        print(f"  k={k}: adjacent |dU| {np.mean(dU[k]):.4f} | "
              f"zero-card share {z / zn:.1%} | "
              f"flipped share (noisy levels) {f / n:.1%} | "
              f"best-vs-2nd gap {np.mean(top_gap[k]):.4f} | "
              f"gaps observed {zn}")

    print("\n=== flipped share by k and nominal noise level ===")
    print(f"    {'':>8}" + "".join(f"{'k=' + str(k):>10}" for k in K_LEVELS))
    for inc in INC_LEVELS:
        row = f"    inc={inc:<4}"
        for k in K_LEVELS:
            f, n = flip_ki[(k, inc)]
            row += f"{f / max(n, 1):>9.1%} "
        print(row)

    print("\n=== zero-card share by k and nominal noise level ===")
    print(f"    {'':>8}" + "".join(f"{'k=' + str(k):>10}" for k in K_LEVELS))
    for inc in INC_LEVELS:
        row = f"    inc={inc:<4}"
        for k in K_LEVELS:
            z, n = zero_ki[(k, inc)]
            row += f"{z / max(n, 1):>9.1%} "
        print(row)

    print("\n=== flipped share by |A| and by m (noisy levels, pooled over k) ===")
    print("  " + "  ".join(f"|A|={a}: {flip_a[a][0] / flip_a[a][1]:.1%}"
                           for a in F1_LEVELS))
    print("  " + "  ".join(f"m={b}: {flip_m[b][0] / flip_m[b][1]:.1%}"
                           for b in F2_LEVELS))

    # ---- AIO against k, separately at each noise level (see the docstring) ----
    print("\n=== AIO by k at each noise level (dip only where flips exist) ===")
    res = load_results()
    arms = [('FTRL_DOR_ADJJ1', 'FTRL'), ('BAYES_DOR_ADJJ1', 'BAYES'),
            ('FTRLDIR_DOR_ADJJ1', 'FTRLd'), ('BAYESDIR_DOR_ADJJ1', 'BAYESd')]
    for inc in INC_LEVELS:
        print(f"  |F_inc| = {inc}%")
        for arm, lbl in arms:
            vals = [pool(res, arm, 'aios', K=k, Inc=inc) for k in K_LEVELS]
            vals = [v[~np.isnan(v)] for v in vals]
            means = [v.mean() for v in vals]
            p = mannwhitneyu(vals[1], vals[2], alternative='two-sided').pvalue
            tag = "monotone" if means[2] >= means[1] else "DIP at k=5"
            print(f"    {lbl:<7} {means[0]:.3f} {means[1]:.3f} {means[2]:.3f}"
                  f"   k4->k5 {means[2] - means[1]:+.3f} (p={p:.3f})  {tag}")

    # ---- the same means with each family on the replications shared with its partner ----
    print("\n=== AIO by k (k = 3/4/5), each family read on the replications it shares with "
          "its direction-only partner (as in Figure 2 and Tables B.1-B.5) ===")
    for inc in INC_LEVELS:
        cells = []
        for arm, lbl in arms:
            means = [mean_ci(pool(res, arm, 'aios', K=k, Inc=inc, match=PARTNER[arm]))[0]
                     for k in K_LEVELS]
            cells.append(f"{lbl} " + "/".join(f"{x:.3f}" for x in means))
        print(f"  |F_inc| = {inc:>2}%:  " + "   ".join(cells))


if __name__ == "__main__":
    main()
