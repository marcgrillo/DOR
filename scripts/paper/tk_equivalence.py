"""Check whether accuracy depends on (T, k) only through n_T = T (k - 1).

Backs the paragraph 'Larger sets or more sessions?' of Sec. 5.3: in the grid design every
session yields k - 1 adjacent gap observations, so n_T = T (k - 1) after T sessions. From
the aggregated grid results it prints
  1. the (T, k) cross-tab of ASR, ASP and AIO (mean +- 95% CI) per arm, with n_T;
  2. the equal-n comparison (T = 5, k = 5) against (T = 10, k = 3), both n_T = 20, pooled
     and, for AIO, separately at each Finc level;
  3. the growth factor per doubling of n_T along T and along k (marginal means).

Configurations that differ in k are generated from different decision problems (the
evaluation matrices of the K = 3 and K = 5 datasets differ), so the two equal-n designs
are not paired and are compared with a two-sided Mann-Whitney test, unlike the paired
signed-rank tests used for same-data comparisons elsewhere.

Every arm is pooled over all its stored replications (bdor.results.pool without match):
20 per cell, 5 for BAYESdir. The SMAA and LP rows are the adapted per-gap baselines
SMAA_DOR_ADJJ1 and DORLP_DOR_ADJJ1 (bdor/baselines/smaa_adapted.py), stored for all ten
sessions, not the paper's SMAA-DOR and DOR (SMAA_DOR_ORIG / DORLP_DOR_ORIG, T = 1 only).

Paper:   Sec. 5.3, 'Larger sets or more sessions?'
Inputs:  results/grid_results_joint_v1.npz
Outputs: stdout only
Runtime: a few seconds

Usage:
    python scripts/paper/tk_equivalence.py

(formerly analyze_tk_equivalence.py)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, so `import bdor` works without installing

import numpy as np
from scipy.stats import mannwhitneyu

from bdor.results import load_results, mean_ci, pool

TS = [1, 3, 5, 10]
KS = [3, 4, 5]
ARMS = [('FTRL_DOR_ADJJ1', 'FTRL'), ('BAYES_DOR_ADJJ1', 'BAYES'),
        ('SMAA_DOR_ADJJ1', 'SMAA'), ('DORLP_DOR_ADJJ1', 'LP'),
        ('FTRLDIR_DOR_ADJJ1', 'FTRLd'), ('BAYESDIR_DOR_ADJJ1', 'BAYESd')]
METS = [('asrs', 'ASR'), ('asps', 'ASP'), ('aios', 'AIO')]


def main():
    """Print the three views described in the module docstring."""
    res = load_results()

    for metric, mname in METS:
        print(f"\n{'=' * 78}\n{mname}: (T, k) cross-tab, mean +- ci  "
              f"[n_T = T*(k-1)]\n{'=' * 78}")
        for arm, label in ARMS:
            print(f"\n  {label}")
            print(f"    {'':>6}" + "".join(f"{'k=' + str(k):>22}" for k in KS))
            for T in TS:
                row = f"    T={T:<4}"
                for k in KS:
                    mean, ci = mean_ci(pool(res, arm, metric, T=T, K=k))
                    row += f"{mean:.3f}+-{ci:.3f} (n={T * (k - 1):>2}) ".rjust(22)
                print(row)

    print(f"\n{'=' * 78}\nEQUAL-n TEST: (T=5,k=5) vs (T=10,k=3), both n_T=20"
          f"\n(unpaired: the two designs use different decision problems)"
          f"\n{'=' * 78}")
    for metric, mname in METS:
        print(f"\n  {mname}")
        for arm, label in ARMS:
            x = pool(res, arm, metric, T=5, K=5)
            y = pool(res, arm, metric, T=10, K=3)
            x, y = x[~np.isnan(x)], y[~np.isnan(y)]
            p = mannwhitneyu(x, y, alternative='two-sided').pvalue
            print(f"    {label:<7} T=5,k=5: {x.mean():.3f}   "
                  f"T=10,k=3: {y.mean():.3f}   diff={y.mean() - x.mean():+.3f}"
                  f"   Mann-Whitney p={p:.3f}")

    # AIO by noise level: a noise effect (larger subsets ask for finer
    # comparisons, which a noisy DM reverses more often; the mechanism of the
    # k = 5 AIO dip, see subset_density.py) should vanish at Finc = 0%, a
    # coverage effect should not.
    print(f"\n{'=' * 78}\nEQUAL-n BY NOISE LEVEL: (T=5,k=5) vs (T=10,k=3), "
          f"AIO\n{'=' * 78}")
    for inc in (0, 15, 35, 50):
        print(f"  |F_inc| = {inc}%")
        for arm, label in ARMS:
            x = pool(res, arm, 'aios', T=5, K=5, Inc=inc)
            y = pool(res, arm, 'aios', T=10, K=3, Inc=inc)
            x, y = x[~np.isnan(x)], y[~np.isnan(y)]
            p = mannwhitneyu(x, y, alternative='two-sided').pvalue
            print(f"    {label:<7} T=5,k=5: {x.mean():.3f}   "
                  f"T=10,k=3: {y.mean():.3f}   diff={y.mean() - x.mean():+.3f}"
                  f"   p={p:.3f}")

    print(f"\n{'=' * 78}\nGROWTH PER DOUBLING OF n_T, along each factor "
          f"(marginal means)\n{'=' * 78}")
    for metric, mname in METS:
        print(f"\n  {mname}")
        for arm, label in ARMS:
            t_lo, _ = mean_ci(pool(res, arm, metric, T=1))
            t_hi, _ = mean_ci(pool(res, arm, metric, T=10))
            k_lo, _ = mean_ci(pool(res, arm, metric, K=3))
            k_hi, _ = mean_ci(pool(res, arm, metric, K=5))
            # n_T = T (k - 1): with the other factor averaged, the mean n_T grows
            # from 3 to 30 (log2(10) doublings) from T = 1 to T = 10, and from
            # 9.5 to 19 (one doubling) from k = 3 to k = 5
            g_t = (t_hi / t_lo) ** (1 / np.log2(10.0))
            g_k = (k_hi / k_lo) ** (1 / np.log2(2.0))
            print(f"    {label:<7} along T: x{g_t:.3f} per doubling   "
                  f"along k: x{g_k:.3f}   ratio {g_t / g_k:.2f}")


if __name__ == "__main__":
    main()
