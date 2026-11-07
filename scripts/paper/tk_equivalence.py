"""T-k equivalence analysis behind the paragraph "Number of sessions or number
of judgements?" of the paper: does accuracy depend on (T, k) only through the
total number of gap observations n_T = T*(k-1)?

Three views:
  1. full (T, k) cross-tab of each metric, per arm, with n_T annotated;
  2. the equal-n test: (T=5, k=5) vs (T=10, k=3), both n_T = 20;
  3. growth per doubling of n_T measured separately along T and along k.

NOTE on (2): configurations that differ in k are generated from different
decision problems (verified: the evaluation matrices of the K=3 and K=5
datasets differ), so the two designs are NOT paired and the equal-n
comparison uses a two-sided Mann-Whitney test, unlike the paired
signed-rank tests used for same-data comparisons elsewhere.
Run from the DOR folder.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, so `import bdor` works without installing
import numpy as np
from scipy.stats import mannwhitneyu

from bdor import results as m

TS = [1, 3, 5, 10]
KS = [3, 4, 5]
ARMS = [('FTRL_DOR_ADJJ1', 'FTRL'), ('BAYES_DOR_ADJJ1', 'BAYES'),
        ('SMAA_DOR_ADJJ1', 'SMAA'), ('DORLP_DOR_ADJJ1', 'LP'),
        ('FTRLDIR_DOR_ADJJ1', 'FTRLd'), ('BAYESDIR_DOR_ADJJ1', 'BAYESd')]
METS = [('asrs', 'ASR'), ('asps', 'ASP'), ('aios', 'AIO')]


def main():
    res = m.load_results()

    for metric, mname in METS:
        print(f"\n{'=' * 78}\n{mname}: (T, k) cross-tab, mean +- ci  "
              f"[n_T = T*(k-1)]\n{'=' * 78}")
        for arm, label in ARMS:
            print(f"\n  {label}")
            print(f"    {'':>6}" + "".join(f"{'k=' + str(k):>22}" for k in KS))
            for T in TS:
                row = f"    T={T:<4}"
                for k in KS:
                    mean, ci = m.mean_ci(m.pool(res, arm, metric, T=T, K=k))
                    row += f"{mean:.3f}+-{ci:.3f} (n={T * (k - 1):>2}) ".rjust(22)
                print(row)

    print(f"\n{'=' * 78}\nEQUAL-n TEST: (T=5,k=5) vs (T=10,k=3), both n_T=20"
          f"\n(unpaired: the two designs use different decision problems)"
          f"\n{'=' * 78}")
    for metric, mname in METS:
        print(f"\n  {mname}")
        for arm, label in ARMS:
            x = m.pool(res, arm, metric, T=5, K=5)
            y = m.pool(res, arm, metric, T=10, K=3)
            x, y = x[~np.isnan(x)], y[~np.isnan(y)]
            p = mannwhitneyu(x, y, alternative='two-sided').pvalue
            print(f"    {label:<7} T=5,k=5: {x.mean():.3f}   "
                  f"T=10,k=3: {y.mean():.3f}   diff={y.mean() - x.mean():+.3f}"
                  f"   Mann-Whitney p={p:.3f}")

    # The equal-n AIO advantage of the more-sessions design exists only under
    # noise: with a consistent DM the two designs are equivalent. Same
    # mechanism as the k=5 AIO dip (see analyze_subset_density_v1.py) --
    # larger subsets ask for finer comparisons, which a noisy DM reverses
    # more often -- and not a coverage effect, which would not vanish at 0%.
    print(f"\n{'=' * 78}\nEQUAL-n BY NOISE LEVEL: (T=5,k=5) vs (T=10,k=3), "
          f"AIO\n{'=' * 78}")
    for inc in (0, 15, 35, 50):
        print(f"  |F_inc| = {inc}%")
        for arm, label in ARMS:
            x = m.pool(res, arm, 'aios', T=5, K=5, Inc=inc)
            y = m.pool(res, arm, 'aios', T=10, K=3, Inc=inc)
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
            t_lo, _ = m.mean_ci(m.pool(res, arm, metric, T=1))
            t_hi, _ = m.mean_ci(m.pool(res, arm, metric, T=10))
            k_lo, _ = m.mean_ci(m.pool(res, arm, metric, K=3))
            k_hi, _ = m.mean_ci(m.pool(res, arm, metric, K=5))
            # marginals: mean (k-1) over k in {3,4,5} is 3; mean T is 4.75
            g_t = (t_hi / t_lo) ** (1 / np.log2(10.0))
            g_k = (k_hi / k_lo) ** (1 / np.log2(2.0))
            print(f"    {label:<7} along T: x{g_t:.3f} per doubling   "
                  f"along k: x{g_k:.3f}   ratio {g_t / g_k:.2f}")


if __name__ == "__main__":
    main()
