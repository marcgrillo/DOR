"""T-k equivalence analysis behind the paragraph "Number of sessions or number
of judgements?" of the paper: does accuracy depend on (T, k) only through the
total number of gap observations n_T = T*(k-1)?

Three views:
  1. full (T, k) cross-tab of each metric, per arm, with n_T annotated;
  2. the equal-n test: (T=5, k=5) vs (T=10, k=3), both n_T = 20;
  3. growth per doubling of n_T measured separately along T and along k.
Run from the DOR folder.
"""
import numpy as np

import make_paper_figs_v1 as m

TS = [1, 3, 5, 10]
KS = [3, 4, 5]
ARMS = [('FTRL_DOR_ADJJ1', 'FTRL'), ('BAYES_DOR_ADJJ1', 'BAYES'),
        ('SMAA_DOR_ADJJ1', 'SMAA'), ('DORLP_DOR_ADJJ1', 'LP'),
        ('FTRLDIR_DOR_ADJJ1', 'FTRLd'), ('BAYESDIR_DOR_ADJJ1', 'BAYESd')]
METS = [('asrs', 'ASR'), ('asps', 'ASP'), ('aios', 'AIO')]


def main():
    res = np.load(m.NPZ, allow_pickle=True)['results'].item()

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

    print(f"\n{'=' * 78}\nEQUAL-n TEST: (T=5,k=5) vs (T=10,k=3), "
          f"both n_T=20\n{'=' * 78}")
    for metric, mname in METS:
        print(f"\n  {mname}")
        for arm, label in ARMS:
            a, ca = m.mean_ci(m.pool(res, arm, metric, T=5, K=5))
            b, cb = m.mean_ci(m.pool(res, arm, metric, T=10, K=3))
            sep = "OVERLAP" if abs(a - b) < (ca + cb) else "DIFFER "
            print(f"    {label:<7} T=5,k=5: {a:.3f}+-{ca:.3f}   "
                  f"T=10,k=3: {b:.3f}+-{cb:.3f}   diff={b - a:+.3f}  {sep}")

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
