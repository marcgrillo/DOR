"""Why AIO peaks at k=4: what enlarging the displayed subset does to the
individual judgements, measured on the whole generated grid.

For each subset size k it reports
  - the mean true utility distance between ADJACENT displayed alternatives
    (order statistics get closer as the subset grows);
  - the share of declared adjacent judgements carrying zero blank cards,
    split by noise level;
  - the realized share of FLIPPED adjacent judgements (declared order
    contradicting the true one), by noise level and pooled;
  - AIO against k SEPARATELY at each noise level, which is the decisive
    evidence: with a consistent DM the zero-card share still grows with k
    yet AIO keeps improving, so the dip is caused by the flipped
    judgements, not by the dilution of the intensity information.
It also reports the flip share by |A| and by m, which is what the claim in
Section~\ref{sec:sim_design} rests on.
Run from the DOR folder.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, so `import bdor` works without installing
import os
from collections import defaultdict

import numpy as np
from scipy.stats import mannwhitneyu

from bdor import utils
from bdor.simulation import expand_subrank
from bdor import results as m
from bdor.paths import DATASETS_DIR

F1_LEVELS = [10, 20, 35, 50]
F2_LEVELS = [3, 5, 7, 9]
INC_LEVELS = [0, 15, 35, 50]
K_LEVELS = [3, 4, 5]
STEPS = 10
ETA_MAX = 5


def main():
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
        print(f"(skipped {missing} missing dataset folders)\n")

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

    # ---- decisive test: AIO against k, separately at each noise level ------
    print("\n=== AIO by k at each noise level (dip only where flips exist) ===")
    res = m.load_results()
    arms = [('FTRL_DOR_ADJJ1', 'FTRL'), ('BAYES_DOR_ADJJ1', 'BAYES'),
            ('FTRLDIR_DOR_ADJJ1', 'FTRLd'), ('BAYESDIR_DOR_ADJJ1', 'BAYESd')]
    for inc in INC_LEVELS:
        print(f"  |F_inc| = {inc}%")
        for arm, lbl in arms:
            vals = [m.pool(res, arm, 'aios', K=k, Inc=inc) for k in K_LEVELS]
            vals = [v[~np.isnan(v)] for v in vals]
            means = [v.mean() for v in vals]
            p = mannwhitneyu(vals[1], vals[2], alternative='two-sided').pvalue
            tag = "monotone" if means[2] >= means[1] else "DIP at k=5"
            print(f"    {lbl:<7} {means[0]:.3f} {means[1]:.3f} {means[2]:.3f}"
                  f"   k4->k5 {means[2] - means[1]:+.3f} (p={p:.3f})  {tag}")


if __name__ == "__main__":
    main()
