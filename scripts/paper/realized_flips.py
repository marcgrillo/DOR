"""Measure the realized share of flipped pairwise comparisons in the grid datasets.

Reads the raw elicitation datasets and, for every declared subset ranking, compares each
declared pair with the true ranks of the replication: a pair is flipped when the
alternative declared better has the worse true rank. Prints the flipped share per nominal
Finc level over all C(k, 2) pairs a ranking implies and over adjacent pairs only, then the
all-pairs share by Finc level and subset size k.

Every folder data/datasets/grid_* with AR == K is pooled. S is not filtered, so besides
the 192 S = 10 grid folders the 37 legacy AR == K folders with S in {1, 3, 5} present locally
are included too; this is how the published Table 1 numbers were obtained (a warning is
printed if a different number of legacy folders is found). --s10-only pools only the 192
folders of the main grid.

Paper:   Table 1; Sec. 5.1 realized flip shares {0.0, 7.7, 15.4, 21.5}% (all pairs) and
         {0.0, 13.6, 23.6, 29.7}% (adjacent pairs). The by-k and by-m shares quoted after
         Table 1 come from subset_density.py.
Inputs:  data/datasets/grid_*/ (the first *.json file of each folder); needs the full local
         datasets folder, the two folders committed for Figure 5 are not enough
Outputs: stdout only
Runtime: a few seconds (JSON parsing of a few hundred dataset files)

Usage:
    python scripts/paper/realized_flips.py [--s10-only]

(formerly measure_realized_flips.py)
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, so `import bdor` works without installing

from bdor.paths import DATASETS_DIR

FOLDER_RE = re.compile(r"grid_F1=(\d+)_F2=(\d+)_S=(\d+)_AR=(\d+)_K=(\d+)_Inc=(\d+)$")
# Legacy AR == K folders (S in {1, 3, 5}; 12 + 14 + 11) present when Table 1 was computed
N_LEGACY_TABLE1 = 37


def main(argv=None):
    """Count the flipped declared pairs over the grid datasets and print the shares."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    ap.add_argument("--s10-only", action="store_true",
                    help="pool only the 192 S = 10 folders of the main grid (by default the "
                         "legacy AR == K folders with S in {1, 3, 5} are pooled too, as for the "
                         "published Table 1)")
    args = ap.parse_args(argv)

    expected = [(f"grid_F1={a}_F2={m}_S=10_AR={k}_K={k}_Inc={i}", f"f1_{a}__f2_{m}__nsteps_10.json")
                for a in (10, 20, 35, 50) for m in (3, 5, 7, 9) for k in (3, 4, 5)
                for i in (0, 15, 35, 50)]
    absent = [f for f, js in expected if not os.path.isfile(os.path.join(DATASETS_DIR, f, js))]
    if absent:
        sys.exit(f"realized_flips.py needs the datasets of the {len(expected)} grid folders with "
                 f"AR == K under {DATASETS_DIR} ({len(absent)} missing). They are not part of the "
                 "public repository; see data/README.md.")
    folders = [d for d in glob.glob(os.path.join(DATASETS_DIR, "grid_*")) if os.path.isdir(d)]
    n_legacy = 0  # pooled AR == K folders with S != 10

    # [flipped, total] counters
    allpairs = defaultdict(lambda: [0, 0])     # Finc -> all declared pairs
    adjpairs = defaultdict(lambda: [0, 0])     # Finc -> adjacent declared pairs
    bylevel_byk = defaultdict(lambda: [0, 0])  # (Finc, k) -> all declared pairs

    for d in folders:
        m = FOLDER_RE.search(os.path.basename(d))
        if not m:
            continue
        f1, f2, s, ar, k, inc = map(int, m.groups())
        if ar != k:  # the paper's cells; S is not filtered by default (see the module docstring)
            continue
        if args.s10_only and s != 10:
            continue
        js = glob.glob(os.path.join(d, "*.json"))
        if not js:
            continue
        n_legacy += s != 10
        with open(js[0]) as fh:
            data = json.load(fh)
        for run in data['runs']:
            tr = run['true_ranks']  # tr[a] = true rank of alternative a (0 = best)
            for sr in run['subranks']:
                items = sr['items']  # declared order, best first
                kk = len(items)
                # every declared pair: position p < q means items[p] was declared better
                for p in range(kk):
                    for q in range(p + 1, kk):
                        a, b = items[p], items[q]
                        flip = 1 if tr[a] > tr[b] else 0
                        allpairs[inc][0] += flip
                        allpairs[inc][1] += 1
                        bylevel_byk[(inc, k)][0] += flip
                        bylevel_byk[(inc, k)][1] += 1
                        if q == p + 1:
                            adjpairs[inc][0] += flip
                            adjpairs[inc][1] += 1

    if not args.s10_only and n_legacy != N_LEGACY_TABLE1:
        print(f"WARNING: pooled {n_legacy} legacy AR == K folders with S in {{1, 3, 5}}; the "
              f"published Table 1 pooled {N_LEGACY_TABLE1}, so the shares will differ from it.\n")
    print("=== Realized FLIPPED pairwise comparisons (AR==K datasets) ===")
    print(f"{'TargetInc':>9} | {'all-pairs flip%':>15} | {'adjacent flip%':>15}")
    for inc in sorted(allpairs):
        ap = 100*allpairs[inc][0]/allpairs[inc][1]
        aj = 100*adjpairs[inc][0]/adjpairs[inc][1]
        print(f"{inc:>9} | {ap:>15.2f} | {aj:>15.2f}   (n_pairs={allpairs[inc][1]})")

    print("\n=== all-pairs flip% by level and subset size k ===")
    for inc in sorted(set(i for i, _ in bylevel_byk)):
        row = []
        for k in [3, 4, 5]:
            f, t = bylevel_byk[(inc, k)]
            row.append(f"k={k}:{100*f/t:.2f}%")
        print(f"Inc={inc}: " + "  ".join(row))


if __name__ == "__main__":
    main()
