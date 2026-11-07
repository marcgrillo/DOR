"""
Add the paper's DOR and SMAA-DOR baselines to the aggregated grid results.

Pipeline step 3 of 3 for the main simulation (after ``run_grid.py`` and
``run_original_baselines.py``). It reads the T = 1 metrics of the DOR and SMAA-DOR
baselines in their original single-session formulation (paper Sec. 2.2;
``bdor/baselines/smaa_original.py``) from ``results/original_comparison.csv`` and
stores them in ``results/grid_results_joint_v1.npz`` as two new arms per configuration:

    SMAA_DOR_ORIG   (from rows SMAAORIG_T1: Hit-and-Run sample of the compatible set)
    DORLP_DOR_ORIG  (from rows DORLPORIG_T1: DOR linear-programming point estimate)

Each arm holds ``asrs`` / ``asps`` / ``aios`` with a single session row (T = 1, the only
horizon at which the one-shot methods are compared) and 20 replications. The npz is
rewritten in place; all other arms are left untouched (the adapted per-gap arms
SMAA_DOR_ADJJ1 / DORLP_DOR_ADJJ1 stay for provenance but no paper output reads them).

Used by: Table 3 (lower block), Figure 3 and Supplementary Tables B.6-B.9.

Usage (seconds):
    python scripts/simulation/inject_original_baselines.py [--npz PATH] [--csv PATH]
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, so `import bdor` works without installing

import numpy as np

from bdor.paths import GRID_RESULTS, RESULTS_DIR

N_TABLES = 20
ARM_MAP = {"SMAAORIG_T1": "SMAA_DOR_ORIG", "DORLPORIG_T1": "DORLP_DOR_ORIG"}


def inject(npz_path, csv_path):
    """Write the SMAAORIG_T1 / DORLPORIG_T1 rows of ``csv_path`` into ``npz_path`` (in place)."""
    data = np.load(npz_path, allow_pickle=True)
    results = data["results"].item()
    extra = {k: data[k] for k in data.files if k != "results"}

    new = {}   # config_str -> arm -> metric -> 1 x N_TABLES
    with open(csv_path) as fh:
        for row in csv.DictReader(fh):
            if row["method"] not in ARM_MAP:
                continue
            cfg = row["fold"].replace("grid_", "")
            arm = ARM_MAP[row["method"]]
            d = new.setdefault(cfg, {}).setdefault(
                arm, {m: np.full((1, N_TABLES), np.nan)
                      for m in ("asrs", "asps", "aios")})
            t = int(row["table"])
            d["asrs"][0, t] = float(row["asr"])
            d["asps"][0, t] = float(row["asp"])
            d["aios"][0, t] = float(row["aio"])

    n_cfg, n_missing = 0, 0
    for cfg, arms in new.items():
        if cfg not in results:
            n_missing += 1
            continue
        for arm, mets in arms.items():
            results[cfg][arm] = {m: v.tolist() for m, v in mets.items()}
        n_cfg += 1

    np.savez_compressed(npz_path, results=results, **extra)
    print(f"injected {list(ARM_MAP.values())} into {n_cfg} configs "
          f"({n_missing} CSV configs not in npz)")
    counts = [sum(np.isfinite(np.array(results[c]['SMAA_DOR_ORIG']['asrs'][0])))
              for c in results if 'SMAA_DOR_ORIG' in results[c]]
    print(f"tables per config: min={min(counts)} max={max(counts)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    ap.add_argument("--npz", default=str(GRID_RESULTS),
                    help="aggregated grid results, rewritten in place (default: %(default)s)")
    ap.add_argument("--csv", default=str(RESULTS_DIR / "original_comparison.csv"),
                    help="output of run_original_baselines.py (default: %(default)s)")
    args = ap.parse_args()
    inject(args.npz, args.csv)


if __name__ == "__main__":
    main()
