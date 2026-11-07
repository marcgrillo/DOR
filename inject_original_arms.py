"""
Inject the faithful original-formulation baselines (compare_original.py,
inference/smaa_original.py: interval-scale eq. (5) of corrente2025deck) into
grid_results_joint_v1.npz as arms

    SMAA_DOR_ORIG   (Hit-and-Run over the original polytope)
    DORLP_DOR_ORIG  (original DOR LP point estimate)

with a single step row (T=1: the only horizon at which the one-shot methods
are compared). Metric values come from original_comparison.csv, computed with
exactly the BenchmarkRunner formulas. The old adapted arms (SMAA_DOR_ADJJ1,
DORLP_DOR_ADJJ1) are left in the npz for provenance but are no longer used by
the paper scripts.
"""
import csv
import numpy as np

NPZ = "grid_results_joint_v1.npz"
CSV = "original_comparison.csv"
N_TABLES = 20
ARM_MAP = {"SMAAORIG_T1": "SMAA_DOR_ORIG", "DORLPORIG_T1": "DORLP_DOR_ORIG"}

data = np.load(NPZ, allow_pickle=True)
results = data["results"].item()
extra = {k: data[k] for k in data.files if k != "results"}

new = {}   # config_str -> arm -> metric -> 1 x N_TABLES
with open(CSV) as fh:
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

np.savez_compressed(NPZ, results=results, **extra)
print(f"injected {list(ARM_MAP.values())} into {n_cfg} configs "
      f"({n_missing} CSV configs not in npz)")
counts = [sum(np.isfinite(np.array(results[c]['SMAA_DOR_ORIG']['asrs'][0])))
          for c in results if 'SMAA_DOR_ORIG' in results[c]]
print(f"tables per config: min={min(counts)} max={max(counts)}")
