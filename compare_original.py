"""
T=1 comparison of the FAITHFUL original DOR / SMAA-DOR formulation
(interval-scale eq. (5) of corrente2025deck: per-alternative deviations,
free offset h, nu increments e+1, 0.01 eta-growth) against our adapted
per-gap baselines and the proposed estimators.

For every grid config that already carries step-1 SMAA_DOR_ADJJ1 samples:
  - runs SMAAORIG (Hit-and-Run, 200 samples) and DORLPORIG (vertex) on the
    FIRST subrank of each table (same data all methods see at T=1);
  - caches the new samples under samples/<fold>/.../{SMAAORIG,DORLPORIG}_T1/;
  - computes ASR / ASP / AIO for the new methods AND for the stored step-1
    states of SMAA_DOR_ADJJ1, DORLP_DOR_ADJJ1, FTRL_DOR_ADJJ1, BAYES_DOR_ADJJ1.

Output: original_comparison.csv (one row per config/table/method) and a
pooled + per-factor paired summary on stdout.
"""

import os
import re
import json
import time
import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

NUM_WORKERS = 12
NUM_SAMPLES = 200
RUN_SEED = 20260806
CSV_PATH = "original_comparison.csv"

STORED_ARMS = ["SMAA_DOR_ADJJ1", "DORLP_DOR_ADJJ1",
               "FTRL_DOR_ADJJ1", "BAYES_DOR_ADJJ1"]
NEW_ARMS = ["SMAAORIG_T1", "DORLPORIG_T1"]

_CFG_RE = re.compile(
    r"grid_F1=(\d+)_F2=(\d+)_S=10_AR=(\d+)_K=(\d+)_Inc=(\d+)$")


def metrics_from_samples(features, true_ranks, sam, p):
    """ASR / ASP / AIO exactly as BenchmarkRunner computes them."""
    sam = np.atleast_2d(np.asarray(sam, dtype=float))
    w = sam[:, :p]
    vals = features @ w.T                     # (f1, R)
    f1, R = vals.shape
    rai = np.zeros((f1, f1))
    poi = np.zeros((f1, f1))
    for col in range(R):
        scores = vals[:, col]
        poi += (scores[:, None] >= scores[None, :])
        ranks = np.argsort(np.argsort(-scores))
        rai[np.arange(f1), ranks] += 1
    rai /= R
    poi /= R
    tr = np.asarray(true_ranks)
    pos = {r: np.argwhere(tr == r)[0, 0] for r in range(f1)}
    asr = float(np.mean([rai[pos[r], r] for r in range(f1)]))
    aio = float(rai[pos[0], 0])
    pref_mask = tr[:, None] > tr[None, :]
    asp = float(np.sum(poi.T * pref_mask) * 2.0 / (f1 * (f1 - 1)))
    return asr, asp, aio


def run_config(fold, f1, f2, ar, k, inc):
    from mcda.models import PiecewiseLinearTransformer
    from inference.smaa_original import OriginalDORSampler

    ds_path = os.path.join("datasets", fold, f"f1_{f1}__f2_{f2}__nsteps_10.json")
    with open(ds_path) as fh:
        runs = json.load(fh)["runs"]
    cfg_dir = os.path.join("samples", fold, f"f1_{f1}_f2_{f2}_nsteps_10")

    rows = []
    for i, run in enumerate(runs):
        smaa_ref = os.path.join(cfg_dir, "SMAA_DOR_ADJJ1", f"table_{i}", "1.npy")
        if not os.path.exists(smaa_ref):
            continue
        table = np.array(run["table"], dtype=float)
        tr = PiecewiseLinearTransformer.from_equal_intervals(table, 3)
        feats = tr.transform(table)
        p = tr.total_params
        sub = run["subranks"][0]
        levels = sub.get("levels", [[a] for a in sub["items"]])
        cards = sub["cards"]
        seed = RUN_SEED + 100003 * i + ar * 7 + inc  # deterministic per table

        # --- new arms (cached) ---
        new_states = {}
        for arm in NEW_ARMS:
            path = os.path.join(cfg_dir, arm, f"table_{i}", "1.npy")
            if os.path.exists(path):
                new_states[arm] = np.load(path)
                continue
            sampler = OriginalDORSampler(feats, levels, cards, p, seed=seed)
            state = (sampler.sample(NUM_SAMPLES) if arm.startswith("SMAA")
                     else sampler.solve_point())
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.save(path, state)
            new_states[arm] = state

        # --- metrics for new + stored arms ---
        for arm, state in new_states.items():
            asr, asp, aio = metrics_from_samples(feats, run["true_ranks"], state, p)
            rows.append((fold, f1, f2, ar, k, inc, i, arm, asr, asp, aio))
        for arm in STORED_ARMS:
            path = os.path.join(cfg_dir, arm, f"table_{i}", "1.npy")
            if not os.path.exists(path):
                continue
            state = np.load(path)
            asr, asp, aio = metrics_from_samples(feats, run["true_ranks"], state, p)
            rows.append((fold, f1, f2, ar, k, inc, i, arm, asr, asp, aio))
    return rows


def main():
    cfgs = []
    for fold in sorted(os.listdir("samples")):
        m = _CFG_RE.match(fold)
        if not m:
            continue
        f1, f2, ar, k, inc = map(int, m.groups())
        ref = os.path.join("samples", fold, f"f1_{f1}_f2_{f2}_nsteps_10",
                           "SMAA_DOR_ADJJ1")
        if os.path.isdir(ref):
            cfgs.append((fold, f1, f2, ar, k, inc))
    print(f"{len(cfgs)} configs")

    t0 = time.time()
    all_rows = []
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as ex:
        futs = {ex.submit(run_config, *c): c[0] for c in cfgs}
        done = 0
        for fut in as_completed(futs):
            all_rows.extend(fut.result())
            done += 1
            if done % 25 == 0 or done == len(cfgs):
                print(f"  {done}/{len(cfgs)} configs ({time.time()-t0:.0f}s)",
                      flush=True)

    header = "fold,f1,f2,ar,k,inc,table,method,asr,asp,aio"
    with open(CSV_PATH, "w") as fh:
        fh.write(header + "\n")
        for r in all_rows:
            fh.write(",".join(map(str, r)) + "\n")
    print(f"wrote {CSV_PATH} ({len(all_rows)} rows)")
    summarize(all_rows)


def summarize(rows):
    from scipy.stats import wilcoxon
    by = {}
    for r in rows:
        key = (r[0], r[6])       # (fold, table) pairing key
        by.setdefault(key, {})[r[7]] = r[8:11]  # method -> (asr, asp, aio)

    methods = NEW_ARMS + STORED_ARMS
    names = {"SMAAORIG_T1": "SMAA-orig", "DORLPORIG_T1": "DOR-orig",
             "SMAA_DOR_ADJJ1": "SMAA-ours", "DORLP_DOR_ADJJ1": "DOR-ours",
             "FTRL_DOR_ADJJ1": "FTRL", "BAYES_DOR_ADJJ1": "BAYES"}

    def pooled(filt=None):
        print(f"\n=== pooled means {filt or ''} (n pairs with all methods) ===")
        keys = [k for k, v in by.items()
                if all(m in v for m in methods)
                and (filt is None or filt(k[0]))]
        print(f"n = {len(keys)}")
        print(f"{'method':10s} {'ASR':>8s} {'ASP':>8s} {'AIO':>8s}")
        for m in methods:
            vals = np.array([by[k][m] for k in keys])
            print(f"{names[m]:10s} " + " ".join(f"{x:8.4f}" for x in vals.mean(0)))
        for a, b in [("SMAAORIG_T1", "SMAA_DOR_ADJJ1"),
                     ("DORLPORIG_T1", "DORLP_DOR_ADJJ1"),
                     ("BAYES_DOR_ADJJ1", "SMAAORIG_T1"),
                     ("FTRL_DOR_ADJJ1", "DORLPORIG_T1")]:
            for mi, met in enumerate(["asr", "asp", "aio"]):
                x = np.array([by[k][a][mi] for k in keys])
                y = np.array([by[k][b][mi] for k in keys])
                d = x - y
                if np.allclose(d, 0):
                    p = 1.0
                else:
                    p = wilcoxon(x, y, alternative="greater").pvalue
                print(f"  {names[a]:9s} - {names[b]:9s} {met}: "
                      f"delta={d.mean():+.4f}  p(greater)={p:.2e}")

    pooled()
    for inc in [0, 15, 35, 50]:
        pooled(lambda f, inc=inc: f.endswith(f"_Inc={inc}"))


if __name__ == "__main__":
    main()
