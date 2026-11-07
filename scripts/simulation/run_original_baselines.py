"""Fit the paper's DOR and SMAA-DOR baselines to session 1 of every grid replication.

Runs the original single-session DOR (LP point estimate) and SMAA-DOR (200 Hit-and-Run draws) of
paper Sec. 2.2 (bdor.baselines.smaa_original.OriginalDORSampler) on the first session of every
grid replication (T = 1 only: both are one-shot methods) and scores them, and the session-1
states of four arms stored by run_grid.py, with ASR / ASP / AIO (eqs. (23)-(25)).
Step 2 of 3 of the main simulation pipeline: it processes only the replications i that have
data/samples/grid_<config>/.../SMAA_DOR_ADJJ1/table_<i>/1.npy, so run_grid.py must run first;
inject_original_baselines.py then copies the baseline rows into the results npz.

Paper:   Table 3 (lower block), Figure 3 and Supplementary Tables B.6-B.9, through
         inject_original_baselines.py
Inputs:  data/datasets/grid_<config>/; data/samples/grid_<config>/f1_<|A|>_f2_<m>_nsteps_10/
         <arm>/table_<i>/1.npy of SMAA_DOR_ADJJ1, DORLP_DOR_ADJJ1, FTRL_DOR_ADJJ1, BAYES_DOR_ADJJ1
Outputs: results/original_comparison.csv; the baseline states, cached as
         data/samples/grid_<config>/f1_<|A|>_f2_<m>_nsteps_10/{SMAAORIG_T1,DORLPORIG_T1}/
         table_<i>/1.npy (reused when present); stdout: pooled paired summary
Runtime: about 30 s with the default 12 worker processes when the baseline states are cached;
         a few minutes on the first run, which computes them (11520 replications x 2 methods).

Usage:
    python scripts/simulation/run_original_baselines.py [--workers N]

(formerly compare_original.py)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, so `import bdor` works without installing
import argparse
import os
import re
import json
import time
import numpy as np

from bdor.paths import DATASETS_DIR, SAMPLES_DIR, RESULTS_DIR

# Single-threaded BLAS in the worker processes, which inherit this environment.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

NUM_WORKERS = 12      # default of --workers
NUM_SAMPLES = 200     # Hit-and-Run draws of SMAA-DOR (as many as BAYES-DOR's posterior draws)
RUN_SEED = 20260806   # seed of replication i: RUN_SEED + 100003 * i + 7 * AR + Inc (run_config)
CSV_PATH = os.path.join(RESULTS_DIR, "original_comparison.csv")
N_GRID_CONFIGS = 576  # 4 |A| x 4 m x 3 AR x 3 K x 4 Finc, all run by run_grid.py

# Arms of run_grid.py whose stored session-1 states are scored as well, for the paired summary.
STORED_ARMS = ["SMAA_DOR_ADJJ1", "DORLP_DOR_ADJJ1",
               "FTRL_DOR_ADJJ1", "BAYES_DOR_ADJJ1"]
# The two baselines, as sample sub-folders and CSV method names; inject_original_baselines.py
# turns them into the npz arms SMAA_DOR_ORIG and DORLP_DOR_ORIG.
NEW_ARMS = ["SMAAORIG_T1", "DORLPORIG_T1"]

# Main-grid sample folders (S = 10); every other folder under data/samples/ is ignored.
_CFG_RE = re.compile(
    r"grid_F1=(\d+)_F2=(\d+)_S=10_AR=(\d+)_K=(\d+)_Inc=(\d+)$")


def metrics_from_samples(features, true_ranks, sam, p):
    """ASR, ASP and AIO of one inference state, with the formulas of BenchmarkRunner.

    Parameters
    ----------
    features : ndarray, shape (|A|, p)
        Piecewise-linear features of the alternatives.
    true_ranks : array_like of int, shape (|A|,)
        True rank of every alternative, 0 = best.
    sam : ndarray, shape (R, >= p) or (>= p,)
        Inference state: R draws (or one point estimate) whose first ``p`` entries are the
        weights; further columns (thresholds, SMAA's k) are ignored.
    p : int
        Number of weights.

    Returns
    -------
    asr, asp, aio : float
        Paper eqs. (23)-(25).
    """
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
    """Score the session-1 states of one grid configuration (one worker task).

    Only the replications ``i`` with a stored SMAA_DOR_ADJJ1 session-1 state are processed.
    A missing baseline state is computed from session 1 with the seed
    ``RUN_SEED + 100003 * i + 7 * ar + inc`` (only the Hit-and-Run sampler uses it) and cached;
    a cached one is loaded.

    Returns
    -------
    list of tuple
        CSV rows ``(fold, f1, f2, ar, k, inc, table, method, asr, asp, aio)``, one per
        processed replication and available method.
    """
    from bdor.features import PiecewiseLinearTransformer
    from bdor.baselines.smaa_original import OriginalDORSampler

    ds_path = os.path.join(DATASETS_DIR, fold, f"f1_{f1}__f2_{f2}__nsteps_10.json")
    with open(ds_path) as fh:
        runs = json.load(fh)["runs"]
    cfg_dir = os.path.join(SAMPLES_DIR, fold, f"f1_{f1}_f2_{f2}_nsteps_10")

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
        # The paper's grid datasets predate set-valued levels and store no 'levels' key:
        # every level is then a single alternative (there are no ties in the grid).
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
    """Score every main-grid configuration with SMAA_DOR_ADJJ1 states; write the CSV, summarize."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    ap.add_argument("--workers", type=int, default=NUM_WORKERS, metavar="N",
                    help="worker processes, one configuration at a time each "
                         "(default: %(default)s)")
    args = ap.parse_args()
    if args.workers < 1:
        ap.error("--workers must be at least 1")

    cfgs = []
    for fold in sorted(os.listdir(SAMPLES_DIR)):
        m = _CFG_RE.match(fold)
        if not m:
            continue
        f1, f2, ar, k, inc = map(int, m.groups())
        ref = os.path.join(SAMPLES_DIR, fold, f"f1_{f1}_f2_{f2}_nsteps_10",
                           "SMAA_DOR_ADJJ1")
        if os.path.isdir(ref):
            cfgs.append((fold, f1, f2, ar, k, inc))
    print(f"{len(cfgs)} configs")
    if not cfgs:  # never replace the committed CSV with an empty one
        sys.exit(f"No main-grid sample folders with SMAA_DOR_ADJJ1 states under {SAMPLES_DIR}; "
                 "run scripts/simulation/run_grid.py first.")
    if len(cfgs) < N_GRID_CONFIGS:
        print(f"WARNING: only {len(cfgs)} of the {N_GRID_CONFIGS} grid configurations have "
              f"samples; {CSV_PATH} will cover only those.")

    t0 = time.time()
    all_rows = []
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_config, *c): c[0] for c in cfgs}
        done = 0
        for fut in as_completed(futs):
            all_rows.extend(fut.result())
            done += 1
            if done % 25 == 0 or done == len(cfgs):
                print(f"  {done}/{len(cfgs)} configs ({time.time()-t0:.0f}s)",
                      flush=True)

    header = "fold,f1,f2,ar,k,inc,table,method,asr,asp,aio"
    tmp = CSV_PATH + ".tmp"  # write next to the target, then replace it in one step
    with open(tmp, "w") as fh:
        fh.write(header + "\n")
        for r in all_rows:
            fh.write(",".join(map(str, r)) + "\n")
    os.replace(tmp, CSV_PATH)
    print(f"wrote {CSV_PATH} ({len(all_rows)} rows)")
    summarize(all_rows)


def summarize(rows):
    """Print pooled means and one-sided paired Wilcoxon tests, overall and per Inc level.

    The pairing unit is a (configuration, replication) with all six methods. Each test checks
    whether the first method of a pair scores higher than the second; 'ours' labels the
    adapted per-gap baselines of run_grid.py (SMAA_DOR_ADJJ1, DORLP_DOR_ADJJ1).
    """
    from scipy.stats import wilcoxon
    by = {}
    for r in rows:
        key = (r[0], r[6])       # (fold, table) pairing key
        by.setdefault(key, {})[r[7]] = r[8:11]  # method -> (asr, asp, aio)

    methods = NEW_ARMS + STORED_ARMS
    names = {"SMAAORIG_T1": "SMAA-orig", "DORLPORIG_T1": "DOR-orig",
             "SMAA_DOR_ADJJ1": "SMAA-ours", "DORLP_DOR_ADJJ1": "DOR-ours",
             "FTRL_DOR_ADJJ1": "FTRL", "BAYES_DOR_ADJJ1": "BAYES"}

    def pooled(label, filt=None):
        print(f"\n=== pooled means, {label} (n pairs with all methods) ===")
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

    pooled("all configurations")
    for inc in [0, 15, 35, 50]:
        pooled(f"Inc={inc}", lambda f, inc=inc: f.endswith(f"_Inc={inc}"))


if __name__ == "__main__":
    main()
