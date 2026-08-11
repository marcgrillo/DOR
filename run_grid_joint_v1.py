"""
joint-v1 grid campaign (migration plan C4-C15; Decisions 4/6 of 2026-08-06 and the
revised Decision 5 of 2026-08-06 evening: theta_0 = 0, JOINT direction+intensity CLM).

Runs FOUR arms under the NEW engine (theta_0 = 0 joint CLM, rho = 1e-6,
fresh per-(table,step) seeds, prequential log-loss + HMC diagnostics recording):

  FTRL_DOR_ADJJ1 / BAYES_DOR_ADJJ1  - adjacent-only likelihood (PRIMARY model)
  FTRL_DOR_ALLJ1 / BAYES_DOR_ALLJ1  - all-pairs composite (ablation arm -> supplement)

Sub_fold names carry the run tag so results can never mix with the legacy
PASSIVE runs (old engine). Datasets are REUSED from datasets/grid_* (the DGP is
unchanged by the migration; regeneration would alter them because the Thurstone
noise is not seeded). Results npz: grid_results_joint_v1.npz.

Resumable: per-arm sample files are skipped if present; configs with all arms
already in the npz are skipped.
"""
import time
import os
import itertools

# Keep each worker (and the parent) single-threaded at the BLAS/XLA level:
# the per-fit matrices are tiny, so parallelism comes from running CONFIGS in
# separate processes, not from intra-op threads.  Must be set before jax loads.
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

from experiments.runner import run_batch_experiments
from experiments.metrics import BenchmarkRunner
from common.utils import parse_subfold_string
import common.utils as utils

# Configs are independent (separate dataset folds, sample dirs and metric
# caches); only the parent process touches the npz.
NUM_WORKERS = 12

TEST_MODE = False

if TEST_MODE:
    print(">>> RUNNING IN TINY GRID TEST MODE <<<")
    F1_list = [10]
    F2_list = [3]
    NUM_STEPS_list = [10]
    A_R_list = [3]
    K_list = [3]
    F_inc_list = [0]
    HM_0 = 4
else:
    print(">>> RUNNING FULL GRID (joint-v1) <<<")
    F1_list = [10, 20, 35, 50]
    F2_list = [3, 5, 7, 9]
    NUM_STEPS_list = [10]
    A_R_list = [3, 4, 5]
    K_list = [3, 4, 5]
    F_inc_list = [0, 15, 35, 50]
    HM_0 = 20

ETA_MAX = 5
RUN_TAG = 'joint-v1'
RUN_SEED = 20260806  # global offset; per-(config, arm, table, step) seeds derived below
SAVE_PATH = "grid_results_joint_v1_tiny.npz" if TEST_MODE else "grid_results_joint_v1.npz"

# (sub_fold, expansion, uses_all_tables)
# Arms 0-3: main campaign (primary adjacent + all-pairs ablation).
# Arms 4-7 (appended 2026-08-07 -- keep order stable, seed_base uses the index):
#   SMAA  = SMAA-DOR baseline (corrente2025deck), Hit-and-Run samples
#   DORLP = original DOR LP point estimate (barbati2024deck), fbnote 884-i
#   *DIR  = direction-only ablation (card counts ignored), fbnote 884-ii
# 2026-08-10: BAYES_DOR_ADJJ1 and SMAA_DOR_ADJJ1 promoted to all 20 tables. They
# are the two arms of the T=1 comparison of Section 6.3 and were the only ones
# still on hm_0/4 = 5 of the 20 decision problems each dataset already contains,
# which is what made their confidence intervals twice as wide as FTRL's/DOR's.
# Tables 0-4 are untouched (per-table seeds are seed_base + 100003*table_index+j
# and overwrite=False skips existing sample files), so no existing number moves.
ARMS = [
    ('FTRL_DOR_ADJJ1', 'adjacent', True),
    ('BAYES_DOR_ADJJ1', 'adjacent', True),
    ('FTRL_DOR_ALLJ1', 'all', True),
    ('BAYES_DOR_ALLJ1', 'all', False),
    ('SMAA_DOR_ADJJ1', 'adjacent', True),
    ('DORLP_DOR_ADJJ1', 'adjacent', True),
    ('FTRLDIR_DOR_ADJJ1', 'adjacent', True),
    ('BAYESDIR_DOR_ADJJ1', 'adjacent', False),
]

# Arms re-run even when the npz already holds a result for them, needed when an
# arm's table count grows: the resume check below is by arm presence, not by
# replication count. Set to an empty set once the extra tables are in the npz.
FORCE_ARMS = {'BAYES_DOR_ADJJ1', 'SMAA_DOR_ADJJ1'}


def config_seed_base(f1, f2, a_r, k, f_inc, arm_idx):
    """Deterministic, distinct seed base per (config, arm)."""
    return (RUN_SEED
            + f1 * 7_000_000 + f2 * 600_000 + a_r * 60_000
            + k * 6_000 + f_inc * 40 + arm_idx)


def collect_sidecar(dataset_fold, sub_fold, f1, f2, steps, hm, name):
    """Aggregates loss_{j}.npy / diag_{j}.npy sidecars -> (steps x hm x width) list."""
    out = []
    for j in range(1, steps + 1):
        row = []
        for i in range(hm):
            path = os.path.join("samples", dataset_fold, f"f1_{f1}_f2_{f2}_nsteps_{steps}",
                                sub_fold, f"table_{i}", f"{name}_{j}.npy")
            row.append(np.load(path).tolist() if os.path.exists(path) else [np.nan, np.nan])
        out.append(row)
    return out


def run_single_config(config, hm_0, arms_to_run=None):
    f1, f2, steps, a_r, k, f_inc = config
    k_tuple = (a_r, k)
    config_str = f"F1={f1}_F2={f2}_S={steps}_AR={a_r}_K={k}_Inc={f_inc}"
    dataset_fold = f"grid_{config_str}"

    dataset_json = os.path.join("datasets", dataset_fold, f"f1_{f1}__f2_{f2}__nsteps_{steps}.json")
    if not os.path.exists(dataset_json):
        raise FileNotFoundError(
            f"{dataset_json} missing - joint-v1 reuses the existing grid datasets; "
            "generate it explicitly first if this config is genuinely new")

    config_results = {}
    for arm_idx, (sub_fold, expansion, all_tables) in enumerate(ARMS):
        if arms_to_run is not None and sub_fold not in arms_to_run:
            continue
        alg_name, _ = parse_subfold_string(sub_fold)
        hm = hm_0 if all_tables else int(max(1, hm_0 / 4))
        seed_base = config_seed_base(f1, f2, a_r, k, f_inc, arm_idx)

        arm_start = time.time()
        run_batch_experiments(
            F1=[f1], F2=[f2], sub_fold=sub_fold, dataset_folds=[dataset_fold],
            alg=alg_name, overwrite=False, hm=hm, k_subset=k_tuple,
            num_steps=steps, eta_max=ETA_MAX, expansion=expansion,
            seed_base=seed_base, log_loss=True
        )
        arm_runtime = time.time() - arm_start

        runner = BenchmarkRunner(
            dataset_fold=dataset_fold, sub_fold=sub_fold, num_subint=3, hm=hm,
            F1=[f1], F2=[f2], NUM_STEPS_LIST=[steps], eta_max=ETA_MAX
        )
        # perc_inc deliberately NOT computed: its direction-probability formula is a
        # legacy 0-cutoff-era heuristic with no meaning under the theta_0=-inf model.
        runner.compute_metrics("poi", force=True)
        runner.compute_metrics("rai", force=True)
        runner.compute_asrs(force=True)
        runner.compute_asps(force=True)
        runner.compute_aios(force=True)

        metrics_dict = {}
        for metric in ['asrs', 'asps', 'aios']:
            data = utils.load_test_results(metric, dataset_fold, sub_fold, f1, f2, steps)
            metrics_dict[metric] = data.tolist() if isinstance(data, np.ndarray) else data
        metrics_dict['log_loss'] = collect_sidecar(dataset_fold, sub_fold, f1, f2, steps, hm, 'loss')
        if sub_fold.startswith('BAYES'):
            metrics_dict['hmc_diag'] = collect_sidecar(dataset_fold, sub_fold, f1, f2, steps, hm, 'diag')
        metrics_dict['runtime_s'] = arm_runtime
        metrics_dict['seed_base'] = seed_base
        config_results[sub_fold] = metrics_dict

    return config_str, config_results


def main():
    print(f"Starting joint-v1 grid campaign ({len(ARMS)} arms)...")
    start_time = time.time()

    combinations = list(itertools.product(F1_list, F2_list, NUM_STEPS_list, A_R_list, K_list, F_inc_list))
    print(f"Total configurations in grid: {len(combinations)}")

    all_results = {}
    if os.path.exists(SAVE_PATH):
        print(f"Loading existing results from {SAVE_PATH}...")
        try:
            all_results = np.load(SAVE_PATH, allow_pickle=True)['results'].item()
            print(f"Loaded {len(all_results)} configurations.")
        except Exception as e:
            print(f"Error loading {SAVE_PATH}: {e}. Starting fresh.")

    arm_names = [a[0] for a in ARMS]
    configs_to_run = []   # (config, missing_arm_names)
    for config in combinations:
        f1, f2, steps, a_r, k, f_inc = config
        config_str = f"F1={f1}_F2={f2}_S={steps}_AR={a_r}_K={k}_Inc={f_inc}"
        done = all_results.get(config_str, {})
        missing = [sf for sf in arm_names if sf not in done or sf in FORCE_ARMS]
        if not missing:
            continue
        configs_to_run.append((config, missing))

    total_to_run = len(configs_to_run)
    if total_to_run == 0:
        print("All joint-v1 configurations have already been run!")
        return
    print(f"Configurations remaining: {total_to_run}")

    completed = 0
    if NUM_WORKERS <= 1:
        for idx, (config, missing) in enumerate(configs_to_run):
            f1, f2, steps, a_r, k, f_inc = config
            config_str = f"F1={f1}_F2={f2}_S={steps}_AR={a_r}_K={k}_Inc={f_inc}"

            config_start = time.time()
            print(f"\n========================================================")
            print(f"[{idx+1}/{total_to_run}] Running Configuration: {config_str} arms={missing}")
            print(f"========================================================")

            try:
                config_str, config_results = run_single_config(config, HM_0, missing)
                all_results.setdefault(config_str, {}).update(config_results)
                np.savez_compressed(SAVE_PATH, results=all_results,
                                    run_tag=RUN_TAG, run_seed=RUN_SEED, eta_max=ETA_MAX)
                completed += 1
                print(f"[{completed}/{total_to_run}] Completed in {time.time() - config_start:.2f}s")
            except Exception as e:
                print(f"Error running configuration {config_str}: {e}")
    else:
        # Parallel over configs; only the parent merges/saves the npz.
        from concurrent.futures import ProcessPoolExecutor, as_completed
        print(f"Running with {NUM_WORKERS} worker processes")
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as ex:
            futures = {}
            for config, missing in configs_to_run:
                f1, f2, steps, a_r, k, f_inc = config
                cs = f"F1={f1}_F2={f2}_S={steps}_AR={a_r}_K={k}_Inc={f_inc}"
                futures[ex.submit(run_single_config, config, HM_0, missing)] = cs
            for fut in as_completed(futures):
                cs = futures[fut]
                try:
                    config_str, config_results = fut.result()
                    all_results.setdefault(config_str, {}).update(config_results)
                    np.savez_compressed(SAVE_PATH, results=all_results,
                                        run_tag=RUN_TAG, run_seed=RUN_SEED, eta_max=ETA_MAX)
                    completed += 1
                    elapsed = (time.time() - start_time) / 3600
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta_h = (total_to_run - completed) / rate if rate > 0 else float('nan')
                    print(f"[{completed}/{total_to_run}] Completed {cs} "
                          f"({elapsed:.2f} h elapsed, ~{eta_h:.2f} h remaining)")
                except Exception as e:
                    print(f"Error running configuration {cs}: {e}")

    print(f"\nFinished! Total time: {(time.time() - start_time)/3600:.2f} h")


if __name__ == "__main__":
    main()
