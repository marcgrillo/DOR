"""Main simulation campaign: run every method arm on every configuration of the grid.

For each configuration of the factorial grid below and each of the eight arms in ``ARMS``, runs
the ten-session elicitation on every replication (bdor.runner), computes the RAI / PWI matrices
and the ASR / ASP / AIO metrics after every session (bdor.metrics.BenchmarkRunner) and merges
them into the results npz, rewritten after each configuration. Step 1 of 3 of the main
simulation pipeline: run_original_baselines.py and inject_original_baselines.py then add the
paper's DOR and SMAA-DOR baselines. Resumable (see main()); parallel over configurations.

Paper:   Table 3, Figures 2, 3 and 5, Supplementary Tables B.1-B.9 (all but the DOR / SMAA-DOR
         rows, which come from inject_original_baselines.py); Sec. 5.1 HMC diagnostics;
         Sec. 5.3 numbers
Inputs:  data/datasets/grid_<config>/f1_<|A|>__f2_<m>__nsteps_10.json
Outputs: results/grid_results_joint_v1.npz (layout in bdor/results.py); per-session states under
         data/samples/grid_<config>/ and metric caches under data/metrics/grid_<config>/
         (--test: data/grid_results_test.npz and test_<config> folders)
Runtime: about 10 h from scratch with the default 12 worker processes on a desktop CPU (about
         110 CPU-hours, mostly the Bayesian arms); --test: a few minutes.

Usage:
    python scripts/simulation/run_grid.py [--workers N] [--force-arms ARM [ARM ...]]
    python scripts/simulation/run_grid.py --test

(formerly run_grid_joint_v1.py)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, so `import bdor` works without installing
import argparse
import itertools
import os
import time

# Keep each worker (and the parent) single-threaded at the BLAS/XLA level: the per-fit
# matrices are tiny, so parallelism comes from running configurations in separate processes,
# not from intra-op threads. Must be set before numpy and jax are imported.
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

from bdor.runner import run_batch_experiments
from bdor.metrics import BenchmarkRunner
from bdor.utils import parse_subfold_string
import bdor.utils as utils
from bdor.paths import DATA_DIR, DATASETS_DIR, SAMPLES_DIR, METRICS_DIR, GRID_RESULTS

# Full factorial grid: 4 x 4 x 1 x 3 x 3 x 4 = 576 configurations. The paper reports the 192
# with AR == K, each read after T in {1, 3, 5, 10} sessions (prefixes of one 10-session run):
# the 768 cells of the design of its Table 2. generate_grid_datasets.py uses the same lists.
F1_list = [10, 20, 35, 50]     # |A|: alternatives
F2_list = [3, 5, 7, 9]         # m: criteria
NUM_STEPS_list = [10]          # S: sessions per replication
A_R_list = [3, 4, 5]           # AR: subset size k of session 1
K_list = [3, 4, 5]             # K: subset size k of sessions 2..S
F_inc_list = [0, 15, 35, 50]   # Inc: nominal inconsistency Finc (%)
HM_0 = 20                      # replications per configuration

ETA_MAX = 5           # e_max, the largest number of blank cards in a gap
RUN_TAG = 'joint-v1'  # stored in the npz together with RUN_SEED and ETA_MAX
RUN_SEED = 20260806   # global offset; per-(config, arm, table, step) seeds derived below
NUM_WORKERS = 12      # default of --workers
SAVE_PATH = str(GRID_RESULTS)

# --test: one tiny configuration with 4 replications. Its folders are named test_<config>
# instead of grid_<config> and its results go to a separate npz, so a test run never touches
# a campaign file; the dataset is generated (seed 0) the first time.
TEST_GRID = ([10], [3], [10], [3], [3], [0])   # F1, F2, S, AR, K, Inc lists, as above
TEST_HM_0 = 4
TEST_FOLD_PREFIX = 'test'
TEST_SAVE_PATH = str(DATA_DIR / "grid_results_test.npz")

# Method arms: (sub_fold, expansion, uses_all_tables). sub_fold = <ALG>_DOR_<TAG> names the
# arm's data folders and its key in the npz; ALG selects the method
# (bdor.simulation.get_sampler_state). expansion turns each session's ranking into gap
# observations: 'adjacent' = consecutive levels only (the paper's likelihood, tag ADJJ1),
# 'all' = every pair of levels (ablation, tag ALLJ1). Arms with uses_all_tables=False run on
# the first HM_0 / 4 = 5 replications only. The list index enters the seeds
# (config_seed_base), so keep the order and append new arms at the end.
#   FTRL_DOR_ADJJ1      FTRL-DOR                                        (paper)
#   BAYES_DOR_ADJJ1     BAYES-DOR                                       (paper)
#   FTRL_DOR_ALLJ1      all-pairs ablation                              (not reported)
#   BAYES_DOR_ALLJ1     all-pairs ablation                              (not reported)
#   SMAA_DOR_ADJJ1      per-gap adaptation of SMAA-DOR (bdor.baselines.smaa_adapted)
#   DORLP_DOR_ADJJ1     per-gap adaptation of the DOR LP point estimate
#   FTRLDIR_DOR_ADJJ1   FTRLdir: direction-only likelihood, cards ignored    (paper)
#   BAYESDIR_DOR_ADJJ1  BAYESdir: direction-only likelihood, cards ignored   (paper)
# The paper's DOR and SMAA-DOR are the original formulations (npz arms *_ORIG, added by
# run_original_baselines.py + inject_original_baselines.py), not the two adapted arms; but
# run_original_baselines.py processes exactly the replications that have a SMAA_DOR_ADJJ1
# session-1 file. BAYES_DOR_ADJJ1 and SMAA_DOR_ADJJ1 first ran on 5 replications and were
# extended to 20 later; the seeds do not depend on the replication count and existing sample
# files are reused, so replications 0-4 are the same as in a run from scratch.
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


def config_string(config):
    """``(F1, F2, S, AR, K, Inc)`` -> ``'F1=.._F2=.._S=.._AR=.._K=.._Inc=..'``, the npz key."""
    f1, f2, steps, a_r, k, f_inc = config
    return f"F1={f1}_F2={f2}_S={steps}_AR={a_r}_K={k}_Inc={f_inc}"


def config_seed_base(f1, f2, a_r, k, f_inc, arm_idx):
    """Deterministic seed base of one (configuration, arm), distinct across the grid.

    bdor.simulation.process_single_table adds ``100003 * i + t`` for replication ``i``
    (0-based) and session ``t`` (1-based); that seed drives the NUTS run (BAYES arms), the
    Hit-and-Run sampler (SMAA) and, at ``t = 1``, the prior draws of the BAYES log-loss.
    FTRL and DORLP fits are deterministic.
    """
    return (RUN_SEED
            + f1 * 7_000_000 + f2 * 600_000 + a_r * 60_000
            + k * 6_000 + f_inc * 40 + arm_idx)


def collect_sidecar(dataset_fold, sub_fold, f1, f2, steps, hm, name):
    """Gather the per-session sidecar files ``<name>_<t>.npy`` of one arm.

    Returns a nested list of shape (steps, hm, 2): for ``name='loss'`` each entry is
    [prequential log-loss of the session's gaps, number of gaps], for ``name='diag'``
    [divergent transitions, max split-Rhat]. Missing files give [nan, nan] (e.g. the loss of
    SMAA / DORLP arms, which write none).
    """
    out = []
    for j in range(1, steps + 1):
        row = []
        for i in range(hm):
            path = os.path.join(SAMPLES_DIR, dataset_fold, f"f1_{f1}_f2_{f2}_nsteps_{steps}",
                                sub_fold, f"table_{i}", f"{name}_{j}.npy")
            row.append(np.load(path).tolist() if os.path.exists(path) else [np.nan, np.nan])
        out.append(row)
    return out


def run_single_config(config, hm_0, arms_to_run=None, fold_prefix='grid'):
    """Run the arms of one configuration and collect their results (one worker task).

    Parameters
    ----------
    config : tuple of int
        ``(F1, F2, S, AR, K, Inc)``.
    hm_0 : int
        Replications of the arms with ``uses_all_tables``; the others use
        ``int(max(1, hm_0 / 4))``.
    arms_to_run : collection of str, optional
        Sub-folds to run; default: every arm of ``ARMS``.
    fold_prefix : str
        The data folders are ``<fold_prefix>_<config_string(config)>``: 'grid' for the
        campaign, 'test' for --test.

    Returns
    -------
    config_str : str
        ``config_string(config)``, the key of the configuration in the npz.
    config_results : dict
        ``{sub_fold: {'asrs' | 'asps' | 'aios': (S, hm) nested lists, 'log_loss': (S, hm, 2),
        'hmc_diag': (S, hm, 2) (BAYES arms only), 'runtime_s': float, 'seed_base': int}}``.
    """
    f1, f2, steps, a_r, k, f_inc = config
    k_tuple = (a_r, k)
    config_str = config_string(config)
    dataset_fold = f"{fold_prefix}_{config_str}"

    dataset_json = os.path.join(DATASETS_DIR, dataset_fold, f"f1_{f1}__f2_{f2}__nsteps_{steps}.json")
    if not os.path.exists(dataset_json):
        raise FileNotFoundError(
            f"{dataset_json} missing - the campaign reads existing datasets; create missing "
            "grid datasets with scripts/simulation/generate_grid_datasets.py")

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
        # The legacy perc_inc metric (no longer in bdor.metrics) is not computed: its
        # direction-probability formula predated the paper's likelihood.
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


def save_results(save_path, all_results):
    """Write the results npz atomically: temporary file in the same folder, then os.replace,
    so an interruption never leaves a truncated results file behind."""
    tmp = save_path + ".tmp.npz"
    np.savez_compressed(tmp, results=all_results,
                        run_tag=RUN_TAG, run_seed=RUN_SEED, eta_max=ETA_MAX)
    os.replace(tmp, save_path)


def clean_test_outputs():
    """Remove what a previous --test run left (its own test_* folders and npz only), so every
    smoke test refits from scratch instead of reusing cached states."""
    import glob
    import shutil
    for root in (DATASETS_DIR, SAMPLES_DIR, METRICS_DIR):
        for d in glob.glob(os.path.join(root, f"{TEST_FOLD_PREFIX}_*")):
            print(f"removing previous test output {d}")
            shutil.rmtree(d)
    if os.path.exists(TEST_SAVE_PATH):
        os.remove(TEST_SAVE_PATH)


def make_test_datasets():
    """Generate the --test datasets (``TEST_HM_0`` replications, seed 0) that do not exist yet."""
    from bdor.datagen import DatasetGenerator  # only needed here; keeps worker start-up light

    for config in itertools.product(*TEST_GRID):
        f1, f2, steps, a_r, k, f_inc = config
        out_dir = os.path.join(DATASETS_DIR, f"{TEST_FOLD_PREFIX}_{config_string(config)}")
        if os.path.exists(os.path.join(out_dir, f"f1_{f1}__f2_{f2}__nsteps_{steps}.json")):
            continue
        np.random.seed(0)
        DatasetGenerator(output_dir=out_dir, n_runs=TEST_HM_0).generate_batch(
            F1=[f1], F2=[f2], num_steps=steps, k_subset=(a_r, k), n_max=ETA_MAX,
            target_consistency=1.0 - f_inc / 100.0, plot_utilities=False)


def parse_args():
    """Parse the command line; called from main() only.

    On Windows every worker process re-imports this module, so nothing may be parsed or
    printed at import time.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    ap.add_argument("--workers", type=int, default=NUM_WORKERS, metavar="N",
                    help="worker processes, one configuration at a time each; 1 runs "
                         "everything in this process (default: %(default)s)")
    ap.add_argument("--force-arms", nargs="+", default=[], metavar="ARM",
                    choices=[sub_fold for sub_fold, _, _ in ARMS],
                    help="re-run these arms even where the npz already holds them, e.g. after "
                         "raising their number of replications; existing per-session sample "
                         "files are still reused (choices: %(choices)s)")
    ap.add_argument("--test", action="store_true",
                    help="smoke test on one tiny configuration (|A|=10, m=3, k=3, Inc=0, "
                         f"{TEST_HM_0} replications) in test_<config> data folders, results in "
                         "data/grid_results_test.npz; never touches the campaign files")
    return ap.parse_args()


def main():
    """Run every (configuration, arm) of the grid that the results npz does not hold yet.

    Resume rules: an arm already in the npz is skipped unless named in --force-arms (the check
    is by presence, not by replication count); the arms that run reuse every existing
    per-session file under data/samples/ (states and log-loss sidecars) and recompute all
    their metric files. Only this (parent) process writes the npz, after every configuration;
    a configuration that raises is reported and left for the next run.
    """
    args = parse_args()
    if args.test:
        print(">>> RUNNING IN TINY GRID TEST MODE <<<")
        grid, hm_0 = TEST_GRID, TEST_HM_0
        fold_prefix, save_path = TEST_FOLD_PREFIX, TEST_SAVE_PATH
        clean_test_outputs()
        make_test_datasets()
    else:
        print(">>> RUNNING FULL GRID (joint-v1) <<<")
        grid = (F1_list, F2_list, NUM_STEPS_list, A_R_list, K_list, F_inc_list)
        hm_0 = HM_0
        fold_prefix, save_path = 'grid', SAVE_PATH
    force_arms = set(args.force_arms)
    num_workers = args.workers

    print(f"Starting joint-v1 grid campaign ({len(ARMS)} arms)...")
    start_time = time.time()

    combinations = list(itertools.product(*grid))
    print(f"Total configurations in grid: {len(combinations)}")

    all_results = {}
    if os.path.exists(save_path):
        print(f"Loading existing results from {save_path}...")
        try:
            with np.load(save_path, allow_pickle=True) as saved:  # closed before any save
                all_results = saved['results'].item()
            print(f"Loaded {len(all_results)} configurations.")
        except Exception as e:
            # Never start fresh on top of an unreadable file: the first save would replace
            # it with only the configurations run now.
            sys.exit(f"Cannot read {save_path} ({e}). Move it away to start a fresh campaign.")

    arm_names = [a[0] for a in ARMS]
    configs_to_run = []   # (config, missing_arm_names)
    for config in combinations:
        done = all_results.get(config_string(config), {})
        missing = [sf for sf in arm_names if sf not in done or sf in force_arms]
        if not missing:
            continue
        configs_to_run.append((config, missing))

    total_to_run = len(configs_to_run)
    if total_to_run == 0:
        print("All joint-v1 configurations have already been run!")
        return
    print(f"Configurations remaining: {total_to_run}")

    completed = 0
    if num_workers <= 1:
        for idx, (config, missing) in enumerate(configs_to_run):
            config_str = config_string(config)

            config_start = time.time()
            print("\n========================================================")
            print(f"[{idx+1}/{total_to_run}] Running Configuration: {config_str} arms={missing}")
            print("========================================================")

            try:
                config_str, config_results = run_single_config(config, hm_0, missing, fold_prefix)
                all_results.setdefault(config_str, {}).update(config_results)
                save_results(save_path, all_results)
                completed += 1
                print(f"[{completed}/{total_to_run}] Completed in {time.time() - config_start:.2f}s")
            except Exception as e:
                print(f"Error running configuration {config_str}: {e}")
    else:
        # Parallel over configurations; only the parent merges and saves the npz.
        from concurrent.futures import ProcessPoolExecutor, as_completed
        print(f"Running with {num_workers} worker processes")
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            futures = {}
            for config, missing in configs_to_run:
                cs = config_string(config)
                futures[ex.submit(run_single_config, config, hm_0, missing, fold_prefix)] = cs
            for fut in as_completed(futures):
                cs = futures[fut]
                try:
                    config_str, config_results = fut.result()
                    all_results.setdefault(config_str, {}).update(config_results)
                    save_results(save_path, all_results)
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
