"""Readers for the synthetic datasets and the per-session metric files, plus a small helper.

Used by ``bdor.runner``, ``bdor.metrics``, ``bdor.regret``, ``bdor.datagen`` and the scripts in
scripts/simulation and scripts/paper.  Locations follow ``bdor.paths``: a dataset is read from
its folder under data/datasets, metric files from data/metrics (``METRICS_DIR``).
"""
import os
import json
import numpy as np

from bdor.paths import METRICS_DIR

def save_path(path: str):
    """Create the directory ``path`` (and its parents) if it does not exist yet."""
    os.makedirs(path, exist_ok=True)

def read_dataset(dataset_fold: str, f1: int, f2: int, num_steps: int):
    """Read one synthetic dataset written by ``bdor.datagen.DatasetGenerator`` (Sec. 5.1).

    Parameters
    ----------
    dataset_fold : str
        Full path of the dataset folder, e.g. ``os.path.join(DATASETS_DIR, 'grid_F1=10_...')``;
        the file read is ``<dataset_fold>/f1_<f1>__f2_<f2>__nsteps_<num_steps>.json``.
    f1, f2, num_steps : int
        Number of alternatives |A|, of criteria m and of sessions of the dataset.

    Returns
    -------
    tables : ndarray, shape (n_runs, f1, f2)
        Performance table of every replication ("run"; 20 per grid dataset).
    rankings : ndarray of int, shape (n_runs, f1)
        True rank of every alternative, 0 = best.
    subranks_list : list (n_runs) of list (num_steps) of dict
        The sessions of every replication: 'items' (declared order of the alternatives, best
        to worst), 'cards' (blank cards between consecutive levels) and, in datasets written
        by the current generator, 'levels' (the declared levels, best to worst, each a list
        of tied alternatives; absent from the grid datasets, whose levels are singletons).
    Us : ndarray, shape (n_runs, f1)
        True utilities U_true(a) of the alternatives.

    Raises
    ------
    FileNotFoundError
        If the JSON file does not exist.
    """
    json_path = os.path.join(dataset_fold, f"f1_{f1}__f2_{f2}__nsteps_{num_steps}.json")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Missing dataset file: {json_path}")

    with open(json_path, 'r') as f:
        dataset = json.load(f)

    runs = dataset['runs']
    tables = []
    rankings = []
    subranks_list = []
    Us = []

    for run in runs:
        tables.append(run['table'])
        rankings.append(run['true_ranks'])
        subranks_list.append(run['subranks'])
        Us.append(run['true_utilities'])

    return np.array(tables), np.array(rankings), subranks_list, np.array(Us)

def load_test_results(test_name: str, dataset_fold: str, sub_fold: str, f1: int, f2: int,
                      num_steps: int):
    """Load the per-session values of one scalar metric for one arm and configuration.

    Reads ``<METRICS_DIR>/<dataset_fold>/<sub_fold>/<test_name>/f1_<f1>_f2_<f2>_nsteps_<num_steps>/
    <test_name>_<j>.npy`` for j = 1, ..., num_steps (one value per replication, as written by
    ``bdor.metrics.BenchmarkRunner``); missing files are reported on stdout and skipped.

    Parameters
    ----------
    test_name : str
        Metric folder, e.g. 'asrs', 'asps' or 'aios'.
    dataset_fold : str
        Name of the dataset folder (joined to ``METRICS_DIR``), e.g.
        'grid_F1=10_F2=3_S=10_AR=5_K=5_Inc=0'.
    sub_fold : str
        Arm tag, e.g. 'BAYES_DOR_ADJJ1'.
    f1, f2, num_steps : int
        Number of alternatives, of criteria and of sessions of the configuration.

    Returns
    -------
    ndarray, shape (n_sessions_found, n_replications)
        Row r holds the (flattened) file of the r-th session found; an empty array if no
        file exists.
    """
    test_fold = os.path.join(METRICS_DIR, dataset_fold, sub_fold, test_name)
    sub_test_fold = os.path.join(test_fold, f"f1_{f1}_f2_{f2}_nsteps_{num_steps}")

    loaded_data = []

    for i in range(1, num_steps + 1):
        path = os.path.join(sub_test_fold, f"{test_name}_{i}.npy")

        if not os.path.exists(path):
            print(path, "not found. Skipping.")
            continue

        d = np.load(path)
        loaded_data.append(d.flatten())

    if not loaded_data:
        return np.array([])

    return np.stack(loaded_data, axis=0)

def parse_subfold_string(s):
    """Split an arm tag '<ALG>_DOR_<TAG>' into ('<ALG>-DOR', '<TAG>').

    E.g. 'BAYES_DOR_ADJJ1' -> ('BAYES-DOR', 'ADJJ1') and 'FTRLDIR_DOR_ADJJ1' ->
    ('FTRLDIR-DOR', 'ADJJ1').  The first value is the ``alg`` string expected by
    ``bdor.runner.run_batch_experiments``; only the first three '_'-separated fields are used.
    """
    parts = s.split('_')
    alg_str = f"{parts[0]}-{parts[1]}"
    active_str = parts[2]
    return alg_str, active_str
