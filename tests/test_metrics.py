"""BenchmarkRunner: RAI and PWI (paper eqs. (4)-(5)) and ASR / ASP / AIO (eqs. (23)-(25)) on a
hand-computed case, read back through the on-disk layout used by every campaign script.

Two decision problems with three alternatives and two criteria. With one interval per
criterion the features equal the tables below, so U(a) = w0 * v0(a) + w1 * v1(a) and the
third alternative of each table can never move: only the order of the other two depends
on whether w0 > w1. States carry three extra columns (threshold increments, n_max = 3),
which the metrics must ignore.
"""
import json

import numpy as np
import pytest

from bdor.metrics import BenchmarkRunner
from bdor.utils import load_test_results

FOLD = "handmade_metrics"
SUB_FOLD = "FTRL_DOR_ADJJ1"
F1, F2, T = 3, 2, 2
JUNK = [9.0, 9.0, 9.0]          # threshold part of the parameter vector, unused by the metrics

TABLES = [
    [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],     # true order: a0 > a1 > a2
    [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],     # true order: a1 > a2 > a0
]
TRUE_RANKS = [[0, 1, 2], [2, 0, 1]]           # 0 = best
TRUE_ORDER, SWAPPED = [0.6, 0.4] + JUNK, [0.3, 0.7] + JUNK

STATES = {   # (table, session) -> state, as the methods store it
    (0, 1): np.array([TRUE_ORDER, SWAPPED]),                  # two draws, one per order
    (0, 2): np.array(TRUE_ORDER),                             # 1-D point estimate (FTRL)
    (1, 1): np.array(SWAPPED),
    (1, 2): np.array([[0.9, 0.1] + JUNK, [0.7, 0.3] + JUNK,   # 3 of 4 draws in the true order
                      SWAPPED, TRUE_ORDER]),
}

# rows: session 1, 2; columns: table 0, 1
EXPECTED = {
    "asrs": [[2 / 3, 1 / 3], [1.0, 5 / 6]],
    "asps": [[5 / 6, 2 / 3], [1.0, 11 / 12]],
    "aios": [[1 / 2, 0.0], [1.0, 3 / 4]],
}
RAI_T0_S1 = [[0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 1.0]]
PWI_T0_S1 = [[1.0, 0.5, 1.0], [0.5, 1.0, 1.0], [0.0, 0.0, 1.0]]   # diagonal: U(a) >= U(a)


@pytest.fixture(scope="module")
def runner(data_dir):
    dataset_dir = data_dir / "datasets" / FOLD
    dataset_dir.mkdir(parents=True, exist_ok=True)
    runs = [{"table": t, "true_ranks": r, "true_utilities": [0.0] * F1,
             "subranks": [{"items": [0, 1], "levels": [[0], [1]], "cards": [0]}] * T}
            for t, r in zip(TABLES, TRUE_RANKS)]
    (dataset_dir / f"f1_{F1}__f2_{F2}__nsteps_{T}.json").write_text(json.dumps({"runs": runs}))

    for (i, j), state in STATES.items():
        table_dir = (data_dir / "samples" / FOLD / f"f1_{F1}_f2_{F2}_nsteps_{T}" / SUB_FOLD
                     / f"table_{i}")
        table_dir.mkdir(parents=True, exist_ok=True)
        np.save(table_dir / f"{j}.npy", state)

    runner = BenchmarkRunner(dataset_fold=FOLD, sub_fold=SUB_FOLD, num_subint=1, hm=10,
                             F1=[F1], F2=[F2], NUM_STEPS_LIST=[T], eta_max=3)
    runner.compute_metrics("poi", force=True)
    runner.compute_metrics("rai", force=True)
    runner.compute_asrs(force=True)
    runner.compute_asps(force=True)
    runner.compute_aios(force=True)
    return runner


def test_number_of_replications_is_capped_by_the_dataset(runner):
    assert runner.hm == 2


def test_rai_and_pwi_matrices(runner, data_dir):
    cfg = f"f1_{F1}_f2_{F2}_nsteps_{T}"
    metrics_dir = data_dir / "metrics" / FOLD / SUB_FOLD
    rai = np.load(metrics_dir / "rais" / cfg / "table_0" / "1.npy")
    pwi = np.load(metrics_dir / "pois" / cfg / "table_0" / "1.npy")
    np.testing.assert_allclose(rai, RAI_T0_S1)
    np.testing.assert_allclose(pwi, PWI_T0_S1)
    # a point estimate gives 0/1 indicators: the identity for RAI on the true order
    np.testing.assert_allclose(np.load(metrics_dir / "rais" / cfg / "table_0" / "2.npy"),
                               np.eye(3))


@pytest.mark.parametrize("metric", ["asrs", "asps", "aios"])
def test_asr_asp_aio_by_hand(runner, metric):
    values = load_test_results(metric, FOLD, SUB_FOLD, F1, F2, T)
    assert values.shape == (T, 2)          # (sessions, replications)
    np.testing.assert_allclose(values, EXPECTED[metric], rtol=1e-12)


def test_pwi_counts_ties_in_both_directions(runner):
    features = np.array(TABLES[0])
    pwi = runner.calc_poi(F1, features, np.array([0.5, 0.5] + JUNK))   # a0 and a1 tie
    assert pwi[0, 1] == pwi[1, 0] == 1.0


def test_missing_metric_files_give_an_empty_array(runner):
    assert load_test_results("asrs", FOLD, "BAYES_DOR_ADJJ1", F1, F2, T).size == 0
