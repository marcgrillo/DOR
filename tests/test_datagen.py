"""Synthetic decision makers and their elicitation data (bdor.datagen, paper Sec. 5.1)."""
import json

import numpy as np
import pytest

from bdor.datagen import DatasetGenerator
from bdor.utils import read_dataset

F1, F2, STEPS, K_SUBSET, N_MAX = 8, 2, 3, (4, 3), 5


def _generate(out_dir, n_runs=3, seed=0, **kwargs):
    np.random.seed(seed)    # the generator draws from NumPy's global random state
    DatasetGenerator(output_dir=str(out_dir), n_runs=n_runs).generate_batch(
        F1=[F1], F2=[F2], num_steps=STEPS, k_subset=K_SUBSET, n_max=N_MAX,
        plot_utilities=False, **kwargs)
    path = out_dir / f"f1_{F1}__f2_{F2}__nsteps_{STEPS}.json"
    return json.loads(path.read_text())


def test_consistent_decision_maker(tmp_path):
    """F_inc = 0 (target_consistency = 1): no noise, so every session declares the true
    order of its subset and the cards are round(e_max * utility gap), eq. (21) with sigma = 0."""
    data = _generate(tmp_path, target_consistency=1.0)
    meta = data["metadata"]
    assert meta["sigma"] == 0.0
    assert (meta["f1"], meta["f2"], meta["num_steps"], meta["n_max"]) == (F1, F2, STEPS, N_MAX)
    assert len(data["runs"]) == meta["n_runs"] == 3

    for i, run in enumerate(data["runs"]):
        table = np.array(run["table"])
        U = np.array(run["true_utilities"])
        assert table.shape == (F1, F2) and table.min() >= 0.0 and table.max() < 1.0
        np.testing.assert_array_equal(run["true_ranks"], np.argsort(np.argsort(-U)))
        assert len(run["subranks"]) == STEPS
        for j, sr in enumerate(run["subranks"], start=1):
            k = K_SUBSET[0] if j == 1 else K_SUBSET[-1]   # session 1 may use another size
            # the shown subset is seeded per (run, session), independently of the global state
            subset = np.random.default_rng(i * 1000 + j).choice(F1, k, replace=False)
            assert sorted(sr["items"]) == sorted(subset.tolist())
            assert sr["levels"] == [[a] for a in sr["items"]]      # merge_threshold = 0
            assert np.all(np.diff(U[sr["items"]]) < 0)             # true order, best first
            assert sr["cards"] == [int(np.clip(np.round(N_MAX * U[a] - N_MAX * U[b]), 0, N_MAX))
                                   for a, b in zip(sr["items"][:-1], sr["items"][1:])]


def test_noise_scale_grows_with_the_share_of_inconsistent_data(tmp_path):
    sigma = {tc: _generate(tmp_path / str(tc), n_runs=1, target_consistency=tc)
             ["metadata"]["sigma"] for tc in (0.85, 0.5)}
    assert 0.0 < sigma[0.85] < sigma[0.5]


def test_indifference_levels(tmp_path):
    """merge_threshold > 0: consecutive alternatives (in true order) closer than the threshold
    share a level, and the cards sit between levels."""
    threshold = 0.05
    data = _generate(tmp_path, target_consistency=1.0, merge_threshold=threshold)
    n_tied = 0
    for run in data["runs"]:
        U = np.array(run["true_utilities"])
        for sr in run["subranks"]:
            levels = sr["levels"]
            assert sr["items"] == [a for lv in levels for a in lv]
            assert len(sr["cards"]) == len(levels) - 1
            for lv in levels:
                n_tied += len(lv) > 1
                inner = U[sorted(lv, key=lambda a: -U[a])]
                assert np.all(-np.diff(inner) < threshold)
            for upper, lower in zip(levels[:-1], levels[1:]):
                assert U[upper].min() - U[lower].max() >= threshold
    assert n_tied > 0


def test_read_dataset(tmp_path):
    _generate(tmp_path, target_consistency=1.0)
    tables, rankings, subranks, utilities = read_dataset(str(tmp_path), F1, F2, STEPS)
    assert tables.shape == (3, F1, F2)
    assert rankings.shape == utilities.shape == (3, F1)
    assert len(subranks) == 3 and len(subranks[0]) == STEPS
    with pytest.raises(FileNotFoundError):
        read_dataset(str(tmp_path), F1 + 1, F2, STEPS)
