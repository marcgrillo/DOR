"""Committed aggregated results (results/grid_results_joint_v1.npz) read through bdor.results:
layout, pooling over the paper's 768 cells, the Table 3 means quoted in the paper text, and
agreement with the independent T = 1 recomputation in results/original_comparison.csv."""
import csv

import numpy as np
import pytest

from bdor import results as m
from bdor.paths import GRID_RESULTS, RESULTS_DIR

# every arm stored for every configuration: (sessions stored, replications per cell)
SHAPES = {
    "FTRL_DOR_ADJJ1": (10, 20), "BAYES_DOR_ADJJ1": (10, 20),
    "FTRLDIR_DOR_ADJJ1": (10, 20), "BAYESDIR_DOR_ADJJ1": (10, 5),
    "FTRL_DOR_ALLJ1": (10, 20), "BAYES_DOR_ALLJ1": (10, 5),
    "SMAA_DOR_ADJJ1": (10, 20), "DORLP_DOR_ADJJ1": (10, 20),
    "SMAA_DOR_ORIG": (1, 20), "DORLP_DOR_ORIG": (1, 20),
}


@pytest.fixture(scope="module")
def results():
    return m.load_results()


def test_npz_metadata():
    with np.load(GRID_RESULTS, allow_pickle=True) as z:
        assert sorted(z.files) == ["eta_max", "results", "run_seed", "run_tag"]
        assert str(z["run_tag"]) == "joint-v1"
        assert int(z["run_seed"]) == 20260806
        assert int(z["eta_max"]) == 5


def test_grid_layout(results):
    assert len(results) == 576
    paper_configs = [c for c in results if m.parse_config(c)["AR"] == m.parse_config(c)["K"]]
    assert len(paper_configs) == 192            # x 4 horizons = the paper's 768 cells
    for config, arms in results.items():
        assert set(arms) == set(SHAPES), config
        for arm, shape in SHAPES.items():
            for metric in m.METRICS:
                values = np.array(arms[arm][metric], dtype=float)
                assert values.shape == shape, (config, arm, metric)
                assert np.all((values >= 0.0) & (values <= 1.0)), (config, arm, metric)


def test_parse_config():
    assert m.parse_config("F1=35_F2=7_S=10_AR=4_K=5_Inc=15") == {
        "F1": 35, "F2": 7, "S": 10, "AR": 4, "K": 5, "Inc": 15}


def test_pool_sizes(results):
    assert len(m.pool(results, "FTRL_DOR_ADJJ1", "asrs")) == 768 * 20
    assert len(m.pool(results, "BAYES_DOR_ADJJ1", "asrs")) == 768 * 20
    # paired with its direction-only version: the 5 replications BAYESdir has
    assert len(m.pool(results, "BAYES_DOR_ADJJ1", "asrs", match="BAYESDIR_DOR_ADJJ1")) == 768 * 5
    assert len(m.pool(results, "SMAA_DOR_ORIG", "aios", T=1)) == 192 * 20
    assert len(m.pool(results, "FTRL_DOR_ADJJ1", "asps", T=1, F1=10, Inc=50)) == 12 * 20


def test_mean_ci_ignores_nan():
    mean, half_width = m.mean_ci(np.array([1.0, np.nan, 3.0]))
    assert mean == 2.0
    assert half_width == pytest.approx(1.96 / np.sqrt(2))


# Table 3 of the paper, as quoted in Sec. 5.3 / 5.5 (ASR, ASP, AIO; 3 decimals).
UPPER_BLOCK = {   # whole grid, all horizons, each arm on the replications shared with its partner
    "FTRL_DOR_ADJJ1": (0.212, 0.846, 0.502),
    "FTRLDIR_DOR_ADJJ1": (0.189, 0.835, 0.469),
    "BAYES_DOR_ADJJ1": (0.175, 0.805, 0.439),
    "BAYESDIR_DOR_ADJJ1": (0.159, 0.793, 0.412),
}
LOWER_BLOCK = {   # first session only (T = 1)
    "FTRL_DOR_ADJJ1": (0.145, 0.792, 0.404),
    "BAYES_DOR_ADJJ1": (0.123, 0.748, 0.330),
    "SMAA_DOR_ORIG": (0.117, 0.718, 0.267),
    "DORLP_DOR_ORIG": (0.121, 0.750, 0.253),
}


@pytest.mark.parametrize("arm", list(UPPER_BLOCK))
def test_table3_upper_block(results, arm):
    got = [m.mean_ci(m.pool(results, arm, met, match=m.PARTNER[arm]))[0] for met in m.METRICS]
    assert [f"{v:.3f}" for v in got] == [f"{v:.3f}" for v in UPPER_BLOCK[arm]]


@pytest.mark.parametrize("arm", list(LOWER_BLOCK))
def test_table3_lower_block(results, arm):
    got = [m.mean_ci(m.pool(results, arm, met, T=1))[0] for met in m.METRICS]
    assert [f"{v:.3f}" for v in got] == [f"{v:.3f}" for v in LOWER_BLOCK[arm]]


def test_original_comparison_csv_agrees_with_the_npz(results):
    """results/original_comparison.csv recomputes the T = 1 metrics of six methods with its own
    implementation (scripts/simulation/run_original_baselines.py); the npz holds the
    BenchmarkRunner values, and its *_ORIG arms were injected from this CSV. Both must agree
    on every (configuration, replication, method). A failure means the two files come from
    different runs: re-run run_original_baselines.py and inject_original_baselines.py."""
    arm_of = {"SMAAORIG_T1": "SMAA_DOR_ORIG", "DORLPORIG_T1": "DORLP_DOR_ORIG"}
    keys, stored, recomputed = [], [], []
    with open(RESULTS_DIR / "original_comparison.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            config = row["fold"].replace("grid_", "")
            arm = arm_of.get(row["method"], row["method"])
            table = int(row["table"])
            for metric, column in (("asrs", "asr"), ("asps", "asp"), ("aios", "aio")):
                keys.append((config, table, row["method"], metric))
                stored.append(results[config][arm][metric][0][table])
                recomputed.append(float(row[column]))
    assert len(keys) == 576 * 20 * 6 * 3
    # equal up to rounding: ASP is summed in a different order by the two implementations
    bad = np.flatnonzero(np.abs(np.array(stored) - np.array(recomputed)) > 1e-12)
    assert bad.size == 0, [keys[i] for i in bad[:5]]
