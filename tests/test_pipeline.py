"""End-to-end smoke test of the simulation pipeline on a tiny dataset.

Runs, in a subprocess whose ``BDOR_DATA_DIR`` points at a temporary directory, the steps of
the main simulation: DatasetGenerator.generate_batch (consistent decision maker, Finc = 0),
then, for each of the four B-DOR arms of the paper, run_batch_experiments, the
BenchmarkRunner metrics and load_test_results, called as scripts/simulation/run_grid.py
calls them, and finally the hindsight benchmark of the regret study
(bdor.regret.hindsight_loss). Configuration: |A| = 8, m = 2, two sessions of three
alternatives, e_max = 5, two replications.

``bdor.paths`` reads ``BDOR_DATA_DIR`` at import time and several modules copy the paths
when imported, so a fresh interpreter is the clean way to relocate the data directory.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from bdor.paths import REPO_ROOT

FOLD = "smoke_F1=8_F2=2"
F1, F2, STEPS, HM, ETA_MAX = 8, 2, 2, 2, 5
N_PARAMS = 3 * F2                     # three pieces per criterion
CFG = f"f1_{F1}_f2_{F2}_nsteps_{STEPS}"
ARMS = [("FTRL_DOR_ADJJ1", "FTRL-DOR"), ("FTRLDIR_DOR_ADJJ1", "FTRLDIR-DOR"),
        ("BAYES_DOR_ADJJ1", "BAYES-DOR"), ("BAYESDIR_DOR_ADJJ1", "BAYESDIR-DOR")]

PIPELINE = r"""
import json, os, sys
import numpy as np
from bdor import paths
from bdor.datagen import DatasetGenerator
from bdor.metrics import BenchmarkRunner
from bdor.regret import hindsight_loss
from bdor.runner import run_batch_experiments
from bdor.utils import load_test_results

fold, report_path = sys.argv[1], sys.argv[2]
F1, F2, STEPS, HM, ETA_MAX = {F1}, {F2}, {STEPS}, {HM}, {ETA_MAX}
arms = {ARMS!r}

np.random.seed(0)
DatasetGenerator(output_dir=os.path.join(paths.DATASETS_DIR, fold), n_runs=HM).generate_batch(
    F1=[F1], F2=[F2], num_steps=STEPS, k_subset=(3, 3), n_max=ETA_MAX,
    target_consistency=1.0, plot_utilities=False)

report = {{"data_dir": str(paths.DATA_DIR), "metrics": {{}}}}
for sub_fold, alg in arms:
    run_batch_experiments(F1=[F1], F2=[F2], sub_fold=sub_fold, dataset_folds=[fold], alg=alg,
                          overwrite=False, hm=HM, k_subset=(3, 3), num_steps=STEPS,
                          eta_max=ETA_MAX, expansion="adjacent", seed_base=0, log_loss=True)
    runner = BenchmarkRunner(dataset_fold=fold, sub_fold=sub_fold, num_subint=3, hm=HM,
                             F1=[F1], F2=[F2], NUM_STEPS_LIST=[STEPS], eta_max=ETA_MAX)
    runner.compute_metrics("poi", force=True)
    runner.compute_metrics("rai", force=True)
    runner.compute_asrs(force=True)
    runner.compute_asps(force=True)
    runner.compute_aios(force=True)
    report["metrics"][sub_fold] = {{
        met: load_test_results(met, fold, sub_fold, F1, F2, STEPS).tolist()
        for met in ("asrs", "asps", "aios")}}

report["hindsight"] = hindsight_loss(fold, F1, F2, STEPS, HM, "adjacent").tolist()
report["hindsight_curves"] = hindsight_loss(fold, F1, F2, STEPS, HM, "adjacent",
                                            return_curves=True).tolist()
with open(report_path, "w") as fh:
    json.dump(report, fh)
""".format(F1=F1, F2=F2, STEPS=STEPS, HM=HM, ETA_MAX=ETA_MAX, ARMS=ARMS)


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    """Run the pipeline once; return (data directory, report written by the subprocess)."""
    work = tmp_path_factory.mktemp("pipeline")
    data_dir, report_path = work / "data", work / "report.json"
    env = dict(os.environ, BDOR_DATA_DIR=str(data_dir), PYTHONIOENCODING="utf-8")
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO_ROOT), env.get("PYTHONPATH")) if p)
    proc = subprocess.run(
        [sys.executable, "-B", "-c", PIPELINE, FOLD, str(report_path)], cwd=REPO_ROOT,
        env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=900)
    if proc.returncode != 0:
        pytest.fail(f"pipeline subprocess failed (exit {proc.returncode}):\n"
                    f"{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}")
    return data_dir, json.loads(report_path.read_text())


def test_data_directory_is_relocated(pipeline):
    data_dir, report = pipeline
    assert os.path.samefile(report["data_dir"], data_dir)
    assert (data_dir / "datasets" / FOLD / f"f1_{F1}__f2_{F2}__nsteps_{STEPS}.json").is_file()


@pytest.mark.parametrize("sub_fold", [arm for arm, _ in ARMS])
def test_metrics_per_session_and_replication(pipeline, sub_fold):
    _, report = pipeline
    for metric, values in report["metrics"][sub_fold].items():
        values = np.array(values)
        assert values.shape == (STEPS, HM), metric
        assert np.all((values >= 0.0) & (values <= 1.0)), metric


@pytest.mark.parametrize("sub_fold", [arm for arm, _ in ARMS])
def test_on_disk_layout(pipeline, sub_fold):
    """File names read back by run_grid.py (collect_sidecar) and the metric loaders."""
    data_dir, _ = pipeline
    bayes = sub_fold.startswith("BAYES")
    for i in range(HM):
        table_dir = data_dir / "samples" / FOLD / CFG / sub_fold / f"table_{i}"
        expected = {f"{j}.npy" for j in range(1, STEPS + 1)}
        expected |= {f"loss_{j}.npy" for j in range(1, STEPS + 1)}
        if bayes:
            expected |= {f"diag_{j}.npy" for j in range(1, STEPS + 1)}
        assert {f.name for f in table_dir.iterdir()} == expected
        for j in range(1, STEPS + 1):
            state = np.load(table_dir / f"{j}.npy")
            assert state.shape == ((200, N_PARAMS + ETA_MAX) if bayes else (N_PARAMS + ETA_MAX,))
            loss, n_gaps = np.load(table_dir / f"loss_{j}.npy")
            assert np.isfinite(loss) and loss > 0.0
            assert n_gaps == 2                      # three alternatives, no ties: two gaps
            if bayes:
                assert np.load(table_dir / f"diag_{j}.npy").shape == (2,)

    metrics_dir = data_dir / "metrics" / FOLD / sub_fold
    for matrix in ("pois", "rais"):                 # PWI and RAI matrices, (|A|, |A|)
        for i in range(HM):
            for j in range(1, STEPS + 1):
                path = metrics_dir / matrix / CFG / f"table_{i}" / f"{j}.npy"
                assert np.load(path).shape == (F1, F1)
    for metric in ("asrs", "asps", "aios"):         # one value per replication
        for j in range(1, STEPS + 1):
            assert np.load(metrics_dir / metric / CFG / f"{metric}_{j}.npy").shape == (HM,)


def test_hindsight_benchmark(pipeline):
    _, report = pipeline
    mean_curve = np.array(report["hindsight"])
    curves = np.array(report["hindsight_curves"])
    assert mean_curve.shape == (STEPS,) and curves.shape == (HM, STEPS)
    assert np.all(np.isfinite(curves)) and np.all(curves > 0.0)
    assert np.all(np.diff(curves, axis=1) >= 0.0)           # cumulative losses
    np.testing.assert_allclose(mean_curve, curves.mean(axis=0))
