"""Central file-system layout of the project.

Every script and library module takes its input/output locations from here, so
nothing depends on the current working directory.

    <repo>/
      data/                 local-only, large (gitignored; see data/README.md)
        datasets/           synthetic elicitation datasets, one folder per configuration
        samples/            inference state after every session (posterior draws / MAP)
        metrics/            per-session RAI/PWI matrices and ASR/ASP/AIO vectors
      results/              aggregated results behind the paper (committed)
      outputs/              figures and tables generated from results/ (committed)
        figures/ tables/ case_study/
        diagnostics/        exploratory plots of the studies (gitignored)

The data directory can be relocated (e.g. to another drive) by setting the
environment variable ``BDOR_DATA_DIR`` before running any script.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.environ.get("BDOR_DATA_DIR", REPO_ROOT / "data"))
DATASETS_DIR = DATA_DIR / "datasets"
SAMPLES_DIR = DATA_DIR / "samples"
METRICS_DIR = DATA_DIR / "metrics"

RESULTS_DIR = REPO_ROOT / "results"
GRID_RESULTS = RESULTS_DIR / "grid_results_joint_v1.npz"

OUTPUTS_DIR = REPO_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
CASE_STUDY_DIR = OUTPUTS_DIR / "case_study"
DIAGNOSTICS_DIR = OUTPUTS_DIR / "diagnostics"
