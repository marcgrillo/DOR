"""Central file-system layout of the project.

The library modules and the pipeline scripts under ``scripts/`` take their input/output
locations from here, so nothing depends on the current working directory (the
interactive example in ``examples/`` writes its plots next to its input file).

    <repo>/
      data/                 local-only, large: gitignored except its README and the two
                            dataset folders Figure 5 needs (see data/README.md)
        datasets/           synthetic elicitation datasets, one folder per configuration
        samples/            inference state after every session (posterior draws, point
                            estimates) with its loss / HMC-diagnostic sidecars
        metrics/            cache of per-session RAI/PWI matrices and ASR/ASP/AIO vectors
      results/              aggregated results behind the paper (committed)
      outputs/              paper figures and tables (committed)
        figures/ tables/    built by scripts/paper/, mainly from results/
        case_study/         written by scripts/case_study/case_study_lea.py
        diagnostics/        exploratory plots of the studies (gitignored)

The data directory can be relocated (e.g. to another drive) by setting the
environment variable ``BDOR_DATA_DIR`` before running any script; ``results/`` and
``outputs/`` always stay in the repository.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.environ.get("BDOR_DATA_DIR") or REPO_ROOT / "data")  # unset or empty: default
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
