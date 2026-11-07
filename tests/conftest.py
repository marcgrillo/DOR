"""Shared set-up of the B-DOR test suite.

Run from the repository root (``pyproject.toml`` puts the root on ``sys.path``)::

    python -B -m pytest -q -p no:cacheprovider      # whole suite, about one minute on a CPU

Safety: the suite never reads or writes the real ``data/`` tree (about 23 GB, local only).
``bdor.paths`` reads ``BDOR_DATA_DIR`` once, when it is first imported, and several modules
copy its paths at import time. The variable is therefore pointed at a throw-away directory
here, before any test module imports ``bdor``; tests that write under the data directory
request the ``data_dir`` fixture, which checks that the redirection took effect. The
end-to-end pipeline test runs in a subprocess with its own ``BDOR_DATA_DIR``. Matplotlib
uses the non-interactive Agg backend unless ``MPLBACKEND`` is already set.
"""
import os
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_TEST_DATA_DIR = Path(tempfile.mkdtemp(prefix="bdor-tests-data-"))
os.environ["BDOR_DATA_DIR"] = str(_TEST_DATA_DIR)
os.environ.setdefault("MPLBACKEND", "Agg")


def pytest_unconfigure(config):
    shutil.rmtree(_TEST_DATA_DIR, ignore_errors=True)


@pytest.fixture(scope="session")
def data_dir():
    """The redirected data directory (``bdor.paths.DATA_DIR``), never ``<repo>/data``."""
    from bdor import paths
    if Path(paths.DATA_DIR).resolve() != _TEST_DATA_DIR.resolve():
        pytest.fail("bdor.paths was imported before tests/conftest.py redirected "
                    "BDOR_DATA_DIR; refusing to write into the real data directory")
    return Path(paths.DATA_DIR)


def make_problem(n_alt=10, n_crit=2, n_intervals=3, n_max=5, n_sessions=6, k=4,
                 scale=6.0, band=0.5, seed=7):
    """Synthetic decision problem with a known value function and noise-free answers.

    The true weights are ``scale * Dirichlet(1)`` over the piecewise-linear features of a
    random ``(n_alt, n_crit)`` performance table (``n_intervals`` pieces per criterion), and
    the true thresholds are ``theta_m = band * m`` (paper eq. (10)). Each of the
    ``n_sessions`` sessions shows ``k`` random alternatives; the declared levels are their
    true order (singletons) and the card count of a gap is the band of eq. (10) that
    contains its true strength ``z = x^T w``. With the defaults the counts cover all of
    0..n_max and every alternative is shown at least once.

    Returns a ``SimpleNamespace`` with ``table``, ``X`` (features, ``(n_alt, n_params)``),
    ``n_params``, ``n_max``, ``w_true``, ``theta_true``, ``U`` (true values), ``true_ranks``
    (0 = best), ``subranks`` (dataset format: ``items`` / ``levels`` / ``cards``) and
    ``gaps`` (adjacent-gap observations ``(upper_members, lower_members, eta)``).
    """
    from bdor.features import PiecewiseLinearTransformer
    from bdor.simulation import expand_subrank

    rng = np.random.default_rng(seed)
    table = rng.random((n_alt, n_crit))
    transformer = PiecewiseLinearTransformer.from_equal_intervals(table, n_intervals)
    X = transformer.transform(table)
    n_params = transformer.total_params
    w_true = scale * rng.dirichlet(np.ones(n_params))
    theta_true = band * np.arange(1, n_max + 1)
    U = X @ w_true

    subranks = []
    for _ in range(n_sessions):
        subset = rng.choice(n_alt, size=k, replace=False)
        order = [int(a) for a in subset[np.argsort(-U[subset])]]
        cards = [int(np.sum(theta_true <= U[a] - U[b])) for a, b in zip(order[:-1], order[1:])]
        subranks.append({"items": order, "levels": [[a] for a in order], "cards": cards})
    gaps = [g for sr in subranks for g in expand_subrank(sr["levels"], sr["cards"], "adjacent")]

    return SimpleNamespace(
        table=table, X=X, n_params=n_params, n_max=n_max, w_true=w_true,
        theta_true=theta_true, U=U, true_ranks=np.argsort(np.argsort(-U)),
        subranks=subranks, gaps=gaps)


@pytest.fixture(scope="session")
def problem():
    """The default :func:`make_problem` instance (10 alternatives, 2 criteria, 6 sessions of 4)."""
    return make_problem()
