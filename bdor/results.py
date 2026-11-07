"""Loading and pooling of the aggregated grid results (``results/grid_results_joint_v1.npz``).

File layout
-----------
``np.load(GRID_RESULTS, allow_pickle=True)`` holds ``run_tag`` ('joint-v1'),
``run_seed`` (20260806), ``eta_max`` (5) and ``results``, a 0-d object array whose
``.item()`` is a nested dict::

    results[config][arm][key]

* ``config`` -- ``'F1=<|A|>_F2=<m>_S=10_AR=<k of session 1>_K=<k of sessions 2..10>_Inc=<Finc %>'``
  (576 configurations; the paper uses the 192 with ``AR == K``).
* ``arm`` -- sub-fold tag ``<ALG>_DOR_<TAG>`` (see README, "Naming conventions"). The six
  arms reported in the paper are FTRL_DOR_ADJJ1 (FTRL-DOR), BAYES_DOR_ADJJ1 (BAYES-DOR),
  FTRLDIR_DOR_ADJJ1 (FTRLdir), BAYESDIR_DOR_ADJJ1 (BAYESdir), DORLP_DOR_ORIG (DOR) and
  SMAA_DOR_ORIG (SMAA-DOR).
* ``key`` -- ``'asrs' | 'asps' | 'aios'``: lists of shape (sessions, replications); row
  ``t-1`` is the metric after session ``t``. The ``*_ORIG`` arms have a single row (T = 1).
  Most arms also store ``'log_loss'`` (sessions x replications x [prequential loss, number
  of gaps]), the Bayesian arms ``'hmc_diag'`` ([divergences, max split-Rhat]), and every
  arm ``'runtime_s'`` and ``'seed_base'``.

The paper's 768 cells are the 192 ``AR == K`` configurations, each read after
``T in {1, 3, 5, 10}`` sessions (prefixes of one 10-session run).
"""
import re

import numpy as np

from bdor.paths import GRID_RESULTS

METRICS = ['asrs', 'asps', 'aios']
MET_LABEL = {'asrs': 'ASR', 'asps': 'ASP', 'aios': 'AIO'}
T_LEVELS = [1, 3, 5, 10]

# The direction-only ablation is a paired comparison, so the two arms of a family
# are read on the replications they share: BAYES-DOR runs on all 20 replications
# per cell, BAYESdir on the first 5 (FTRL-DOR and FTRLdir both on 20).
PARTNER = {
    'FTRL_DOR_ADJJ1': 'FTRLDIR_DOR_ADJJ1',
    'FTRLDIR_DOR_ADJJ1': 'FTRL_DOR_ADJJ1',
    'BAYES_DOR_ADJJ1': 'BAYESDIR_DOR_ADJJ1',
    'BAYESDIR_DOR_ADJJ1': 'BAYES_DOR_ADJJ1',
}


def load_results(path=GRID_RESULTS):
    """Return the ``results`` dict stored in the aggregated grid npz."""
    return np.load(path, allow_pickle=True)['results'].item()


def parse_config(c):
    """``'F1=10_F2=3_S=10_AR=3_K=3_Inc=0'`` -> ``{'F1': 10, 'F2': 3, 'S': 10, 'AR': 3, 'K': 3, 'Inc': 0}``."""
    m = re.match(r"F1=(\d+)_F2=(\d+)_S=(\d+)_AR=(\d+)_K=(\d+)_Inc=(\d+)", c)
    return {k: int(v) for k, v in zip(['F1', 'F2', 'S', 'AR', 'K', 'Inc'], m.groups())}


def pool(results, arm, metric, T=None, match=None, **filters):
    """Pooled per-replication values of ``arm``/``metric`` over the paper's cells.

    Only configurations with ``AR == K`` that match every ``filters`` item
    (e.g. ``F1=10, Inc=50``) are used; each contributes its values after the
    sessions in ``T_LEVELS`` (or only after session ``T`` if given).  With
    ``match=<other arm>`` every cell is cut to the replications the two arms
    share, so the mean is read on the same decision problems as its partner.
    """
    vals = []
    for c, arms in results.items():
        p = parse_config(c)
        if p['AR'] != p['K'] or arm not in arms:
            continue
        if any(p[k] != v for k, v in filters.items()):
            continue
        data = np.array(arms[arm][metric])
        n = data.shape[1]
        if match is not None and match in arms:
            n = min(n, np.array(arms[match][metric]).shape[1])
        for t in ([T] if T else T_LEVELS):
            vals.extend(data[t - 1][:n])
    return np.array(vals)


def mean_ci(v):
    """Mean and half-width of the normal 95% confidence interval, ignoring NaNs.

    NaNs occur when a session collapses to a single level (ties study): it yields
    no gap observation, so at t = 1 BAYES-DOR has no posterior sample.
    """
    v = v[~np.isnan(v)]
    return v.mean(), 1.96 * v.std() / np.sqrt(len(v))
