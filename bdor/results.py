"""Loading and pooling of the aggregated grid results (``results/grid_results_joint_v1.npz``).

The paper scripts (``scripts/paper/``) read this file through the helpers below to build
Table 3, Figures 2, 3 and 5, Supplementary Tables B.1-B.9 and the Sec. 5.3 numbers.

File layout
-----------
``np.load(GRID_RESULTS, allow_pickle=True)`` holds the 0-d arrays ``run_tag``
('joint-v1'), ``run_seed`` (20260806) and ``eta_max`` (5), and ``results``, a 0-d
object array whose ``.item()`` is a nested dict::

    results[config][arm][key]

* ``config`` -- ``'F1=<|A|>_F2=<m>_S=10_AR=<k_1>_K=<k>_Inc=<Finc %>'``, with k_1 the
  subset size of session 1 and k that of sessions 2-10 (576 configurations; the paper
  uses the 192 with ``AR == K``). Its dataset folder is ``'grid_' + config``.
* ``arm`` -- sub-fold tag ``<ALG>_DOR_<TAG>`` (see README, "Naming conventions"). Every
  configuration holds ten arms. The six reported in the paper are FTRL_DOR_ADJJ1
  (FTRL-DOR), BAYES_DOR_ADJJ1 (BAYES-DOR), FTRLDIR_DOR_ADJJ1 (FTRLdir),
  BAYESDIR_DOR_ADJJ1 (BAYESdir), DORLP_DOR_ORIG (DOR) and SMAA_DOR_ORIG (SMAA-DOR); the
  other four are the all-pairs ablation FTRL_DOR_ALLJ1 / BAYES_DOR_ALLJ1 and the adapted
  per-gap baselines DORLP_DOR_ADJJ1 / SMAA_DOR_ADJJ1.
* ``key`` -- ``'asrs' | 'asps' | 'aios'`` (ASR / ASP / AIO): nested lists of shape
  (sessions, replications); row ``t-1`` holds the metric after session ``t`` and column
  ``i`` replication ``i``. There are 20 replications, except for BAYESDIR_DOR_ADJJ1 and
  BAYES_DOR_ALLJ1 (the first 5). The ``*_ORIG`` arms, added by
  ``scripts/simulation/inject_original_baselines.py``, hold a single row (T = 1) and only
  these three keys. The arms written by ``scripts/simulation/run_grid.py`` also store

  - ``'log_loss'``: sessions x replications x [prequential log-loss of the session's
    gaps, number of gaps]; NaN for the SMAA / DORLP arms, which have no likelihood;
  - ``'runtime_s'``: wall-clock seconds of the run that wrote the entry (small when the
    inference states were already cached);
  - ``'seed_base'``: session t of replication i is seeded with
    ``seed_base + 100003 * i + t``;
  - ``'hmc_diag'`` (BAYES* arms only): sessions x replications x [divergent transitions,
    max split-Rhat].

The paper's 768 cells are the 192 ``AR == K`` configurations, each read after
``T in {1, 3, 5, 10}`` sessions (prefixes of one 10-session run).
"""
import re

import numpy as np

from bdor.paths import GRID_RESULTS

METRICS = ['asrs', 'asps', 'aios']                         # metric keys in the npz
MET_LABEL = {'asrs': 'ASR', 'asps': 'ASP', 'aios': 'AIO'}  # their names in the paper
T_LEVELS = [1, 3, 5, 10]                                   # horizons T of the paper's grid

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
    """Return the ``results`` dict stored in the aggregated grid npz (layout above)."""
    return np.load(path, allow_pickle=True)['results'].item()


def parse_config(c):
    """Factor levels of a configuration string.

    ``'F1=10_F2=3_S=10_AR=3_K=3_Inc=0'`` ->
    ``{'F1': 10, 'F2': 3, 'S': 10, 'AR': 3, 'K': 3, 'Inc': 0}``.
    """
    m = re.match(r"F1=(\d+)_F2=(\d+)_S=(\d+)_AR=(\d+)_K=(\d+)_Inc=(\d+)", c)
    return {k: int(v) for k, v in zip(['F1', 'F2', 'S', 'AR', 'K', 'Inc'], m.groups())}


def pool(results, arm, metric, T=None, match=None, **filters):
    """Pooled per-replication values of ``arm``/``metric`` over the paper's cells.

    Only configurations with ``AR == K`` that match every ``filters`` item
    (e.g. ``F1=10, Inc=50``) are used; each contributes its values after the
    sessions in ``T_LEVELS`` (or only after session ``T`` if given).  With
    ``match=<other arm>`` every cell is cut to the replications the two arms
    share, so the mean is read on the same decision problems as its partner.

    Parameters
    ----------
    results : dict
        Output of ``load_results``.
    arm : str
        Arm tag, e.g. ``'FTRL_DOR_ADJJ1'``; configurations without it are skipped.
    metric : str
        ``'asrs'``, ``'asps'`` or ``'aios'``.
    T : int, optional
        Single horizon to read; by default all of ``T_LEVELS``. The ``*_ORIG`` arms
        hold only the first session, so they need ``T=1``.
    match : str, optional
        Partner arm (see ``PARTNER``): each cell then contributes only its first
        min(n_arm, n_match) replications.
    **filters : int
        Fixed factor levels, keyed as in ``parse_config`` (``F1``, ``F2``, ``S``,
        ``AR``, ``K``, ``Inc``).

    Returns
    -------
    ndarray, shape (n_values,)
        The values, concatenated over configurations, horizons and replications.
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

    The half-width is 1.96 * std / sqrt(n) over the n non-NaN values, with the
    population standard deviation (ddof = 0). NaNs occur when a session collapses to
    a single level (ties study): it yields no gap observation, so at t = 1 BAYES-DOR
    has no posterior sample.

    Parameters
    ----------
    v : ndarray, shape (n_values,)

    Returns
    -------
    mean, half_width : float
    """
    v = v[~np.isnan(v)]
    return v.mean(), 1.96 * v.std() / np.sqrt(len(v))
