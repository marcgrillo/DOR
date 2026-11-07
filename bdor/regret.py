"""Best-fixed-in-hindsight benchmark for the regret study (paper Sec. 5.7, Figure 5).

The regret of an online learner after session T is its cumulative prequential
log-loss minus the cumulative loss of the single parameter vector chosen with
hindsight.  The comparator used in the paper is the *unpenalised* maximum-likelihood
fit on all sessions' gap observations, computed here by L-BFGS-B with exact JAX
gradients (box ``omega >= 1e-6``, no prior).

The online losses themselves are recorded during the simulation
(``loss_<j>.npy`` sidecars, stored in the grid npz under ``'log_loss'``).
"""
import os

import numpy as np

from bdor.paths import DATASETS_DIR

ETA_MAX = 5


def hindsight_loss(dataset_fold, f1, f2, steps, hm, expansion, return_curves=False,
                   eta_max=ETA_MAX):
    """Cumulative per-session loss of the best fixed parameter in hindsight.

    Parameters
    ----------
    dataset_fold : str
        Dataset folder name under ``data/datasets`` (e.g. ``'grid_F1=10_F2=3_S=10_AR=5_K=5_Inc=0'``).
    f1, f2, steps : int
        Number of alternatives |A|, criteria m and sessions stored in the dataset.
    hm : int
        Number of replications (tables) to use, starting from the first.
    expansion : {'adjacent', 'all'}
        How a declared ranking is turned into gap observations (paper: 'adjacent').
    return_curves : bool
        If True return the per-table curves (hm x steps), otherwise their mean (steps,).
    eta_max : int
        Maximum number of blank cards e_max of the likelihood.
    """
    from scipy.optimize import minimize as sp_minimize
    import jax
    import jax.numpy as jnp
    from bdor import utils
    from bdor.features import PiecewiseLinearTransformer
    from bdor.engine import _jax_log_prob_per_obs, RHO
    from bdor.simulation import expand_subrank

    tables, rankings, dm_prefs, _ = utils.read_dataset(
        os.path.join(DATASETS_DIR, dataset_fold), f1, f2, steps)
    curves = []
    for i in range(min(hm, len(tables))):
        transformer = PiecewiseLinearTransformer.from_equal_intervals(tables[i], num_intervals=3)
        Xf = transformer.transform(tables[i])
        n_params = transformer.total_params
        batches = []
        for j in range(steps):
            sr = dm_prefs[i][j]
            gaps = expand_subrank(sr.get('levels', sr['items']), sr['cards'], expansion)
            X = np.array([Xf[u].mean(axis=0) - Xf[l].mean(axis=0) for u, l, _ in gaps])
            batches.append((jnp.array(X), jnp.array([g[2] for g in gaps])))
        X_all = jnp.concatenate([b[0] for b in batches])
        e_all = jnp.concatenate([b[1] for b in batches])
        ndim = n_params + eta_max

        def neg_ll(om):
            return -jnp.sum(_jax_log_prob_per_obs(jnp.array(om), X_all, e_all,
                                                  n_params, eta_max, RHO))
        val_grad = jax.value_and_grad(neg_ll)

        def neg_ll_np(om):
            v, g = val_grad(jnp.array(om))
            return float(v), np.array(g, dtype=np.float64)

        res = sp_minimize(neg_ll_np, np.ones(ndim) * 0.5, method='L-BFGS-B', jac=True,
                          bounds=[(1e-6, None)] * ndim)
        om_star = jnp.array(res.x)
        per_step = [float(-jnp.sum(_jax_log_prob_per_obs(om_star, X, e, n_params, eta_max, RHO)))
                    for X, e in batches]
        curves.append(np.cumsum(per_step))
    return np.array(curves) if return_curves else np.mean(curves, axis=0)
