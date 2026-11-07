"""DOR and SMAA-DOR baselines: the original single-session formulation reported in the paper
(arms *_ORIG, bdor.baselines.smaa_original) and the earlier per-gap adaptation (arms *_ADJJ1,
bdor.baselines.smaa_adapted).

Tiny session: 6 alternatives, 2 criteria (3 pieces each, so 6 weights), one session that
shows 3 of them in their true order with the cards of a consistent decision maker.
"""
import numpy as np
import pytest

from bdor.baselines.smaa_adapted import SMAADORSampler, sample_smaa_dor, solve_dor_lp
from bdor.baselines.smaa_original import (OriginalDORSampler, sample_smaa_dor_original,
                                          solve_dor_lp_original)
from bdor.features import PiecewiseLinearTransformer
from bdor.simulation import expand_subrank

N_SAMPLES = 50


@pytest.fixture(scope="module")
def session():
    rng = np.random.default_rng(5)
    table = rng.random((6, 2))
    transformer = PiecewiseLinearTransformer.from_equal_intervals(table, 3)
    X = transformer.transform(table)
    p = transformer.total_params
    U = X @ rng.dirichlet(np.ones(p))
    order = sorted([0, 2, 4], key=lambda a: -U[a])
    levels = [[a] for a in order]
    cards = [int(np.clip(np.round(5 * (U[a] - U[b])), 0, 5))
             for a, b in zip(order[:-1], order[1:])]
    gaps = expand_subrank(levels, cards, "adjacent")
    X_diff = np.array([X[u].mean(axis=0) - X[l].mean(axis=0) for u, l, _ in gaps])
    etas = np.array([g[2] for g in gaps])
    return dict(X=X, p=p, levels=levels, cards=cards, X_diff=X_diff, etas=etas)


def _assert_on_simplex(w, p):
    w = np.atleast_2d(w)
    assert w.shape[1] == p
    assert w.min() >= 0.0
    np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-8)


def _assert_declared_order(X, levels, w, tol=1e-9):
    """U(a) >= U(b) for every a of a level and b of the next one, for every weight row."""
    w = np.atleast_2d(w)
    for upper, lower in zip(levels[:-1], levels[1:]):
        for a in upper:
            for b in lower:
                assert ((X[a] - X[b]) @ w.T).min() >= -tol


# ------------------------------------------------------------------- original (paper)

def test_original_dor_point_estimate(session):
    sampler = OriginalDORSampler(session["X"], session["levels"], session["cards"], session["p"])
    w = sampler.solve_point()
    assert w.shape == (session["p"],)
    _assert_on_simplex(w, session["p"])
    assert sampler.info["hard_ordering"] and sampler.info["k_star"] > 0
    _assert_declared_order(session["X"], session["levels"], w)
    np.testing.assert_array_equal(
        w, solve_dor_lp_original(session["X"], session["levels"], session["cards"], session["p"]))


def test_original_smaa_dor_sample(session):
    p = session["p"]
    sampler = OriginalDORSampler(session["X"], session["levels"], session["cards"], p, seed=0)
    draws = sampler.sample(N_SAMPLES)
    assert draws.shape == (N_SAMPLES, p + 1)            # rows [w, k]
    _assert_on_simplex(draws[:, :p], p)
    assert draws[:, p].min() > 0.0                       # value of a blank card k > 0
    assert sampler.info["hard_ordering"] and not sampler.info["degenerate"]
    _assert_declared_order(session["X"], session["levels"], draws[:, :p])
    assert np.ptp(draws[:, :p], axis=0).max() > 0.0      # Hit-and-Run moved
    np.testing.assert_array_equal(draws, sample_smaa_dor_original(
        session["X"], session["levels"], session["cards"], p, num_samples=N_SAMPLES, seed=0))


def test_original_methods_without_information(session):
    p = session["p"]
    sampler = OriginalDORSampler(session["X"], [[0, 1, 2]], [], p, seed=0)   # one level
    np.testing.assert_allclose(sampler.solve_point(), np.full(p, 1.0 / p))
    draws = sampler.sample(5)
    assert draws.shape == (5, p + 1)
    _assert_on_simplex(draws[:, :p], p)
    np.testing.assert_array_equal(draws[:, p], 0.0)


# ------------------------------------------------------------------- adapted (per gap)

def test_adapted_dor_point_estimate(session):
    p = session["p"]
    w = solve_dor_lp(session["X_diff"], session["etas"], p)
    _assert_on_simplex(w, p)
    assert (session["X_diff"] @ w).min() >= -1e-9
    np.testing.assert_allclose(solve_dor_lp(np.empty((0, p)), [], p), np.full(p, 1.0 / p))


def test_adapted_smaa_dor_sample(session):
    p = session["p"]
    sampler = SMAADORSampler(session["X_diff"], session["etas"], p, seed=0)
    draws = sampler.sample(N_SAMPLES)
    assert draws.shape == (N_SAMPLES, p + 1)
    _assert_on_simplex(draws[:, :p], p)
    assert draws[:, p].min() > 0.0
    assert sampler.info["hard_ordering"] and not sampler.info["degenerate"]
    assert (session["X_diff"] @ draws[:, :p].T).min() >= -1e-9
    np.testing.assert_array_equal(draws, sample_smaa_dor(
        session["X_diff"], session["etas"], p, num_samples=N_SAMPLES, seed=0))
