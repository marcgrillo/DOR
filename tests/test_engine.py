"""PreferenceSampler: gap likelihood (paper eqs. (11)-(13)), priors (eq. (14)), FTRL-DOR (MAP)
and BAYES-DOR (NUTS).

Parameter vectors are ``omega = [w (n_params), delta_raw (n_max)]`` with thresholds
``theta_m = sum_{i<=m} (RHO + delta_raw_i)`` and ``theta_0 = 0``.
"""
import numpy as np
import pytest
from scipy.special import expit
from scipy.stats import kendalltau

from bdor.engine import RHO, PreferenceSampler


def _fitted_sampler(problem, likelihood="joint"):
    sampler = PreferenceSampler(problem.X, [], problem.n_params, problem.n_max,
                                likelihood=likelihood)
    for upper, lower, eta in problem.gaps:
        sampler.add_gap(upper, lower, eta)
    return sampler


def _tau(problem, omega):
    """Kendall tau between the true values and those of the weight part of ``omega``."""
    return kendalltau(problem.U, problem.X @ np.asarray(omega)[:problem.n_params])[0]


@pytest.fixture(scope="module")
def omega_test(problem):
    """A fixed, moderate parameter vector (probabilities well away from 0 and 1)."""
    rng = np.random.default_rng(3)
    return np.concatenate([rng.uniform(0.2, 1.5, problem.n_params),
                           rng.uniform(0.1, 0.6, problem.n_max)])


@pytest.fixture(scope="module")
def posterior(problem):
    """BAYES-DOR draws on the synthetic problem (short chain: 100 warm-up + 100 draws)."""
    sampler = _fitted_sampler(problem)
    return sampler, sampler.run_hmc(num_samples=100, num_warmup=100, seed=0)


# --------------------------------------------------------------------------- observations

def test_gap_rows_are_level_centroid_differences(problem):
    X = problem.X
    by_pair = PreferenceSampler(X, [(0, 1, 2)], problem.n_params, problem.n_max)
    by_gap = PreferenceSampler(X, [], problem.n_params, problem.n_max)
    by_gap.add_gap([0], [1], 2)
    np.testing.assert_array_equal(by_pair.X_diff, by_gap.X_diff)
    np.testing.assert_array_equal(by_pair.etas, by_gap.etas)
    np.testing.assert_array_equal(by_pair.X_diff[0], X[0] - X[1])

    by_gap.add_gap([2, 3], [4, 5, 6], 0)   # set-valued levels (ties): eq. (8)
    assert by_gap.X_diff.shape == (2, problem.n_params)
    np.testing.assert_allclose(by_gap.X_diff[1], X[[2, 3]].mean(0) - X[[4, 5, 6]].mean(0))
    np.testing.assert_array_equal(by_gap.etas, [2, 0])


def test_invalid_card_counts_and_likelihoods_are_rejected(problem):
    sampler = PreferenceSampler(problem.X, [], problem.n_params, problem.n_max)
    with pytest.raises(ValueError):
        sampler.add_gap([0], [1], problem.n_max + 1)       # above the protocol cap e_max
    with pytest.raises(ValueError):
        sampler.add_preference(0, 1, -1)
    with pytest.raises(ValueError):
        sampler.log_predictive_loss(np.ones(problem.n_params + problem.n_max),
                                    problem.X[:1] - problem.X[1:2], [problem.n_max + 1])
    with pytest.raises((AssertionError, ValueError)):
        PreferenceSampler(problem.X, [], problem.n_params, problem.n_max, likelihood="cards")


# ----------------------------------------------------------------------------- likelihood

def _probability(sampler, omega, x, eta):
    """P(observed direction, eta | x, omega) through the public plug-in log-loss."""
    return float(np.exp(-sampler.log_predictive_loss(omega, x[None, :], [eta])))


def test_joint_likelihood_matches_eq_11(problem, omega_test):
    n_params, n_max = problem.n_params, problem.n_max
    sampler = PreferenceSampler(problem.X, [], n_params, n_max)
    w, delta_raw = omega_test[:n_params], omega_test[n_params:]
    theta = np.concatenate([[0.0], np.cumsum(RHO + delta_raw), [np.inf]])

    for a, b in [(0, 5), (3, 8), (9, 1)]:
        x = problem.X[a] - problem.X[b]
        z = x @ w
        expected = expit(z - theta[:-1]) - expit(z - theta[1:])     # e = 0, ..., n_max
        got = [_probability(sampler, omega_test, x, e) for e in range(n_max + 1)]
        np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-6)
        # the card categories of one direction add up to sigma(z) ...
        assert sum(got) == pytest.approx(expit(z), abs=1e-5)
        # ... and both directions together to one (paper, after eq. (11))
        both = sum(got) + sum(_probability(sampler, omega_test, -x, e) for e in range(n_max + 1))
        assert both == pytest.approx(1.0, abs=1e-5)


def test_direction_only_likelihood_ignores_the_cards(problem, omega_test):
    sampler = PreferenceSampler(problem.X, [], problem.n_params, problem.n_max,
                                likelihood="direction")
    x = problem.X[0] - problem.X[5]
    z = x @ omega_test[:problem.n_params]
    for e in range(problem.n_max + 1):
        assert _probability(sampler, omega_test, x, e) == pytest.approx(expit(z), rel=1e-5)


def test_log_loss_is_additive_and_mixes_probabilities_over_draws(problem, omega_test):
    sampler = PreferenceSampler(problem.X, [], problem.n_params, problem.n_max)
    X_new = np.array([problem.X[0] - problem.X[5], problem.X[3] - problem.X[8]])
    etas_new = [1, 3]
    singles = [sampler.log_predictive_loss(omega_test, X_new[i:i + 1], etas_new[i:i + 1])
               for i in range(2)]
    # composite likelihood (eqs. (12)-(13)): the batch loss is the sum over gaps
    assert sampler.log_predictive_loss(omega_test, X_new, etas_new) == pytest.approx(
        sum(singles), rel=1e-5)

    # 2-D state: posterior (or prior) predictive, -log of the MEAN probability over draws
    other = omega_test * 1.7
    state = np.stack([omega_test, other])
    p1 = _probability(sampler, omega_test, X_new[0], 1)
    p2 = _probability(sampler, other, X_new[0], 1)
    assert sampler.log_predictive_loss(state, X_new[:1], [1]) == pytest.approx(
        -np.log(0.5 * (p1 + p2)), rel=1e-5)

    assert sampler.log_predictive_loss(omega_test, np.empty((0, problem.n_params)), []) == 0.0
    assert np.isnan(sampler.log_predictive_loss(
        np.empty((0, problem.n_params + problem.n_max)), X_new, etas_new))


# --------------------------------------------------------------------------------- priors

def test_prior_draws_are_gamma_1_1(problem):
    sampler = PreferenceSampler(problem.X, [], problem.n_params, problem.n_max)
    draws = sampler.sample_prior(num_samples=4000, seed=0)
    assert draws.shape == (4000, problem.n_params + problem.n_max)
    assert draws.min() >= 0.0
    # Gamma(shape 1, scale 1) = Exp(1): mean 1 and variance 1
    assert draws.mean() == pytest.approx(1.0, abs=0.03)
    assert draws.var() == pytest.approx(1.0, abs=0.08)
    np.testing.assert_array_equal(draws, sampler.sample_prior(num_samples=4000, seed=0))


# ----------------------------------------------------------------------------- FTRL-DOR

def test_find_map_without_data_is_the_prior_mode(problem):
    """FTRL-DOR regularises with Gamma(2, 1), whose mode is 1: this is the state used for
    the prequential loss of the first session."""
    sampler = PreferenceSampler(problem.X, [], problem.n_params, problem.n_max)
    omega = sampler.find_map()
    assert omega.shape == (problem.n_params + problem.n_max,)
    np.testing.assert_allclose(omega, 1.0, atol=1e-3)


# Observed values: tau = 0.956 (joint) and 0.867 (direction only, which ignores the cards).
@pytest.mark.parametrize("likelihood, min_tau", [("joint", 0.8), ("direction", 0.7)])
def test_find_map_recovers_the_true_ranking(problem, likelihood, min_tau):
    sampler = _fitted_sampler(problem, likelihood)
    omega = sampler.find_map()
    assert omega.shape == (problem.n_params + problem.n_max,)
    assert np.all(np.isfinite(omega)) and omega.min() >= 1e-6 * (1 - 1e-6)   # box omega >= 1e-6
    assert _tau(problem, omega) >= min_tau
    np.testing.assert_array_equal(omega, sampler.find_map())   # deterministic


def test_find_map_fits_the_observed_gaps_better_than_the_prior_mode(problem):
    sampler = _fitted_sampler(problem)
    loss_map = sampler.log_predictive_loss(sampler.find_map(), sampler.X_diff, sampler.etas)
    loss_mode = sampler.log_predictive_loss(np.ones(problem.n_params + problem.n_max),
                                            sampler.X_diff, sampler.etas)
    assert np.isfinite(loss_map) and 0.0 < loss_map < loss_mode


# ---------------------------------------------------------------------------- BAYES-DOR

def test_run_hmc_returns_draws_on_the_support(problem, posterior):
    _, draws = posterior
    assert draws.shape == (100, problem.n_params + problem.n_max)
    assert np.all(np.isfinite(draws))
    assert draws.min() >= 0.0


def test_run_hmc_posterior_mean_ranks_like_the_truth(problem, posterior):
    _, draws = posterior
    assert _tau(problem, draws.mean(axis=0)) >= 0.7   # observed 0.911


def test_posterior_predicts_the_data_better_than_the_prior(problem, posterior):
    sampler, draws = posterior
    loss_post = sampler.log_predictive_loss(draws, sampler.X_diff, sampler.etas)
    loss_prior = sampler.log_predictive_loss(sampler.sample_prior(200, seed=0),
                                             sampler.X_diff, sampler.etas)
    assert np.isfinite(loss_post) and loss_post > 0.0
    assert loss_post < loss_prior


def test_run_hmc_is_reproducible_and_reports_diagnostics(problem, posterior):
    sampler, draws = posterior
    again, diag = sampler.run_hmc(num_samples=100, num_warmup=100, seed=0,
                                  return_diagnostics=True)
    np.testing.assert_array_equal(again, draws)
    assert set(diag) == {"divergences", "max_rhat"}
    assert isinstance(diag["divergences"], int) and diag["divergences"] >= 0
    assert np.isfinite(diag["max_rhat"])

    other_seed = sampler.run_hmc(num_samples=100, num_warmup=100, seed=1)
    assert not np.array_equal(other_seed, draws)


def test_run_hmc_without_data_returns_no_draws(problem):
    sampler = PreferenceSampler(problem.X, [], problem.n_params, problem.n_max)
    assert sampler.run_hmc().shape == (0, problem.n_params + problem.n_max)
    draws, diag = sampler.run_hmc(return_diagnostics=True)
    assert draws.shape == (0, problem.n_params + problem.n_max)
    assert diag["divergences"] == 0 and np.isnan(diag["max_rhat"])


def test_run_hmc_direction_only_likelihood(problem):
    sampler = _fitted_sampler(problem, likelihood="direction")
    draws = sampler.run_hmc(num_samples=100, num_warmup=100, seed=0)
    assert draws.shape == (100, problem.n_params + problem.n_max)
    assert np.all(np.isfinite(draws)) and draws.min() >= 0.0
    loss = sampler.log_predictive_loss(draws, sampler.X_diff, sampler.etas)
    assert np.isfinite(loss) and loss > 0.0
