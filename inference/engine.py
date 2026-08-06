import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.diagnostics import split_gelman_rubin

# Threshold-increment floor (Delta_m = RHO + delta_raw_m, delta_raw_m >= 0).
# Theory only requires RHO > 0; smallest numerically meaningful value chosen.
RHO = 1e-6

def robust_sigmoid(x):
    return expit(x)

def _jax_log_prob_per_obs_untraced(omega, X_diff, etas, n_params, n_max, rho):
    """
    Per-observation log P(eta | x, omega) under the conditional cumulative-link
    model with theta_0 = -inf and theta_k = sum_{m<=k} (rho + delta_raw_m):
      P(eta=0)          = sigma(theta_1 - u)
      P(eta=k), 0<k<max = sigma(u - theta_k) - sigma(u - theta_{k+1})
      P(eta=n_max)      = sigma(u - theta_{n_max})
    Direction is conditioned on, not modelled (adjacent-v1 likelihood).
    """
    w = omega[:n_params]
    delta_raw = omega[n_params:]
    theta = jnp.cumsum(delta_raw + rho)

    u = jnp.dot(X_diff, w)

    def stable_log_sigmoid_diff(ub, lb, u_val):
        A = lb - u_val
        B = ub - u_val
        softplus_A = jnp.logaddexp(0.0, -A)
        softplus_B = jnp.logaddexp(0.0, -B)
        logdiffexp = -A + jnp.log(1.0 - jnp.exp(-B - (-A)) + 1e-15)
        return logdiffexp - softplus_A - softplus_B

    logp = jnp.zeros_like(u)
    for k in range(n_max + 1):
        mask = (etas == k)

        if k == 0:
            term = -jnp.logaddexp(0.0, u - theta[0])
        elif k < n_max:
            term = stable_log_sigmoid_diff(theta[k], theta[k - 1], u)
        else:
            term = -jnp.logaddexp(0.0, -(u - theta[k - 1]))

        logp = logp + jnp.where(mask, term, 0.0)

    return logp

def _jax_log_likelihood_untraced(omega, X_diff, etas, n_params, n_max, rho):
    return jnp.sum(_jax_log_prob_per_obs_untraced(omega, X_diff, etas, n_params, n_max, rho))

_jax_log_prob_per_obs = jax.jit(_jax_log_prob_per_obs_untraced, static_argnums=(3, 4))
_jax_log_likelihood = jax.jit(_jax_log_likelihood_untraced, static_argnums=(3, 4))

def _jax_neg_log_posterior_untraced(omega, X_diff, etas, n_params, n_max, rho, prior_a, prior_b):
    ll = _jax_log_likelihood_untraced(omega, X_diff, etas, n_params, n_max, rho)
    log_prior = jnp.sum((prior_a - 1.0) * jnp.log(omega + 1e-12) - omega / prior_b)
    return -(ll + log_prior)

_jax_neg_log_posterior = jax.jit(_jax_neg_log_posterior_untraced, static_argnums=(3, 4))


class PreferenceSampler:
    """
    Bayesian (HMC) and FTRL Inference Engine for Difference of Responses (DOR) model.
    """
    def __init__(self, feature_matrix, preferences, n_params, n_max, rho=RHO):
        self.X = feature_matrix
        self.n_params = n_params
        self.n_max = n_max
        self.rho = rho

        # Gamma Prior Params
        self.gamma_a = 1.0
        self.gamma_b = 1.0

        self.prefs = np.empty((0, 3), dtype=int)
        self._update_diff_vectors()
        for up_idx, down_idx, eta in preferences:
            self.add_preference(up_idx, down_idx, eta)

    def add_preference(self, up_idx, down_idx, eta):
        if not (0 <= eta <= self.n_max):
            raise ValueError(
                f"eta={eta} outside [0, {self.n_max}] for pair ({up_idx}, {down_idx}); "
                "responses must be raw card counts within the protocol cap"
            )
        new_pref = np.array([[up_idx, down_idx, eta]], dtype=int)
        self.prefs = np.vstack([self.prefs, new_pref]) if self.prefs.size > 0 else new_pref
        self._update_diff_vectors()

    def _update_diff_vectors(self):
        if self.prefs.size == 0:
            self.X_diff = np.empty((0, self.n_params))
            self.etas = np.empty(0, dtype=int)
        else:
            self.X_diff = self.X[self.prefs[:, 0]] - self.X[self.prefs[:, 1]]
            self.etas = self.prefs[:, 2]

        # Convert to JAX arrays to avoid overhead during iterations
        self.jax_X_diff = jnp.array(self.X_diff)
        self.jax_etas = jnp.array(self.etas)

    def sample_prior(self, num_samples=200, seed=0):
        """Draws of the raw parameter vector from the Gamma prior (prior predictive at t=1)."""
        rng = np.random.default_rng(seed)
        return rng.gamma(self.gamma_a, self.gamma_b, size=(num_samples, self.n_params + self.n_max))

    def log_predictive_loss(self, state, X_diff_new, etas_new):
        """
        Negative log predictive probability of a new batch of (x, eta) observations.
        state 1-D  -> plug-in (FTRL / MAP) loss;
        state 2-D  -> posterior/prior predictive: -sum_i log( mean_s P(eta_i | x_i, omega_s) ).
        """
        e_np = np.asarray(etas_new)
        if e_np.size and (e_np.min() < 0 or e_np.max() > self.n_max):
            raise ValueError(
                f"etas outside [0, {self.n_max}] in log_predictive_loss; out-of-range "
                "observations would silently contribute zero loss (probability 1)"
            )
        X = jnp.asarray(X_diff_new)
        e = jnp.asarray(etas_new)
        if X.shape[0] == 0:
            return 0.0
        state = np.asarray(state)
        if state.ndim == 1:
            lp = _jax_log_prob_per_obs(jnp.asarray(state), X, e, self.n_params, self.n_max, self.rho)
            return float(-jnp.sum(lp))
        if state.shape[0] == 0:
            return float('nan')
        lps = jax.vmap(
            lambda om: _jax_log_prob_per_obs(om, X, e, self.n_params, self.n_max, self.rho)
        )(jnp.asarray(state))
        pred = jax.scipy.special.logsumexp(lps, axis=0) - jnp.log(state.shape[0])
        return float(-jnp.sum(pred))

    def run_hmc(self, num_samples=200, num_warmup=200, seed=0, return_diagnostics=False):
        if self.prefs.size == 0:
            empty = np.empty((0, self.n_params + self.n_max))
            return (empty, {'divergences': 0, 'max_rhat': np.nan}) if return_diagnostics else empty

        # We define a proper NumPyro model to automatically handle the (0, inf) boundary
        # via unconstraining bijections instead of raw potential_fn.
        def model():
            omega = numpyro.sample(
                'omega',
                dist.Gamma(self.gamma_a, 1.0 / self.gamma_b).expand([self.n_params + self.n_max])
            )
            ll = _jax_log_likelihood(omega, self.jax_X_diff, self.jax_etas, self.n_params, self.n_max, self.rho)
            numpyro.factor('obs', ll)

        nuts_kernel = NUTS(model)
        mcmc = MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples, progress_bar=False)

        mcmc.run(jax.random.PRNGKey(seed), extra_fields=('diverging',))
        samples = np.array(mcmc.get_samples()['omega'])
        if not return_diagnostics:
            return samples
        divergences = int(np.sum(np.array(mcmc.get_extra_fields()['diverging'])))
        max_rhat = float(np.max(split_gelman_rubin(samples[None, ...])))
        return samples, {'divergences': divergences, 'max_rhat': max_rhat}

    def find_map(self):
        """MAP estimation using L-BFGS-B with JAX exact gradients"""
        ndim = self.n_params + self.n_max

        # FTRL uses alpha=2, beta=1
        alpha = 2.0
        beta = 1.0

        # We need a function that returns value and gradient as numpy arrays for scipy.optimize
        def obj_and_grad(omega):
            val, grad = jax.value_and_grad(_jax_neg_log_posterior)(
                jnp.array(omega),
                self.jax_X_diff,
                self.jax_etas,
                self.n_params,
                self.n_max,
                self.rho,
                alpha,
                beta
            )
            # return as float and float64 numpy array for scipy minimize
            return float(val), np.array(grad, dtype=np.float64)

        bounds = [(1e-6, None) for _ in range(ndim)]
        x0 = np.ones(ndim) * 0.5

        res = minimize(obj_and_grad, x0, method='L-BFGS-B', bounds=bounds, jac=True)
        return res.x
