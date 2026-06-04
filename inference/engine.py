import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

def robust_sigmoid(x):
    return expit(x)

def _jax_log_likelihood_untraced(omega, X_diff, etas, n_params, n_max):
    w = omega[:n_params]
    delta = omega[n_params:]
    theta = jnp.cumsum(delta)
    
    u = jnp.dot(X_diff, w)
    
    def stable_log_sigmoid_diff(ub, lb, u_val):
        A = lb - u_val
        B = ub - u_val
        softplus_A = jnp.logaddexp(0.0, -A)
        softplus_B = jnp.logaddexp(0.0, -B)
        logdiffexp = -A + jnp.log(1.0 - jnp.exp(-B - (-A)) + 1e-15)
        return logdiffexp - softplus_A - softplus_B

    ll = 0.0
    for k in range(n_max + 1):
        mask = (etas == k)
        
        # We handle k=0 and k>0
        lb_val = jnp.where(k == 0, 0.0, theta[k-1] if k > 0 else 0.0)
            
        if k < n_max:
            ub_val = theta[k]
            term = stable_log_sigmoid_diff(ub_val, lb_val, u)
        else:
            term = -jnp.logaddexp(0.0, -(u - lb_val))
            
        ll += jnp.sum(jnp.where(mask, term, 0.0))

    return ll

_jax_log_likelihood = jax.jit(_jax_log_likelihood_untraced, static_argnums=(3, 4))

def _jax_neg_log_posterior_untraced(omega, X_diff, etas, n_params, n_max, prior_a, prior_b):
    ll = _jax_log_likelihood_untraced(omega, X_diff, etas, n_params, n_max)
    log_prior = jnp.sum((prior_a - 1.0) * jnp.log(omega + 1e-12) - omega / prior_b)
    return -(ll + log_prior)

_jax_neg_log_posterior = jax.jit(_jax_neg_log_posterior_untraced, static_argnums=(3, 4))


class PreferenceSampler:
    """
    Bayesian (HMC) and FTRL Inference Engine for Difference of Responses (DOR) model.
    """
    def __init__(self, feature_matrix, preferences, n_params, n_max):
        self.X = feature_matrix
        self.prefs = np.array(preferences, dtype=int) if len(preferences) > 0 else np.empty((0, 3), dtype=int)
        self.n_params = n_params
        self.n_max = n_max
        
        # Gamma Prior Params
        self.gamma_a = 1.0 
        self.gamma_b = 1.0

        self._update_diff_vectors()

    def add_preference(self, up_idx, down_idx, eta):
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

    def run_hmc(self, num_samples=200, num_warmup=200):
        if self.prefs.size == 0:
            return np.empty((0, self.n_params + self.n_max))

        # We define a proper NumPyro model to automatically handle the (0, inf) boundary
        # via unconstraining bijections instead of raw potential_fn.
        def model():
            omega = numpyro.sample(
                'omega', 
                dist.Gamma(self.gamma_a, 1.0 / self.gamma_b).expand([self.n_params + self.n_max])
            )
            ll = _jax_log_likelihood(omega, self.jax_X_diff, self.jax_etas, self.n_params, self.n_max)
            numpyro.factor('obs', ll)

        nuts_kernel = NUTS(model)
        mcmc = MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples, progress_bar=False)
        
        mcmc.run(jax.random.PRNGKey(0))
        return np.array(mcmc.get_samples()['omega'])

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
                alpha, 
                beta
            )
            # return as float and float64 numpy array for scipy minimize
            return float(val), np.array(grad, dtype=np.float64)

        bounds = [(1e-6, None) for _ in range(ndim)]
        x0 = np.ones(ndim) * 0.5
        
        res = minimize(obj_and_grad, x0, method='L-BFGS-B', bounds=bounds, jac=True)
        return res.x