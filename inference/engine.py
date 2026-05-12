import numpy as np
import dynesty
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import gamma

def robust_sigmoid(x):
    return expit(x)

class PreferenceSampler:
    """
    Bayesian and FTRL Inference Engine for Difference of Responses (DOR) model.
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

    def _stable_log_sigmoid_diff(self, ub, lb, u):
        """
        ln(sigma(ub-u) - sigma(lb-u))
        """
        A = lb - u
        B = ub - u
        def softplus(x): return np.logaddexp(0, x)
        def logdiffexp(x, y): return x + np.log(1 - np.exp(y - x) + 1e-15)
        return logdiffexp(-A, -B) - softplus(-A) - softplus(-B)

    def log_likelihood_dor(self, omega):
        """
        Log-likelihood for the DOR model.
        omega = [w_1, ..., w_N, Delta_1, ..., Delta_{n_max}]
        """
        if self.prefs.size == 0:
            return 0.0

        w = omega[:self.n_params]
        delta = omega[self.n_params:]
        theta = np.cumsum(delta)
        
        u = self.X_diff @ w
        ll = 0.0
        
        for k in range(self.n_max + 1):
            idx = (self.etas == k)
            if not np.any(idx): continue
            
            u_k = u[idx]
            if k < self.n_max:
                lb_val = 0.0 if k == 0 else theta[k-1]
                ub_val = theta[k]
                ll += np.sum(self._stable_log_sigmoid_diff(ub_val, lb_val, u_k))
            else:
                lb_val = theta[k-1]
                ll += np.sum(-np.logaddexp(0, -(u_k - lb_val)))
                
        return ll

    def ptform_gamma(self, u_vec):
        return gamma.ppf(u_vec, a=self.gamma_a, scale=self.gamma_b)

    def run_nested(self, nlive=500):
        ndim = self.n_params + self.n_max
        sampler = dynesty.NestedSampler(
            loglikelihood=self.log_likelihood_dor,
            prior_transform=self.ptform_gamma,
            ndim=ndim,
            bound='multi',
            sample='rwalk',
            nlive=max(ndim * 3, nlive)
        )
        sampler.run_nested(print_progress=False, dlogz=0.5)
        return sampler.results.samples_equal()

    def find_map(self):
        """MAP estimation using L-BFGS-B"""
        ndim = self.n_params + self.n_max
        
        def neg_reg_ll(omega):
            ll = self.log_likelihood_dor(omega)
            # Log-prior: (alpha-1)*log(x) - x/beta (with beta=1, alpha=2 for FTRL)
            # Let's use alpha=2, beta=1 for FTRL MAP regularization
            alpha = 2.0
            beta = 1.0
            log_prior = np.sum((alpha - 1) * np.log(omega + 1e-12) - omega / beta)
            return -(ll + log_prior)

        bounds = [(1e-6, None) for _ in range(ndim)]
        x0 = np.ones(ndim) * 0.5
        
        res = minimize(neg_reg_ll, x0, method='L-BFGS-B', bounds=bounds)
        return res.x