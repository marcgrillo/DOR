"""Probability model and inference engine of B-DOR (paper Sec. 3).

Gap observations (Sec. 3.1).  A session, in which the DM sorts a subset of alternatives into
levels and places blank cards between consecutive levels, is turned by
``bdor.simulation.expand_subrank`` (``expansion='adjacent'``, the paper model) into one
observation per gap between adjacent levels: the feature vector

    x = mean_{a in upper level} v(a) - mean_{b in lower level} v(b)                eq. (8)

(v(a) = piecewise-linear features of ``bdor.features``, length N) and the card count
e in {0, ..., e_max}.  By linearity of U(a) = v(a)^T w (eq. (2)), the latent strength
z = x^T w is the difference between the mean values of the two levels.  In the code e_max is
``n_max`` and a card count is ``eta``.

Card model (Sec. 3.2).  Thresholds theta_0 = 0, theta_m = sum_{i <= m} Delta_i with
Delta_i >= rho > 0 (m = 1, ..., e_max) and theta_{e_max + 1} = +inf (eq. (10)).  The
probability that the DM confirms the direction of the gap and places e cards is

    P(>, e | x, phi) = sigma(z - theta_e) - sigma(z - theta_{e+1}),                eq. (11)

so the top category e = e_max is right-censored, sigma(z - theta_{e_max}).  The categories
sum to sigma(z); the remaining 1 - sigma(z) is the probability of the reversed direction.
The gaps of a session are combined by the adjacent-gap composite likelihood (eq. (12)), the
product of these terms; its negative log is the session log-loss (eq. (13)).

Parameter vector.  The states of BAYES-DOR and FTRL-DOR (and of their direction-only
variants) are

    omega = [w, Delta - rho],   shape (N + e_max,), or (S, N + e_max) for S draws,

with omega[:N] = w >= 0 the value-function weights and omega[N:] = Delta - rho >= 0 the
threshold increments above the floor.  The thresholds are rebuilt as
theta = cumsum(omega[N:] + rho), so that theta[m - 1] = theta_m.

Priors (eq. (14)), as implemented: independent Gamma distributions on all coordinates of
omega (hence a shifted Gamma on every Delta_m), with SHAPE a and SCALE b: ``gamma_a`` /
``gamma_b`` (a = b = 1) for BAYES-DOR, ``prior_a`` / ``prior_b`` in the MAP objectives
(a = 2, b = 1 for FTRL-DOR).  With b = 1 scale and rate coincide, which gives the unit rates
of Sec. 3.5.

Inference (``PreferenceSampler``):
    BAYES-DOR (Sec. 3.3)   ``run_hmc``: posterior draws of omega by NUTS (NumPyro).
    FTRL-DOR (Sec. 3.4)    ``find_map``: constrained MAP estimate (eq. (20)) by L-BFGS-B
                           (SciPy) with exact JAX gradients.
    direction-only         ``likelihood='direction'``: the ablation of Sec. 5.1 (arms
                           BAYESDIR / FTRLDIR, paper BAYESdir / FTRLdir); every gap
                           contributes sigma(z) and the card counts are ignored.
    prequential loss       ``log_predictive_loss`` (Sec. 5.7); ``sample_prior`` supplies the
                           prior predictive of the first session (BAYES arms).

All JAX computations use JAX's default float32 precision (x64 is not enabled).
"""
import numpy as np
from scipy.optimize import minimize
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.diagnostics import split_gelman_rubin

# Threshold-increment floor rho of eq. (10): Delta_m = RHO + omega[N + m - 1], omega >= 0.
# The theory only requires RHO > 0; the smallest numerically meaningful value is used.
RHO = 1e-6

def _jax_log_prob_per_obs_untraced(omega, X_diff, etas, n_params, n_max, rho):
    """Per-gap log-probability log P(>, eta | x, omega) of the card model, eq. (11).

    With z = x^T w and theta = cumsum(omega[n_params:] + rho) (theta[m - 1] = theta_m):

      eta = 0             log[sigma(z) - sigma(z - theta_1)]                band [0, theta_1)
      0 < eta < n_max     log[sigma(z - theta_eta) - sigma(z - theta_{eta+1})]
      eta = n_max         log sigma(z - theta_{n_max})                      right-censored

    The n_max + 1 categories sum to sigma(z), so each term scores both the declared order of
    the two levels and the card count.  Requires n_max >= 1.  The jitted version is also used
    by ``bdor.regret`` (hindsight comparator).

    Parameters
    ----------
    omega : array, shape (n_params + n_max,)
        Parameter vector [w, Delta - rho].
    X_diff : array, shape (n_obs, n_params)
        Gap feature vectors x.
    etas : int array, shape (n_obs,)
        Card counts.  A value outside [0, n_max] matches no category and contributes 0, so
        callers validate them (see ``PreferenceSampler``).
    n_params, n_max : int
        N and e_max (static arguments of the jitted version).
    rho : float
        Threshold-increment floor.

    Returns
    -------
    array, shape (n_obs,)
        log P(>, eta_i | x_i, omega) of every gap.
    """
    w = omega[:n_params]
    delta_raw = omega[n_params:]
    theta = jnp.cumsum(delta_raw + rho)

    u = jnp.dot(X_diff, w)

    def stable_log_sigmoid_diff(ub, lb, u_val):
        # log[sigma(u - lb) - sigma(u - ub)] for lb < ub.  With A = lb - u and B = ub - u it
        # equals -A + log(1 - exp(A - B)) - log(1 + e^-A) - log(1 + e^-B); 1e-15 guards log(0).
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
            term = stable_log_sigmoid_diff(theta[0], 0.0, u)
        elif k < n_max:
            term = stable_log_sigmoid_diff(theta[k], theta[k - 1], u)
        else:
            term = -jnp.logaddexp(0.0, -(u - theta[k - 1]))

        logp = logp + jnp.where(mask, term, 0.0)

    return logp

def _jax_log_likelihood_untraced(omega, X_diff, etas, n_params, n_max, rho):
    """Log of the composite likelihood (12) of all gaps: the sum of the eq. (11) terms."""
    return jnp.sum(_jax_log_prob_per_obs_untraced(omega, X_diff, etas, n_params, n_max, rho))

# Jitted versions; n_params and n_max are static (they fix the slicing and the loop length).
_jax_log_prob_per_obs = jax.jit(_jax_log_prob_per_obs_untraced, static_argnums=(3, 4))
_jax_log_likelihood = jax.jit(_jax_log_likelihood_untraced, static_argnums=(3, 4))

def _jax_dir_log_prob_per_obs_untraced(omega, X_diff, n_params):
    """Per-gap log-probability of the direction-only ablation: log sigma(x^T w).

    This is the ordinal-regression likelihood of Grillo et al. (2025), used by the BAYESDIR /
    FTRLDIR arms (paper BAYESdir / FTRLdir): only the declared direction of each gap is
    scored and the card counts are ignored.  omega keeps its threshold block, so states have
    the same shape as under the card model, but the data do not inform it: BAYESdir draws it
    from the prior and FTRLdir sets it to the prior mode.
    """
    w = omega[:n_params]
    u = jnp.dot(X_diff, w)
    return -jnp.logaddexp(0.0, -u)

_jax_dir_log_prob_per_obs = jax.jit(_jax_dir_log_prob_per_obs_untraced, static_argnums=(2,))

def _jax_dir_neg_log_posterior_untraced(omega, X_diff, n_params, prior_a, prior_b):
    """Negative log-posterior of the direction-only model, up to an additive constant.

    Prior: i.i.d. Gamma(shape prior_a, scale prior_b) on every coordinate of omega; the
    1e-12 only guards log(0).
    """
    ll = jnp.sum(_jax_dir_log_prob_per_obs_untraced(omega, X_diff, n_params))
    log_prior = jnp.sum((prior_a - 1.0) * jnp.log(omega + 1e-12) - omega / prior_b)
    return -(ll + log_prior)

_jax_dir_neg_log_posterior = jax.jit(_jax_dir_neg_log_posterior_untraced, static_argnums=(2,))

def _jax_neg_log_posterior_untraced(omega, X_diff, etas, n_params, n_max, rho, prior_a, prior_b):
    """Negative log-posterior of the card model up to a constant (the FTRL-DOR objective (20)).

    Prior: i.i.d. Gamma(shape prior_a, scale prior_b) on every coordinate of omega; the
    1e-12 only guards log(0).
    """
    ll = _jax_log_likelihood_untraced(omega, X_diff, etas, n_params, n_max, rho)
    log_prior = jnp.sum((prior_a - 1.0) * jnp.log(omega + 1e-12) - omega / prior_b)
    return -(ll + log_prior)

_jax_neg_log_posterior = jax.jit(_jax_neg_log_posterior_untraced, static_argnums=(3, 4))


class PreferenceSampler:
    """Gap-observation store and inference engine of B-DOR.

    B-DOR is the Bayesian Deck-of-cards-based Ordinal Regression of the paper.  Gaps are
    added with ``add_gap`` (set-valued levels) or ``add_preference`` (two alternatives);
    ``run_hmc`` (BAYES-DOR) and ``find_map`` (FTRL-DOR) condition on all gaps added so far.

    Parameters
    ----------
    feature_matrix : ndarray, shape (n_alternatives, n_params)
        Features v(a) of every alternative, e.g. from ``PiecewiseLinearTransformer.transform``.
    preferences : iterable of (up_idx, down_idx, eta)
        Initial alternative-level gaps, passed to ``add_preference`` (the callers in this
        repository pass ``[]``).
    n_params : int
        Number N of value-function weights (columns of ``feature_matrix``).
    n_max : int
        Card cap e_max: card counts lie in [0, n_max] and omega has n_max threshold
        coordinates.  Must be >= 1 with ``likelihood='joint'``.
    rho : float, default RHO
        Threshold-increment floor.
    likelihood : {'joint', 'direction'}
        'joint': the card model of eqs. (11)-(12) (paper model); 'direction': the
        direction-only ablation (BAYESdir / FTRLdir).

    Attributes
    ----------
    X_diff : ndarray, shape (n_obs, n_params)
        Gap feature vectors x, in insertion order.
    etas : ndarray of int, shape (n_obs,)
        Their card counts.
    jax_X_diff, jax_etas : jax arrays
        The same as JAX arrays (float32 / int32), rebuilt after every added gap.
    prefs : ndarray of int, shape (n_pairs, 3)
        (up_idx, down_idx, eta) of the gaps added with ``add_preference`` (introspection only).
    gamma_a, gamma_b : float
        Shape and scale of the BAYES-DOR prior used by ``run_hmc`` and ``sample_prior``.
    """
    def __init__(self, feature_matrix, preferences, n_params, n_max, rho=RHO,
                 likelihood='joint'):
        self.X = feature_matrix
        self.n_params = n_params
        self.n_max = n_max
        self.rho = rho
        # 'joint' = card model of eqs. (11)-(12) (paper model);
        # 'direction' = direction-only ablation (card counts ignored).
        assert likelihood in ('joint', 'direction')
        self.likelihood = likelihood

        # BAYES-DOR prior (run_hmc, sample_prior): Gamma with shape gamma_a and SCALE gamma_b.
        self.gamma_a = 1.0
        self.gamma_b = 1.0

        # Gap observations are stored as explicit feature-difference rows so
        # that set-valued DOR levels (centroid features) and plain alternative
        # pairs share one code path.  self.prefs keeps the index pairs for the
        # alternative-level observations (introspection only).
        self._xdiff_rows = []
        self._eta_list = []
        self.prefs = np.empty((0, 3), dtype=int)
        self._update_diff_vectors()
        for up_idx, down_idx, eta in preferences:
            self.add_preference(up_idx, down_idx, eta)

    def _validate_eta(self, eta, what):
        """Raise ValueError unless 0 <= eta <= n_max (a raw card count within the cap)."""
        if not (0 <= eta <= self.n_max):
            raise ValueError(
                f"eta={eta} outside [0, {self.n_max}] for {what}; "
                "responses must be raw card counts within the protocol cap"
            )

    def add_preference(self, up_idx, down_idx, eta):
        """Add the gap between two single alternatives, x = v(up) - v(down).

        Parameters
        ----------
        up_idx, down_idx : int
            Row indices (in ``feature_matrix``) of the preferred and of the other alternative.
        eta : int
            Number of blank cards between them, in [0, n_max] (ValueError otherwise).
            The triple is also appended to ``prefs``.
        """
        self._validate_eta(eta, f"pair ({up_idx}, {down_idx})")
        new_pref = np.array([[up_idx, down_idx, eta]], dtype=int)
        self.prefs = np.vstack([self.prefs, new_pref]) if self.prefs.size > 0 else new_pref
        self._xdiff_rows.append(self.X[int(up_idx)] - self.X[int(down_idx)])
        self._eta_list.append(int(eta))
        self._update_diff_vectors()

    def add_gap(self, upper_members, lower_members, eta):
        """Add the gap between two adjacent levels of a declared preorder (eq. (8)).

        The feature vector is the difference of the level centroids,

            x = mean_{a in upper} v(a) - mean_{b in lower} v(b),

        so by linearity x^T w is the mean-utility gap between the two sets.  Singleton levels
        give the same row as ``add_preference`` (but are not recorded in ``prefs``).

        Parameters
        ----------
        upper_members, lower_members : iterable of int
            Row indices of the alternatives in the upper (preferred) and in the lower level.
        eta : int
            Number of blank cards between the two levels, in [0, n_max] (ValueError otherwise).
        """
        upper = [int(i) for i in upper_members]
        lower = [int(i) for i in lower_members]
        self._validate_eta(eta, f"gap ({upper}, {lower})")
        self._xdiff_rows.append(self.X[upper].mean(axis=0) - self.X[lower].mean(axis=0))
        self._eta_list.append(int(eta))
        self._update_diff_vectors()

    def _update_diff_vectors(self):
        """Rebuild X_diff / etas and their JAX copies from the stored observations."""
        if not self._xdiff_rows:
            self.X_diff = np.empty((0, self.n_params))
            self.etas = np.empty(0, dtype=int)
        else:
            self.X_diff = np.array(self._xdiff_rows)
            self.etas = np.array(self._eta_list, dtype=int)

        # Convert to JAX arrays to avoid overhead during iterations
        self.jax_X_diff = jnp.array(self.X_diff)
        self.jax_etas = jnp.array(self.etas)

    def sample_prior(self, num_samples=200, seed=0):
        """Draws of omega from the BAYES-DOR prior: i.i.d. Gamma(shape gamma_a, scale gamma_b).

        ``bdor.simulation`` uses 200 such draws as the prior predictive state that scores the
        first session of the BAYES / BAYESDIR arms in the prequential log-loss.

        Parameters
        ----------
        num_samples : int
            Number of draws.
        seed : int
            Seed of ``numpy.random.default_rng`` (NumPy, not JAX).

        Returns
        -------
        ndarray, shape (num_samples, n_params + n_max)
        """
        rng = np.random.default_rng(seed)
        return rng.gamma(self.gamma_a, self.gamma_b, size=(num_samples, self.n_params + self.n_max))

    def log_predictive_loss(self, state, X_diff_new, etas_new):
        """Prequential log-loss of a new batch of gaps under a given state (paper Sec. 5.7).

        For a prequential evaluation pass the state inferred before the batch was observed.
        The per-gap probabilities follow this sampler's ``likelihood`` (eq. (11), or sigma(z)
        for 'direction'); the gaps stored in the sampler are not used.

        Parameters
        ----------
        state : array, shape (D,) or (S, D), D = n_params + n_max
            1-D: a point estimate (FTRL-DOR); returns the plug-in loss -sum_i log P_i(omega).
            2-D: S draws (posterior, or prior from ``sample_prior``); every gap's probability
            is averaged over the draws separately, -sum_i log(mean_s P_i(omega_s)) (computed
            with logsumexp), rather than averaging the product over the batch.
        X_diff_new : array, shape (n_new, n_params)
            Feature vectors of the new gaps.
        etas_new : int array, shape (n_new,)
            Their card counts, in [0, n_max] (ValueError otherwise; checked also under
            ``likelihood='direction'``, which ignores them).

        Returns
        -------
        float
            The loss in nats: 0.0 for an empty batch, nan for a 2-D state without draws.
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
        if self.likelihood == 'direction':
            per_obs = lambda om: _jax_dir_log_prob_per_obs(om, X, self.n_params)
        else:
            per_obs = lambda om: _jax_log_prob_per_obs(om, X, e, self.n_params, self.n_max,
                                                       self.rho)
        if state.ndim == 1:
            return float(-jnp.sum(per_obs(jnp.asarray(state))))
        if state.shape[0] == 0:
            return float('nan')
        lps = jax.vmap(per_obs)(jnp.asarray(state))
        pred = jax.scipy.special.logsumexp(lps, axis=0) - jnp.log(state.shape[0])
        return float(-jnp.sum(pred))

    def run_hmc(self, num_samples=200, num_warmup=200, seed=0, return_diagnostics=False):
        """BAYES-DOR: posterior draws of omega given all gaps added so far (Sec. 3.3).

        One NUTS chain (NumPyro) with the prior ``gamma_a`` / ``gamma_b`` and the likelihood
        selected by ``likelihood``.

        Parameters
        ----------
        num_samples : int
            Retained draws (200 in the simulation studies, 2000 in the case study).
        num_warmup : int
            Warm-up (adaptation) iterations (200 in the simulations, 1000 in the case study).
        seed : int
            Seed of ``jax.random.PRNGKey``.
        return_diagnostics : bool
            Also return the sampler diagnostics.

        Returns
        -------
        samples : ndarray, shape (num_samples, n_params + n_max)
            Posterior draws of omega (float32); an empty (0, n_params + n_max) array when no
            gap has been added.
        diagnostics : dict, only if ``return_diagnostics``
            'divergences': number of divergent transitions among the retained draws;
            'max_rhat': largest split-Rhat over the coordinates of omega (the single chain is
            split in halves).  {'divergences': 0, 'max_rhat': nan} when no gap has been added.
        """
        if len(self._eta_list) == 0:
            empty = np.empty((0, self.n_params + self.n_max))
            return (empty, {'divergences': 0, 'max_rhat': np.nan}) if return_diagnostics else empty

        # A NumPyro model (rather than a raw potential_fn) lets NUTS handle the (0, inf)
        # support through the automatic unconstraining (log) bijection.  NumPyro's Gamma takes
        # (concentration, rate), hence rate = 1 / gamma_b.
        def model():
            omega = numpyro.sample(
                'omega',
                dist.Gamma(self.gamma_a, 1.0 / self.gamma_b).expand([self.n_params + self.n_max])
            )
            if self.likelihood == 'direction':
                ll = jnp.sum(_jax_dir_log_prob_per_obs(omega, self.jax_X_diff, self.n_params))
            else:
                ll = _jax_log_likelihood(omega, self.jax_X_diff, self.jax_etas, self.n_params,
                                         self.n_max, self.rho)
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
        """FTRL-DOR: constrained MAP estimate of omega given all gaps added so far (eq. (20)).

        Minimises the negative log-likelihood (of the selected ``likelihood``) plus the negative
        log of an i.i.d. Gamma(shape 2, scale 1) prior (Sec. 3.5; hard-coded here, ``gamma_a``
        / ``gamma_b`` are not used) by SciPy L-BFGS-B with exact JAX gradients, over the box
        omega >= 1e-6 and starting from omega = 0.5.  The l1 radius M of eq. (18) is not
        imposed.  With no gaps the minimiser is the prior mode, omega = 1 (up to the optimiser
        tolerance); ``bdor.simulation`` uses it to score the first session of the FTRL arms.

        Returns
        -------
        ndarray, shape (n_params + n_max,), float64
            The estimate ``res.x`` (the optimiser's success flag is not checked).
        """
        ndim = self.n_params + self.n_max

        # FTRL-DOR prior: Gamma shape alpha = 2, scale beta = 1 on every coordinate.
        alpha = 2.0
        beta = 1.0

        # SciPy needs the value as a Python float and the gradient as a float64 array.
        def obj_and_grad(omega):
            if self.likelihood == 'direction':
                val, grad = jax.value_and_grad(_jax_dir_neg_log_posterior)(
                    jnp.array(omega), self.jax_X_diff, self.n_params, alpha, beta
                )
            else:
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
            return float(val), np.array(grad, dtype=np.float64)

        bounds = [(1e-6, None) for _ in range(ndim)]
        x0 = np.ones(ndim) * 0.5

        res = minimize(obj_and_grad, x0, method='L-BFGS-B', bounds=bounds, jac=True)
        return res.x
