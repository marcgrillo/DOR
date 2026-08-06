"""
SMAA-DOR baseline (Corrente, Greco, Zappala - arXiv:2405.04313, Sec. 3.2).

Samples value functions compatible with the DM's Deck-of-Cards preference
information via Hit-And-Run over the polytope E^DM'':

    E^Model:   U(a) = w . x(a),  w >= 0,  sum(w) = 1        (additive PWL model)
    ordering:  U(a) >= U(b)                    for each elicited pair a > b
    cards:     U(a) - U(b) - s+ + s- = k * eta for each pair with eta blank cards
    k >= eps,  s+ >= 0,  s- >= 0
    sum(s+ + s-) <= sigma_bar + eta_budget     (paper's admitted deterioration)

where sigma_bar is the minimal total deviation (LP problem (1) of the paper)
and k is maximized as in LP problem (3) to fix eps and the k range.

Notes on the adaptation to this benchmark:
- Preference information arrives as pairwise triples (a, b, eta) expanded from
  subranks, so the proportionality constraint is written on differences
  (the paper's interval-scale form, eqs. (4)-(5)) with nu(a) - nu(b) = eta.
  The dataset generator produces cards as round(scaled utility difference),
  hence eta (not eta + 1) is the difference in nu-units.
- With inconsistent (cyclic) preference information the hard ordering
  constraints U(a) >= U(b) can make the polytope empty; in that case they are
  dropped and only the slack-based card constraints are kept (soft mode).
"""

import numpy as np
from scipy.optimize import linprog
from scipy.linalg import null_space

_K_CAP = 1.0          # k upper bound when no pair carries cards (k unconstrained)
_K_MIN = 1e-4         # minimal acceptable k* before growing the error budget
_INTERIOR_TOL = 1e-7  # minimal Chebyshev-like radius to accept a starting point


def _solve_lp(c, A_ub, b_ub, A_eq, b_eq, bounds):
    return linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                   bounds=bounds, method="highs")


class SMAADORSampler:
    def __init__(self, X_diff, etas, n_params, seed=0):
        self.D = np.atleast_2d(np.asarray(X_diff, dtype=float))
        self.etas = np.asarray(etas, dtype=float).ravel()
        self.p = int(n_params)
        self.m = self.D.shape[0]
        self.rng = np.random.default_rng(seed)
        self.info = {}

    # ------------------------------------------------------------------
    # Constraint construction (variables x = [w (p), k, s+ (m), s- (m)])
    # ------------------------------------------------------------------

    def _equalities(self):
        p, m = self.p, self.m
        n = p + 1 + 2 * m
        A = np.zeros((1 + m, n))
        b = np.zeros(1 + m)
        A[0, :p] = 1.0
        b[0] = 1.0
        for t in range(m):
            A[1 + t, :p] = self.D[t]
            A[1 + t, p] = -self.etas[t]
            A[1 + t, p + 1 + t] = -1.0
            A[1 + t, p + 1 + m + t] = 1.0
        return A, b

    def _ordering_rows(self):
        """-d_t . w <= 0  (U(a) >= U(b))."""
        p, m = self.p, self.m
        n = p + 1 + 2 * m
        G = np.zeros((m, n))
        G[:, :p] = -self.D
        return G, np.zeros(m)

    def _phase1(self, hard_ordering):
        """sigma_bar = min sum(s+ + s-)."""
        p, m = self.p, self.m
        c = np.concatenate([np.zeros(p + 1), np.ones(2 * m)])
        A_eq, b_eq = self._equalities()
        A_ub, b_ub = (self._ordering_rows() if hard_ordering else (None, None))
        bounds = [(0, None)] * p + [(0, _K_CAP)] + [(0, None)] * (2 * m)
        return _solve_lp(c, A_ub, b_ub, A_eq, b_eq, bounds)

    def _phase2(self, hard_ordering, budget):
        """k* = max k subject to the budget on total deviation."""
        p, m = self.p, self.m
        n = p + 1 + 2 * m
        c = np.zeros(n)
        c[p] = -1.0
        A_eq, b_eq = self._equalities()
        rows, rhs = [], []
        if hard_ordering:
            Go, ho = self._ordering_rows()
            rows.append(Go)
            rhs.append(ho)
        brow = np.zeros(n)
        brow[p + 1:] = 1.0
        rows.append(brow[None, :])
        rhs.append(np.array([budget]))
        A_ub = np.vstack(rows)
        b_ub = np.concatenate(rhs)
        bounds = [(0, None)] * p + [(0, _K_CAP)] + [(0, None)] * (2 * m)
        return _solve_lp(c, A_ub, b_ub, A_eq, b_eq, bounds)

    def _full_inequalities(self, hard_ordering, budget, eps, k_ub):
        """All inequalities as G x <= h (variable bounds included as rows)."""
        p, m = self.p, self.m
        n = p + 1 + 2 * m
        rows, rhs = [], []
        eye = np.eye(n)
        # w >= 0, s+ >= 0, s- >= 0
        nonneg = list(range(p)) + list(range(p + 1, n))
        rows.append(-eye[nonneg])
        rhs.append(np.zeros(len(nonneg)))
        # eps <= k <= k_ub
        rows.append(-eye[[p]])
        rhs.append(np.array([-eps]))
        rows.append(eye[[p]])
        rhs.append(np.array([k_ub]))
        if hard_ordering:
            Go, ho = self._ordering_rows()
            rows.append(Go)
            rhs.append(ho)
        brow = np.zeros(n)
        brow[p + 1:] = 1.0
        rows.append(brow[None, :])
        rhs.append(np.array([budget]))
        return np.vstack(rows), np.concatenate(rhs)

    def _interior_point(self, G, h, A_eq, b_eq):
        """max delta s.t. G x + delta * ||row|| <= h, A_eq x = b_eq."""
        n = G.shape[1]
        norms = np.linalg.norm(G, axis=1)
        norms[norms == 0] = 1.0
        c = np.zeros(n + 1)
        c[-1] = -1.0
        A_ub = np.hstack([G, norms[:, None]])
        A_eq_x = np.hstack([A_eq, np.zeros((A_eq.shape[0], 1))])
        bounds = [(None, None)] * n + [(0, 10.0)]
        res = _solve_lp(c, A_ub, h, A_eq_x, b_eq, bounds)
        if not res.success:
            return None, 0.0
        return res.x[:n], res.x[-1]

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def sample(self, num_samples=200, burn_in=None, thin=None):
        p = self.p
        if self.m == 0:
            # No information yet: uniform Dirichlet over the weight simplex.
            w = self.rng.dirichlet(np.ones(p), size=num_samples)
            return np.hstack([w, np.zeros((num_samples, 1))])

        hard = True
        res1 = self._phase1(hard)
        if not res1.success:
            hard = False
            res1 = self._phase1(hard)
            if not res1.success:
                raise RuntimeError(f"SMAA-DOR phase-1 LP failed: {res1.message}")
        sigma_bar = max(res1.fun, 0.0)

        # Grow the admitted deterioration until k* > 0 (paper Sec. 2) and the
        # polytope has an interior to sample from.
        step = max(0.01, 0.05 * sigma_bar)
        x0 = None
        for trial in range(60):
            budget = sigma_bar + step * (trial + 1)
            res2 = self._phase2(hard, budget)
            if not res2.success:
                continue
            k_star = res2.x[p]
            if k_star < _K_MIN:
                continue
            eps = max(1e-6, 0.01 * k_star)
            k_ub = min(_K_CAP, k_star)
            G, h = self._full_inequalities(hard, budget, eps, k_ub)
            A_eq, b_eq = self._equalities()
            x0, radius = self._interior_point(G, h, A_eq, b_eq)
            if x0 is not None and radius > _INTERIOR_TOL:
                break
            x0 = None
        else:
            pass

        if x0 is None and hard:
            # Cyclic information can make the hard-ordering polytope degenerate.
            hard = False
            res1 = self._phase1(hard)
            sigma_bar = max(res1.fun, 0.0)
            step = max(0.01, 0.05 * sigma_bar)
            for trial in range(60):
                budget = sigma_bar + step * (trial + 1)
                res2 = self._phase2(hard, budget)
                if not res2.success:
                    continue
                k_star = res2.x[p]
                if k_star < _K_MIN:
                    continue
                eps = max(1e-6, 0.01 * k_star)
                k_ub = min(_K_CAP, k_star)
                G, h = self._full_inequalities(hard, budget, eps, k_ub)
                A_eq, b_eq = self._equalities()
                x0, radius = self._interior_point(G, h, A_eq, b_eq)
                if x0 is not None and radius > _INTERIOR_TOL:
                    break
                x0 = None

        if x0 is None:
            # Degenerate polytope: fall back to the phase-2 vertex.
            xv = res2.x
            w = np.tile(np.clip(xv[:p], 0, None), (num_samples, 1))
            self.info = {"hard_ordering": hard, "sigma_bar": sigma_bar,
                         "degenerate": True}
            return np.hstack([w, np.full((num_samples, 1), xv[p])])

        self.info = {"hard_ordering": hard, "sigma_bar": sigma_bar,
                     "budget": budget, "k_star": k_star, "radius": radius,
                     "degenerate": False}

        samples = self._hit_and_run(G, h, A_eq, x0, num_samples, burn_in, thin)
        w = np.clip(samples[:, :p], 0, None)
        k = samples[:, p:p + 1]
        return np.hstack([w, k])

    def _hit_and_run(self, G, h, A_eq, x0, num_samples, burn_in, thin):
        Z = null_space(A_eq)
        dim = Z.shape[1]
        if burn_in is None:
            burn_in = max(1000, 50 * dim)
        if thin is None:
            thin = max(10, dim)

        GZ = G @ Z
        slack0 = h - G @ x0  # >= 0 by construction
        z = np.zeros(dim)
        out = np.empty((num_samples, G.shape[1]))
        n_out = 0
        total = burn_in + num_samples * thin
        it = 0
        while n_out < num_samples and it < 10 * total:
            it += 1
            u = self.rng.standard_normal(dim)
            u /= np.linalg.norm(u)
            a = GZ @ u
            b = slack0 - GZ @ z
            b = np.maximum(b, 0.0)
            pos = a > 1e-12
            neg = a < -1e-12
            t_hi = np.min(b[pos] / a[pos]) if np.any(pos) else np.inf
            t_lo = np.max(b[neg] / a[neg]) if np.any(neg) else -np.inf
            if not np.isfinite(t_hi) or not np.isfinite(t_lo) or t_hi <= t_lo:
                continue
            z = z + self.rng.uniform(t_lo, t_hi) * u
            if it > burn_in and (it - burn_in) % thin == 0:
                out[n_out] = x0 + Z @ z
                n_out += 1
        if n_out < num_samples:
            out[n_out:] = x0 + (Z @ z)
        return out


def sample_smaa_dor(X_diff, etas, n_params, num_samples=200, seed=0):
    """Convenience wrapper used by the experiment pipeline."""
    sampler = SMAADORSampler(X_diff, etas, n_params, seed=seed)
    return sampler.sample(num_samples=num_samples)
