"""Adapted per-gap DOR / SMAA-DOR: grid arms DORLP_DOR_ADJJ1 and SMAA_DOR_ADJJ1 (not reported).

An earlier adaptation of DOR (Barbati et al. 2024) and SMAA-DOR (Corrente, Greco and
Zappalà 2025, arXiv:2405.04313) to the sequential gap observations of B-DOR. The grid
campaign (``scripts/simulation/run_grid.py``) still runs it, through
``bdor.simulation.get_sampler_state``, as the arms ``DORLP_DOR_ADJJ1`` (``solve_dor_lp``)
and ``SMAA_DOR_ADJJ1`` (``sample_smaa_dor``), but no paper output reads these arms: the
baselines reported in the paper are the original formulation in ``smaa_original.py``.
This module is still required because ``OriginalDORSampler`` subclasses
``SMAADORSampler`` and reuses ``_solve_lp``, ``_interior_point`` and ``_hit_and_run``.

Formulation
-----------
The data are the m gap observations accumulated over all sessions so far
(``PreferenceSampler.X_diff`` / ``.etas``, adjacent levels only in the grid arms): gap t
has the feature difference d_t = (mean of v over the upper level) - (mean over the lower
level) and eta_t blank cards. The variables are x = [w (p), k, s+ (m), s- (m)], with ONE
deviation pair PER GAP:

    E^Model    U(a) = v(a)^T w,  w >= 0,  sum(w) = 1
    cards      d_t . w - s+_t + s-_t = k * eta_t          for every gap t
    ordering   d_t . w >= 0                               for every gap t (hard mode)
    bounds     0 <= k <= 1 (_K_CAP),  s+ >= 0,  s- >= 0

Phase 1 minimises the total deviation sum(s+ + s-), giving sigma_bar; phase 2 maximises k
subject to sum(s+ + s-) <= budget. The proportionality is written on differences (the
interval-scale form, no offset) with target k * eta_t, the raw card count (capped at e_max
by the simulation) rather than eta_t + 1: the dataset generator sets the cards to the
rounded difference of the perceived utilities scaled by e_max, so eta_t is the gap in
position units. With zero cards on every gap k would be unbounded, hence the cap
``_K_CAP``.

* ``solve_dor_lp`` (DOR): phase 2 at budget sigma_bar + 1e-9 (no deterioration admitted);
  returns the weight part of the vertex.
* ``SMAADORSampler.sample`` (SMAA-DOR): the budget is sigma_bar + j * step, j = 1, ..., 60,
  step = max(0.01, 0.05 sigma_bar), increased until k* >= 1e-4 (``_K_MIN``) and the
  polytope {E^Model, cards, ordering, sum(s+ + s-) <= budget,
  max(1e-6, 0.01 k*) <= k <= min(1, k*)} has an interior point; Hit-and-Run started there
  samples it uniformly.

With inconsistent information (e.g. answers that contradict each other across sessions)
the hard ordering rows can make the polytope empty or flat; they are then dropped and only
the card equalities with their deviations are kept (soft mode).
"""

import numpy as np
from scipy.optimize import linprog
from scipy.linalg import null_space

_K_CAP = 1.0          # upper bound on k (k is unbounded when no gap carries cards)
_K_MIN = 1e-4         # minimal acceptable k*; smaller values grow the error budget
_INTERIOR_TOL = 1e-7  # minimal Chebyshev-like radius to accept a starting point


def _solve_lp(c, A_ub, b_ub, A_eq, b_eq, bounds):
    """``scipy.optimize.linprog`` with the HiGHS solver; returns its ``OptimizeResult``."""
    return linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                   bounds=bounds, method="highs")


class SMAADORSampler:
    """Adapted per-gap SMAA-DOR over the accumulated gap observations (arm SMAA_DOR_ADJJ1).

    Parameters
    ----------
    X_diff : array_like, shape (m, p)
        Feature difference d_t of each gap observation (upper minus lower level centroid).
    etas : array_like, shape (m,)
        Blank cards eta_t of each gap.
    n_params : int
        Number p of weights (piecewise-linear increments).
    seed : int
        Seed of the generator used by ``sample`` (Dirichlet draws or Hit-and-Run).

    Attributes
    ----------
    info : dict
        Diagnostics of the last ``sample`` call: ``hard_ordering``, ``sigma_bar``,
        ``degenerate`` and, unless degenerate, ``budget``, ``k_star`` and ``radius``.
    """

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
        """sum(w) = 1 and d_t . w - k eta_t - s+_t + s-_t = 0 for each gap (A x = b)."""
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
        """Chebyshev-like centre: max delta s.t. G x + delta * ||row|| <= h, A_eq x = b_eq.

        Returns ``(x, delta)`` with delta in [0, 10], or ``(None, 0.0)`` if the LP fails.
        Row norms are taken in the full variable space (not projected on the equalities).
        """
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
        """Adapted SMAA-DOR: Hit-and-Run draws, uniform over the compatible polytope.

        Parameters
        ----------
        num_samples : int
            Number of draws returned.
        burn_in, thin : int or None
            Hit-and-Run burn-in iterations and thinning interval; ``None`` uses
            max(1000, 50 dim) and max(10, dim), with dim = p + m.

        Returns
        -------
        ndarray, shape (num_samples, p + 1)
            Rows [w, k]: weights (clipped at 0) and value k of a blank card. Without gap
            observations: Dirichlet(1, ..., 1) weights and k = 0. When no budget gives
            the polytope an interior point, in hard and then in soft mode:
            ``num_samples`` copies of the last phase-2 vertex (``info['degenerate']`` is
            True).

        Raises
        ------
        RuntimeError
            If the phase-1 LP fails even without the ordering rows.
        """
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

        # Grow the admitted deterioration until k* >= _K_MIN and the polytope has an
        # interior point to start Hit-and-Run from.
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
        """Hit-and-Run over {x : G x <= h, A_eq x = A_eq x0}, started at the interior point x0.

        Each iteration draws a uniformly random direction in the null space of ``A_eq``
        (dimension dim) and moves to a uniform point of the feasible chord. After
        ``burn_in`` iterations (default max(1000, 50 dim)) the current point is recorded
        every ``thin`` iterations (default max(10, dim)); iterations with a degenerate
        chord count but move and record nothing. After 10 * (burn_in + num_samples * thin)
        iterations the remaining rows repeat the last point.

        Returns
        -------
        ndarray, shape (num_samples, G.shape[1])
            Draws of the full variable vector x.
        """
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
    """Adapted SMAA-DOR draws (arm SMAA_DOR_ADJJ1), as called by ``get_sampler_state``.

    ``SMAADORSampler(X_diff, etas, n_params, seed).sample(num_samples)``: an array of
    shape (num_samples, p + 1) with rows [w, k].
    """
    sampler = SMAADORSampler(X_diff, etas, n_params, seed=seed)
    return sampler.sample(num_samples=num_samples)


def solve_dor_lp(X_diff, etas, n_params):
    """Adapted per-gap DOR point estimate (arm DORLP_DOR_ADJJ1).

    Phase 1 gives the minimal total deviation sigma_bar; phase 2 maximises the
    discrimination k with total deviation at most sigma_bar + 1e-9 (no deterioration is
    admitted, so k* may be 0), and the weight part of that vertex is returned. The hard
    ordering rows are dropped when either LP fails with them, mirroring
    ``SMAADORSampler``.

    Parameters
    ----------
    X_diff, etas, n_params
        As in ``SMAADORSampler``.

    Returns
    -------
    ndarray, shape (p,)
        Weights clipped at 0 and renormalised to sum 1; uniform when there are no gap
        observations or the LPs fail.
    """
    p = int(n_params)
    uniform = np.full(p, 1.0 / p)
    D = np.atleast_2d(np.asarray(X_diff, dtype=float))
    if D.size == 0 or len(np.asarray(etas).ravel()) == 0:
        return uniform
    solver = SMAADORSampler(X_diff, etas, n_params)
    for hard in (True, False):
        res1 = solver._phase1(hard)
        if not res1.success:
            continue
        sigma_bar = float(res1.fun)
        res2 = solver._phase2(hard, sigma_bar + 1e-9)
        if res2.success:
            w = np.clip(res2.x[:p], 0.0, None)
            s = w.sum()
            return w / s if s > 0 else uniform
    return uniform
