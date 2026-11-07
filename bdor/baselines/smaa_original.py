"""DOR and SMAA-DOR in their original single-session formulation (paper Sec. 2.2).

These are the two existing methods of the DOR family that the paper compares with B-DOR
after the first session (Sec. 5.5: Table 3 lower block, Figure 3, Supplementary Tables
B.6-B.9). ``scripts/simulation/run_original_baselines.py`` runs them on the first session
of every grid replication (sample folders ``DORLPORIG_T1`` / ``SMAAORIG_T1``), and
``scripts/simulation/inject_original_baselines.py`` stores their metrics in the grid
results as the arms ``DORLP_DOR_ORIG`` (DOR) and ``SMAA_DOR_ORIG`` (SMAA-DOR). Only the
single-session case is implemented: both methods are one-shot.

Model
-----
The linear programme of paper eq. (3), in the interval-scale variant of Corrente et al.
(2025, Remark 1, eq. (5)) that needs no blank cards between L_1 and a fictitious zero
level L_0. The DM sorts the n alternatives shown in the session into levels L_1, ..., L_s
(paper numbering: L_1 is the worst) with e_r blank cards between L_r and L_{r+1}. The
positions are

    nu(a) = 0 for a in L_1,    nu(a) = sum_{q < r} (e_q + 1) for a in L_r, r >= 2,

so a gap without cards still counts one unit. The variables are

    x = [w (p), k, h, sigma+ (n), sigma- (n)],

with ONE deviation pair per shown alternative (not per gap), and the constraints are

    E^Model   U(a) = v(a)^T w,  w >= 0,  sum(w) = 1     (eq. (2) with normalised weights)
    E^DM      U(a) - sigma+(a) + sigma-(a) = k * nu(a) + h       for every shown a
              U(a) >= U(b)  for a in L_{r+1}, b in L_r  (adjacent levels; the other pairs
                                                         follow by transitivity)
              U(a) = U(b)   for a, b in the same level
              k >= 0,  sigma+(a) >= 0,  sigma-(a) >= 0

where h is the free offset (delta in eq. (3)). The code adds two bounds that the paper does
not state: k <= 1 (``_K_CAP``, shared with the adapted baseline) and the loose box
|h| <= nu_max + 2.

Methods
-------
``OriginalDORSampler.solve_point`` (DOR). Phase 1 minimises the total deviation
sigma = sum_a (sigma+(a) + sigma-(a)) subject to the constraints above (eq. (3)), giving
sigma_bar. Phase 2 maximises k subject to the same constraints and
sigma <= sigma_bar + eta, with the admitted deterioration eta = 0, 0.01, 0.02, ... grown
until k* > 1e-9 (at most 400 values). It returns the weight part of this most
discriminant vertex.

``OriginalDORSampler.sample`` (SMAA-DOR). The same two phases, with eta grown until, in
addition, the lifted polytope

    {x : E^Model, E^DM,  sigma <= sigma_bar + eta,  max(1e-6, 0.01 k*) <= k <= min(1, k*)}

has an interior point. Hit-and-Run (inherited from ``SMAADORSampler``) started there
draws points uniformly from this polytope in the space of all variables, so the returned
weights follow the marginal of that uniform distribution. Burn-in is max(1000, 50 dim)
iterations and thinning max(10, dim), with dim = p + n + 1 the dimension of the affine
hull of the equalities. If no admitted deterioration gives an interior point, the samples
are copies of the last phase-2 vertex.

Both methods fall back from hard to soft ordering: when no weight vector satisfies the
ordering rows (e.g. an alternative is ranked above one that dominates it), phase 1 is
solved again without them, keeping only the proportionality equalities with deviations.

Differences from the adapted per-gap baseline (``smaa_adapted.py``, arms ``*_ADJJ1``):
deviations per alternative instead of per gap, position increments e + 1 instead of e,
a free offset h, and eta grown in fixed steps of 0.01 starting from 0.
"""

import numpy as np

from bdor.baselines.smaa_adapted import SMAADORSampler, _solve_lp, _K_CAP, _INTERIOR_TOL

_ETA_STEP = 0.01      # admitted deterioration eta = 0, 0.01, 0.02, ... (phase 2)
_MAX_ETA_STEPS = 400  # safety cap on the eta growth loop
_K_POS = 1e-9         # numerical "k* > 0" test that stops the eta growth


def _build_nu(levels, cards):
    """Position nu of each declared level, aligned with ``levels`` (best first).

    Built worst first, as in the paper: nu(L_1) = 0 and nu(L_{r+1}) = nu(L_r) + e_r + 1.
    Starting from 0 is what the free offset h allows: h absorbs the shift that the
    fictitious zero level L_0 (and its e_0 cards) would add to every position.
    ``cards[r]`` is the number of blank cards between ``levels[r]`` and ``levels[r + 1]``;
    the positions depend on the cards only (``levels`` is unused).
    """
    nu_worst_first = [0.0]
    cards_wf = list(cards)[::-1]
    for e in cards_wf:
        nu_worst_first.append(nu_worst_first[-1] + float(e) + 1.0)
    return nu_worst_first[::-1]  # align to best-to-worst levels


class OriginalDORSampler(SMAADORSampler):
    """DOR / SMAA-DOR for the levels and blank cards declared in one session.

    Parameters
    ----------
    feature_matrix : array_like, shape (n_alternatives, p)
        Piecewise-linear features v(a) of every alternative of the problem
        (``PiecewiseLinearTransformer.transform``); only the shown ones are used.
    levels : list of list of int
        Declared levels, BEST first (dataset convention), each a list of alternative
        indices; alternatives in the same level are tied.
    cards : list of int
        ``cards[r]`` blank cards between ``levels[r]`` and ``levels[r + 1]``; length
        ``len(levels) - 1``.
    n_params : int
        Number p of weights (piecewise-linear increments).
    seed : int
        Seed of the generator used by ``sample`` (Dirichlet draws or Hit-and-Run).

    Attributes
    ----------
    alts : list of int
        The n shown alternatives, best level first; deviation pair i belongs to ``alts[i]``.
    nus : ndarray, shape (n,)
        Position nu of each shown alternative.
    info : dict
        Diagnostics of the last ``solve_point`` / ``sample`` call: ``sigma_bar``,
        ``k_star``, ``hard_ordering`` and, depending on the call and its outcome,
        ``budget``, ``radius`` and ``degenerate``.

    Notes
    -----
    Only ``_interior_point`` and ``_hit_and_run`` are inherited and used. The constructor,
    the LP builders (with their own signatures) and ``sample`` are redefined for the
    per-alternative variable layout.
    """

    def __init__(self, feature_matrix, levels, cards, n_params, seed=0):
        self.X = np.asarray(feature_matrix, dtype=float)
        self.levels = [list(map(int, lv)) for lv in levels]
        self.cards = [int(c) for c in cards]
        assert len(self.cards) == len(self.levels) - 1
        self.p = int(n_params)
        self.rng = np.random.default_rng(seed)
        self.info = {}

        nu_lv = _build_nu(self.levels, self.cards)
        self.alts = [a for lv in self.levels for a in lv]
        self.nus = np.array([nu_lv[r] for r, lv in enumerate(self.levels)
                             for _ in lv])
        self.n = len(self.alts)
        self.nu_max = float(self.nus.max()) if self.n else 0.0
        # h is free in eq. (3); a loose box keeps it explicitly bounded in the LPs and in
        # the sampled polytope.
        self.h_bound = _K_CAP * self.nu_max + 2.0

    # variable layout ----------------------------------------------------
    @property
    def _nvar(self):
        return self.p + 2 + 2 * self.n

    def _equalities(self):
        """sum(w) = 1 and U(a) - s+(a) + s-(a) = k nu(a) + h for each shown a (A x = b)."""
        p, n = self.p, self.n
        nv = self._nvar
        A = np.zeros((1 + n, nv))
        b = np.zeros(1 + n)
        A[0, :p] = 1.0
        b[0] = 1.0
        for i, (a, nu) in enumerate(zip(self.alts, self.nus)):
            A[1 + i, :p] = self.X[a]          # U(a) = v(a) . w
            A[1 + i, p] = -nu                 # - k * nu(a)
            A[1 + i, p + 1] = -1.0            # - h
            A[1 + i, p + 2 + i] = -1.0        # - s+
            A[1 + i, p + 2 + n + i] = 1.0     # + s-
        return A, b

    def _ordering_rows(self):
        """Ordering constraints as rows G x <= 0, without deviation variables (hard).

        U(a) >= U(b) for every a in the better and b in the worse of two ADJACENT
        levels (non-adjacent pairs follow by transitivity); within a level,
        U(a) = U(b) for consecutive members, written as two opposite rows.
        """
        p = self.p
        nv = self._nvar
        rows = []
        for r in range(len(self.levels) - 1):
            for a in self.levels[r]:            # better level (best-first)
                for b in self.levels[r + 1]:
                    row = np.zeros(nv)
                    row[:p] = -(self.X[a] - self.X[b])
                    rows.append(row)
        for lv in self.levels:                  # exact indifference in a set
            for a, b in zip(lv[:-1], lv[1:]):
                row = np.zeros(nv)
                row[:p] = self.X[a] - self.X[b]
                rows.append(row.copy())
                rows.append(-row)
        if not rows:
            return np.zeros((0, nv)), np.zeros(0)
        G = np.vstack(rows)
        return G, np.zeros(G.shape[0])

    def _bounds(self):
        return ([(0, None)] * self.p                 # w >= 0
                + [(0, _K_CAP)]                      # 0 <= k <= cap
                + [(-self.h_bound, self.h_bound)]    # h free (bounded box)
                + [(0, None)] * (2 * self.n))        # s+/- >= 0

    # the two LPs --------------------------------------------------------
    def _phase1(self, hard=True):
        """Phase 1 (eq. (3)): minimise sum(s+ + s-); ``hard=False`` drops the ordering rows."""
        c = np.concatenate([np.zeros(self.p + 2), np.ones(2 * self.n)])
        A_eq, b_eq = self._equalities()
        G, hvec = (self._ordering_rows() if hard
                   else (np.zeros((0, self._nvar)), np.zeros(0)))
        return _solve_lp(c, G if G.size else None, hvec if G.size else None,
                         A_eq, b_eq, self._bounds())

    def _phase2(self, budget, hard=True):
        """Phase 2: maximise k subject to sum(s+ + s-) <= budget."""
        nv = self._nvar
        c = np.zeros(nv)
        c[self.p] = -1.0
        A_eq, b_eq = self._equalities()
        G, hvec = (self._ordering_rows() if hard
                   else (np.zeros((0, nv)), np.zeros(0)))
        brow = np.zeros(nv)
        brow[self.p + 2:] = 1.0
        A_ub = np.vstack([G, brow[None, :]]) if G.size else brow[None, :]
        b_ub = np.concatenate([hvec, [budget]]) if G.size else np.array([budget])
        return _solve_lp(c, A_ub, b_ub, A_eq, b_eq, self._bounds())

    def _full_inequalities(self, budget, eps, k_ub, hard=True):
        """Inequalities of the sampled polytope as rows G x <= rhs (bounds included)."""
        p, n = self.p, self.n
        nv = self._nvar
        eye = np.eye(nv)
        rows, rhs = [], []
        nonneg = list(range(p)) + list(range(p + 2, nv))
        rows.append(-eye[nonneg]); rhs.append(np.zeros(len(nonneg)))
        rows.append(-eye[[p]]); rhs.append(np.array([-eps]))       # k >= eps
        rows.append(eye[[p]]); rhs.append(np.array([k_ub]))        # k <= k_ub
        rows.append(-eye[[p + 1]]); rhs.append(np.array([self.h_bound]))
        rows.append(eye[[p + 1]]); rhs.append(np.array([self.h_bound]))
        if hard:
            G, hvec = self._ordering_rows()
            if G.size:
                rows.append(G); rhs.append(hvec)
        brow = np.zeros(nv)
        brow[p + 2:] = 1.0
        rows.append(brow[None, :]); rhs.append(np.array([budget]))
        return np.vstack(rows), np.concatenate(rhs)

    # entry points -------------------------------------------------------
    def solve_point(self):
        """DOR point estimate: the weights of the most discriminant compatible vertex.

        Phase 1 gives sigma_bar; phase 2 maximises k with total deviation at most
        sigma_bar + eta, for eta = 0, 0.01, 0.02, ... until k* > 1e-9 (at most 400 values;
        if none qualifies, the last feasible phase-2 vertex is used). Does not use the
        random generator.

        Returns
        -------
        ndarray, shape (p,)
            Weights of that vertex, clipped at 0 and renormalised to sum 1. Uniform
            weights when the session has fewer than two levels or the LPs fail.
        """
        uniform = np.full(self.p, 1.0 / self.p)
        if self.n == 0 or len(self.levels) < 2:
            return uniform
        # Same fallback rule as the adapted baseline (smaa_adapted.py): a declared
        # order that violates dominance makes the hard-ordering polytope empty; drop the
        # ordering rows and keep the slack-based proportionality.
        hard = True
        res1 = self._phase1(hard)
        if not res1.success:
            hard = False
            res1 = self._phase1(hard)
            if not res1.success:
                return uniform
        sigma_bar = max(res1.fun, 0.0)
        res2, k_star, budget = None, 0.0, sigma_bar
        for step in range(_MAX_ETA_STEPS):
            budget = sigma_bar + _ETA_STEP * step
            r = self._phase2(budget, hard)
            if r.success:
                res2 = r
                k_star = r.x[self.p]
                if k_star > _K_POS:
                    break
        self.info = {"sigma_bar": sigma_bar, "k_star": k_star,
                     "budget": budget, "hard_ordering": hard}
        if res2 is None:
            return uniform
        w = np.clip(res2.x[:self.p], 0.0, None)
        s = w.sum()
        return w / s if s > 0 else uniform

    def sample(self, num_samples=200, burn_in=None, thin=None):
        """SMAA-DOR: Hit-and-Run draws, uniform over the compatible polytope.

        Parameters
        ----------
        num_samples : int
            Number of draws returned (the paper uses 200).
        burn_in, thin : int or None
            Hit-and-Run burn-in iterations and thinning interval; ``None`` uses
            max(1000, 50 dim) and max(10, dim), with dim = p + n + 1.

        Returns
        -------
        ndarray, shape (num_samples, p + 1)
            Rows [w, k]: weights (clipped at 0) and value k of a blank card. With fewer
            than two levels: Dirichlet(1, ..., 1) weights and k = 0. When no admitted
            deterioration gives the polytope an interior point: ``num_samples`` copies of
            the last feasible phase-2 vertex, or of uniform weights with k = 0 if no
            phase-2 LP succeeded (``info['degenerate']`` is True).

        Raises
        ------
        RuntimeError
            If the phase-1 LP fails even without the ordering rows.
        """
        p = self.p
        if self.n == 0 or len(self.levels) < 2:
            w = self.rng.dirichlet(np.ones(p), size=num_samples)
            return np.hstack([w, np.zeros((num_samples, 1))])

        hard = True
        res1 = self._phase1(hard)
        if not res1.success:
            hard = False
            res1 = self._phase1(hard)
            if not res1.success:
                raise RuntimeError(f"original phase-1 LP failed: {res1.message}")
        sigma_bar = max(res1.fun, 0.0)

        # Unlike solve_point, keep growing eta until the polytope also has an interior
        # point to start Hit-and-Run from.
        x0, k_star, budget, res2 = None, 0.0, sigma_bar, None
        for step in range(_MAX_ETA_STEPS):
            budget = sigma_bar + _ETA_STEP * step
            r = self._phase2(budget, hard)
            if not r.success:
                continue
            res2 = r
            k_star = r.x[p]
            if k_star <= _K_POS:
                continue
            eps = max(1e-6, 0.01 * k_star)
            k_ub = min(_K_CAP, k_star)
            G, h = self._full_inequalities(budget, eps, k_ub, hard)
            A_eq, b_eq = self._equalities()
            x0, radius = self._interior_point(G, h, A_eq, b_eq)
            if x0 is not None and radius > _INTERIOR_TOL:
                break
            x0 = None

        if x0 is None:
            xv = (res2.x if res2 is not None else
                  np.concatenate([np.full(p, 1.0 / p), np.zeros(self._nvar - p)]))
            w = np.tile(np.clip(xv[:p], 0, None), (num_samples, 1))
            self.info = {"sigma_bar": sigma_bar, "k_star": k_star,
                         "hard_ordering": hard, "degenerate": True}
            return np.hstack([w, np.full((num_samples, 1), xv[p])])

        self.info = {"sigma_bar": sigma_bar, "k_star": k_star,
                     "budget": budget, "radius": radius,
                     "hard_ordering": hard, "degenerate": False}
        samples = self._hit_and_run(G, h, A_eq, x0, num_samples, burn_in, thin)
        w = np.clip(samples[:, :p], 0, None)
        k = samples[:, p:p + 1]
        return np.hstack([w, k])


def sample_smaa_dor_original(feature_matrix, levels, cards, n_params,
                             num_samples=200, seed=0):
    """SMAA-DOR draws for one session: ``OriginalDORSampler(...).sample(num_samples)``.

    Arguments as in ``OriginalDORSampler``; returns an array of shape
    (num_samples, p + 1) with rows [w, k].
    """
    s = OriginalDORSampler(feature_matrix, levels, cards, n_params, seed=seed)
    return s.sample(num_samples=num_samples)


def solve_dor_lp_original(feature_matrix, levels, cards, n_params):
    """DOR point estimate for one session: ``OriginalDORSampler(...).solve_point()``.

    Arguments as in ``OriginalDORSampler``; returns the weights, shape (p,).
    """
    s = OriginalDORSampler(feature_matrix, levels, cards, n_params)
    return s.solve_point()
