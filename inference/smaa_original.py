"""
FAITHFUL implementation of the original DOR / SMAA-DOR formulation in its
interval-scale variant (corrente2025deck, Remark 1, eq. (5)):

    U(a) - sigma^+(a) + sigma^-(a) = k * nu(a) + h,   h in R free,
    U(a) >= U(a')  whenever  nu(a) >= nu(a'),

with ONE pair of deviations PER ALTERNATIVE (not per gap), position values
nu(a) built with the paper's e+1 increments (a zero-card gap is still one
nu-unit), and the free offset h making the zero level L_0 redundant.

Differences w.r.t. inference/smaa.py (our adapted baseline):
  - deviations sigma^+/- attached to alternatives, objective sum_a |eps_a|
    (theirs) instead of per-gap sum_q |eps_q| (ours);
  - nu increments e+1 (theirs) instead of eta = e (ours);
  - phase 2 grows the admitted deterioration eta(sigma_bar) in increments of
    0.01 starting from 0 until k* > 0, exactly as their LP problem (3).

Only the single-session (T=1) case is implemented: the original methods are
one-shot and the paper's comparison is run at T=1 only.
"""

import numpy as np

from inference.smaa import SMAADORSampler, _solve_lp, _K_CAP, _INTERIOR_TOL

_ETA_STEP = 0.01      # paper's LP (3): increments of 0.01, starting from 0
_MAX_ETA_STEPS = 400  # safety cap on the eta growth loop
_K_POS = 1e-9         # "k* > 0" threshold of the paper's iteration


def _build_nu(levels, cards):
    """
    Position values, worst-first, with the paper's +1 rule:
    nu(L_1) = 0 (h absorbs e_0), nu(L_{h+1}) = nu(L_h) + e_h + 1.
    `levels` is best-to-worst (dataset convention); returns per-level nu
    aligned with the ORIGINAL best-to-worst ordering.
    """
    L = len(levels)
    nu_worst_first = [0.0]
    cards_wf = list(cards)[::-1]
    for e in cards_wf:
        nu_worst_first.append(nu_worst_first[-1] + float(e) + 1.0)
    return nu_worst_first[::-1]  # align to best-to-worst levels


class OriginalDORSampler(SMAADORSampler):
    """
    Variables x = [w (p), k, h, s+ (n), s- (n)], one deviation pair per
    alternative shown in the session.
    """

    def __init__(self, feature_matrix, levels, cards, n_params, seed=0):
        # levels: best-to-worst list of lists of alternative indices
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
        # h is free; bound it loosely so the sampling polytope is bounded.
        self.h_bound = _K_CAP * self.nu_max + 2.0

    # variable layout ----------------------------------------------------
    @property
    def _nvar(self):
        return self.p + 2 + 2 * self.n

    def _equalities(self):
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
        """U(a) >= U(b) for every a in the better of two ADJACENT levels
        (non-adjacent pairs follow by transitivity; rows have no slacks).
        Within-level ties: both directions, i.e. two opposite rows."""
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
        c = np.concatenate([np.zeros(self.p + 2), np.ones(2 * self.n)])
        A_eq, b_eq = self._equalities()
        G, hvec = (self._ordering_rows() if hard
                   else (np.zeros((0, self._nvar)), np.zeros(0)))
        return _solve_lp(c, G if G.size else None, hvec if G.size else None,
                         A_eq, b_eq, self._bounds())

    def _phase2(self, budget, hard=True):
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
        """Original DOR: LP (1) then LP (3) with the 0.01-increment eta
        growth until k* > 0; returns the weight part of the optimal vertex."""
        uniform = np.full(self.p, 1.0 / self.p)
        if self.n == 0 or len(self.levels) < 2:
            return uniform
        # Same fallback rule as the adapted baseline (inference/smaa.py): a
        # declared order violating dominance makes the hard-ordering polytope
        # empty; drop the ordering rows and keep the slack-based proportionality.
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
    s = OriginalDORSampler(feature_matrix, levels, cards, n_params, seed=seed)
    return s.sample(num_samples=num_samples)


def solve_dor_lp_original(feature_matrix, levels, cards, n_params):
    s = OriginalDORSampler(feature_matrix, levels, cards, n_params)
    return s.solve_point()
