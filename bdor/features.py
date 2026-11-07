"""Piecewise-linear marginal value features v(a) (paper Sec. 2.1, eqs. (1)-(2)).

Every criterion g_j (gain type: the larger, the better) is divided by characteristic points
c_j0 < c_j1 < ... < c_j,gamma_j into gamma_j segments, and alternative a gets one feature per
segment,

    v_jk(a) = clip((g_j(a) - c_j,k-1) / (c_jk - c_j,k-1), 0, 1)  in [0, 1],   k = 1..gamma_j,

i.e. 0 below the segment, 1 above it and linear in between.  With the increments w_jk >= 0 of
eq. (1) the marginal value is u_j(a) = v_j(a)^T w_j, and the global value is U(a) = v(a)^T w
(eq. (2)), with v(a) the concatenation of the v_j(a), criterion by criterion
(N = gamma_1 + ... + gamma_m features).

The simulations use ``from_equal_intervals(table, num_intervals=3)`` on the performance table
of each replication (gamma_j = 3: four equally spaced characteristic points between the
column's minimum and maximum, Sec. 5.1).  The case study (scripts/case_study/case_study_lea.py)
uses gamma_j = 1 and gamma_j = 3 on the [0, 100] scale (Sec. 6).
"""
import numpy as np

class PiecewiseLinearTransformer:
    """Maps a performance table to the piecewise-linear features v(a) of Sec. 2.1.

    Parameters
    ----------
    ch_p : sequence of 1-D arrays
        Characteristic points of every criterion, ``ch_p[j] = (c_j0, ..., c_j,gamma_j)``,
        strictly increasing: a zero-length segment divides by zero (a constant column passed
        to ``from_equal_intervals`` gives NaN features).

    Attributes
    ----------
    ch_p : sequence of 1-D arrays
        The characteristic points, as given.
    num_criteria : int
        Number m of criteria.
    n_w : list of int
        gamma_j = ``len(ch_p[j]) - 1``, the number of features (and weights) of criterion j.
    total_params : int
        N = ``sum(n_w)``, the length of v(a) and of the weight vector w.
    """
    def __init__(self, ch_p):
        self.ch_p = ch_p
        self.num_criteria = len(ch_p)
        self.n_w = [len(cp) - 1 for cp in ch_p]
        self.total_params = sum(self.n_w)

    @classmethod
    def from_equal_intervals(cls, data_matrix, num_intervals):
        """Build the transformer with equally spaced characteristic points for every column.

        Parameters
        ----------
        data_matrix : ndarray, shape (n_alternatives, m)
            Performance table (rows = alternatives, columns = criteria).
        num_intervals : int
            Number gamma of segments, the same for every criterion.

        Returns
        -------
        PiecewiseLinearTransformer
            With ``ch_p[j] = linspace(min_j, max_j, num_intervals + 1)``, min_j and max_j being
            the observed minimum and maximum of column j; ``total_params = m * num_intervals``.
        """
        mini = np.min(data_matrix, axis=0)
        maxi = np.max(data_matrix, axis=0)
        ch_p = []

        for j in range(data_matrix.shape[1]):
            # Create breakpoints: min, min+step, ..., max
            breakpoints = np.linspace(mini[j], maxi[j], num_intervals + 1)
            ch_p.append(breakpoints)

        return cls(ch_p)

    def transform(self, data_matrix):
        """Feature vectors v(a) of the alternatives (vectorised over the rows).

        Parameters
        ----------
        data_matrix : ndarray, shape (n_alternatives, m)
            Performance table on the same criteria (columns) as ``ch_p``.  Values below c_j0
            (above c_j,gamma_j) give all-zero (all-one) features for criterion j.

        Returns
        -------
        ndarray, shape (n_alternatives, total_params)
            Row i is v(a_i): the n_w[0] features of criterion 0, then those of criterion 1,
            and so on; every entry lies in [0, 1].
        """
        N, M = data_matrix.shape
        features = np.zeros((N, self.total_params))
        current_col_idx = 0

        for j in range(M):
            col_data = data_matrix[:, j]
            breakpoints = self.ch_p[j]
            n_segments = len(breakpoints) - 1

            for k in range(n_segments):
                bp_low = breakpoints[k]
                bp_high = breakpoints[k+1]
                val = (col_data - bp_low) / (bp_high - bp_low)
                features[:, current_col_idx + k] = np.clip(val, 0, 1)

            current_col_idx += n_segments

        return features
