"""Piecewise-linear marginal value features v(a) (paper Sec. 2.1, eqs. (1)-(2))."""
import numpy as np
import pytest

from bdor.features import PiecewiseLinearTransformer

# Criterion 0 hits the breakpoints exactly; criterion 1 has a range of 40 split in three.
TABLE = np.array([
    [0.00, 10.0],
    [0.30, 20.0],
    [0.60, 50.0],
    [0.90, 40.0],
    [0.45, 30.0],
])
EXPECTED = np.array([
    [0.0, 0.0, 0.0, 0.00, 0.0, 0.00],
    [1.0, 0.0, 0.0, 0.75, 0.0, 0.00],
    [1.0, 1.0, 0.0, 1.00, 1.0, 1.00],
    [1.0, 1.0, 1.0, 1.00, 1.0, 0.25],
    [1.0, 0.5, 0.0, 1.00, 0.5, 0.00],
])


def test_equal_interval_breakpoints_and_known_features():
    tr = PiecewiseLinearTransformer.from_equal_intervals(TABLE, num_intervals=3)
    assert tr.num_criteria == 2
    assert tr.n_w == [3, 3]
    assert tr.total_params == 6
    np.testing.assert_allclose(tr.ch_p[0], [0.0, 0.3, 0.6, 0.9])
    np.testing.assert_allclose(tr.ch_p[1], [10.0, 10.0 + 40 / 3, 10.0 + 80 / 3, 50.0])

    features = tr.transform(TABLE)
    assert features.shape == (5, 6)
    np.testing.assert_allclose(features, EXPECTED, atol=1e-12)


def test_custom_breakpoints_and_clipping_outside_the_range():
    tr = PiecewiseLinearTransformer([np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.2, 0.4, 1.0])])
    assert tr.n_w == [2, 3]
    assert tr.total_params == 5
    features = tr.transform(np.array([[-1.0, 2.0], [0.75, 0.1]]))
    np.testing.assert_allclose(features, [[0, 0, 1, 1, 1], [1, 0.5, 0.5, 0, 0]])


def _random_table(seed=0, n=40, m=3):
    return np.random.default_rng(seed).random((n, m))


@pytest.mark.parametrize("num_intervals", [1, 3, 4])
def test_features_in_unit_interval_and_monotone(num_intervals):
    table = _random_table()
    tr = PiecewiseLinearTransformer.from_equal_intervals(table, num_intervals)
    features = tr.transform(table)
    assert features.shape == (table.shape[0], table.shape[1] * num_intervals)
    assert features.min() >= 0.0 and features.max() <= 1.0

    for j in range(table.shape[1]):
        block = features[:, j * num_intervals:(j + 1) * num_intervals]
        ordered = block[np.argsort(table[:, j])]
        # non-decreasing in the criterion value, piece by piece
        assert np.all(np.diff(ordered, axis=0) >= -1e-12)
        # worst value of the criterion -> all pieces 0; best value -> all pieces 1
        np.testing.assert_allclose(ordered[0], 0.0)
        np.testing.assert_allclose(ordered[-1], 1.0)


def test_linear_value_function_equals_piecewise_interpolation():
    """U(a) = v(a)^T w (eq. (2)) is the additive piecewise-linear value function of eq. (1):
    on criterion j the marginal value interpolates linearly between the characteristic
    points, where it equals 0, w_j1, w_j1 + w_j2, ..."""
    table = _random_table(seed=1)
    tr = PiecewiseLinearTransformer.from_equal_intervals(table, num_intervals=3)
    w = np.random.default_rng(2).gamma(1.0, 1.0, size=tr.total_params)

    U = tr.transform(table) @ w

    expected = np.zeros(table.shape[0])
    start = 0
    for j, breakpoints in enumerate(tr.ch_p):
        w_j = w[start:start + tr.n_w[j]]
        expected += np.interp(table[:, j], breakpoints, np.concatenate([[0.0], np.cumsum(w_j)]))
        start += tr.n_w[j]
    np.testing.assert_allclose(U, expected, rtol=1e-12, atol=1e-12)
    # the best attainable value is ||w||_1 (all features equal to 1)
    assert U.max() <= w.sum() + 1e-12
