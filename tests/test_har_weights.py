"""The section weights of the HAR kernel, checked against the explicit basis."""
import itertools
import math

import numpy as np
import pytest

from kernel_ridge import HighlyAdaptiveRidgeCV
from kernel_ridge.kernels import HighlyAdaptiveRidge


def factor(shell, x, v, order):
    """One coordinate of a basis function of the given order, with knot coordinate v.

    Shell 0 means that the coordinate is not in the section (factor 1). Shells
    1, ..., order give the polynomial factor x^shell / shell!, and shell order + 1
    gives the spline (x - v)_+^order / order!.
    """
    if shell == 0:
        return 1.0
    if shell <= order:
        return x ** shell / math.factorial(shell)
    return float(v <= x) * (x - v) ** order / math.factorial(order)


def explicit_kernel(X, X_test, order, w):
    """sum over knots and bases of w_{|s|} h(x) h(x'), with |s| the number of coordinates in the section."""
    n, p = X.shape
    K = np.zeros((len(X_test), n))
    for knot in X:
        for shells in itertools.product(range(order + 2), repeat=p):
            size = sum(s > 0 for s in shells)
            if size == 0:
                continue  # the intercept is fit separately
            h = np.array([[math.prod(factor(s, x[j], knot[j], order) for j, s in enumerate(shells))] for x in X])
            h_test = np.array([[math.prod(factor(s, x[j], knot[j], order) for j, s in enumerate(shells))] for x in X_test])
            K += w[size] * h_test @ h.T
    return K


def data(n, n_test, p, seed):
    rng = np.random.default_rng(seed)
    return rng.random((n, p)), rng.random((n_test, p))


p = 4
settings = {
    "plain": dict(),
    "random weights": dict(weights=[0.3, 1.7, 0.05, 2.0]),
    "zero weight in the middle": dict(weights=[1.0, 0.0, 0.5, 0.0]),
    "depth": dict(depth=2),
    "decay": dict(decay=0.4),
    "decay and depth": dict(decay=0.4, depth=3),
    "callable": dict(weights=lambda p: 1.0 / np.arange(1, p + 1)),
}


@pytest.mark.parametrize("order", [0, 1, 2])
@pytest.mark.parametrize("name", list(settings))
@pytest.mark.parametrize("seed", range(2))
def test_weighted_kernel_matches_explicit_basis(order, name, seed):
    X, X_test = data(6, 3, p, seed)
    kernel = HighlyAdaptiveRidge(order=order, **settings[name])
    w = kernel.section_weights(p)
    np.testing.assert_allclose(kernel(X, X_test), explicit_kernel(X, X_test, order, w), rtol=1e-10)
    np.testing.assert_allclose(kernel(X), explicit_kernel(X, X, order, w), rtol=1e-10)


def test_section_weights_compose():
    w = HighlyAdaptiveRidge(weights=[1.0, 2.0, 3.0, 4.0], decay=0.5, depth=3).section_weights(4)
    np.testing.assert_array_equal(w, [0.0, 0.5, 0.5, 0.375, 0.0])


@pytest.mark.parametrize("order", [0, 1])
def test_depth_is_a_zero_weight(order):
    X, X_test = data(8, 4, 5, 0)
    by_depth = HighlyAdaptiveRidge(order=order, depth=2)(X, X_test)
    by_weights = HighlyAdaptiveRidge(order=order, weights=[1, 1, 0, 0, 0])(X, X_test)
    np.testing.assert_array_equal(by_depth, by_weights)


def test_infinite_depth_keeps_every_section():
    X, X_test = data(8, 4, 5, 0)
    np.testing.assert_array_equal(
        HighlyAdaptiveRidge(depth=np.inf)(X, X_test), HighlyAdaptiveRidge()(X, X_test)
    )


def test_decay_has_the_closed_form_at_large_p():
    # A knot at 0 is below both points in all 1200 coordinates. C(1200, 600) does
    # not fit in a float, but (1 + rate)^1200 - 1 does.
    X, X_test = data(3, 2, 1200, 0)
    X[0] = 0.0
    rate = 0.1
    m = np.minimum(X_test[:, None, :], X[None, :, :])        # test point x training point x coordinate
    c = (X[None, None, :, :] <= m[:, :, None, :]).sum(-1)    # test point x training point x knot
    expected = ((1 + rate) ** c - 1).sum(-1)
    np.testing.assert_allclose(HighlyAdaptiveRidge(decay=rate)(X, X_test), expected, rtol=1e-10)


def test_plain_kernel_overflow_is_an_error():
    # 2^1200 does not fit in a float.
    X, X_test = data(3, 2, 1200, 0)
    X[0] = 0.0
    with pytest.raises(ValueError, match="overflows"):
        HighlyAdaptiveRidge()(X, X_test)


def test_plain_kernel_at_large_p_is_finite_when_no_knot_overflows():
    X, X_test = data(3, 2, 1200, 0)
    assert np.all(np.isfinite(HighlyAdaptiveRidge()(X, X_test)))


@pytest.mark.parametrize(
    "kwargs",
    [dict(weights=[1.0, 1.0]), dict(weights=[1.0, -1.0, 1.0]), dict(decay=-0.5), dict(depth=0)],
)
def test_bad_weights_are_errors(kwargs):
    with pytest.raises(ValueError):
        HighlyAdaptiveRidge(**kwargs).section_weights(3)


def test_cv_passes_the_weights_on():
    X, _ = data(30, 1, 5, 0)
    Y = X.sum(1)
    har = HighlyAdaptiveRidgeCV(decay=0.5, depth=2, n_alphas=5).fit(X, Y)
    assert har.best.kernel == HighlyAdaptiveRidge(depth=2, decay=0.5)
