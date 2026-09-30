"""The section weights of the HAR kernel, checked against the explicit basis."""
import itertools
import math

import numpy as np
import pytest
from sklearn.model_selection import KFold

from kernel_ridge import HighlyAdaptiveRidgeCV
from kernel_ridge.fast import _prep
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


# Paths over depth or decay, with early stopping.

def path_data():
    rng = np.random.default_rng(0)
    X = rng.random((40, 4))
    Y = 2 * X[:, 0] * X[:, 1] * X[:, 2] + np.sin(3 * X[:, 3]) + 0.1 * rng.normal(size=40)
    return X, Y


# On path_data, the eig path over decays has its best kernel inside the path, at 0.3.
paths = {"depths": dict(depths=[1, 2, 3, 4]), "decays": dict(decays=[0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0])}
cvs = {"eig": None, "brute": KFold(3)}


def test_paths_build_the_kernels_in_order():
    by_depth = HighlyAdaptiveRidgeCV(decay=0.5, depths=[1, 3])
    assert by_depth.kernels == [HighlyAdaptiveRidge(depth=1, decay=0.5), HighlyAdaptiveRidge(depth=3, decay=0.5)]
    by_decay = HighlyAdaptiveRidgeCV(depth=2, order=1, decays=[0.1, 1.0])
    assert by_decay.kernels == [HighlyAdaptiveRidge(depth=2, order=1, decay=0.1), HighlyAdaptiveRidge(depth=2, order=1, decay=1.0)]


def test_depths_and_decays_together_are_an_error():
    with pytest.raises(ValueError, match="not both"):
        HighlyAdaptiveRidgeCV(depths=[1, 2], decays=[0.1, 1.0])


@pytest.mark.parametrize("patience", [0, 1.5])
def test_bad_patience_is_an_error(patience):
    X, Y = path_data()
    with pytest.raises(ValueError, match="patience"):
        HighlyAdaptiveRidgeCV(depths=[1, 2], patience=patience, n_alphas=5).fit(X, Y)


@pytest.mark.parametrize("cv", list(cvs))
@pytest.mark.parametrize("path", list(paths))
def test_no_patience_evaluates_every_kernel(path, cv):
    X, Y = path_data()
    m = HighlyAdaptiveRidgeCV(n_alphas=10, cv=cvs[cv], **paths[path]).fit(X, Y)
    assert len(m.kernel_mses_) == len(m.kernels)
    assert m.best.kernel == m.kernels[int(np.argmin(m.kernel_mses_))]


@pytest.mark.parametrize("patience", [1, 2])
@pytest.mark.parametrize("cv", list(cvs))
@pytest.mark.parametrize("path", list(paths))
def test_patience_stops_after_that_many_kernels_without_a_lower_error(path, cv, patience):
    X, Y = path_data()
    full = HighlyAdaptiveRidgeCV(n_alphas=10, cv=cvs[cv], **paths[path]).fit(X, Y)
    m = HighlyAdaptiveRidgeCV(n_alphas=10, cv=cvs[cv], patience=patience, **paths[path]).fit(X, Y)
    i_best = int(np.argmin(m.kernel_mses_))
    assert len(m.kernel_mses_) == min(len(m.kernels), i_best + patience + 1)
    if path == "decays":
        assert len(m.kernel_mses_) < len(m.kernels)  # it stopped early on this data
    assert m.best.kernel == m.kernels[i_best]
    # the kernels that it evaluated have the same errors as in the walk over every kernel
    np.testing.assert_array_equal(m.kernel_mses_, full.kernel_mses_[: len(m.kernel_mses_)])


def test_alpha_grid_takes_the_smallest_eigenvalue_from_the_eigendecomposition():
    X, Y = path_data()
    kernel = HighlyAdaptiveRidge()
    K = kernel(X)
    _, min_eig = _prep(K, Y)
    np.testing.assert_allclose(
        kernel.alpha_grid(Y, 20, 1e-3, K=K, min_eig=min_eig), kernel.alpha_grid(Y, 20, 1e-3, K=K), rtol=1e-12
    )
