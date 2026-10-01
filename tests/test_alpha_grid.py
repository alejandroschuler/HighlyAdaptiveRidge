"""Each kernel of a KernelRidgeCV gets its own alpha grid, by the paper's appendix D,
from 1e-12 times the largest eigenvalue of its kernel matrix up to lambda_0."""
import numpy as np
import pytest
from numpy.linalg import eigvalsh, norm
from sklearn.model_selection import KFold

from kernel_ridge import HighlyAdaptiveRidgeCV, KernelRidge, kernels


def appendix_d_grid(K, Y, n_alphas, eps, alpha_min=None):
    """lambda_0 = max_i ||K_i|| ||Y|| / (eps max_i |y_i|) - eig_n(K), and a log grid up
    to it from alpha_min, by default 1e-12 eig_1(K). K_i is row i of the kernel matrix K."""
    eig = eigvalsh(K)
    lam0 = np.max(norm(K, axis=1)) * norm(Y) / (eps * np.max(np.abs(Y))) - eig[0]
    return np.geomspace(1e-12 * eig[-1] if alpha_min is None else alpha_min, lam0, n_alphas)


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(60, 4))
    Y = np.sin(3 * X[:, 0]) + X[:, 1] * X[:, 2] + 0.1 * rng.normal(size=60)
    return X, Y


@pytest.mark.parametrize("path", [
    dict(decays=[0.05, 0.25, 1.0]),
    dict(depths=[1, 2, 4]),
    dict(order=1, decays=[0.05, 0.25, 1.0]),
], ids=["decay", "depth", "order1-decay"])
@pytest.mark.parametrize("cv", [None, KFold(3, shuffle=True, random_state=0)], ids=["loocv", "3-fold"])
def test_each_kernel_gets_its_own_appendix_d_grid(data, path, cv):
    X, Y = data
    m = HighlyAdaptiveRidgeCV(n_alphas=20, cv=cv, **path).fit(X, Y)
    assert len(m.alpha_grids_) == len(m.kernels) == len(m.kernel_mses_)
    for kernel, grid in zip(m.kernels, m.alpha_grids_):
        np.testing.assert_allclose(grid, appendix_d_grid(kernel(X), Y, n_alphas=20, eps=m.eps), rtol=1e-8)
    # The kernels differ, and so do the tops of their grids.
    tops = [grid[-1] for grid in m.alpha_grids_]
    assert len(set(np.round(np.log(tops), 6))) == len(tops)
    # The chosen alpha comes from the grid of the chosen kernel.
    i = int(np.argmin(m.kernel_mses_))
    assert m.best.kernel is m.kernels[i]
    assert np.any(np.isclose(m.alpha_grids_[i], m.best.alpha, rtol=1e-12, atol=0))


def test_early_stopping_records_only_the_evaluated_kernels(data):
    X, Y = data
    m = HighlyAdaptiveRidgeCV(n_alphas=20, decays=[0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12],
                              patience=1).fit(X, Y)
    assert len(m.alpha_grids_) == len(m.kernel_mses_) <= len(m.kernels)


@pytest.mark.parametrize("kernel", [
    kernels.HighlyAdaptiveRidge(), kernels.HighlyAdaptiveRidge(decay=0.05), kernels.MixedSobolev(),
    kernels.RadialBasis(1.0), kernels.Linear(),
], ids=["HAR", "HAR-decay", "mixed-Sobolev", "RBF", "linear"])
def test_the_grid_runs_from_1e_12_of_the_largest_eigenvalue_to_lambda_0(data, kernel):
    X, Y = data
    K = kernel(X)
    grid = kernel.alpha_grid(Y, 20, 1e-3, X=X)
    np.testing.assert_allclose(grid, appendix_d_grid(K, Y, 20, 1e-3), rtol=1e-10)
    np.testing.assert_allclose(grid[0], 1e-12 * eigvalsh(K)[-1], rtol=1e-10)


@pytest.mark.parametrize("cv", [None, KFold(3, shuffle=True, random_state=0)], ids=["loocv", "3-fold"])
def test_a_fixed_alpha_min_gives_the_grid_of_the_first_version(data, cv):
    X, Y = data
    kernel = kernels.HighlyAdaptiveRidge()
    np.testing.assert_allclose(
        kernel.alpha_grid(Y, 20, 1e-3, alpha_min=1e-8, X=X), appendix_d_grid(kernel(X), Y, 20, 1e-3, alpha_min=1e-8),
        rtol=1e-10,
    )
    m = HighlyAdaptiveRidgeCV(n_alphas=20, cv=cv, alpha_min=1e-8).fit(X, Y)
    assert m.alpha_grids_[0][0] == 1e-8


@pytest.mark.parametrize("cv, scales", [
    (None, [2.0 ** -40, 2.0 ** 40]),
    # The k-fold path solves the bordered system, whose border does not scale, so
    # its accuracy drops for a kernel far from unit scale. Smaller factors suffice.
    (KFold(3, shuffle=True, random_state=0), [2.0 ** -10, 2.0 ** 10]),
], ids=["loocv", "3-fold"])
def test_the_scale_of_the_kernel_does_not_change_the_fit(data, cv, scales):
    """Both ends of the grid scale with K, so K and s K give the same fit, with
    alpha scaled by s. A power of 2 scales the kernel exactly in floating point."""
    X, Y = data
    X_test = np.random.default_rng(1).uniform(size=(20, 4))
    fit = lambda s: HighlyAdaptiveRidgeCV(n_alphas=20, cv=cv, weights=lambda p: s * np.ones(p)).fit(X, Y)
    base = fit(1.0)
    for s in scales:
        m = fit(s)
        np.testing.assert_allclose(m.best.alpha / s, base.best.alpha, rtol=1e-10)
        np.testing.assert_allclose(m.predict(X_test), base.predict(X_test), rtol=1e-8)


def test_the_floor_gives_the_unpenalized_fit_of_a_full_rank_kernel(data):
    """At the bottom of the grid, the fit of a full-rank kernel is the interpolating
    fit (alpha = 0), so the grid reaches no penalty."""
    X, Y = data
    X_test = np.random.default_rng(1).uniform(size=(20, 4))
    kernel = kernels.HighlyAdaptiveRidge()
    K = kernel(X)
    eig = eigvalsh(K)
    assert eig[0] > 1e-8 * eig[-1]  # full rank, far above roundoff
    m = KernelRidge(kernel, alpha=kernel.alpha_grid(Y, 20, 1e-3, K=K)[0])
    m.fit(X, Y)
    n = len(Y)
    bordered = np.block([[K, np.ones((n, 1))], [np.ones((1, n)), np.zeros((1, 1))]])
    interpolant = np.linalg.solve(bordered, np.append(Y, 0.0))
    np.testing.assert_allclose(m.predict(X_test), KernelRidge._predict_kernel(kernel(X, X_test), interpolant), rtol=1e-6)
