"""Each kernel of a KernelRidgeCV gets its own alpha grid, by the paper's appendix D."""
import numpy as np
import pytest
from numpy.linalg import eigvalsh, norm
from sklearn.model_selection import KFold

from kernel_ridge import HighlyAdaptiveRidgeCV


def appendix_d_grid(K, Y, n_alphas, eps, alpha_min=1e-8):
    """lambda_0 = max_i ||K_i|| ||Y|| / (eps max_i |y_i|) - eig_n(K), and a log grid
    from alpha_min up to it. K_i is row i of the kernel matrix K."""
    lam0 = np.max(norm(K, axis=1)) * norm(Y) / (eps * np.max(np.abs(Y))) - np.min(eigvalsh(K))
    return np.geomspace(alpha_min, lam0, n_alphas)


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
