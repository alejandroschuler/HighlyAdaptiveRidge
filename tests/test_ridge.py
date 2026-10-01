import numpy as np
import pytest
from sklearn.linear_model import Ridge

from kernel_ridge import KernelRidge, RidgeRegressionCV, kernels, ridge_alpha_grid


def make_data(seed, n=40, p=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)) + 2
    Y = X @ rng.normal(size=p) + rng.normal(size=n) + 3
    X_ = rng.normal(size=(10, p)) + 2
    return X, Y, X_


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("alpha", [1e-2, 1.0, 1e2])
def test_ridge_alpha_is_the_kernel_ridge_penalty(seed, alpha):
    """Ridge(alpha=a) and kernel ridge with the linear kernel and alpha=a fit
    the same function, so the two alphas are one regularization parameter."""
    X, Y, X_ = make_data(seed)
    ridge = Ridge(alpha=alpha).fit(X, Y).predict(X_)
    krr = KernelRidge(kernels.Linear(), alpha=alpha)
    krr.fit(X, Y)
    np.testing.assert_allclose(krr.predict(X_), ridge, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("eps", [1e-3, 1e-1])
def test_top_of_ridge_grid_regularizes_fully(seed, eps):
    """At the largest penalty of the grid, every fitted value is within
    eps max|Y - mean(Y)| of mean(Y), which is the bound of appendix D."""
    X, Y, _ = make_data(seed)
    grid = ridge_alpha_grid(X, Y, n_alphas=50, eps=eps)
    fitted = Ridge(alpha=grid[-1]).fit(X, Y).predict(X)
    assert np.max(np.abs(fitted - Y.mean())) < eps * np.max(np.abs(Y - Y.mean()))
    assert len(grid) == 50 and np.all(np.diff(grid) > 0)


def test_ridge_grid_keeps_the_fixed_floor():
    """Unlike the kernel methods, ridge starts its grid at 1e-8, not at a fraction of
    the largest eigenvalue: its covariates are on raw scales (see ridge_alpha_grid)."""
    X, Y, _ = make_data(0)
    assert ridge_alpha_grid(X, Y)[0] == 1e-8


def test_ridge_cv_uses_five_folds_over_the_grid():
    X, Y, X_ = make_data(0, n=100)
    m = RidgeRegressionCV().fit(X, Y)
    assert m.search_.n_splits_ == 5
    assert m.alpha_ in m.alphas_
    np.testing.assert_allclose(m.predict(X_), Ridge(alpha=m.alpha_).fit(X, Y).predict(X_))
