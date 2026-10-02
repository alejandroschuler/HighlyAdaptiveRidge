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


def test_ridge_grid_starts_at_the_squared_floor_of_the_largest_singular_value():
    """Ridge factors X, not K = X X', so its floor is (1e-12 s_1)^2, with s_1 the largest
    singular value of the centered covariates. alpha_min fixes another floor."""
    X, Y, _ = make_data(0)
    s_1 = np.linalg.svd(X - X.mean(axis=0), compute_uv=False)[0]
    np.testing.assert_allclose(ridge_alpha_grid(X, Y)[0], (1e-12 * s_1) ** 2, rtol=1e-10)
    assert ridge_alpha_grid(X, Y, alpha_min=1e-8)[0] == 1e-8
    assert RidgeRegressionCV(alpha_min=1e-8).fit(X, Y).alphas_[0] == 1e-8


@pytest.mark.parametrize("scale", [2.0 ** -20, 2.0 ** 20])
def test_the_scale_of_the_covariates_does_not_change_the_ridge_fit(scale):
    """Both ends of the ridge grid scale with X'X, so X and s X give the same fit, with
    alpha scaled by s^2. A power of 2 scales X exactly in floating point."""
    X, Y, X_ = make_data(0, n=100)
    base = RidgeRegressionCV().fit(X, Y)
    m = RidgeRegressionCV().fit(scale * X, Y)
    np.testing.assert_allclose(m.alpha_ / scale ** 2, base.alpha_, rtol=1e-10)
    np.testing.assert_allclose(m.predict(scale * X_), base.predict(X_), rtol=1e-8)


def test_ridge_cv_uses_five_folds_over_the_grid():
    X, Y, X_ = make_data(0, n=100)
    m = RidgeRegressionCV().fit(X, Y)
    assert m.search_.n_splits_ == 5
    assert m.alpha_ in m.alphas_
    np.testing.assert_allclose(m.predict(X_), Ridge(alpha=m.alpha_).fit(X, Y).predict(X_))
