"""The tuning of the Section 4 estimators, against direct computations."""
import numpy as np
import pytest

from estimators import SLUGS, load
from estimators.folds import folds
from estimators.hal import HALDepthPath
from estimators.kernel_path import KernelPathCV, loocv_fit
from highly_adaptive_regression import HighlyAdaptiveLassoCV
from kernel_ridge import KernelRidgeCV
from kernel_ridge.kernels import HighlyAdaptiveRidge, RadialBasis


@pytest.fixture
def data():
    rng = np.random.default_rng(1)
    X = rng.uniform(size=(80, 3))
    Y = np.sin(4 * X[:, 0]) + X[:, 1] * X[:, 2] + 0.2 * rng.normal(size=80)
    return X, Y


def direct_fold_risk(kernel, X, Y, splits, n_alphas, eps):
    """The mean over the folds of the validation MSE at the penalty that LOOCV
    chooses on each fold's training rows, from KernelRidgeCV."""
    errors = []
    for train, val in splits:
        m = KernelRidgeCV([kernel], n_alphas=n_alphas, eps=eps).fit(X[train], Y[train])
        errors.append(np.mean((m.predict(X[val]) - Y[val]) ** 2))
    return np.mean(errors)


def test_path_risk_is_the_mean_validation_error_at_the_loocv_penalty(data):
    X, Y = data
    splits = folds(len(Y), 0)
    kernels = [HighlyAdaptiveRidge(depth=d) for d in [1, 2, 3]]
    m = KernelPathCV(kernels, splits, n_alphas=20).fit(X, Y)
    for k, kernel in enumerate(kernels):
        assert m.cv_risk_[k] == pytest.approx(direct_fold_risk(kernel, X, Y, splits, 20, m.eps), rel=1e-8)
    assert m.best_ == int(np.argmin(m.cv_risk_))


def test_refit_is_the_loocv_fit_of_the_chosen_kernel(data):
    X, Y = data
    splits = folds(len(Y), 0)
    kernels = [RadialBasis(gamma=g) for g in [0.3, 3.0]]
    m = KernelPathCV(kernels, splits, n_alphas=20).fit(X, Y)
    ref = KernelRidgeCV([kernels[m.best_]], n_alphas=20, eps=m.eps).fit(X, Y)
    assert m.alpha_ == pytest.approx(ref.best.alpha, rel=1e-12)
    np.testing.assert_allclose(m.predict(X[:7]), ref.predict(X[:7]), rtol=1e-8, atol=1e-10)


def test_one_kernel_skips_the_folds(data):
    X, Y = data
    m = KernelPathCV([HighlyAdaptiveRidge()], folds(len(Y), 0), n_alphas=20).fit(X, Y)
    assert m.cv_risk_ == [] and m.best_ == 0
    coef, alphas, mses, j, _ = loocv_fit(HighlyAdaptiveRidge(), X, Y, 20, m.eps)
    assert m.alpha_ == alphas[j]


def test_patience_stops_the_walk(data):
    X, Y = data
    kernels = [RadialBasis(gamma=g) for g in [1.0, 1e3, 1e4, 1e5, 1e6]]  # narrower and narrower
    m = KernelPathCV(kernels, folds(len(Y), 0), n_alphas=10, patience=2).fit(X, Y)
    assert m.stop_ == "risk"
    assert len(m.cv_risk_) == int(np.argmin(m.cv_risk_)) + 3


def test_budget_stops_before_a_kernel_that_would_not_fit(data):
    X, Y = data
    kernels = [HighlyAdaptiveRidge(depth=d) for d in [1, 2, 3]]
    m = KernelPathCV(kernels, folds(len(Y), 0), n_alphas=10, budget=1e-9, cost=lambda k: 1.0).fit(X, Y)
    assert m.stop_ == "budget" and len(m.cv_risk_) == 1


def test_hal_takes_the_shared_folds(data):
    X, Y = data
    splits = folds(len(Y), 0)
    a = HighlyAdaptiveLassoCV(n_alphas=10, cv=splits, max_degree=2, n_jobs=1).fit(X, Y)

    class Splitter:
        def split(self, X):
            return iter(splits)

    b = HighlyAdaptiveLassoCV(n_alphas=10, cv=Splitter(), max_degree=2, n_jobs=1).fit(X, Y)
    np.testing.assert_array_equal(a.mse_path_, b.mse_path_)


def test_hal_depth_walk_keeps_the_depth_with_the_smallest_risk(data):
    X, Y = data
    m = HALDepthPath(folds(len(Y), 0), n_alphas=10).fit(X, Y)
    assert m.depth_ == m.depths_[int(np.argmin(m.cv_risk_))]
    assert m.stop_ in {"risk", "full depth"}
    ref = HighlyAdaptiveLassoCV(n_alphas=10, eps=m.eps, cv=m.folds, max_degree=m.depth_, n_jobs=1).fit(X, Y)
    np.testing.assert_allclose(m.predict(X[:5]), ref.predict(X[:5]))


@pytest.mark.parametrize("slug", SLUGS)
def test_every_estimator_fits_and_reports(data, slug):
    X, Y = data
    module = load(slug)
    m = module.learner(folds(len(Y), 0), 0, X.shape[1]).fit(X, Y)
    assert np.all(np.isfinite(m.predict(X[:5])))
    assert isinstance(module.chosen(m), dict)
    for row in module.tuning(m):
        assert np.isfinite(row["cv_risk"])
