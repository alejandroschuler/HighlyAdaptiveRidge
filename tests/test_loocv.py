"""The leave-one-out error of kernel ridge, on the eig path (fast.loocv_path) and the
brute path (KernelRidge.loocv), against a direct solve of the bordered system and
against refitting without each point.

Both paths compute the residual as c_i / [A^-1]_ii. The earlier form
(Y_i - Yhat_i) / (1 - h_i) divides two differences that both go to 0 with alpha, so its
error grew like 1 / alpha.
"""
import numpy as np
import pytest

from kernel_ridge import KernelRidge, KernelRidgeCV, kernels
from kernel_ridge.fast import loocv_path


def bordered(K, alpha):
    """A = [[K + alpha I, 1], [1', 0]], the matrix that KernelRidge.fit solves."""
    n = K.shape[0]
    return np.block([[K + alpha * np.eye(n), np.ones((n, 1))], [np.ones((1, n)), np.zeros((1, 1))]])


def loocv_bordered(K, Y, alpha):
    """The mean of (c_i / [A^-1]_ii)^2, from a direct solve of the bordered system."""
    n = len(Y)
    A_inv = np.linalg.inv(bordered(K, alpha))
    c = (A_inv @ np.append(Y, 0.0))[:n]
    return np.mean((c / np.diag(A_inv)[:n]) ** 2)


def loocv_refit(K, Y, alpha):
    """The definition: fit without point i, predict point i, for each i."""
    n = len(Y)
    R = np.empty(n)
    for i in range(n):
        keep = np.arange(n) != i
        coef = np.linalg.solve(bordered(K[np.ix_(keep, keep)], alpha), np.append(Y[keep], 0.0))
        R[i] = Y[i] - KernelRidge._predict_kernel(K[[i]][:, keep], coef)[0]
    return np.mean(R ** 2)


def loocv_brute(kernel, X, Y, K, alpha):
    m = KernelRidge(kernel, alpha=alpha)
    m.fit(X, Y, K=K)
    return m.loocv(Y)


rng = np.random.default_rng(0)
X = rng.uniform(size=(40, 4))
f = np.sin(3 * X[:, 0]) + X[:, 1] * X[:, 2]
OUTCOMES = {
    "centered": f + 0.1 * rng.normal(size=40),
    # A large mean next to a small spread, as on naval. The earlier form lost the
    # most digits here, because Y - Yhat is a difference of numbers of size |Y|.
    "offset": 10 + 0.1 * f + 0.01 * rng.normal(size=40),
}
# Kernels of full rank on X, with eig_n / eig_1 between 2e-6 and 1e-2.
KERNELS = {
    "HAR": kernels.HighlyAdaptiveRidge(),
    "HAR order 1": kernels.HighlyAdaptiveRidge(order=1),
    "HAR decay": kernels.HighlyAdaptiveRidge(decay=0.25),
    "mixed Sobolev": kernels.MixedSobolev(),
    "RBF": kernels.RadialBasis(1.0),
}
# alpha as a fraction of mean(diag K), from a large penalty down to none.
FRACTIONS = [1e-2, 1e-6, 1e-10, 1e-12, 1e-14, 0.0]


@pytest.mark.parametrize("y", OUTCOMES, ids=list(OUTCOMES))
@pytest.mark.parametrize("name", KERNELS, ids=list(KERNELS))
def test_both_paths_are_the_bordered_solve_down_to_alpha_0(name, y):
    kernel, Y = KERNELS[name], OUTCOMES[y]
    K = kernel(X)
    alphas = np.array(FRACTIONS) * np.mean(np.diag(K))
    expected = [loocv_bordered(K, Y, a) for a in alphas]
    np.testing.assert_allclose(loocv_path(K, Y, alphas)[0], expected, rtol=1e-9)
    np.testing.assert_allclose([loocv_brute(kernel, X, Y, K, a) for a in alphas], expected, rtol=1e-9)


@pytest.mark.parametrize("y", OUTCOMES, ids=list(OUTCOMES))
@pytest.mark.parametrize("name", ["HAR", "RBF"])
def test_the_bordered_solve_is_refitting_without_each_point(name, y):
    """c_i / [A^-1]_ii is the leave-one-out residual, also at alpha = 0, where the
    fit interpolates."""
    Y = OUTCOMES[y]
    K = KERNELS[name](X)
    for a in np.array(FRACTIONS) * np.mean(np.diag(K)):
        np.testing.assert_allclose(loocv_bordered(K, Y, a), loocv_refit(K, Y, a), rtol=1e-9)


@pytest.mark.parametrize("name", KERNELS, ids=list(KERNELS))
def test_loocv_goes_to_the_loocv_of_the_interpolant(name):
    """For a full-rank kernel, a tiny penalty gives the fit, and the LOOCV error, of no
    penalty. The penalty itself moves the error by about alpha / eig_n(K), which is 1e-8
    at 1e-12 mean(diag K) for the kernels with eig_n / eig_1 = 2e-6, hence the tolerance."""
    Y = OUTCOMES["offset"]
    K = KERNELS[name](X)
    md = np.mean(np.diag(K))
    mses = loocv_path(K, Y, [1e-12 * md, 1e-14 * md, 0.0])[0]
    np.testing.assert_allclose(mses, loocv_refit(K, Y, 0.0), rtol=1e-7)


@pytest.mark.parametrize("name", ["HAR", "RBF"])
def test_loocv_of_a_singular_kernel_is_the_bordered_solve(name):
    """Ten points appear twice, so K has a null space of dimension 10 and the
    coefficients grow like 1 / alpha along it. The earlier form computed Yhat from
    them, and gave 3.4 times the correct value at alpha = 1e-8 mean(diag K) on the HAR
    kernel."""
    X2 = np.vstack([X[:30], X[:10]])
    Y = OUTCOMES["centered"]
    kernel = KERNELS[name]
    K = kernel(X2)
    alphas = np.array([1e-2, 1e-4, 1e-6, 1e-8]) * np.mean(np.diag(K))
    expected = [loocv_bordered(K, Y, a) for a in alphas]
    np.testing.assert_allclose(loocv_path(K, Y, alphas)[0], expected, rtol=1e-6)
    np.testing.assert_allclose([loocv_brute(kernel, X2, Y, K, a) for a in alphas], expected, rtol=1e-6)


@pytest.mark.parametrize("name", KERNELS, ids=list(KERNELS))
def test_eig_and_brute_paths_choose_the_same_alpha(name):
    """Over a grid that reaches 1e-14 mean(diag K), both paths give the same LOOCV error
    and choose the same alpha."""
    Y = OUTCOMES["centered"]
    kernel = KERNELS[name]
    K = kernel(X)
    grid = [np.geomspace(1e-14 * np.mean(np.diag(K)), np.mean(np.diag(K)), 20)]
    eig = KernelRidgeCV([kernel], alphas=grid, method="eig").fit(X, Y)
    brute = KernelRidgeCV([kernel], alphas=grid, method="brute").fit(X, Y)
    np.testing.assert_allclose(eig.kernel_mses_, brute.kernel_mses_, rtol=1e-9)
    assert eig.best.alpha == brute.best.alpha
