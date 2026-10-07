"""The kernels of the mixed Sobolev comparators (estimators/sobolev_kernels.py)."""
import itertools

import numpy as np
import pytest

from estimators.sobolev_kernels import (
    AnchoredSobolev, UsualSobolev, _basis, usual_factors, usual_kernel_1d,
)
from kernel_ridge.kernels import MixedSobolev


def _points(n, p, seed):
    return np.random.default_rng(seed).uniform(size=(n, p))


def _anchored_factor(a, b, order):
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    return lo if order == 0 else a * b + lo ** 2 * hi / 2 - lo ** 3 / 6


@pytest.mark.parametrize("order", [0, 1])
def test_anchored_depth_is_the_sum_over_sections(order):
    """Below the full depth, the kernel is the sum over the sections of at most
    D coordinates of the product of the factors, written out by brute force."""
    X, Z = _points(7, 4, 0), _points(5, 4, 1)
    for depth in [1, 2, 3, 4, -1]:
        D = 4 if depth < 0 else depth
        c = _anchored_factor(Z[:, None, :], X[None, :, :], order)
        want = np.zeros((5, 7))
        for k in range(1, D + 1):
            for s in itertools.combinations(range(4), k):
                want += np.prod(c[:, :, list(s)], axis=2)
        got = AnchoredSobolev(order=order, depth=depth)(X, Z)
        np.testing.assert_allclose(got, want, rtol=1e-13)


@pytest.mark.parametrize("order", [0, 1])
def test_anchored_full_depth_is_the_product(order):
    X = _points(20, 6, 2)
    c = _anchored_factor(X[:, None, :], X[None, :, :], order)
    np.testing.assert_allclose(AnchoredSobolev(order=order)(X), np.prod(1 + c, axis=2) - 1, rtol=1e-13)
    np.testing.assert_allclose(AnchoredSobolev(order=order, depth=6)(X), AnchoredSobolev(order=order)(X),
                               rtol=1e-13)


def test_anchored_first_order_factor_is_the_note_integral():
    """c_1(a, b) = ab + int_0^1 (a - u)_+ (b - u)_+ du, the first-order R_1 - 1."""
    u, w = np.polynomial.legendre.leggauss(40)
    rng = np.random.default_rng(3)
    for a, b in rng.uniform(size=(20, 2)):
        lo = min(a, b)
        nodes, weights = lo * (u + 1) / 2, lo * w / 2   # the integrand is zero above a ^ b
        integral = np.sum(weights * (a - nodes) * (b - nodes))
        assert _anchored_factor(a, b, 1) == pytest.approx(a * b + integral, rel=1e-13)
        got = AnchoredSobolev(order=1, depth=1)(np.array([[b]]), np.array([[a]]))[0, 0]
        assert got == pytest.approx(a * b + integral, rel=1e-13)


@pytest.mark.parametrize("kernel", [AnchoredSobolev(order=0, depth=2), AnchoredSobolev(order=1),
                                    UsualSobolev(order=0), UsualSobolev(order=1)])
def test_cross_kernel_is_a_block_of_the_kernel(kernel):
    X, Z = _points(9, 5, 4), _points(6, 5, 5)
    full = kernel(np.vstack([X, Z]))
    np.testing.assert_allclose(kernel(X, Z), full[9:, :9], rtol=1e-12)
    np.testing.assert_allclose(kernel(X), full[:9, :9], rtol=1e-12)


def test_usual_order_zero_is_the_cosh_kernel():
    """For t = 0 the construction gives the paper's kernel K_S."""
    X, Z = _points(15, 3, 6), _points(8, 3, 7)
    np.testing.assert_allclose(UsualSobolev(order=0)(X), MixedSobolev()(X), rtol=1e-12)
    np.testing.assert_allclose(UsualSobolev(order=0)(X, Z), MixedSobolev()(X, Z), rtol=1e-12)


def _derivative(a, b, k, m):
    """The k-th derivative in a of the one-dimensional kernel G(a, b), for a != b."""
    A, N1 = usual_factors(m)
    if a <= b:
        return float((_basis([a], k, m) @ A)[0] @ (_basis([b], 0, m) @ N1)[0])
    return float((_basis([b], 0, m) @ A)[0] @ (_basis([a], k, m) @ N1)[0])


@pytest.mark.parametrize("m", [1, 2, 3])
def test_usual_kernel_reproduces(m):
    """<f, G(., b)> = f(b) under <f, g> = sum_{k <= m} int f^(k) g^(k), by
    Gauss-Legendre quadrature on [0, b] and [b, 1], where G(., b) is smooth."""
    u, w = np.polynomial.legendre.leggauss(60)
    # f(x) = exp(0.7 x) sin(3 x) + x^2 and its derivatives, from complex exponentials
    r = 0.7 + 3j

    def f(x, k):
        poly = [x ** 2, 2 * x, 2 + 0 * x][k] if k <= 2 else 0 * x
        return np.imag(r ** k * np.exp(r * x)) + poly

    for b in [0.13, 0.5, 0.91]:
        total = 0.0
        for lo, hi in [(0.0, b), (b, 1.0)]:
            nodes = lo + (hi - lo) * (u + 1) / 2
            weights = (hi - lo) * w / 2
            for k in range(m + 1):
                g = np.array([_derivative(a, b, k, m) for a in nodes])
                total += np.sum(weights * f(nodes, k) * g)
        assert total == pytest.approx(f(np.array(b), 0), rel=1e-10)


@pytest.mark.parametrize("m", [1, 2])
def test_usual_kernel_is_symmetric_and_positive(m):
    a = np.random.default_rng(8).uniform(size=60)
    G = usual_kernel_1d(np.repeat(a, 60), np.tile(a, 60), m).reshape(60, 60)
    np.testing.assert_allclose(G, G.T, rtol=1e-12)
    assert np.linalg.eigvalsh(G)[0] > -1e-12 * np.linalg.eigvalsh(G)[-1]
    K = UsualSobolev(order=m - 1)(_points(40, 4, 9))
    assert np.linalg.eigvalsh(K)[0] > -1e-12 * np.linalg.eigvalsh(K)[-1]


# The usual-norm kernels with depth weights (estimators/sobolev_depth.py).

from estimators.sobolev_depth import UsualSobolevDepth  # noqa: E402


@pytest.mark.parametrize("order", [0, 1])
def test_usual_depth_is_the_sum_over_sections(order):
    """Below the full depth, the kernel is the sum over the sections of at most
    D coordinates of the product of the factors G - 1, written out by brute force."""
    X, Z = _points(7, 4, 10), _points(5, 4, 11)
    a = np.repeat(Z[:, None, :], 7, axis=1).ravel()
    b = np.repeat(X[None, :, :], 5, axis=0).ravel()
    g0 = (usual_kernel_1d(a, b, order + 1) - 1).reshape(5, 7, 4)
    for depth in [1, 2, 3, 4, -1]:
        D = 4 if depth < 0 else depth
        want = np.zeros((5, 7))
        for k in range(1, D + 1):
            for s in itertools.combinations(range(4), k):
                want += np.prod(g0[:, :, list(s)], axis=2)
        np.testing.assert_allclose(UsualSobolevDepth(order=order, depth=depth)(X, Z), want, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("order", [0, 1])
def test_usual_depth_full_is_the_product_less_one(order):
    X, Z = _points(20, 6, 12), _points(9, 6, 13)
    np.testing.assert_allclose(UsualSobolevDepth(order=order)(X), UsualSobolev(order=order)(X) - 1, rtol=1e-12)
    np.testing.assert_allclose(UsualSobolevDepth(order=order, depth=6)(X, Z), UsualSobolev(order=order)(X, Z) - 1,
                               rtol=1e-12)


@pytest.mark.parametrize("order", [0, 1])
def test_usual_depth_cross_block_and_positive(order):
    X, Z = _points(30, 5, 14), _points(6, 5, 15)
    for depth in [1, 2, 5]:
        kernel = UsualSobolevDepth(order=order, depth=depth)
        full = kernel(np.vstack([X, Z]))
        np.testing.assert_allclose(kernel(X, Z), full[30:, :30], rtol=1e-12, atol=1e-14)
        eig = np.linalg.eigvalsh(kernel(X))
        assert eig[0] > -1e-12 * eig[-1]


@pytest.mark.parametrize("m", [1, 2])
def test_usual_kernel_integrates_to_one(m):
    """int_0^1 G(a, u) du = 1 for every a, so the constants are orthogonal to G - 1."""
    u, w = np.polynomial.legendre.leggauss(60)
    for a in [0.0, 0.2, 0.55, 1.0]:
        total = 0.0
        for lo, hi in [(0.0, a), (a, 1.0)]:
            if hi <= lo:
                continue
            nodes, weights = lo + (hi - lo) * (u + 1) / 2, (hi - lo) * w / 2
            total += np.sum(weights * usual_kernel_1d(np.full_like(nodes, a), nodes, m))
        assert total == pytest.approx(1.0, rel=1e-12)
