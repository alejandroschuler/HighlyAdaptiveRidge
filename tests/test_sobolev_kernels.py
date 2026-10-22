"""The paper's mixed Sobolev kernel (estimators/sobolev_depth.py): the
one-dimensional kernel of the usual norm, and the kernel on [0, 1]^p with
depth weights."""
import itertools

import numpy as np
import pytest

from estimators.sobolev_depth import UsualSobolevDepth, _basis, usual_factors, usual_halves
from kernel_ridge.kernels import MixedSobolev


def _points(n, p, seed):
    return np.random.default_rng(seed).uniform(size=(n, p))


def usual_kernel_1d(a, b, m):
    """G(a, b) at the pairs (a[i], b[i])."""
    a, b = np.ravel(a), np.ravel(b)
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    u, _ = usual_halves(lo, m)
    _, v = usual_halves(hi, m)
    return np.sum(u * v, axis=1)


def _product(X, Z, m):
    """The product kernel prod_j G(Z[i, j], X[k, j]), written out from the
    one-dimensional kernel, with a row for each point of Z and a column for
    each point of X."""
    nz, nx, p = len(Z), len(X), X.shape[1]
    a = np.repeat(Z[:, None, :], nx, axis=1).ravel()
    b = np.repeat(X[None, :, :], nz, axis=0).ravel()
    return usual_kernel_1d(a, b, m).reshape(nz, nx, p).prod(axis=2)


def test_usual_order_zero_is_the_cosh_kernel():
    """For t = 0 the full-depth kernel is the paper's kernel K_S less one."""
    X, Z = _points(15, 3, 6), _points(8, 3, 7)
    np.testing.assert_allclose(UsualSobolevDepth(order=0)(X), MixedSobolev()(X) - 1, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(UsualSobolevDepth(order=0)(X, Z), MixedSobolev()(X, Z) - 1, rtol=1e-12,
                               atol=1e-14)


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
    X = _points(40, 4, 9)
    K = _product(X, X, m)
    assert np.linalg.eigvalsh(K)[0] > -1e-12 * np.linalg.eigvalsh(K)[-1]


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
    np.testing.assert_allclose(UsualSobolevDepth(order=order)(X), _product(X, X, order + 1) - 1, rtol=1e-12)
    np.testing.assert_allclose(UsualSobolevDepth(order=order, depth=6)(X, Z), _product(X, Z, order + 1) - 1,
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
