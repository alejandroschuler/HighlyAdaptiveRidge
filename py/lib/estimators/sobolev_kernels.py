"""The kernels of the mixed Sobolev comparators that notes/sobolev-mars.tex adds
to Section 4, on the unit cube, for the order t = 0 or 1 of HAR.

AnchoredSobolev is the limit of the scaled HAR kernel of order t for uniform
covariates, with HAR's depth weights (notes/higher-order-sobolev.tex, the
remark on section weights):

    K(x, x') = sum_{1 <= |s| <= D} prod_{j in s} c_t(x_j, x'_j),

where c_t = R_t - 1, with R_t the one-dimensional anchored kernel of that note:

    c_0(a, b) = a ^ b,
    c_1(a, b) = ab + (a ^ b)^2 (a v b) / 2 - (a ^ b)^3 / 6,

with a ^ b and a v b the minimum and the maximum. Like the HAR kernels, it
leaves out the empty section, whose function is the intercept that the fit
leaves unpenalized. At the full depth D = p, K = prod_j (1 + c_t) - 1.

UsualSobolev is the product over the coordinates of the reproducing kernel of
the Sobolev space of order m = t + 1 on [0, 1] under the usual norm, whose
square is sum_{k=0}^{m} int_0^1 (f^(k))^2. For t = 0 this is the paper's
kernel K_S, a product of cosh functions. The one-dimensional kernel is the
Green's function of L = sum_{k=0}^{m} (-1)^k D^(2k) under the natural boundary
conditions of that norm (Thomas-Agnan 1996), computed here from a real
fundamental system of L:

    G(a, b) = u(a ^ b)' M v(a v b),

where the entries of u span the solutions that satisfy the boundary
conditions at 0, those of v span the solutions that satisfy them at 1, and the
m x m matrix M makes G and its first 2m - 2 derivatives continuous at a = b,
with a jump of (-1)^m in the derivative of order 2m - 1, which gives the
reproducing property.

Both are product kernels with no sum over knots, so a kernel matrix costs
O(n^2 p D) operations below the full depth and O(n^2 p) at it.
"""
import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numba import njit, prange

from kernel_ridge.kernels import Kernel


# ---------------------------------------------------------------------------
# The anchored kernel

@njit(inline="always")
def _anchored_factor(a, b, order):
    """c_t(a, b) = R_t(a, b) - 1 for t = order, 0 or 1."""
    lo = min(a, b)
    if order == 0:
        return lo
    hi = max(a, b)
    return a * b + lo * lo * (0.5 * hi - lo / 6.0)


@njit(parallel=True)
def _anchored_kernel(X_test, X, equal, order, depth):
    """K[te, tr] = sum_{1 <= k <= depth} e_k(c_t(X_test[te], X[tr])), with e_k the
    elementary symmetric polynomial of degree k of the p factors. A depth below
    1 or at least p keeps every section, through the closed form
    prod_j (1 + c_j) - 1, computed as q <- q + c_j (1 + q), which has no
    subtraction."""
    n, p = X.shape
    n_test = X_test.shape[0]
    full = depth < 1 or depth >= p
    D = p if full else depth
    K = np.empty((n_test, n), dtype=np.float64)
    for tr in prange(n):
        e = np.empty(D + 1, dtype=np.float64)
        max_index = tr + 1 if equal else n_test
        for te in range(max_index):
            if full:
                q = 0.0
                for j in range(p):
                    c = _anchored_factor(X_test[te, j], X[tr, j], order)
                    q += c * (1.0 + q)
                value = q
            else:
                e[0] = 1.0
                for k in range(1, D + 1):
                    e[k] = 0.0
                for j in range(p):
                    c = _anchored_factor(X_test[te, j], X[tr, j], order)
                    for k in range(min(j + 1, D), 0, -1):
                        e[k] += c * e[k - 1]
                value = 0.0
                for k in range(1, D + 1):
                    value += e[k]
            K[te, tr] = value
            if equal:
                K[tr, te] = value
    return K


@dataclass
class AnchoredSobolev(Kernel):
    """The anchored mixed Sobolev kernel of order t = `order` (0 or 1) on [0, 1]^p,
    with the sections of more than `depth` coordinates left out. A negative depth,
    or one of at least p, keeps every section."""
    order: int = 0
    depth: int = -1

    def __post_init__(self):
        if self.order not in (0, 1):
            raise ValueError(f"order must be 0 or 1, not {self.order!r}")
        if self.depth == 0 or self.depth != int(self.depth):
            raise ValueError(f"depth must be a nonzero integer, not {self.depth!r}")

    def __call__(self, X, X_test=None):
        """The kernel matrix, with a row for each point of X_test and a column for each point of X."""
        X = np.ascontiguousarray(X, dtype=np.float64)
        if X_test is None:
            return _anchored_kernel(X, X, True, self.order, int(self.depth))
        X_test = np.ascontiguousarray(X_test, dtype=np.float64)
        return _anchored_kernel(X_test, X, False, self.order, int(self.depth))


# ---------------------------------------------------------------------------
# The kernel of the usual norm

def _roots(m):
    """The roots of the characteristic polynomial sum_{k=0}^m (-1)^k r^(2k) of L,
    one from each conjugate pair (imaginary part >= 0), with the real roots
    kept as they are. There are 2m roots in all, none repeated."""
    coef = np.zeros(2 * m + 1)
    for k in range(m + 1):
        coef[2 * k] = (-1) ** k  # the coefficient of r^(2k), lowest first
    r = np.roots(coef[::-1])
    half = r[r.imag > 1e-12]
    real = np.sort(r[np.abs(r.imag) <= 1e-12].real)
    return half, real


def _basis(x, k, m):
    """The k-th derivatives of the real fundamental system of L at the points x:
    an array of shape (len(x), 2m). For a complex root r the system has
    Re exp(r x) and Im exp(r x), whose k-th derivatives are Re and Im of
    r^k exp(r x); a real root r gives exp(r x)."""
    half, real = _roots(m)
    x = np.asarray(x, dtype=np.float64)[:, None]
    cols = []
    if half.size:
        z = half[None, :] ** k * np.exp(half[None, :] * x)
        cols += [z.real, z.imag]
    if real.size:
        cols.append(real[None, :] ** k * np.exp(real[None, :] * x))
    return np.concatenate(cols, axis=1)


def _boundary(x, m):
    """The natural boundary conditions of the usual norm at the point x, as an
    m x 2m matrix that the coefficients of a solution must annihilate:
    B_l(g) = sum_{k=l+1}^{m} (-1)^(k-1-l) g^(2k-1-l)(x) = 0 for l = 0, ..., m-1."""
    rows = []
    for l in range(m):
        row = np.zeros(2 * m)
        for k in range(l + 1, m + 1):
            row += (-1) ** (k - 1 - l) * _basis([x], 2 * k - 1 - l, m)[0]
        rows.append(row)
    return np.array(rows)


def _null(B):
    """An orthonormal basis of the null space of B, as columns."""
    _, s, vt = np.linalg.svd(B)
    rank = int(np.sum(s > 1e-10 * s[0]))
    return vt[rank:].T


@lru_cache(maxsize=None)
def usual_factors(m):
    """(A, N1) for the order m: the kernel's halves at a point x are
    u_M(x) = A' y(x) and v(x) = N1' y(x), with y(x) the fundamental system at x,
    and G(a, b) = u_M(a ^ b) . v(a v b). A = N0 M, with N0 and N1 the null
    spaces of the boundary conditions at 0 and 1.

    M solves the conditions at the diagonal, which are linear in M:
    u(b)' M v^(k)(b) - u^(k)(b)' M v(b) = 0 for k = 1, ..., 2m-2, and (-1)^m
    for k = 2m-1, at several points b. The least-squares residual is checked,
    since the system is consistent exactly when the construction is right."""
    N0, N1 = _null(_boundary(0.0, m)), _null(_boundary(1.0, m))
    if N0.shape[1] != m or N1.shape[1] != m:
        raise RuntimeError(f"the boundary conditions of order {m} do not leave {m} solutions at each end")
    points = np.linspace(0.1, 0.9, 7)
    rows, rhs = [], []
    u0 = _basis(points, 0, m) @ N0
    v0 = _basis(points, 0, m) @ N1
    for k in range(1, 2 * m):
        uk = _basis(points, k, m) @ N0
        vk = _basis(points, k, m) @ N1
        for i in range(len(points)):
            rows.append(np.outer(u0[i], vk[i]).ravel() - np.outer(uk[i], v0[i]).ravel())
            rhs.append((-1.0) ** m if k == 2 * m - 1 else 0.0)
    rows, rhs = np.array(rows), np.array(rhs)
    vec, *_ = np.linalg.lstsq(rows, rhs, rcond=None)
    residual = np.max(np.abs(rows @ vec - rhs))
    if residual > 1e-9:
        raise RuntimeError(f"the conditions at the diagonal have no exact solution (residual {residual:.2e})")
    M = vec.reshape(m, m)
    return N0 @ M, N1


def usual_halves(x, m):
    """The halves (u_M, v) of the one-dimensional kernel at the points x, each
    of shape (len(x), m)."""
    A, N1 = usual_factors(m)
    y = _basis(np.ravel(x), 0, m)
    return y @ A, y @ N1


def usual_kernel_1d(a, b, m):
    """G(a, b) at the pairs (a[i], b[i]); for the tests and the checks."""
    a, b = np.ravel(a), np.ravel(b)
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    u, _ = usual_halves(lo, m)
    _, v = usual_halves(hi, m)
    return np.sum(u * v, axis=1)


@njit(parallel=True)
def _usual_kernel(X_test, X, U_test, V_test, U, V, equal):
    """K[te, tr] = prod_j U[lo, j] . V[hi, j], where lo is whichever of the two
    points has the smaller coordinate j and hi the other."""
    n, p = X.shape
    n_test = X_test.shape[0]
    m = U.shape[2]
    K = np.empty((n_test, n), dtype=np.float64)
    for tr in prange(n):
        max_index = tr + 1 if equal else n_test
        for te in range(max_index):
            value = 1.0
            for j in range(p):
                g = 0.0
                if X_test[te, j] <= X[tr, j]:
                    for c in range(m):
                        g += U_test[te, j, c] * V[tr, j, c]
                else:
                    for c in range(m):
                        g += U[tr, j, c] * V_test[te, j, c]
                value *= g
            K[te, tr] = value
            if equal:
                K[tr, te] = value
    return K


@dataclass
class UsualSobolev(Kernel):
    """The mixed Sobolev kernel of order t + 1, t = `order`, on [0, 1]^p under the
    usual norm: the product over the coordinates of the reproducing kernel of
    the Sobolev space of order t + 1 on [0, 1] with the norm whose square is
    sum_{k <= t+1} int (f^(k))^2. order = 0 gives the paper's cosh kernel."""
    order: int = 1

    def __post_init__(self):
        if self.order != int(self.order) or self.order < 0:
            raise ValueError(f"order must be an integer >= 0, not {self.order!r}")

    def _halves(self, X):
        n, p = X.shape
        u, v = usual_halves(X.ravel(), int(self.order) + 1)
        m = u.shape[1]
        return np.ascontiguousarray(u.reshape(n, p, m)), np.ascontiguousarray(v.reshape(n, p, m))

    def __call__(self, X, X_test=None):
        """The kernel matrix, with a row for each point of X_test and a column for each point of X."""
        X = np.ascontiguousarray(X, dtype=np.float64)
        U, V = self._halves(X)
        if X_test is None:
            return _usual_kernel(X, X, U, V, U, V, True)
        X_test = np.ascontiguousarray(X_test, dtype=np.float64)
        U_test, V_test = self._halves(X_test)
        return _usual_kernel(X_test, X, U_test, V_test, U, V, False)
