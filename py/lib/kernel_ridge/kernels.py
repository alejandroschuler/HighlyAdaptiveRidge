"""
Kernel functions for use with kernel ridge regression. All functions are compiled 
with numba so that they run quickly and in parallel.
"""

import math
from dataclasses import dataclass
from numba import njit, prange
import numpy as np
from numpy.linalg import norm, eigvalsh


class Kernel:

    def alpha_grid(self, Y, n_alphas, eps, alpha_min=1e-8, K=None, X=None, min_eig=None):
        """
        see HAR paper appendix D. min_eig is the smallest eigenvalue of K, if it is
        already known; otherwise it is computed.
        """
        if K is None:
            K = self.kernel(X, X, equal=True)
        if min_eig is None:
            min_eig = np.min(eigvalsh(K))
        alpha_max = norm(Y) * np.max(norm(K, axis=1)) / (eps * np.max(np.abs(Y))) - min_eig
        return np.geomspace(alpha_min, alpha_max, num=n_alphas)


@dataclass
class Linear(Kernel):
    """K(x, x') = x'x. Kernel ridge regression with this kernel is ridge regression."""

    def __call__(self, X, X_test=None):
        if X_test is None:
            return self.kernel(X, X, equal=True)
        return self.kernel(X_test, X, equal=False)

    @staticmethod
    def kernel(X_test, X, equal):
        return X_test @ X.T


@dataclass
class RadialBasis(Kernel):
    gamma: float = 1

    def __call__(self, X, X_test=None):
        if X_test is None:
            return self.kernel(X, X, equal=True, gamma=self.gamma)
        return self.kernel(X_test, X, equal=False, gamma=self.gamma)

    @staticmethod
    @njit(parallel=True)
    def kernel(X_test, X, equal, gamma):
        n, d = X.shape
        n_test, d = X_test.shape
        
        K = np.empty((n_test, n), dtype=np.float64)
        for tr in prange(n):
            max_index = tr + 1 if equal else n_test
            for te in range(max_index):
                sum_val = 0.0
                for j in range(d):
                    x, x_te = X[tr,j], X_test[te,j]
                    sum_val += (x - x_te)**2
                element = np.exp(-sum_val * gamma)
                K[te, tr] = element
                if equal:
                    K[tr, te] = element
        return K
        

@dataclass
class HighlyAdaptiveRidge(Kernel):
    """
    The HAR kernel with one weight for each section size:

        K(x, x') = sum_i sum_{s nonempty} w_{|s|} h_{i,s}(x) h_{i,s}(x'),

    where h_{i,s} are the order-`order` basis functions with knot X_i, and |s| is
    the number of coordinates that a basis function depends on (|s_0| for the
    higher-order bases). Kernel ridge with this kernel is ridge regression on the
    bases with the penalty sum_{i,s} beta_{i,s}^2 / w_{|s|}, so w_k = 0 drops the
    sections with k coordinates. The empty section (the intercept) is not in the
    kernel, because KernelRidge fits an unpenalized intercept.

    The weights are built from three settings, which compose:
        weights: None for w_k = 1 (plain HAR), a sequence (w_1, ..., w_p), or a
            callable that takes p and returns one.
        decay: multiplies w_k by decay**k. decay = 1 changes nothing.
        depth: sets w_k = 0 for every k > depth. A negative or infinite depth
            keeps every section.

    section_weights(p) turns the settings into (w_0, ..., w_p), and har_kernel
    builds the kernel from those weights and the order.
    """
    depth: int = -1
    order: int = 0
    decay: float = 1.0
    weights: object = None

    def __call__(self, X, X_test=None):
        if X_test is None:
            return self.kernel(X, X, equal=True)
        return self.kernel(X, X_test, equal=False)

    def section_weights(self, p):
        """The weights (w_0, ..., w_p) of the sections with 0, ..., p coordinates. w_0 = 0."""
        if self.weights is None:
            w = np.ones(p)
        else:
            w = self.weights(p) if callable(self.weights) else self.weights
            w = np.array(w, dtype=np.float64)
            if w.shape != (p,):
                raise ValueError(
                    f"weights needs one entry for each section size 1, ..., {p}, not shape {w.shape}"
                )
        w = w * float(self.decay) ** np.arange(1, p + 1)
        if 0 <= self.depth < p:
            w[int(self.depth):] = 0.0
        return _check_section_weights(np.concatenate([[0.0], w]), p)

    def kernel(self, X, X_test, equal):
        """The kernel matrix, with a row for each point of X_test and a column for each point of X. The knots are the rows of X."""
        w = self.section_weights(np.shape(X)[1])
        return har_kernel(X, w, order=self.order, X_test=None if equal else X_test)


# The bits or spline factors of one block of knots take at most this many bytes.
_BLOCK_BYTES = 2 ** 28


def har_kernel(X, w, order=0, X_test=None, block_bytes=_BLOCK_BYTES):
    """The HAR kernel of order `order` with the section weights w = (w_0, ..., w_p).

    K[a, b] = sum_i sum_s w_{|s|} h_{i,s}(X_test[a]) h_{i,s}(X[b]), with the knots X_i
    the rows of X. X_test = None gives the symmetric kernel of X with itself, of
    which only half is computed. w_0 must be 0, because KernelRidge fits the intercept.

    The builder depends on the order and the weights:
        order 0: a knot that is below both points in c coordinates adds
            t[c] = sum_k w_k C(c, k), from a table. The comparisons of each point
            with each knot are packed into bits, 64 coordinates to a word, so c is
            a popcount of an AND. Any weights, at the same cost.
        order >= 1: a knot adds sum_k w_k e_k(a), where e_k is the k-th elementary
            symmetric polynomial of a_j = poly_j + U_j(x) U_j(x'). The polynomial
            part poly_j does not depend on the knot, and the spline factors
            U_j(x) = (x_j - X_ij)_+^order / order! are computed once for each point
            and knot. Geometric weights (plain HAR, or a decay without a depth) use
            the closed form scale * (prod_j (1 + rate a_j) - 1), in O(p) for each
            knot, computed without the cancellation of the - 1. Other weights use
            the recursion for e_1, ..., e_m, in O(p m) for each knot, where m is the
            largest size with a nonzero weight.

    The knots go in blocks, so that the bits or the factors of one block take at
    most block_bytes. In all, the bits take n (n + n_test) ceil(p / 64) 8 bytes and
    the factors n (n + n_test) p 8 bytes (n_test = 0 for the symmetric kernel). The
    sum over the knots continues across the blocks in knot order, so the block size
    does not change the kernel.
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    n, p = X.shape
    equal = X_test is None
    X_test = X if equal else np.ascontiguousarray(X_test, dtype=np.float64)
    if X_test.shape[1] != p:
        raise ValueError(f"X has {p} columns but X_test has {X_test.shape[1]}")
    w = _check_section_weights(w, p)
    if order != int(order) or order < 0:
        raise ValueError(f"order must be an integer >= 0, not {order!r}")
    order = int(order)
    n_points = n if equal else n + X_test.shape[0]
    K = np.zeros((X_test.shape[0], n), dtype=np.float64)
    if order == 0:
        table = _size_table(w)
        block = _block_size(n, n_points * ((p + 63) // 64) * 8, block_bytes)
        for k0 in range(0, n, block):
            knots = X[k0:k0 + block]
            B = _pack_bits(knots, X)
            B_test = B if equal else _pack_bits(knots, X_test)
            _add_order0(K, B, B_test, table, equal)
    else:
        fact_sq = np.array([float(math.factorial(k)) ** 2 for k in range(order + 1)])
        inv_fact = 1.0 / math.factorial(order)
        scale, rate = _geometric(w)
        max_size = int(np.flatnonzero(w)[-1])
        block = _block_size(n, n_points * p * 8, block_bytes)
        for k0 in range(0, n, block):
            knots = X[k0:k0 + block]
            U = _spline_factors(knots, X, order, inv_fact)
            U_test = U if equal else _spline_factors(knots, X_test, order, inv_fact)
            if rate > 0:
                _add_higher_geometric(K, U, U_test, X, X_test, order, fact_sq, rate, scale, equal)
            else:
                _add_higher_general(K, U, U_test, X, X_test, order, fact_sq, w, max_size, equal)
    if not np.all(np.isfinite(K)):
        raise ValueError("the kernel overflows a float; use a smaller decay or a depth")
    return K


def _check_section_weights(w, p):
    """w as a float array, after a check that it is a valid (w_0, ..., w_p) with w_0 = 0."""
    w = np.asarray(w, dtype=np.float64)
    if w.shape != (p + 1,):
        raise ValueError(f"the section weights need one entry for each size 0, ..., {p}, not shape {w.shape}")
    if not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError("the section weights must be finite and nonnegative")
    if w[0] != 0:
        raise ValueError("the weight w_0 of the empty section must be 0, because KernelRidge fits the intercept")
    if not np.any(w > 0):
        raise ValueError("every section weight is 0, so the kernel is 0")
    return w


def _block_size(n, bytes_per_knot, block_bytes):
    """The number of knots in a block: at most n, at least 1."""
    return int(max(1, min(n, block_bytes // max(bytes_per_knot, 1))))


def _size_table(w):
    """t[c] = sum_k w_k C(c, k) for c = 0, ..., p.

    A zero-order knot that is below both points in c coordinates adds t[c] to the kernel.
    An entry that does not fit in a float is inf, which is an error only if a knot uses it.
    """
    p = len(w) - 1
    sizes = [k for k in range(1, p + 1) if w[k] > 0]
    t = np.zeros(p + 1)
    for c in range(1, p + 1):
        ks = [k for k in sizes if k <= c]
        try:
            t[c] = math.fsum(float(w[k]) * math.comb(c, k) for k in ks)
        except OverflowError:
            # C(c, k) is too large for a float, but the weighted sum may not be.
            logs = [math.log(w[k]) + math.lgamma(c + 1) - math.lgamma(k + 1) - math.lgamma(c - k + 1) for k in ks]
            top = max(logs)
            try:
                t[c] = math.exp(top + math.log(math.fsum(math.exp(v - top) for v in logs)))
            except OverflowError:
                t[c] = np.inf
    return t


def _geometric(w):
    """(scale, rate) with w_k = scale * rate**k for k = 1, ..., p, or (0, 0) if the weights are not geometric.

    A small decay at high p makes the last weights fall below the smallest normal
    float, to a subnormal or to 0. Those weights count as geometric when the sequence
    of the larger weights is also below the smallest normal float at those sizes.
    The closed form then keeps their terms, which are negligible. A depth cap at a
    size where the weights are still normal stays a cap.
    """
    v = w[1:]
    tiny = np.finfo(np.float64).tiny
    normal = v >= tiny
    m = len(v) if normal.all() else int(np.argmin(normal))  # the first m weights are normal
    if m == 0 or normal[m:].any():
        return 0.0, 0.0
    if len(v) == 1:
        return float(v[0]), 1.0
    if m == 1:
        return 0.0, 0.0
    rate = v[1] / v[0]
    k = np.arange(len(v))
    if not np.allclose(v[:m], v[0] * rate ** k[:m], rtol=1e-12, atol=0):
        return 0.0, 0.0
    if not np.all(v[0] * rate ** k[m:] < tiny):
        return 0.0, 0.0
    return float(v[0] / rate), float(rate)


# uint64 constants only: numba promotes a mix of uint64 and int64 to float64.
_M1 = np.uint64(0x5555555555555555)
_M2 = np.uint64(0x3333333333333333)
_M4 = np.uint64(0x0F0F0F0F0F0F0F0F)
_H01 = np.uint64(0x0101010101010101)
_S1, _S2, _S4, _S56 = np.uint64(1), np.uint64(2), np.uint64(4), np.uint64(56)


@njit(inline="always")
def _popcount64(x):
    """The number of 1 bits in a uint64."""
    x = x - ((x >> _S1) & _M1)
    x = (x & _M2) + ((x >> _S2) & _M2)
    x = (x + (x >> _S4)) & _M4
    return np.int64((x * _H01) >> _S56)


@njit(parallel=True)
def _pack_bits(knots, P):
    """B[a, i] holds the bits 1(knots[i, j] <= P[a, j]) for j = 0, ..., p - 1.

    Bit j % 64 of word j // 64.
    """
    n_knots, p = knots.shape
    m = P.shape[0]
    W = (p + 63) // 64
    B = np.zeros((m, n_knots, W), dtype=np.uint64)
    for a in prange(m):
        for i in range(n_knots):
            for j in range(p):
                if knots[i, j] <= P[a, j]:
                    B[a, i, j // 64] |= np.uint64(1) << np.uint64(j % 64)
    return B


@njit(parallel=True)
def _add_order0(K, B, B_test, table, equal):
    """Add one block of knots to K. A knot is below both points in c coordinates,
    the popcount of the AND of their bits, and adds table[c]. Each sum continues
    from K, in knot order.
    """
    n, n_knots, W = B.shape
    n_test = B_test.shape[0]
    for tr in prange(n):
        max_index = tr + 1 if equal else n_test
        for te in range(max_index):
            sum_val = K[te, tr]
            for knot in range(n_knots):
                c = 0
                for w in range(W):
                    c += _popcount64(B[tr, knot, w] & B_test[te, knot, w])
                sum_val += table[c]
            K[te, tr] = sum_val
            if equal:
                K[tr, te] = sum_val


# The order >= 1 loops run over this many knots at once. The knots do not depend on
# each other, so the compiler can vectorize over them. It does so only when the
# inner loop reads 1D views, such as U[tr, j, c0:c0 + L], and not 3D indices.
_CHUNK = 64


@njit(parallel=True)
def _spline_factors(knots, P, order, inv_fact):
    """U[a, j, i] = (P[a, j] - knots[i, j])_+^order / order!, with inv_fact = 1 / order!.

    The knots are the last axis, so that the kernel loops read several knots at once.
    """
    n_knots, p = knots.shape
    m = P.shape[0]
    U = np.empty((m, p, n_knots), dtype=np.float64)
    for a in prange(m):
        for j in range(p):
            for i in range(n_knots):
                d = P[a, j] - knots[i, j]
                U[a, j, i] = d ** order * inv_fact if d >= 0 else 0.0
    return U


@njit(inline="always")
def _poly_part(poly, X, X_test, tr, te, order, fact_sq):
    """poly[j] = sum_{k=1}^{order} (x_j x'_j)^k / (k!)^2, the part that does not depend on the knot."""
    for j in range(poly.shape[0]):
        x_x_te = X[tr, j] * X_test[te, j]
        t = 0.0
        for k in range(1, order + 1):
            t += x_x_te ** k / fact_sq[k]
        poly[j] = t


@njit(parallel=True)
def _add_higher_geometric(K, U, U_test, X, X_test, order, fact_sq, rate, scale, equal):
    """Add one block of knots to K, for w_k = scale * rate**k. A knot adds
    sum_k w_k e_k(a) = scale * q, with q = prod_j (1 + t_j) - 1 and t_j = rate a_j.
    q comes from the recursion q <- q + t_j (1 + q), which has no subtraction, so
    it keeps its digits when rate is small and the product is close to 1. Each sum
    continues from K, in knot order.
    """
    n, p, n_knots = U.shape
    n_test = U_test.shape[0]
    for tr in prange(n):
        T = np.empty(p, dtype=np.float64)
        q = np.empty(_CHUNK, dtype=np.float64)
        max_index = tr + 1 if equal else n_test
        for te in range(max_index):
            _poly_part(T, X, X_test, tr, te, order, fact_sq)
            for j in range(p):
                T[j] = rate * T[j]  # the part of t_j that does not depend on the knot
            sum_val = K[te, tr]
            for c0 in range(0, n_knots, _CHUNK):
                L = min(_CHUNK, n_knots - c0)
                for l in range(L):
                    q[l] = 0.0
                for j in range(p):
                    T_j = T[j]
                    u = U[tr, j, c0:c0 + L]
                    v = U_test[te, j, c0:c0 + L]
                    for l in range(L):
                        t = T_j + rate * u[l] * v[l]
                        q[l] += t * (1.0 + q[l])
                for l in range(L):
                    sum_val += scale * q[l]
            K[te, tr] = sum_val
            if equal:
                K[tr, te] = sum_val


@njit(parallel=True)
def _add_higher_general(K, U, U_test, X, X_test, order, fact_sq, w, max_size, equal):
    """Add one block of knots to K, for any weights. A knot adds sum_k w_k e_k(a)
    for k <= max_size, from the recursion for e_1, ..., e_{max_size}. Each sum
    continues from K, in knot order.
    """
    n, p, n_knots = U.shape
    n_test = U_test.shape[0]
    for tr in prange(n):
        poly = np.empty(p, dtype=np.float64)
        a = np.empty(_CHUNK, dtype=np.float64)
        e = np.empty((max_size + 1, _CHUNK), dtype=np.float64)
        max_index = tr + 1 if equal else n_test
        for te in range(max_index):
            _poly_part(poly, X, X_test, tr, te, order, fact_sq)
            sum_val = K[te, tr]
            for c0 in range(0, n_knots, _CHUNK):
                L = min(_CHUNK, n_knots - c0)
                for l in range(L):
                    e[0, l] = 1.0
                for k in range(1, max_size + 1):
                    for l in range(L):
                        e[k, l] = 0.0
                for j in range(p):
                    poly_j = poly[j]
                    u = U[tr, j, c0:c0 + L]
                    v = U_test[te, j, c0:c0 + L]
                    for l in range(L):
                        a[l] = poly_j + u[l] * v[l]
                    for k in range(min(j + 1, max_size), 0, -1):
                        e_k = e[k]
                        e_below = e[k - 1]
                        for l in range(L):
                            e_k[l] += a[l] * e_below[l]
                for l in range(L):
                    for k in range(1, max_size + 1):
                        sum_val += w[k] * e[k, l]
            K[te, tr] = sum_val
            if equal:
                K[tr, te] = sum_val


@dataclass
class MixedSobolev(Kernel):
    """
    The kernel described in eq. 39, example B.9 of Zhang and Simon:
    https://projecteuclid.org/journals/electronic-journal-of-statistics/volume-17/issue-2/Regression-in-tensor-product-spaces-by-the-method-of-sieves/10.1214/23-EJS2188.full
    """
    def __call__(self, X, X_test=None):
        if X_test is None:
            return self.kernel(X, X, equal=True)
        return self.kernel(X_test, X, equal=False)

    @staticmethod
    @njit(parallel=True)
    def kernel(X_test, X, equal):
        n, d = X.shape
        n_test, d = X_test.shape
        factor = np.sinh(1)**(-d)
        
        K = np.empty((n_test, n), dtype=np.float64)
        for tr in prange(n):
            max_index = tr + 1 if equal else n_test
            for te in range(max_index):
                prod_val = 1.0
                for j in range(d):
                    x, x_te = X[tr,j], X_test[te,j]
                    prod_val *= np.cosh(min(x, x_te))
                    prod_val *= np.cosh(1-max(x, x_te))
                element = prod_val * factor
                K[te, tr] = element
                if equal:
                    K[tr, te] = element
        return K