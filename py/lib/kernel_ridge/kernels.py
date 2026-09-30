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

    def alpha_grid(self, Y, n_alphas, eps, alpha_min=1e-8, K=None, X=None):
        """
        see HAR paper appendix D
        """
        if K is None:
            K = self.kernel(X, X, equal=True)
        alpha_max = norm(Y) * np.max(norm(K, axis=1)) / (eps * np.max(np.abs(Y))) - np.min(eigvalsh(K))
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
        if not np.all(np.isfinite(w)) or np.any(w < 0):
            raise ValueError("the section weights must be finite and nonnegative")
        if not np.any(w > 0):
            raise ValueError("every section weight is 0, so the kernel is 0")
        return np.concatenate([[0.0], w])

    def kernel(self, X, X_test, equal):
        """The kernel matrix, with a row for each point of X_test and a knot and a column for each point of X."""
        X = np.ascontiguousarray(X, dtype=np.float64)
        X_test = np.ascontiguousarray(X_test, dtype=np.float64)
        w = self.section_weights(X.shape[1])
        if self.order == 0:
            K = _har_kernel_order0(X, X_test, _size_table(w), equal)
        else:
            fact_sq = np.array([float(math.factorial(k)) ** 2 for k in range(self.order + 1)])
            scale, rate = _geometric(w)
            max_size = int(np.flatnonzero(w)[-1])
            K = _har_kernel_higher(X, X_test, self.order, fact_sq, w, max_size, rate, scale, equal)
        if not np.all(np.isfinite(K)):
            raise ValueError("the kernel overflows a float; use a smaller decay or a depth")
        return K


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
    """(scale, rate) with w_k = scale * rate**k for k = 1, ..., p, or (0, 0) if the weights are not geometric."""
    v = w[1:]
    if np.any(v <= 0):
        return 0.0, 0.0
    if len(v) == 1:
        return float(v[0]), 1.0
    rate = v[1] / v[0]
    if np.allclose(v, v[0] * rate ** np.arange(len(v)), rtol=1e-12, atol=0):
        return float(v[0] / rate), float(rate)
    return 0.0, 0.0


@njit(parallel=True)
def _har_kernel_order0(X, X_test, table, equal):
    n, d = X.shape
    n_test = X_test.shape[0]
    K = np.empty((n_test, n), dtype=np.float64)
    for tr in prange(n):
        max_index = tr + 1 if equal else n_test
        for te in range(max_index):
            sum_val = 0.0
            for knot in range(n):
                c = 0
                for j in range(d):
                    if X[knot, j] <= X[tr, j] and X[knot, j] <= X_test[te, j]:
                        c += 1
                sum_val += table[c]
            K[te, tr] = sum_val
            if equal:
                K[tr, te] = sum_val
    return K


@njit(parallel=True)
def _har_kernel_higher(X, X_test, order, fact_sq, w, max_size, rate, scale, equal):
    """Each coordinate j gives a knot the value a_j, and the knot adds
    sum_k w_k e_k(a_1, ..., a_d) to the kernel, where e_k is the k-th elementary
    symmetric polynomial. Geometric weights (rate > 0) use the closed form
    scale * (prod_j (1 + rate * a_j) - 1). Other weights use the recursion for
    e_1, ..., e_{max_size}.
    """
    n, d = X.shape
    n_test = X_test.shape[0]
    K = np.empty((n_test, n), dtype=np.float64)
    for tr in prange(n):
        a = np.empty(d, dtype=np.float64)
        term2 = np.empty(d, dtype=np.float64)
        e = np.empty(max_size + 1, dtype=np.float64)
        max_index = tr + 1 if equal else n_test
        for te in range(max_index):
            # the polynomial part of each coordinate does not depend on the knot
            for j in range(d):
                x_x_te = X[tr, j] * X_test[te, j]
                term2[j] = 0.0
                for k in range(1, order + 1):
                    term2[j] += x_x_te ** k / fact_sq[k]
            sum_val = 0.0
            for knot in range(n):
                for j in range(d):
                    diff = X[tr, j] - X[knot, j]
                    diff_te = X_test[te, j] - X[knot, j]
                    if (diff >= 0) and (diff_te >= 0):
                        a[j] = (diff * diff_te) ** order / fact_sq[order] + term2[j]
                    else:
                        a[j] = term2[j]
                if rate > 0:
                    prod_val = 1.0
                    for j in range(d):
                        prod_val *= 1.0 + rate * a[j]
                    sum_val += scale * (prod_val - 1.0)
                else:
                    e[0] = 1.0
                    for k in range(1, max_size + 1):
                        e[k] = 0.0
                    for j in range(d):
                        for k in range(min(j + 1, max_size), 0, -1):
                            e[k] += a[j] * e[k - 1]
                    for k in range(1, max_size + 1):
                        sum_val += w[k] * e[k]
            K[te, tr] = sum_val
            if equal:
                K[tr, te] = sum_val
    return K


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