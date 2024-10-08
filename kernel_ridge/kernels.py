"""
Kernel functions for use with kernel ridge regression. All functions are compiled 
with numba so that they run quickly and in parallel.
"""

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
    depth: int = -1
    order: int = 0

    def __call__(self, X, X_test=None):
        # this is done like this so that the numba functions can be simple and compile nicely
        depth = min([X.shape[1], self.depth])
        if X_test is None:
            return self.kernel(X, X, depth, order=self.order, equal=True) 
        return self.kernel(X, X_test, depth, order=self.order, equal=False)
        
    @staticmethod
    @njit(parallel=True)
    def kernel(X, X_test, depth, order, equal):
        n, d = X.shape
        n_test, d = X_test.shape
        t_fac_sq = fact_seq(order)**2
        
        K = np.empty((n_test, n), dtype=np.float64)
        for tr in prange(n):
            max_index = tr + 1 if equal else n_test
            for te in range(max_index):
                sum_val = 0
                for knot in range(n):
                    prod_val = 1
                    term2 = np.empty((d), dtype=np.float64)
                    for j in range(d):
                        x, x_te, x_knot = X[tr,j], X_test[te,j], X[knot,j]
                        diff = x - x_knot
                        diff_te = x_te - x_knot
                        if (diff>=0) and (diff_te>=0):
                            term1 = (diff * diff_te)**order / t_fac_sq[-1]
                        else:
                            term1 = 0
                        if knot == 0: 
                            # this only needs to be computed once for each j across all knots
                            term2[j] = sum([(x*x_te)**k / t_fac_sq[k] for k in np.arange(1,order+1)]) 
                        prod_val *= (term1 + term2[j] + 1)
                    sum_val += prod_val - 1
                K[te, tr] = sum_val
                if equal:
                    K[tr, te] = sum_val
        return K

LOOKUP_TABLE = np.array([1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800, 479001600, 6227020800, 87178291200, 1307674368000, 20922789888000, 355687428096000, 6402373705728000, 121645100408832000, 2432902008176640000], dtype='int64')
@njit
def fact_seq(n):
    if n <= 20:
        return LOOKUP_TABLE[:n+1]
    seq = np.empty((n+1), dtype='int64')
    seq[:21] = LOOKUP_TABLE
    for i in np.arange(21, n+1):
        seq[i] = seq[i-1]*i
    return seq


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