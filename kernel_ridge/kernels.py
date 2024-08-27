
from sklearn.metrics.pairwise import rbf_kernel
from numba import njit, prange
import numpy as np


class Kernel:
    def alpha_grid(self, alpha_max, n_alphas, eps):
        return np.geomspace(alpha_max, alpha_max * eps, num=n_alphas)


class GaussianKernel(Kernel):
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, X, X_test=None):
        if X_test is None:
            return rbf_kernel(X, X, gamma=self.gamma)
        return rbf_kernel(X_test, X, gamma=self.gamma)
            
    def alpha_grid(self, X, Y, max_alpha_coef_norm, n_alphas, eps):
        """
        Computes ||H^T Y|| using a low-dim approximate feature map
        see: https://en.wikipedia.org/wiki/Radial_basis_function_kernel#Approximations
        """
        d_ = 40
        n, d = X.shape
        
        # create an approximate feature map
        W = np.random.normal(0, 1/np.sqrt(2*self.gamma), size=(d, d_))
        WX =  X @ W
        cos_WX = np.cos(WX)  
        sin_WX = np.sin(WX) 
        H = np.hstack((cos_WX, sin_WX)) / np.sqrt(d_)  

        alpha_max = np.linalg.norm(H.T @ Y)  / (max_alpha_coef_norm * np.max(np.abs(Y)))
        return super().alpha_grid(alpha_max, n_alphas, eps)


class HighlyAdaptiveRidgeKernel(Kernel):

    def __init__(self, depth=np.inf, order=0):
        self.depth = depth
        self.order = order

    def alpha_grid(self, X, Y, max_alpha_coef_norm, n_alphas, eps):
        """
        see: https://chatgpt.com/share/01b765c4-8fc3-41e7-b5af-9276d67be2e8
        each element of H^T @ Y is at most sum(|Y|) in magnitude
        so ||H^T @ Y|| <= sqrt(d) sum(|Y|)
        """
        n, p = X.shape
        alpha_max = np.sqrt(float(n*(2**p-1))) * np.sum(np.abs(Y)) / (max_alpha_coef_norm * np.max(np.abs(Y)))
        return super().alpha_grid(alpha_max, n_alphas, eps)
    
    def __call__(self, X, X_test=None):
        depth = min([X.shape[1], self.depth])
        if X_test is None:
            return self.kernel(X, X, depth, equal=True) 
        return self.kernel(X, X_test, depth, equal=False)

    @staticmethod
    @njit(parallel=True)
    def kernel(X, X_test, depth, equal):
        n, d = X.shape
        n_test, d = X_test.shape
        
        K = np.empty((n_test, n), dtype=np.float64)
        for tr in prange(n):
            max_index = tr + 1 if equal else n_test
            for te in range(max_index):
                sum_val = 0
                for knot in range(n):
                    count = 0
                    for feature in range(d):
                        if X[knot, feature] <= min(X[tr, feature], X_test[te, feature]):
                            count += 1
                    sum_val += comb_sum(count, depth)
                    # sum_val += 2**count
                K[te, tr] = sum_val
                if equal:
                    K[tr, te] = sum_val
        return K


    @staticmethod
    @njit(parallel=True)
    def kernel_t(X, X_test, depth, equal, t=0):
        n, d = X.shape
        n_test, d = X_test.shape
        t_fac_sq = fact_seq(t)**2
        
        K = np.empty((n_test, n), dtype=np.float64)
        for tr in prange(n):
            max_index = tr + 1 if equal else n_test
            for te in range(max_index):
                sum_val = 0
                for knot in range(n):
                    prod_val = 1
                    term2 = np.empty((d), dtype=np.float64)
                    for j in range(d):
                        x, x_te, x_knot =X[tr,j], X_test[te,j], X[knot,j]
                        diff = x - x_knot
                        diff_te = x_te - x_knot
                        if (diff>0) and (diff_te>0):
                            term1 = (diff * diff_te)**t / t_fac_sq[-1]
                        else:
                            term1 = 0
                        if knot == 0: 
                            # this only needs to be computed once for each j across all knots
                            term2[j] = sum([(x*x_te)**k / t_fac_sq[k] for k in np.arange(1,t+1)]) 
                        prod_val *= (term1 + term2[j] + 1)
                    sum_val += prod_val
                K[te, tr] = sum_val
                if equal:
                    K[tr, te] = sum_val
        return K

@njit
def comb_sum(n, k):
    """
    sum_i=1^k (n Choose i) 
    """
    bincoef, total = 1, 0
    for i in np.arange(1, k+1):
        bincoef = bincoef * (n-i+1) / i
        total += bincoef
    return total

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