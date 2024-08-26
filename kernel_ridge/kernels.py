
from sklearn.metrics.pairwise import rbf_kernel
from numba import njit, prange
import numpy as np

class GaussianKernel:
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, X, X_test=None):
        if X_test is None:
            return rbf_kernel(X, X, gamma=self.gamma)
        return rbf_kernel(X_test, X, gamma=self.gamma)
            

class HighlyAdaptiveRidgeKernel:

    def __call__(self, X, X_test=None):
        # two separate helper methods are required because numba parallelization does not like the if-then 
        if X_test is None:
            return self.eval_train(X)
        else:
            return self.eval_test(X, X_test)

    @staticmethod
    @njit(parallel=True)
    def eval_train(X):
        n, d = X.shape
        K = np.empty((n, n), dtype=np.int64)

        for tr in prange(n):
            max_index = tr + 1
            for te in prange(max_index):
                sum_val = 0
                for knot in range(n):
                    count = 0
                    for feature in range(d):
                        if X[knot, feature] <= min(X[tr, feature], X[te, feature]):
                            count += 1
                    sum_val += 2 ** count
                K[te, tr] = sum_val
                K[tr, te] = sum_val
                
        return K - n # account for the n "intercepts"

    @staticmethod
    @njit(parallel=True)
    def eval_test(X, X_test):
        n, d = X.shape
        n_test, d = X_test.shape
        
        K = np.empty((n_test, n), dtype=np.int64)
        for tr in prange(n):
            for te in range(n_test):
                sum_val = 0
                for knot in range(n):
                    count = 0
                    for feature in range(d):
                        if X[knot, feature] <= min(X[tr, feature], X_test[te, feature]):
                            count += 1
                    sum_val += 2 ** count
                K[te, tr] = sum_val
                
        return K - n # account for the n "intercepts"