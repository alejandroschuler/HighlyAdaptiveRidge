from sklearn.base import BaseEstimator, RegressorMixin
from numba import njit, prange
import numpy as np
from scipy.linalg import solve

from timer import Timer


class KRR(BaseEstimator, RegressorMixin):

    def __init__(self, alpha=1, kernel=None, verbose=False):
        self.alpha = alpha # regularization strength
        self.X = None # training data
        self.timer = Timer(verbose)
        self.kernel = kernel

    def fit(self, X, Y):
        self.X = X
        n, _ = X.shape

        with self.timer.task('compute kernel'):
            self.K = self.kernel._kernel(self.X)

        with self.timer.task('solve equation'):
            self.B = np.vstack([
                np.hstack([
                    self.K + self.alpha*np.eye(n), np.ones((n,1))
                ]),
                np.hstack([
                    np.ones((1,n)), np.zeros((1,1))
                ])
            ])
            self.coef = solve(self.B, np.hstack([Y,np.zeros((1))]))

    def predict(self, X):
        with self.timer.task('compute test kernel'):
            k = self.kernel._kernel(self.X, X)
        return self._predict_kernel(k)

    def _predict_kernel(self, K):
        n, _ = K.shape
        return np.hstack([K, np.ones((n,1))]) @ self.coef

    def loocv(self, Y):
        """
        Uses LOOCV for efficiency.
        See https://is.mpg.de/fileadmin/user_upload/files/publications/pcw2005a7_[0].pdf#page=10.15

        Returns LOOCV MSE
        """
        n, _ = self.K.shape
        H = solve(
            self.B.T, 
            np.vstack([self.K, np.ones((1,n))])
        )
        Yhat = self._predict_kernel(self.K)
        R = (Y - Yhat) / (1- np.diag(H))
        return np.mean(R ** 2)

class KernelHAR(KRR):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kernel = HARKernel()



# ~~~ KERNELS ~~~


class HARKernel:
    @staticmethod
    def _kernel(X, X_test=None):
        # two separate helper methods are required because numba parallelization does not like the if-then 
        if X_test is None:
            return HARKernel._kernel_train(X)
        else:
            return HARKernel._kernel_test(X, X_test)

    @staticmethod
    @njit(parallel=True)
    def _kernel_train(X):
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
    def _kernel_test(X, X_test):
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