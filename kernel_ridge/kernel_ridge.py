import numpy as np
from scipy.linalg import solve
from sklearn.base import BaseEstimator, RegressorMixin
from timer import Timer
from .kernels import HighlyAdaptiveRidgeKernel


class KernelRidge(BaseEstimator, RegressorMixin):

    def __init__(self, kernel, alpha=1, verbose=False):
        self.alpha = alpha # regularization strength
        self.X = None # training data
        self.timer = Timer(verbose)
        self.kernel = kernel

    def fit(self, X, Y):
        self.X = X
        n, _ = X.shape

        with self.timer.task('compute kernel'):
            self.K = self.kernel(self.X)

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
            k = self.kernel(self.X, X)
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


class KernelRidgeCV(KernelRidge, BaseEstimator, RegressorMixin):

    def __init__(
        self, kernel, alphas=None, 
        max_alpha_coef_norm=0.01, n_alphas=10, eps=1e-8, 
        verbose=False
    ):
        self.kernel=kernel
        self.verbose = verbose
        self.max_alpha_coef_norm = max_alpha_coef_norm # ||beta||2 (or smaller) desired for max alpha
        self.n_alphas = n_alphas
        self.eps = eps
        self.alphas = alphas

    def _pre_fit(self, X, Y):
        if self.alphas is None:
            self.alphas = self.kernel.alpha_grid(X, Y, self.max_alpha_coef_norm, self.n_alphas, self.eps)

    def fit(self, X, Y):
        self._pre_fit(X, Y)
        self.models = []
        for alpha in self.alphas:
            m = KernelRidge(kernel=self.kernel, alpha=alpha, verbose=self.verbose)
            m.fit(X,Y)
            self.models += [(m, m.loocv(Y))]

        models, errors = zip(*self.models)
        self.best = models[np.argmin(errors)]

    def predict(self, X):
        return self.best.predict(X)


class HighlyAdaptiveRidge(KernelRidge):
    def __init__(self, *args, depth=np.inf, **kwargs):
        super().__init__(*args, kernel=HighlyAdaptiveRidgeKernel(depth=depth), **kwargs)

class HighlyAdaptiveRidgeCV(KernelRidgeCV):
    def __init__(self, *args, depth=np.inf, **kwargs):
        super().__init__(*args, kernel=HighlyAdaptiveRidgeKernel(depth=depth), **kwargs)


