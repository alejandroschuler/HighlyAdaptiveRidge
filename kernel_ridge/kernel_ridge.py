import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from timer import Timer
from . import kernels
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

class KernelRidge(BaseEstimator, RegressorMixin):

    def __init__(self, kernel, alpha=1, verbose=False):
        self.kernel = kernel
        self.alpha = alpha # regularization strength
        self.timer = Timer(verbose)
        self.X = None # training data
        self.K = None # kernel matrix

    def fit(self, X, Y, K=None):
        """
        Fit kernel ridge regression with an unregularized intercept
        See https://is.mpg.de/fileadmin/user_upload/files/publications/pcw2005a7_[0].pdf#page=10.15
        """
        self.X = X
        self.K = K
        n, _ = X.shape

        with self.timer.task('compute kernel'):
            if self.K is None:
                self.K = self.kernel(self.X)
            self.K_ = np.vstack([
                np.hstack([
                    self.K + self.alpha*np.eye(n), np.ones((n,1))
                ]),
                np.hstack([
                    np.ones((1,n)), np.zeros((1,1))
                ])
            ])
            Y_ = np.hstack([Y,np.zeros((1))])

        with self.timer.task('solve equation'):
            self.coef = self.solve(self.K_, Y_)

    @staticmethod
    def solve(A, B):
        try:
            ans = np.linalg.solve(A, B)
        except np.linalg.LinAlgError as e:
            if 'Singular matrix' in str(e): # this kills the theory but at least it returns something
                return np.linalg.pinv(A) @ B, "Warning: Singular matrix, solution using pseudoinverse."
            else:
                return f"Error: {str(e)}"
        return ans

    def predict(self, X, k=None):
        with self.timer.task('compute test kernel'):
            if k is None:
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
        H = self.solve(
            self.K_.T, 
            np.vstack([self.K, np.ones((1,n))])
        )
        Yhat = self._predict_kernel(self.K)
        R = (Y - Yhat) / (1- np.diag(H))
        return np.mean(R ** 2)


class KernelRidgeCV(KernelRidge, BaseEstimator, RegressorMixin):

    def __init__(
        self, kernels, alphas=None, 
        max_alpha_coef_norm=0.01, n_alphas=10, eps=1e-8, 
        verbose=False
    ):
        self.kernels=kernels
        self.verbose = verbose
        self.max_alpha_coef_norm = max_alpha_coef_norm # ||beta||2 (or smaller) desired for max alpha
        self.n_alphas = n_alphas
        self.eps = eps
        self.alphas = alphas

    def _pre_fit(self, X, Y):
        if self.alphas is None:
            self.alphas = [
                k.alpha_grid(X, Y, self.max_alpha_coef_norm, self.n_alphas, self.eps) 
                for k in self.kernels
            ]

    def fit(self, X, Y):
        self._pre_fit(X, Y)
        self.cv_results = []
        for kernel, alphas in zip(self.kernels, self.alphas):
            kernel_cv_results = []
            K = kernel(X) # compute kernel once for all alpha, huge time saver
            for alpha in alphas:
                m = KernelRidge(kernel=kernel, alpha=alpha, verbose=self.verbose)
                m.fit(X,Y, K=K)
                kernel_cv_results += [(m, m.loocv(Y))]

            fits, errors = zip(*kernel_cv_results)
            best_index = np.argmin(errors)
            if best_index == 0 and errors[0]/errors[1] < 0.95:
                print(f"Warning: selected regularization is the largest grid value for {kernel}")
            if best_index == len(fits)-1 and errors[-1]/errors[-2] < 0.95:
                print(f"Warning: selected regularization is the smallest grid value for {kernel}")
            self.cv_results += kernel_cv_results

        fits, errors = zip(*self.cv_results)
        best_index = np.argmin(errors)
        self.best = fits[best_index]

    def predict(self, X):
        return self.best.predict(X)


class HighlyAdaptiveRidgeCV(KernelRidgeCV):
    def __init__(self, depth=np.inf, order=0, **kwargs):
        super().__init__(kernels=[kernels.HighlyAdaptiveRidge(depth=depth, order=order)], **kwargs)


class RadialBasisKernelRidgeCV(KernelRidgeCV):
    def __init__(self, gammas, **kwargs):
        super().__init__(kernels=[kernels.RadialBasis(g) for g in gammas], **kwargs)



class ClippedMinMaxScaler(MinMaxScaler):
    def transform(self, X):
        return np.clip(super().transform(X), 0, 1)

class UnscaledMixedSobolevRidgeCV(KernelRidgeCV):
    def __init__(self, **kwargs):
        super().__init__(kernels=[kernels.MixedSobolev()], **kwargs)

class MixedSobolevRidgeCV(Pipeline):
    def __init__(self, **kwargs):
        super().__init__([
            ('scaler', ClippedMinMaxScaler()),
            ('learner', UnscaledMixedSobolevRidgeCV(**kwargs)),
        ])

    def __getattr__(self, name):
        if hasattr(self.named_steps['learner'], name):
            return getattr(self.named_steps['learner'], name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")