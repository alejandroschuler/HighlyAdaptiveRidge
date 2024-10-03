import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from timer import Timer
from . import kernels
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold

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
        if self.K is None:
            with self.timer.task('compute kernel'):
                self.K = self.kernel(self.X)
        with self.timer.task('solve equation'):
            self.coef = self._solve(*self._prep_fit(self.K, Y))
        
    def _prep_fit(self, K, Y):
        n = len(Y)
        K_ = np.vstack([
            np.hstack([ K + self.alpha*np.eye(n), np.ones((n,1))  ]),
            np.hstack([ np.ones((1,n))          , np.zeros((1,1)) ])
        ])
        Y_ = np.hstack([Y, np.zeros((1))])
        return K_, Y_

    @staticmethod
    def _solve(A, B):
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
        return self._predict_kernel(k, self.coef)

    @staticmethod
    def _predict_kernel(k, coef):
        n, _ = k.shape
        return np.hstack([k, np.ones((n,1))]) @ coef

    def loocv(self, Y):
        """
        Uses LOOCV for efficiency.
        This technically doesn't work for HAR since the kernel is data-adaptive- actually need to recompute kernel.
        See https://is.mpg.de/fileadmin/user_upload/files/publications/pcw2005a7_[0].pdf#page=10.15

        Returns LOOCV MSE
        """
        n, _ = self.K.shape
        K_, _ = self._prep_fit(self.K, Y)
        H = self._solve(
            K_.T, 
            np.vstack([self.K, np.ones((1,n))])
        )
        Yhat = self._predict_kernel(self.K, self.coef)
        R = (Y - Yhat) / (1- np.diag(H))
        return np.mean(R ** 2)
    
    def cv(self, Y, cv=None):
        if cv is None:
            return self.loocv(Y)
        errors = []
        for tr, te in cv.split(self.K):
            coef = self._solve(*self._prep_fit(self.K[np.ix_(tr,tr)], Y[tr]))
            Yhat = self._predict_kernel(self.K[np.ix_(te,tr)], coef)
            errors.append(np.mean((Yhat - Y[te]) ** 2))
        return np.mean(errors)


class KernelRidgeCV(KernelRidge, BaseEstimator, RegressorMixin):

    def __init__(
        self, kernels, alphas=None,
        n_alphas=50, eps=1e-3, 
        cv=None, verbose=False
    ):
        self.kernels = kernels
        self.alphas = [None for k in kernels] if alphas is None else alphas
        self.n_alphas = n_alphas
        self.eps = eps # largest value allowable in Yhat/sup(Y) at max regularization
        self.cv = cv
        self.verbose = verbose

    def _errors(self, Y, cv):
        errors = [m.cv(Y, cv=cv) for m in self.models]

    def fit(self, X, Y):
        self.models = []
        for kernel, alphas in zip(self.kernels, self.alphas):
            K = kernel(X) # compute kernel once for all alpha, huge time saver
            if alphas is None:
                alphas = kernel.alpha_grid(
                    Y, K=K, 
                    n_alphas = self.n_alphas, 
                    eps = self.eps
                ) 
            for alpha in alphas:
                m = KernelRidge(kernel=kernel, alpha=alpha, verbose=self.verbose)
                m.fit(X,Y, K=K)
                self.models.append(m)

        errors = self._errors(Y, cv=self.cv)
        self.best = self.models[np.argmin(errors)]

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

    @property
    def scaler(self):
        return self.__dict__['steps'][0][1]