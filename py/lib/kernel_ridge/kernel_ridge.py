import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from .timer import Timer
from . import kernels
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import Ridge

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

        The leave-one-out residual is R_i = c_i / [A^-1]_ii, with c the kernel coefficients
        and A the bordered matrix of _prep_fit. It equals (Y_i - Yhat_i) / (1 - h_i), but
        that form divides two differences that both go to 0 with alpha, so in double
        precision its error grows like 1 / alpha. fast.py has the derivation.

        Returns LOOCV MSE
        """
        n, _ = self.K.shape
        K_, _ = self._prep_fit(self.K, Y)
        A_inv = self._solve(K_, np.eye(n + 1))
        R = self.coef[:n] / np.diag(A_inv)[:n]
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
    """Kernel ridge with the kernel and alpha chosen by cross-validation.

    patience: None evaluates every kernel. An integer patience >= 1 walks the
        kernels in order, builds each one only when it gets there, and stops
        after `patience` kernels in a row that do not lower the best CV error
        so far. Order the kernels along a path, for example by depth.

    alpha_min: the bottom of each grid that Kernel.alpha_grid builds. None
        starts it at kernels.ALPHA_FLOOR times the largest eigenvalue of the
        kernel matrix; a number fixes it (1e-8 gives the earlier fixed floor).

    After fit, kernel_mses_ holds the best CV error of each kernel that was
    evaluated, in order (inf where none was finite), and alpha_grids_ holds the
    alpha grid of each of those kernels. A kernel given no grid gets its own,
    from its own kernel matrix, by Kernel.alpha_grid (the paper's appendix D).
    The entry is None for a kernel whose eigendecomposition failed.
    """

    def __init__(
        self, kernels, alphas=None,
        n_alphas=50, eps=1e-3,
        cv=None, verbose=False, method='eig', patience=None, alpha_min=None,
    ):
        self.kernels = kernels
        self.alphas = [None for k in kernels] if alphas is None else alphas
        self.n_alphas = n_alphas
        self.eps = eps # largest value allowable in Yhat/sup(Y) at max regularization
        self.cv = cv
        self.verbose = verbose
        # 'eig': single-eigendecomposition LOOCV (fast, numerically identical to 'brute'
        #        for leave-one-out). 'brute': original per-alpha bordered solve.
        self.method = method
        self.patience = patience
        self.alpha_min = alpha_min

    def fit(self, X, Y):
        if self.patience is not None and not (isinstance(self.patience, (int, np.integer)) and self.patience >= 1):
            raise ValueError(f"patience must be None or an integer >= 1, not {self.patience!r}")
        # The eig path reproduces the bordered-system LOOCV exactly and is far faster,
        # but only covers cv=None (leave-one-out), which every paper experiment uses.
        # Any explicit cv (k-fold), or an eig path that cannot select a model (a kernel so
        # ill-conditioned that every alpha's LOOCV is non-finite), falls back to the
        # original per-alpha solve.
        if self.method == 'eig' and self.cv is None:
            self._fit_eig(X, Y)
            if self.best is not None:
                return self
        return self._fit_brute(X, Y)

    def _stop(self):
        """True when the last `patience` kernels did not lower the best CV error."""
        if self.patience is None:
            return False
        return len(self.kernel_mses_) - 1 - int(np.argmin(self.kernel_mses_)) >= self.patience

    def _fit_brute(self, X, Y):
        self.models = []
        self.kernel_mses_ = []
        self.alpha_grids_ = []
        errors = []
        for kernel, alphas in zip(self.kernels, self.alphas):
            K = kernel(X) # compute kernel once for all alpha, huge time saver
            if alphas is None:
                alphas = kernel.alpha_grid(
                    Y, K=K,
                    n_alphas = self.n_alphas,
                    eps = self.eps,
                    alpha_min = self.alpha_min,
                )
            self.alpha_grids_.append(np.asarray(alphas, dtype=float))
            kernel_errors = []
            for alpha in alphas:
                m = KernelRidge(kernel=kernel, alpha=alpha, verbose=self.verbose)
                m.fit(X,Y, K=K)
                self.models.append(m)
                e = m.cv(Y, cv=self.cv)
                kernel_errors.append(e if np.isfinite(e) else np.inf)
            errors.extend(kernel_errors)
            self.kernel_mses_.append(float(min(kernel_errors, default=np.inf)))
            if self._stop():
                break

        self.best = self.models[np.argmin(errors)]
        return self

    def _fit_eig(self, X, Y):
        from .fast import _prep, loocv_path, coef_at
        self.best = None
        self.kernel_mses_ = []
        self.alpha_grids_ = []
        for kernel, alphas in zip(self.kernels, self.alphas):
            K = kernel(X) # compute kernel once for all alpha, huge time saver
            mse = np.inf
            grid = None
            try:
                cache, min_eig, max_eig = _prep(K, Y)
            except np.linalg.LinAlgError:
                cache = None  # eigendecomposition failed for this kernel; let brute handle it
            if cache is not None:
                if alphas is None:
                    alphas = kernel.alpha_grid(
                        Y, K=K,
                        n_alphas=self.n_alphas,
                        eps=self.eps,
                        alpha_min=self.alpha_min,
                        min_eig=min_eig,
                        max_eig=max_eig,
                    )
                alphas = np.asarray(alphas, dtype=float)
                grid = alphas
                mses, cache = loocv_path(K, Y, alphas, cache=cache)
                finite = np.isfinite(mses)
                if finite.any():  # else every alpha non-finite (ill-conditioned); fall back to brute
                    idx = np.flatnonzero(finite)
                    j = int(idx[np.argmin(mses[idx])])
                    mse = float(mses[j])
                    if mse < min(self.kernel_mses_, default=np.inf):
                        m = KernelRidge(kernel=kernel, alpha=float(alphas[j]), verbose=self.verbose)
                        m.X = X
                        m.K = K
                        m.coef = coef_at(cache, float(alphas[j]))
                        self.best = m
            self.kernel_mses_.append(mse)
            self.alpha_grids_.append(grid)
            if self._stop():
                break
        return self

    def predict(self, X):
        return self.best.predict(X)


class HighlyAdaptiveRidgeCV(KernelRidgeCV):
    """HAR with alpha chosen by cross-validation, and optionally the depth or the decay.

    order, depth, decay and weights are the settings of kernels.HighlyAdaptiveRidge,
    which turns depth, decay and weights into one weight for each section size.
    kernels.har_kernel then builds each kernel from those weights and the order.

    depths: a sequence of depths to choose from, in increasing order. Each kernel
        uses the given decay.
    decays: a sequence of decays to choose from, from small to large. Each kernel
        uses the given depth. At high p a useful scale is decay = gamma / p, because
        a knot below both points in all p coordinates then adds
        (1 + gamma / p)^p - 1 < e^gamma to the kernel.
    Give at most one of the two, because early stopping (patience, see
    KernelRidgeCV) walks one path. With neither, there is one kernel.
    """

    def __init__(self, depth=np.inf, order=0, decay=1.0, weights=None,
                 depths=None, decays=None, patience=None, **kwargs):
        if depths is not None and decays is not None:
            raise ValueError("give depths or decays, not both: early stopping walks one path")
        if depths is not None:
            ks = [kernels.HighlyAdaptiveRidge(depth=d, order=order, decay=decay, weights=weights) for d in depths]
        elif decays is not None:
            ks = [kernels.HighlyAdaptiveRidge(depth=depth, order=order, decay=r, weights=weights) for r in decays]
        else:
            ks = [kernels.HighlyAdaptiveRidge(depth=depth, order=order, decay=decay, weights=weights)]
        super().__init__(kernels=ks, patience=patience, **kwargs)


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


def ridge_alpha_grid(X, Y, n_alphas=50, eps=1e-3, alpha_min=None):
    """The regularization grid of the paper's appendix D, for ridge regression.

    Ridge does not penalize its intercept, so it solves the ridge problem of
    the centered covariates and the centered outcome, which is kernel ridge
    with the linear kernel of the centered covariates and no intercept. The
    bound of appendix D is for that problem, so the grid comes from it: at the
    largest value, every fitted value is within eps max|Y - mean(Y)| of mean(Y).

    alpha_min = None starts the grid at (kernels.ALPHA_FLOOR * s_1)^2, where s_1
    is the largest singular value of the centered covariates. This is the
    floor of the kernel methods, applied to the singular values of X instead
    of the eigenvalues of K = X X'. The kernel methods factor K, and roundoff
    hides the directions below about 1e-13 eig_1(K). The SVD solve of the ridge
    fits factors X, and it resolves directions down to about 1e-30 eig_1. Ridge
    needs them: its covariates are on raw scales, and on naval some directions
    that carry the signal have eigenvalues 1e-13 to 1e-17 times eig_1. A floor
    of ALPHA_FLOOR * eig_1 cut them off and made the test MSE on naval about 6
    times worse. A number fixes the floor instead; 1e-8 gives the earlier grid.
    """
    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean()
    if alpha_min is None:
        alpha_min = (kernels.ALPHA_FLOOR * np.linalg.norm(Xc, 2)) ** 2
    linear = kernels.Linear()
    return linear.alpha_grid(Yc, n_alphas=n_alphas, eps=eps, alpha_min=alpha_min, K=linear(Xc))


class RidgeRegressionCV(BaseEstimator, RegressorMixin):
    """Ridge regression with its penalty chosen by cv-fold cross-validation
    over the grid of the paper's appendix D.

    The penalty alpha is the lambda of ||Y - X beta - b||^2 + lambda ||beta||^2,
    where the intercept b is not penalized. scikit-learn's Ridge uses this
    scale, so its alpha is the same parameter as the alpha of KernelRidge with
    the linear kernel.

    The grid reaches alpha = (1e-12 s_1)^2 (see ridge_alpha_grid; alpha_min
    sets another floor), where X'X can be ill-conditioned (naval has constant
    columns, for example). The default Cholesky solve then loses accuracy, so
    the fits use the SVD solve, which is exact for every alpha.
    """

    def __init__(self, n_alphas=50, eps=1e-3, cv=5, alpha_min=None):
        self.n_alphas = n_alphas
        self.eps = eps
        self.cv = cv
        self.alpha_min = alpha_min

    def fit(self, X, Y):
        self.alphas_ = ridge_alpha_grid(X, Y, n_alphas=self.n_alphas, eps=self.eps, alpha_min=self.alpha_min)
        self.search_ = GridSearchCV(
            Ridge(solver="svd"), {"alpha": self.alphas_}, cv=self.cv,
            scoring="neg_mean_squared_error",
        ).fit(X, Y)
        self.alpha_ = self.search_.best_params_["alpha"]
        return self

    def predict(self, X):
        return self.search_.predict(X)
