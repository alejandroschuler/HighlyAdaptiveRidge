from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso, LassoCV, RidgeCV, Ridge
from sklearn.linear_model._coordinate_descent import _alpha_grid
from sklearn.preprocessing import StandardScaler

from numba import njit, prange
import numpy as np
from scipy.linalg import solve

from timer import Timer


class HABaseCV:
    """
    Highly Adaptive Base (HABase) class. Implements the Highly Adaptive Lasso/Ridge algorithms.
    """

    @classmethod
    def _basis_products(cls, arr, index=0, current=None, result=None):
        """
        Recursive helper function for computing basis products.
        
        Args:
            arr (ndarray): Array of boolean values.
            index (int): Current index for recursion (default: 0).
            current (ndarray): Current basis product (default: None).
            result (list): List to store the computed basis products (default: None).

        Returns:
            list: List of computed basis products.
        """
        
        if result is None:
            result = []
        if current is None:
            current = np.ones_like(arr[0], dtype=bool)

        if index == len(arr):
            result.append(current)
        else:
            cls._basis_products(arr, index + 1, current & arr[index], result)
            cls._basis_products(arr, index + 1, current, result)

        return result

    def _bases(self, X):
        """
        Computes the basis functions for the given knots and input data.
        Args:
            X (ndarray): Input data.
        Returns:
            ndarray: Array of computed basis functions.
        """
        one_way_bases = np.stack([
            np.less_equal.outer(self.knots[:,j], X[:,j])
            for j in range(self.knots.shape[1])
        ])
        bases = self._basis_products(one_way_bases)
        
        return np.concatenate(bases[:-1]).T

    def _pre_fit(self, X,Y):
        pass

    def fit(self, X, Y):
        """
        Fits the HA_ model to the given input data and target values.
        Args:
            X (ndarray): Input data.
            Y (ndarray): Target values.

        """
        self._pre_fit(X,Y)
        self.knots = X
        self.regression.fit(self._bases(X), Y)

    def predict(self, X):
        """
        Predicts the target values for the given input data.

        Args:
            X (ndarray): Input data.

        Returns:
            ndarray: Array of predicted target values.

        """
        return self.regression.predict(self._bases(X)) 


class HALCV(HABaseCV):

    def __init__(self, *args, **kwargs):
        self.regression = LassoCV(*args, **kwargs)


class HARCV(HABaseCV):

    def __init__(
        self, *args, kernel=True, verbose=False,
        n_alphas=100, eps=1e-3, alphas=None,
        cv=None, **kwargs
        ):
        self.scaler = StandardScaler(with_mean=True, with_std=False)
        self.regression = RidgeCV(*args, cv=cv, **kwargs)
        self.verbose = verbose
        self.n_alphas = n_alphas
        self.eps = eps
        self.alphas = alphas
        self.cv = cv

        if kernel is True:
            self.fit = self.fit_kernel
            self.predict = self.predict_kernel

    def _pre_fit(self, X, Y):
        '''
        RidgeCV uses a very small grid by default and it probably does not use the same path algorithm
        as LassoCV. Moreover the CV in RidgeCV is LOOCV since the LOOCV error can be exactly computed for 
        ridge from the model fit on full data.
        '''
        if self.alphas is None:
            # n, d = X.shape
            # m = Ridge(alpha=0.01)
            # m.fit(X,Y)
            # mse = np.mean((m.predict(X) - Y)**2)
            # R2 = 1 - np.mean(mse/np.var(Y))
            # alpha = (n * (3/2)**d) * (1-R2) / R2 
            # self.alphas = alpha * np.array([2**k for k in range(-4,1)])
            self.alphas = _alpha_grid(X,Y, l1_ratio=1e-3, eps=self.eps, n_alphas=self.n_alphas)
        self.regression.alphas = self.alphas

    def fit_kernel(self, X, Y):
        self._pre_fit(X, Y)

        self.models = []
        for alpha in self.alphas:
            m = KernelHAR(alpha=alpha, verbose=self.verbose)
            m.fit(X,Y)
            self.models += [(m, m.loocv(Y))]

        models, errors = zip(*self.models)
        self.best = models[np.argmin(errors)]

    def predict_kernel(self, X):
        # return self.search.best_estimator_.predict(X)
        return self.best.predict(X)


class KernelHAR(BaseEstimator, RegressorMixin):

    def __init__(self, alpha=1, verbose=False):
        self.alpha = alpha # regularization strength
        self.X = None # training data
        self.timer = Timer(verbose)

    def fit(self, X, Y):
        self.X = X
        n, _ = X.shape

        with self.timer.task('compute kernel'):
            self.K = self._kernel(self.X)

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
            k = self._kernel(self.X, X)
        return self._predict_kernel(k)

    def _predict_kernel(self, K):
        n, _ = K.shape
        return np.hstack([K, np.ones((n,1))]) @ self.coef

    @staticmethod
    def _kernel(X_train, X_test=None):
        # two separate helper methods are required because numba parallelization does not like the if-then 
        if X_test is None:
            return KernelHAR._kernel_train(X_train)
        else:
            return KernelHAR._kernel_test(X_train, X_test)

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
    def _kernel_test(X_train, X_test):
        n_train, d = X_train.shape
        n_test, d = X_test.shape
        
        K = np.empty((n_test, n_train), dtype=np.int64)
        for tr in prange(n_train):
            for te in range(n_test):
                sum_val = 0
                for knot in range(n_train):
                    count = 0
                    for feature in range(d):
                        if X_train[knot, feature] <= min(X_train[tr, feature], X_test[te, feature]):
                            count += 1
                    sum_val += 2 ** count
                K[te, tr] = sum_val
                
        return K - n_train # account for the n "intercepts"

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




#### TESTS ####



# ~~~ Test kernel function and centering ~~~

# import numpy as np
# from highly_adaptive_regression import KernelHAR, HARCV, HALCV
# from sklearn.preprocessing import KernelCenterer, StandardScaler

# n, n_ ,d  = 10, 4, 3
# X = np.random.rand(n, d)  
# X_ = np.random.rand(n_, d)   
# Y = np.random.rand(n)  

# khar = KernelHAR(1)
# K = khar._kernel(X,X)
# Kk = khar._kernel(X,X_)

# har = HARCV(kernel=False, alphas=1)
# har.knots = X
# H = har._bases(X).astype(int) 
# H_ = har._bases(X_).astype(int) 
# K_ = (H @ H.T)
# Kk_ = (H_ @ H.T)

# np.all(Kk == Kk_)
# np.all(K == K_)



## ~~~ Test HAR vs. kernel HAR ~~~

# import numpy as np
# from highly_adaptive_regression import KernelHAR, HARCV, HALCV
# n, n_ ,d  = 100, 100, 3

# X = np.random.rand(n, d)  
# X_ = np.random.rand(n_, d)   
# Y = 10*np.random.rand(n)  

# har = HARCV(kernel=False)
# har.fit(X,Y)
# Y_har = har.predict(X_)
# khar = KernelHAR(har.regression.alpha_)
# khar.fit(X,Y)
# Y_khar = khar.predict(X_)

# rmse_diff = np.sqrt(np.mean((Y_har - Y_khar)**2))
# rmse_diff / np.std(Y) < 0.1

# rmse_har = np.sqrt(np.mean((Y_har - Y)**2))
# rmse_khar = np.sqrt(np.mean((Y_khar - Y)**2))
# np.abs((rmse_har - rmse_khar)/rmse_har) < 1e-2

# rmse_har, rmse_khar, rmse_diff



## ~~~ Test LOOCV for HAR vs. kernel HAR ~~~

# import numpy as np
# from highly_adaptive_regression import KernelHAR, HARCV, HALCV
# n, n_ ,d  = 100, 100, 3

# X = np.random.rand(n, d)  
# X_ = np.random.rand(n_, d)   
# Y = 10*np.random.rand(n)  

# har = HARCV(n_alphas = 5, kernel=False)
# har.fit(X,Y)
# Y_har = har.predict(X_)
# rmse_har = np.sqrt(np.mean((Y_har - Y)**2))

# khar = HARCV(n_alphas = 5)
# khar.fit(X,Y)
# Y_khar = khar.predict(X_)
# rmse_khar = np.sqrt(np.mean((Y_khar - Y)**2))

# rmse_diff = np.sqrt(np.mean((Y_har - Y_khar)**2))
# rmse_har, rmse_khar, rmse_diff
