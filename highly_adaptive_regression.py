from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso, LassoCV, RidgeCV
from sklearn.linear_model._coordinate_descent import _alpha_grid
from sklearn.preprocessing import StandardScaler

from numba import njit
import numpy as np
from scipy.linalg import solve


class KernelCenterer():
    """
    Centering transformer for kernel matrices. see: https://www.mlpack.org/papers/kpca.pdf#page=18.50
    """

    def __init__(self):
        pass

    def transform(self, K):
        return K - np.mean(K, axis=0) - np.mean(self.K, axis=1, keepdims=True) + np.mean(self.K)

    def fit_transform(self, K):
        self.K = K
        return self.transform(K)


class HABaseCV:
    """
    Highly Adaptive Base (HABase) class.

    This class implements the Highly Adaptive Lasso/Ridge algorithms.
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
        Computes the basis functions for the given input data.

        This function computes the basis functions using the knot points and the input data.

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
        Fits the HAL model to the given input data and target values.

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
        self, *args, kernel=True, 
        n_alphas=100, eps=1e-3, alphas=None,
        **kwargs
        ):
        self.scaler = StandardScaler(with_mean=True, with_std=False)
        self.regression = RidgeCV(*args, **kwargs)
        self.n_alphas = n_alphas
        self.eps = eps
        self.alphas = alphas

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
            self.regression.alphas = _alpha_grid(X,Y, l1_ratio=1e-3, eps=self.eps, n_alphas=self.n_alphas)
        else:
            self.regression.alphas = self.alphas

    def fit_kernel(self, X, Y):
        self._pre_fit(X, Y)

        self.search = GridSearchCV(
            estimator = KernelHAR(), 
            param_grid = {'alpha': self.regression.alphas}, 
            cv=5, scoring='neg_mean_squared_error', 
            n_jobs=-1
        )
        self.search.fit(X, Y)

    def predict_kernel(self, X):
        return self.search.best_estimator_.predict(X)


class KernelHAR(BaseEstimator, RegressorMixin):

    def __init__(self, alpha=1):
        self.alpha = alpha # regularization strength
        self.X = None # training data
        self.centerer = KernelCenterer() 

    def fit(self, X, Y):
        self.X = X
        self.intercept = np.mean(Y)
        K = self.centerer.fit_transform(self._kernel(self.X, X))
        self.coef = solve(
            K + self.alpha*np.eye(K.shape[0]), 
            Y-self.intercept
        )

    def predict(self, X):
        k = self.centerer.transform(self._kernel(self.X, X))
        return k.T @ self.coef + self.intercept

    @staticmethod
    @njit
    def _kernel(X_train, X_prime):
        n_train = X_train.shape[0]
        n_prime = X_prime.shape[0]
        n_features = X_train.shape[1]
        
        K = np.empty((n_train, n_prime), dtype=np.int64)
        equal = False
        for tr in range(n_train):
            if equal:
                max_index = tr + 1
            else:
                max_index = n_prime
            for te in range(max_index):
                sum_val = 0
                for knot in range(n_train):
                    count = 0
                    for feature in range(n_features):
                        if X_train[knot, feature] <= min(X_train[tr, feature], X_prime[te, feature]):
                            count += 1
                    sum_val += 2 ** count
                K[tr, te] = sum_val
                if equal:
                    K[te, tr] = sum_val
                
        return K - n_train # account for the n "intercepts"



#### TESTS ####



## ~~~ Test kernel function and centering ~~~

# import numpy as np
# from highly_adaptive_regression import KernelHAR, HARCV, HALCV
# from sklearn.preprocessing import KernelCenterer, StandardScaler

# n, n_ ,d  = 10, 4, 3
# X = np.random.rand(n, d)  
# X_ = np.random.rand(n_, d)   
# Y = np.random.rand(n)  

# khar = KernelHAR(1)
# K = khar._kernel(X,X)
# centerer = KernelCenterer()
# Kc = centerer.fit_transform(K)
# K, Kc

# har = HARCV(kernel=False, alphas=1)
# har.knots = X
# scaler = StandardScaler(with_mean=True, with_std=False)
# H = har._bases(X).astype(int) 
# H_ = har._bases(X_).astype(int) 
# Hc = scaler.fit_transform(H)
# Hc_ = scaler.transform(H)
# K_ = (H @ H.T)
# Kc_ = Hc @ Hc.T

# np.all(K == K_)
# np.all(Kc == Kc_)

# Kkc = Kk - np.mean(Kk, axis=0) - np.mean(K, axis=1, keepdims=True) + np.mean(K)
# Kkc_ = ((H @ H_.T - np.mean(H @ H_.T, axis=0)).T - np.mean(H @ H.T, axis=1) + np.mean(H @ H.T)).T
# np.all(Kkc_ = KKc)



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