from sklearn.linear_model import LassoCV, RidgeCV
from kernel_ridge import KernelHAR
import numpy as np


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
        max_alpha_coef_norm=0.1, n_alphas=10, eps=1e-4, alphas=None,
        cv=None, **kwargs
        ):
        self.regression = RidgeCV(*args, cv=cv, **kwargs)
        self.verbose = verbose

        self.max_alpha_coef_norm = max_alpha_coef_norm # ||beta||2 (or smaller) desired for max alpha
        self.n_alphas = n_alphas
        self.eps = eps
        self.alphas = alphas
        self.cv = cv

        if kernel is True:
            self.fit = self.fit_kernel
            self.predict = self.predict_kernel

    def _pre_fit(self, X, Y):
        # see: https://chatgpt.com/share/01b765c4-8fc3-41e7-b5af-9276d67be2e8
        # each element of H^T @ Y is at most sum(|Y|) in magnitude
        # so ||H^T @ Y|| <= sqrt(n) sum(|Y|)
        if self.alphas is None:
            alpha_max = np.sqrt(len(Y)) * np.sum(np.abs(Y)) / self.max_alpha_coef_norm
            self.alphas = np.geomspace(alpha_max, alpha_max * self.eps, num=self.n_alphas)
            
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
