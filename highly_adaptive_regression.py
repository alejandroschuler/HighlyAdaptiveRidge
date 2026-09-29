from sklearn.linear_model import LassoCV, RidgeCV
from kernel_ridge import HighlyAdaptiveRidgeCV as kHARCV
from kernel_ridge.kernels import HighlyAdaptiveRidge as HARKernel
import numpy as np


class HighlyAdaptiveBaseCV:
    """Highly Adaptive Base (HABase) class. Implements the Highly Adaptive Lasso/Ridge algorithms."""

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
        self._pre_fit(X,Y)
        self.knots = X
        self.regression.fit(self._bases(X), Y)

    def predict(self, X):
        return self.regression.predict(self._bases(X)) 


class HighlyAdaptiveLassoCV(HighlyAdaptiveBaseCV):

    def __init__(self, *args, sparse=False, **kwargs):
        # `sparse` stores the (2^p-1)*n-column basis as a CSC matrix before handing it to
        # LassoCV. The lasso solution is unchanged; this just avoids the dense memory blowup
        # for higher-dimensional datasets. `n_jobs` (in kwargs) parallelizes the CV folds.
        self.sparse = sparse
        self.regression = LassoCV(*args, **kwargs)

    def _design(self, X):
        B = self._bases(X)
        if self.sparse:
            from scipy.sparse import csc_matrix
            B = csc_matrix(B)
        return B

    def fit(self, X, Y):
        self._pre_fit(X, Y)
        self.knots = X
        self.regression.fit(self._design(X), Y)

    def predict(self, X):
        return self.regression.predict(self._design(X))


class HighlyAdaptiveRidgeCV(HighlyAdaptiveBaseCV, kHARCV):

    # TODO: FIX HOW ALPHAS ARE ASSIGNED, 

    def __init__(self, *args, **kwargs):
        kHARCV.__init__(self, *args, **kwargs) # copy the init signature of kHARCV to get alpha grid params
        self.regression = RidgeCV()

    def _pre_fit(self, X,Y):
        K = HARKernel()
        self.regression.alphas = HARKernel.alpha_grid(X, Y, self.n_alphas, self.eps, K=K(X))  