"""HAR fit on the explicit basis, to check the kernel implementation against.

It lived in highly_adaptive_regression.py. It moved here, with the only import of
kernel_ridge that module had, so that HAL no longer depends on the kernel code.
"""
from sklearn.linear_model import RidgeCV
from kernel_ridge import HighlyAdaptiveRidgeCV as kHARCV
from kernel_ridge.kernels import HighlyAdaptiveRidge as HARKernel
from highly_adaptive_regression import HighlyAdaptiveBaseCV


class HighlyAdaptiveRidgeCV(HighlyAdaptiveBaseCV, kHARCV):

    # TODO: FIX HOW ALPHAS ARE ASSIGNED, 

    def __init__(self, *args, **kwargs):
        kHARCV.__init__(self, *args, **kwargs) # copy the init signature of kHARCV to get alpha grid params
        self.regression = RidgeCV()

    def _pre_fit(self, X,Y):
        K = HARKernel()
        self.regression.alphas = HARKernel.alpha_grid(X, Y, self.n_alphas, self.eps, K=K(X))  
