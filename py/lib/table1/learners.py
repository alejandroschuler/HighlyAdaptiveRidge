"""The five Table 1 methods other than HAL, with the paper's settings.

HAL is in hal.py, so that a change here does not rerun the HAL cells.
"""
from sklearn.ensemble import RandomForestRegressor

from kernel_ridge import (
    HighlyAdaptiveRidgeCV,
    MixedSobolevRidgeCV,
    RadialBasisKernelRidgeCV,
    RidgeRegressionCV,
)

from .design import SEED

# The kernel methods choose their penalty by leave-one-out CV, and ridge by
# 5-fold CV, over this many values from the search of the paper's appendix D.
N_ALPHAS = 50
GAMMAS = [0.001, 0.01, 0.1, 1, 10]   # radial basis bandwidths
RF_TREES = 2000


def table1_learners(rep, n_jobs=-1):
    """Fresh estimators for one cell. The kernel learners keep state from a
    fit, so no instance is reused across cells. The thread count does not
    change any fit."""
    return {
        "HAR": HighlyAdaptiveRidgeCV(n_alphas=N_ALPHAS, order=0),
        "Mixed Sobolev KRR": MixedSobolevRidgeCV(n_alphas=N_ALPHAS),
        "Radial Basis KRR": RadialBasisKernelRidgeCV(gammas=GAMMAS, n_alphas=N_ALPHAS),
        "Random Forest": RandomForestRegressor(
            n_estimators=RF_TREES, n_jobs=n_jobs, random_state=SEED + rep
        ),
        "Ridge Regression": RidgeRegressionCV(n_alphas=N_ALPHAS, cv=5),
    }
