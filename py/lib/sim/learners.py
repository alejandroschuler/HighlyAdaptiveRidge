"""The learners of the simulations, with the paper's settings."""
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from highly_adaptive_regression import HighlyAdaptiveLassoCV
from kernel_ridge import (
    HighlyAdaptiveRidgeCV,
    MixedSobolevRidgeCV,
    RadialBasisKernelRidgeCV,
    RidgeRegressionCV,
)

N_ALPHAS = 50
GAMMAS = [0.001, 0.01, 0.1, 1, 10]
RF_TREES = 2000


def demo_learners(rep, n_jobs=-1):
    """The six methods of Figure 1. The regularization searches of the kernel
    methods keep the values of the original notebook: eps = 1e-10 for HAR and
    radial basis KRR, and 1e-6 for mixed Sobolev KRR. Ridge uses the Table 1
    search."""
    return {
        "Ridge Regression": RidgeRegressionCV(n_alphas=N_ALPHAS, cv=5),
        "Random Forest": RandomForestRegressor(
            n_estimators=RF_TREES, n_jobs=n_jobs, random_state=rep
        ),
        "Radial Basis KRR": RadialBasisKernelRidgeCV(gammas=GAMMAS, eps=1e-10),
        "Mixed Sobolev KRR": MixedSobolevRidgeCV(eps=1e-6),
        "HAL": HighlyAdaptiveLassoCV(),
        "HAR": HighlyAdaptiveRidgeCV(eps=1e-10, order=0),
    }


def har_learner():
    """HAR alone, for the convergence study."""
    return HighlyAdaptiveRidgeCV(eps=1e-10, order=0, n_alphas=N_ALPHAS)


def dimension_learners(rep, n_jobs=-1):
    """HAR and mixed Sobolev KRR against the other methods the referee asked
    for."""
    return {
        "HAR": HighlyAdaptiveRidgeCV(n_alphas=N_ALPHAS, order=0),
        "Mixed Sobolev KRR": MixedSobolevRidgeCV(n_alphas=N_ALPHAS),
        "Gradient Boosting": HistGradientBoostingRegressor(random_state=rep),
        "Random Forest": RandomForestRegressor(
            n_estimators=RF_TREES, n_jobs=n_jobs, random_state=rep
        ),
        "Neural Net": make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=2000, random_state=rep),
        ),
        "k-NN": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=10)),
    }
