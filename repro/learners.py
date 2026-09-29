"""Learner factories with the paper's hyperparameters.

Each call returns fresh estimator instances (the kernel learners cache fit state, so
they must not be reused across fits).

Two regularization-search configurations appear in the paper:
  * Table 1 / convergence: class defaults (eps=1e-3) for the 50-point log grid.
  * Figure 1 demonstration: the notebook used eps=1e-10 (HAR, radial) and eps=1e-6
    (mixed Sobolev). We keep those and drop the undocumented `max_alpha_coef_norm`
    kwarg, standardizing on the paper's eps-based search (appendix D).
"""
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from kernel_ridge import (
    HighlyAdaptiveRidgeCV,
    RadialBasisKernelRidgeCV,
    MixedSobolevRidgeCV,
)
from highly_adaptive_regression import HighlyAdaptiveLassoCV

from .config import GAMMAS, RF_TREES, RIDGE_ALPHA, N_ALPHAS


def krr_learners(random_state=0):
    """The five fast Table 1 methods (everything except HAL).

    HAL is split out because its design matrix has (2^p - 1)*n columns, so its LassoCV is
    orders of magnitude slower than the kernel methods; see hal_learner / the --phase hal
    path in table1_benchmark.
    """
    return {
        "HAR": HighlyAdaptiveRidgeCV(n_alphas=N_ALPHAS, order=0),
        "Mixed Sobolev KRR": MixedSobolevRidgeCV(n_alphas=N_ALPHAS),
        "Radial Basis KRR": RadialBasisKernelRidgeCV(gammas=GAMMAS, n_alphas=N_ALPHAS),
        "Random Forest": RandomForestRegressor(
            n_estimators=RF_TREES, n_jobs=-1, random_state=random_state
        ),
        "Ridge Regression": Ridge(alpha=RIDGE_ALPHA),
    }


def hal_learner():
    """HAL with parallel cross-validation and a sparse design matrix.

    Both speedups are exact (same lasso solution): n_jobs parallelizes the CV folds, and a
    sparse basis avoids materializing the dense (2^p-1)*n-column matrix that makes HAL
    intractable for the higher-dimensional datasets (e.g. boston, p=13)."""
    return HighlyAdaptiveLassoCV(n_jobs=-1, sparse=True)


def benchmark_learners(random_state=0):
    """All six Table 1 methods (used by the equivalence/demo paths)."""
    return {**{"HAR": krr_learners(random_state)["HAR"]},
            "HAL": hal_learner(),
            **{k: v for k, v in krr_learners(random_state).items() if k != "HAR"}}


def demo_learners(random_state=0):
    """The six methods in the Figure 1 demonstration (notebook eps values)."""
    return {
        "Ridge Regression": Ridge(alpha=RIDGE_ALPHA),
        "Random Forest": RandomForestRegressor(
            n_estimators=RF_TREES, n_jobs=-1, random_state=random_state
        ),
        "Radial Basis KRR": RadialBasisKernelRidgeCV(gammas=GAMMAS, eps=1e-10),
        "Mixed Sobolev KRR": MixedSobolevRidgeCV(eps=1e-6),
        "HAL": HighlyAdaptiveLassoCV(),
        "HAR": HighlyAdaptiveRidgeCV(eps=1e-10, order=0),
    }


def har_only(eps=1e-10):
    """HAR alone, used for the convergence simulation (Figure 2)."""
    return {"HAR": HighlyAdaptiveRidgeCV(eps=eps, order=0, n_alphas=N_ALPHAS)}


def sota_learners(random_state=0):
    """Additional state-of-the-art baselines requested by the referee (comment 7),
    on top of HAR and mixed Sobolev KRR."""
    return {
        "HAR": HighlyAdaptiveRidgeCV(n_alphas=N_ALPHAS, order=0),
        "Mixed Sobolev KRR": MixedSobolevRidgeCV(n_alphas=N_ALPHAS),
        "Gradient Boosting": HistGradientBoostingRegressor(random_state=random_state),
        "Random Forest": RandomForestRegressor(
            n_estimators=RF_TREES, n_jobs=-1, random_state=random_state
        ),
        "Neural Net": make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=2000, random_state=random_state),
        ),
        "k-NN": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=10)),
    }
