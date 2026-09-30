"""The HAL learner of Table 1, run only on the small datasets.

Its design matrix has (2^p - 1) n columns, so a single fit on boston or
concrete takes hours.
"""
from highly_adaptive_regression import HighlyAdaptiveLassoCV


def hal_learner(n_jobs=-1):
    """HAL with the CV folds fit in parallel and the basis stored as a sparse
    matrix. Neither changes the lasso solution."""
    return HighlyAdaptiveLassoCV(n_jobs=n_jobs, sparse=True)
