"""The HAL learner of Table 1, run only on the small datasets.

Its basis has (2^p - 1) n columns, 3.3 million on boston, but most of them are equal at the
training points, and the fit never forms the basis matrix. A fit takes seconds.
"""
from highly_adaptive_regression import HighlyAdaptiveLassoCV


def hal_learner(n_jobs=-1):
    """HAL with the CV folds fit in parallel. The thread count does not change the fit."""
    return HighlyAdaptiveLassoCV(n_jobs=n_jobs)
