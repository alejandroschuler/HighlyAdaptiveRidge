"""HAL: the zero-order highly adaptive lasso, with its depth tuned by early
stopping and its penalty by 5-fold CV.

At depth D the basis has the functions of the sections of at most D
coordinates, n_train * (C(p, 1) + ... + C(p, D)) of them, one for each training
point and section. The walk starts at depth 1. At each depth, 5-fold CV over the
penalty grid gives the smallest cross-validated risk, and the walk stops at the
first depth that does not lower it (a patience of 1). It also stops before a
depth whose predicted time would take the fit past BUDGET seconds: the time of
the depth before it, times the ratio of the two numbers of basis functions. The
depth with the smallest risk is refit on all the training rows at its chosen
penalty. The implementation stores a section as the bits of a 64-bit integer,
so it handles at most 63 covariates.

The penalty grid is that of scikit-learn's LassoCV, from the smallest penalty
that sets every coefficient to zero down to EPS times it, with N_ALPHAS values
evenly spaced on the log scale. EPS is 1e-5 rather than LassoCV's 1e-3,
because the smallest penalty of the shorter grid was chosen on every split of
yacht and energy; N_ALPHAS keeps LassoCV's spacing of about 33 values per
factor of ten.
"""
import time
from math import comb

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from highly_adaptive_regression import HighlyAdaptiveLassoCV

NAME = "HAL"
N_ALPHAS = 167
EPS = 1e-5
PATIENCE = 1
BUDGET = 20 * 60
# The HAL code refuses a basis of more than 10^8 functions, which needs gigabytes.
MAX_COLUMNS = 10 ** 8


def n_columns(n, p, depth):
    """The number of basis functions of HAL at a depth: one per training point and
    nonempty section of at most `depth` coordinates."""
    return n * sum(comb(p, k) for k in range(1, depth + 1))


class HALDepthPath(BaseEstimator, RegressorMixin):
    """HAL with the depth tuned by early stopping and the penalty by CV over `folds`.

    After fit: depths_ and cv_risk_ hold each depth walked and its smallest
    cross-validated risk, columns_ the size of its basis, seconds_ the time it
    took, alpha_index_ the position of the chosen penalty in its grid, and
    stop_ why the walk ended ("risk", "budget", "columns" or "full depth").
    """

    def __init__(self, folds, n_alphas=N_ALPHAS, eps=EPS, patience=PATIENCE, budget=BUDGET,
                 max_columns=MAX_COLUMNS):
        self.folds = folds
        self.n_alphas = n_alphas
        self.eps = eps
        self.patience = patience
        self.budget = budget
        self.max_columns = max_columns

    def fit(self, X, Y):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        n, p = X.shape
        self.depths_, self.cv_risk_, self.columns_, self.alpha_index_list_ = [], [], [], []
        self.seconds_ = []
        self.stop_ = "full depth"
        self.model_ = None
        start = time.perf_counter()
        for depth in range(1, p + 1):
            cols = n_columns(n, p, depth)
            if cols > self.max_columns:
                self.stop_ = "columns"
                break
            if self.budget is not None and self.depths_:
                predicted = self.seconds_[-1] * cols / self.columns_[-1]
                if time.perf_counter() - start + predicted > self.budget:
                    self.stop_ = "budget"
                    break
            t0 = time.perf_counter()
            m = HighlyAdaptiveLassoCV(n_alphas=self.n_alphas, eps=self.eps, cv=self.folds,
                                      max_degree=depth, n_jobs=1).fit(X, Y)
            self.seconds_.append(time.perf_counter() - t0)
            risk = m.mse_path_.mean(axis=1)
            self.depths_.append(depth)
            self.columns_.append(cols)
            self.cv_risk_.append(float(risk.min()))
            self.alpha_index_list_.append(int(np.argmin(risk)))
            if self.cv_risk_[-1] <= min(self.cv_risk_):
                self.model_ = m
            if len(self.cv_risk_) - 1 - int(np.argmin(self.cv_risk_)) >= self.patience:
                self.stop_ = "risk"
                break
        if self.model_ is None:
            raise RuntimeError(f"no depth fits in {self.max_columns} basis columns")
        self.best_ = int(np.argmin(self.cv_risk_))
        self.depth_ = self.depths_[self.best_]
        self.alpha_ = self.model_.alpha_
        self.alpha_index_ = self.alpha_index_list_[self.best_]
        return self

    def predict(self, X):
        return self.model_.predict(np.asarray(X, dtype=float))


def learner(folds, seed, p):
    return HALDepthPath(folds)


def chosen(fitted):
    return {
        "depth": fitted.depth_, "depths_walked": len(fitted.depths_), "stop": fitted.stop_,
        "columns": fitted.columns_[fitted.best_], "budget": fitted.budget,
        "alpha": fitted.alpha_, "alpha_index": fitted.alpha_index_, "n_alphas": fitted.n_alphas,
        "eps": fitted.eps, "folds": len(fitted.folds), "patience": fitted.patience,
        "n_nonzero": fitted.model_.n_nonzero_, "n_distinct": fitted.model_.n_distinct_,
    }


def tuning(fitted):
    return [{"depth": d, "cv_risk": r, "columns": c, "alpha_index": a, "seconds": t}
            for d, r, c, a, t in zip(fitted.depths_, fitted.cv_risk_, fitted.columns_,
                                     fitted.alpha_index_list_, fitted.seconds_)]
