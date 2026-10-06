"""Kernel ridge regression tuned over a path of kernels, as Section 4 describes.

The kernels form a path from simple to complex, such as the HAR kernels of
depth 1, 2, 3, ..., or the radial basis kernels from wide to narrow. In each
training fold, each kernel gets its own grid of penalties, from its own kernel
matrix (Kernel.alpha_grid, the paper's appendix on the regularization search),
and leave-one-out cross-validation (LOOCV) on the fold's training rows chooses
the penalty. The fit at that penalty predicts the fold's validation rows, and
the kernel's cross-validated risk is the mean over the folds of the mean
squared error on the validation rows. The kernel with the smallest risk is
refit on all the training rows, with the penalty chosen again by LOOCV over a
grid from the full kernel matrix.

With patience = k, the walk along the path stops after k kernels in a row that
do not lower the smallest risk so far, and the kernels after the stop are never
built. With a time budget, the walk also stops before a kernel whose predicted
time would take the walk past the budget: the time of the last kernel walked,
times the ratio of the two kernels' costs (cost, a function of the kernel). With
a single kernel there is nothing for the folds to choose, so the fit is the
LOOCV refit alone.
"""
import time

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from kernel_ridge.fast import _prep, coef_at, loocv_path

# Each penalty grid has N_ALPHAS values, and the epsilon of its top value is EPS:
# at the top penalty, every fitted value is within EPS max|y_i| of zero.
N_ALPHAS = 50
EPS = 1e-3


def loocv_fit(kernel, X, Y, n_alphas, eps):
    """Kernel ridge with the penalty chosen by LOOCV over the kernel's own grid.

    Returns (coef, alphas, mses, j, max_eig): the coefficients [c; b] at the
    chosen penalty alphas[j], the grid, the LOOCV error at each penalty, and the
    largest eigenvalue of the kernel matrix. coef is None when no penalty gives
    a finite LOOCV error, or when the eigendecomposition fails.
    """
    K = kernel(X)
    try:
        cache, min_eig, max_eig = _prep(K, Y)
    except np.linalg.LinAlgError:
        return None, None, None, None, None
    alphas = kernel.alpha_grid(Y, K=K, n_alphas=n_alphas, eps=eps, min_eig=min_eig, max_eig=max_eig)
    mses, cache = loocv_path(K, Y, alphas, cache=cache)
    finite = np.flatnonzero(np.isfinite(mses))
    if finite.size == 0:
        return None, alphas, mses, None, max_eig
    j = int(finite[np.argmin(mses[finite])])
    return coef_at(cache, float(alphas[j])), alphas, mses, j, max_eig


def predict_with(kernel, X_fit, coef, X_new):
    """k(x) c + b, the prediction of kernel ridge with coefficients coef = [c; b]."""
    return kernel(X_fit, X_new) @ coef[:-1] + coef[-1]


class KernelPathCV(BaseEstimator, RegressorMixin):
    """Kernel ridge tuned by cross-validation over a path of kernels.

    kernels: the path, simplest first.
    folds: (train, validation) index arrays of the training rows.
    n_alphas, eps: the size of each penalty grid, and the epsilon of its top
        value (Kernel.alpha_grid).
    patience: None walks the whole path; an integer k stops after k kernels
        in a row that do not lower the smallest cross-validated risk.
    scaler: None, or a class whose instances scale the covariates (fit on the
        training rows of each fit, as in scikit-learn).
    budget, cost: None, or the time budget of the walk in seconds and the
        relative cost of each kernel (see the module docstring).

    After fit:
        cv_risk_: the cross-validated risk of each kernel walked (inf for a
            kernel with no finite LOOCV error in some fold);
        fold_mse_: the validation error of each kernel walked, in each fold;
        fold_alpha_index_: the position, in that fold's grid, of the penalty
            that LOOCV chose for each kernel walked, in each fold (-1 for none);
        best_: the position of the chosen kernel on the path;
        seconds_: the time each kernel walked took, in all the folds;
        stop_: why the walk ended: "risk", "budget" or "end";
        alpha_, alphas_, alpha_index_, loocv_mse_: the penalty of the refit,
            its grid, its position in the grid and the LOOCV error at each penalty.
    """

    def __init__(self, kernels, folds, n_alphas=N_ALPHAS, eps=EPS, patience=None, scaler=None,
                 budget=None, cost=None):
        self.kernels = kernels
        self.folds = folds
        self.n_alphas = n_alphas
        self.eps = eps
        self.patience = patience
        self.scaler = scaler
        self.budget = budget
        self.cost = cost

    def _scale(self, X_fit, *others):
        if self.scaler is None:
            return (X_fit, *others)
        s = self.scaler().fit(X_fit)
        return (s.transform(X_fit), *[s.transform(o) for o in others])

    def _fold_risk(self, kernel, X, Y):
        """The validation error of one kernel in each fold, and the chosen penalty's grid position."""
        errors, positions = [], []
        for train, val in self.folds:
            X_tr, X_val = self._scale(X[train], X[val])
            coef, _, _, j, _ = loocv_fit(kernel, X_tr, Y[train], self.n_alphas, self.eps)
            if coef is None:
                errors.append(np.inf)
                positions.append(-1)
                continue
            pred = predict_with(kernel, X_tr, coef, X_val)
            errors.append(float(np.mean((pred - Y[val]) ** 2)))
            positions.append(j)
        return np.array(errors), np.array(positions)

    def fit(self, X, Y):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        if self.patience is not None and not (isinstance(self.patience, (int, np.integer)) and self.patience >= 1):
            raise ValueError(f"patience must be None or an integer >= 1, not {self.patience!r}")
        self.cv_risk_, fold_mse, fold_pos, self.seconds_ = [], [], [], []
        self.stop_ = "end"
        if len(self.kernels) == 1:
            self.best_ = 0
        else:
            start = time.perf_counter()
            for k, kernel in enumerate(self.kernels):
                if self.budget is not None and k > 0:
                    predicted = self.seconds_[-1] * self.cost(kernel) / self.cost(self.kernels[k - 1])
                    if time.perf_counter() - start + predicted > self.budget:
                        self.stop_ = "budget"
                        break
                t0 = time.perf_counter()
                errors, positions = self._fold_risk(kernel, X, Y)
                self.seconds_.append(time.perf_counter() - t0)
                fold_mse.append(errors)
                fold_pos.append(positions)
                self.cv_risk_.append(float(np.mean(errors)))
                best = int(np.argmin(self.cv_risk_))
                if self.patience is not None and len(self.cv_risk_) - 1 - best >= self.patience:
                    self.stop_ = "risk"
                    break
            if not np.isfinite(min(self.cv_risk_)):
                raise RuntimeError("no kernel on the path has a finite cross-validated risk")
            self.best_ = int(np.argmin(self.cv_risk_))
        self.fold_mse_ = np.array(fold_mse)
        self.fold_alpha_index_ = np.array(fold_pos)
        self.kernel_ = self.kernels[self.best_]
        self.X_fit_, = self._scale(X)
        if self.scaler is not None:
            self.scaler_ = self.scaler().fit(X)
        coef, alphas, mses, j, max_eig = loocv_fit(self.kernel_, self.X_fit_, Y, self.n_alphas, self.eps)
        if coef is None:
            raise RuntimeError("the refit has no penalty with a finite LOOCV error")
        self.coef_, self.alphas_, self.loocv_mse_, self.alpha_index_ = coef, alphas, mses, j
        self.max_eig_ = float(max_eig)
        self.alpha_ = float(alphas[j])
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if self.scaler is not None:
            X = self.scaler_.transform(X)
        return predict_with(self.kernel_, self.X_fit_, self.coef_, X)


def path_tuning(fitted, setting, values):
    """One row for each kernel walked: its setting on the path (named `setting`),
    its cross-validated risk, and how often the penalty that LOOCV chose sat at
    the bottom or the top of the fold's grid."""
    rows = []
    n_alphas = fitted.n_alphas
    for k, risk in enumerate(fitted.cv_risk_):
        pos = fitted.fold_alpha_index_[k]
        rows.append({
            setting: values[k], "cv_risk": risk,
            "folds_alpha_floor": int(np.sum(pos == 0)), "folds_alpha_top": int(np.sum(pos == n_alphas - 1)),
        })
    return rows


def path_chosen(fitted, setting, values):
    """The chosen kernel's setting, the path, how the walk ended, and the refit's
    penalty and its place in its grid."""
    return {
        setting: values[fitted.best_], "path": [v if isinstance(v, str) else float(v) for v in values],
        "stop": fitted.stop_,
        "budget": -1 if fitted.budget is None else fitted.budget,
        "kernels_walked": len(fitted.cv_risk_) if fitted.cv_risk_ else 1,
        "kernels_in_path": len(fitted.kernels),
        "alpha": fitted.alpha_, "alpha_index": fitted.alpha_index_,
        "n_alphas": fitted.n_alphas, "eps": fitted.eps,
        "alpha_floor": float(fitted.alphas_[0]), "alpha_top": float(fitted.alphas_[-1]),
        "alpha_floor_ratio": float(fitted.alphas_[0] / fitted.max_eig_),
        "folds": len(fitted.folds), "patience": -1 if fitted.patience is None else fitted.patience,
    }
