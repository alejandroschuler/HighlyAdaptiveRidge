"""MARS: multivariate adaptive regression splines (Friedman 1991), from pymars
(github.com/alejandroschuler/mars), which follows the rules of the R package
earth. Each fit runs earth's forward pass and its backward pruning, which
chooses the number of terms by generalized cross-validation (GCV), with
earth's defaults except for the term limit of the forward pass. Earth's
default limit, min(200, max(20, 2p)) + 1, stopped the forward pass in 140 of
the 270 pilot fits on the nine datasets with p < 20 (reps 0 and 1, degrees 1
to 3, the shared folds; scratch/sobolev_mars), and a limit of 101 lowered the
best cross-validated risk over the degrees by 17 to 25 percent on boston (one
split), concrete and kin8nm, left it unchanged on five datasets, and raised it
by at most 4 percent on wine and the other boston split. So the limit is
MAX_TERMS, the largest value of earth's default, and the forward pass stops by
its threshold on the change in R squared instead.

The largest degree of a term, the number of covariates that one term can
multiply, plays the role of HAR's depth, and it is tuned as HAR's depth is
(har.py): along the depth path (depths.py), by 5-fold CV over the shared folds,
with the same early stopping. The degree with the smallest cross-validated
risk is refit on all the training rows.

pymars needs scikit-learn 1.6 or later, and the main environment pins 1.5.1
so that the fits of Table 1 reproduce, so the MARS fits run in the environment
of envs/mars (see the Snakefile).
"""
import json
from importlib.metadata import distribution, version

import numpy as np
from pymars import EarthRegressor
from sklearn.base import BaseEstimator, RegressorMixin

from .depths import PATIENCE, depth_path

NAME = "MARS"
MAX_TERMS = 201


def _commit():
    """The commit of the installed pymars, from its record of a git install."""
    try:
        info = json.loads(distribution("mars-earth").read_text("direct_url.json"))
        return info["vcs_info"]["commit_id"]
    except (TypeError, KeyError, json.JSONDecodeError):
        return "unknown"


class DegreePathCV(BaseEstimator, RegressorMixin):
    """MARS with its degree tuned by cross-validation along a path of degrees.

    degrees: the path, smallest first. folds: (train, validation) index arrays
    of the training rows. patience: the walk stops after this many degrees in
    a row that do not lower the smallest cross-validated risk.

    After fit: cv_risk_ and fold_mse_ for each degree walked, best_ (the
    position of the chosen degree), stop_ ("risk" or "end"), and model_, the
    refit at the chosen degree on all the training rows.
    """

    def __init__(self, degrees, folds, patience=PATIENCE):
        self.degrees = degrees
        self.folds = folds
        self.patience = patience

    def fit(self, X, Y):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        self.cv_risk_, fold_mse = [], []
        self.stop_ = "end"
        if len(self.degrees) == 1:
            self.best_ = 0
        else:
            for degree in self.degrees:
                errors = []
                for train, val in self.folds:
                    fit = EarthRegressor(max_degree=degree, max_terms=MAX_TERMS).fit(X[train], Y[train])
                    errors.append(float(np.mean((fit.predict(X[val]) - Y[val]) ** 2)))
                fold_mse.append(errors)
                self.cv_risk_.append(float(np.mean(errors)))
                best = int(np.argmin(self.cv_risk_))
                if len(self.cv_risk_) - 1 - best >= self.patience:
                    self.stop_ = "risk"
                    break
            self.best_ = int(np.argmin(self.cv_risk_))
        self.fold_mse_ = np.array(fold_mse)
        self.model_ = EarthRegressor(max_degree=self.degrees[self.best_], max_terms=MAX_TERMS).fit(X, Y)
        return self

    def predict(self, X):
        return self.model_.predict(np.asarray(X, dtype=float))


def learner(folds, seed, p):
    return DegreePathCV(depth_path(p), folds)


def chosen(fitted):
    """The chosen degree, the path and how its walk ended, and the size of the
    refit: its terms after pruning (with the intercept), the terms that its
    forward pass kept, its term limit, why its forward pass stopped, and its
    GCV penalty. Also the pymars commit and the scikit-learn version."""
    model = fitted.model_
    forward = model.mars_.forward
    return {
        "degree": fitted.degrees[fitted.best_], "path": [int(d) for d in fitted.degrees],
        "stop": fitted.stop_, "degrees_walked": len(fitted.cv_risk_) if fitted.cv_risk_ else 1,
        "terms": int(model.dirs_.shape[0]), "forward_terms": int(len(forward.kept)),
        "max_terms": int(model.max_terms_), "forward_stop": forward.termination.name,
        "penalty": float(model.penalty_),
        "folds": len(fitted.folds), "patience": fitted.patience,
        "pymars_commit": _commit(), "sklearn": version("scikit-learn"),
    }


def tuning(fitted):
    """One row for each degree walked, with its cross-validated risk."""
    return [{"degree": int(d), "cv_risk": r} for d, r in zip(fitted.degrees, fitted.cv_risk_)]
