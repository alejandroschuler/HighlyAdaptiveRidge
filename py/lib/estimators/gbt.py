"""Gradient-boosted trees: xgboost, with the depth, the learning rate and the
number of trees tuned by 5-fold CV.

Every setting not tuned is at the xgboost default, except that a leaf has no
minimum size (min_child_weight = 0; under squared error the default of 1 asks
for one row per leaf, so the two give the same trees). Each pair of a depth
and a learning rate is fit in every fold, with up to MAX_TREES trees, and the
mean over the folds of the validation mean squared error after each tree gives
the CV risk of every number of trees at once. The fits of a pair stop when that
mean has not improved for PATIENCE trees. The setting with the smallest CV risk
is refit on all the training rows with its number of trees.

The grid follows notes/har-comparators.tex. There depths up to 6 were chosen
on most datasets and 8 once, and the learning rates 0.03 and 0.1 most often;
1.0 was never chosen. A larger learning rate reaches further along the boosting
path in the same number of trees, so the cap stays at 2000 trees.
"""
import numpy as np
import xgboost as xgb
from sklearn.base import BaseEstimator, RegressorMixin

NAME = "Gradient Boosted Trees"
DEPTHS = [1, 2, 3, 4, 6, 8]
RATES = [0.03, 0.1, 0.3]
MAX_TREES = 2000
PATIENCE = 100
MIN_CHILD_WEIGHT = 0


def _mse(predt, dmatrix):
    return "mse", float(np.mean((predt - dmatrix.get_label()) ** 2))


class TunedXGB(BaseEstimator, RegressorMixin):
    """xgboost tuned over DEPTHS x RATES and the number of trees, by CV over `folds`.

    After fit, cv_ holds one row for each pair: its best number of trees, its
    CV risk there, and the number of trees fit before the stop.
    """

    def __init__(self, folds, seed, depths=DEPTHS, rates=RATES, max_trees=MAX_TREES, patience=PATIENCE):
        self.folds = folds
        self.seed = seed
        self.depths = depths
        self.rates = rates
        self.max_trees = max_trees
        self.patience = patience

    def _params(self, depth, rate):
        return {"objective": "reg:squarederror", "max_depth": depth, "learning_rate": rate,
                "min_child_weight": MIN_CHILD_WEIGHT, "nthread": 1, "seed": self.seed}

    def fit(self, X, Y):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        dtrain = xgb.DMatrix(X, label=Y, nthread=1)
        self.cv_ = []
        for depth in self.depths:
            for rate in self.rates:
                params = {**self._params(depth, rate), "disable_default_eval_metric": 1}
                res = xgb.cv(params, dtrain, num_boost_round=self.max_trees, folds=self.folds,
                             custom_metric=_mse, early_stopping_rounds=self.patience,
                             as_pandas=False, verbose_eval=False, seed=self.seed)
                curve = np.asarray(res["test-mse-mean"])
                t = int(np.argmin(curve)) + 1
                self.cv_.append({"depth": depth, "learning_rate": rate, "n_trees": t,
                                 "cv_risk": float(curve[t - 1]), "trees_fit": len(curve)})
        best = min(self.cv_, key=lambda r: r["cv_risk"])
        self.depth_, self.learning_rate_, self.n_trees_ = best["depth"], best["learning_rate"], best["n_trees"]
        p = self._params(self.depth_, self.learning_rate_)
        self.model_ = xgb.XGBRegressor(
            n_estimators=self.n_trees_, max_depth=p["max_depth"], learning_rate=p["learning_rate"],
            min_child_weight=p["min_child_weight"], n_jobs=1, random_state=self.seed,
        ).fit(X, Y)
        return self

    def predict(self, X):
        return self.model_.predict(np.asarray(X, dtype=float))


def learner(folds, seed, p):
    return TunedXGB(folds, seed)


def chosen(fitted):
    config = fitted.model_.get_booster().save_config()
    return {"depth": fitted.depth_, "learning_rate": fitted.learning_rate_, "n_trees": fitted.n_trees_,
            "depths": list(fitted.depths), "rates": list(fitted.rates), "max_trees": fitted.max_trees,
            "patience": fitted.patience, "folds": len(fitted.folds),
            "min_child_weight": MIN_CHILD_WEIGHT, "xgboost_config": config}


def tuning(fitted):
    return fitted.cv_
