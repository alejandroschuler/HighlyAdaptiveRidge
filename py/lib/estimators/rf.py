"""Random forest: scikit-learn's RandomForestRegressor with N_TREES trees and every
other setting at its default, untuned."""
from sklearn.ensemble import RandomForestRegressor

NAME = "Random Forest"
N_TREES = 2000
# The defaults that shape the trees, recorded from each fit.
REPORTED = ["n_estimators", "max_features", "min_samples_leaf", "min_samples_split",
            "max_depth", "bootstrap", "criterion"]


def learner(folds, seed, p):
    return RandomForestRegressor(n_estimators=N_TREES, n_jobs=1, random_state=seed)


def chosen(fitted):
    params = fitted.get_params()
    return {k: params[k] for k in REPORTED}


def tuning(fitted):
    return []
