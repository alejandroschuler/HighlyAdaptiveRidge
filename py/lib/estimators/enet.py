"""The elastic net: scikit-learn's ElasticNetCV on standardized covariates, with
the penalty and the mixing weight tuned by 5-fold CV over the shared folds.

For each mixing weight, the penalty grid runs from the smallest penalty that
sets every coefficient to zero down to EPS times it, with N_ALPHAS values evenly
spaced on the log scale (scikit-learn's construction). The mixing weight is the
lasso's share of the penalty: 0 is ridge and 1 is the lasso.
"""
import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

NAME = "Elastic Net"
MIXTURES = [0.1, 0.5, 0.9]
N_ALPHAS = 100
EPS = 1e-4
MAX_ITER = 10_000


def learner(folds, seed, p):
    return make_pipeline(StandardScaler(), ElasticNetCV(
        l1_ratio=MIXTURES, n_alphas=N_ALPHAS, eps=EPS, cv=folds, max_iter=MAX_ITER, n_jobs=1))


def _net(fitted):
    return fitted[-1]


def chosen(fitted):
    e = _net(fitted)
    k = list(e.l1_ratio).index(e.l1_ratio_)
    j = int(np.argmin(np.abs(e.alphas_[k] - e.alpha_)))
    return {"mixture": e.l1_ratio_, "alpha": e.alpha_, "alpha_index": j, "n_alphas": e.alphas_.shape[1],
            "mixtures": [float(m) for m in e.l1_ratio], "eps": e.eps, "max_iter": e.max_iter,
            "folds": e.mse_path_.shape[2]}


def tuning(fitted):
    """One row for each mixing weight: its best penalty's grid position and CV risk."""
    e = _net(fitted)
    risk = e.mse_path_.mean(axis=2)
    return [{"mixture": m, "alpha_index": int(np.argmin(risk[k])), "cv_risk": float(risk[k].min())}
            for k, m in enumerate(e.l1_ratio)]
