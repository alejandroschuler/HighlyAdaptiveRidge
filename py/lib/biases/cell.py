"""One cell of the Section 4.3 simulation: one estimator on one DGP, over a
block of repetitions.

A dataset is named by its DGP and its repetition, and its draws and folds come
from the repetition alone, so a cell run alone returns the same rows as a cell
run with others, and every estimator of a repetition gets the same folds.
"""
import json
import time

import numpy as np
import pandas as pd

import estimators
from estimators import more_depth
from estimators.folds import folds

from . import design, dgps

more_depth.register()   # mixed Sobolev KRR with depth is registered there

# Settings of the chosen fit that are too long for a row of the results.
DROP = {"xgboost_config"}


def _plain(v):
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, (list, tuple, dict)):
        return json.dumps(v, default=float)
    return v


def run(dgp, slug, reps, learner=None, method=None):
    """Return (results, tuning): for each repetition, one row with the test MSE
    against the truth and the chosen setting, and one row for each setting that
    the tuning scored, with its CV risk.

    `learner(folds, seed, p)` replaces the estimator's own learner, with its
    chosen() and tuning() kept, and `method` names the result rows (default:
    the slug); the edge checks (variants.py) use both."""
    module = estimators.load(slug)
    learner = module.learner if learner is None else learner
    method = slug if method is None else method
    results, tunings = [], []
    for rep in reps:
        data = dgps.draw(dgp, rep)
        (X, Y), (Xte, fte) = data["train"], data["test"]
        seed = design.SEED + rep
        start = time.perf_counter()
        fitted = learner(folds(len(Y), seed), seed, X.shape[1]).fit(X, Y)
        pred = fitted.predict(Xte)
        seconds = time.perf_counter() - start
        row = {"dgp": dgp, "method": method, "rep": rep, "n": len(Y), "n_test": len(fte),
               "p": X.shape[1], "sigma": design.SIGMA,
               "mse": float(np.mean((pred - fte) ** 2)), "seconds": seconds}
        row.update({f"chosen_{k}": _plain(v) for k, v in module.chosen(fitted).items()
                    if k not in DROP})
        results.append(row)
        for t in module.tuning(fitted):
            tunings.append({"dgp": dgp, "method": method, "rep": rep,
                            **{k: _plain(v) for k, v in dict(t).items()}})
    return pd.DataFrame(results), pd.DataFrame(tunings)
