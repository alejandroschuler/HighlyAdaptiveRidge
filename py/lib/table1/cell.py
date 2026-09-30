"""Fit learners on one Table 1 cell and record each one's test error."""
import time

import numpy as np
import pandas as pd

from .data import split


def run(X, Y, learners, dataset, rep):
    """One row per learner: the test MSE on repetition `rep`, and the time the
    fit and the prediction took. n and d are the size of the loaded data,
    before the split."""
    n, d = X.shape
    Xtr, Xte, Ytr, Yte = split(X, Y, rep)
    rows = []
    for name, learner in learners.items():
        t0 = time.time()
        learner.fit(Xtr, Ytr)
        t_fit = time.time() - t0
        t0 = time.time()
        pred = learner.predict(Xte)
        t_pred = time.time() - t0
        rows.append({
            "data": dataset, "n": n, "d": d, "learner": name, "rep": rep,
            "mse": float(np.mean((pred - Yte) ** 2)),
            "time fitting": t_fit, "time predicting": t_pred,
        })
    return pd.DataFrame(rows)
