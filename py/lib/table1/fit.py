"""One Table 1 fit: one estimator on one dataset and repetition, timed on one core.

The fit time is the time to tune and fit the estimator on the training part,
all of its cross-validation included, and the prediction time is the time to
predict the test part. The estimators whose code numba compiles are first fit
once, untimed, on a few training rows, so that no recorded time includes the
compilation.
"""
import json
import platform
import subprocess
import time

import numpy as np
import pandas as pd

import estimators
from estimators.folds import folds

from .data import split
from .design import SEED

# The estimators that numba compiles, and the rows of their untimed warm-up fit.
COMPILED = {"har", "har1", "mixed_sobolev", "rbf", "hal"}
WARM_ROWS = 60


def cpu():
    """The processor's name, for the methods text."""
    try:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return platform.processor()


def _flat(d, prefix):
    """A dict with prefixed keys and its values as plain scalars, for one CSV row."""
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.integer,)):
            v = int(v)
        elif isinstance(v, (np.floating,)):
            v = float(v)
        elif isinstance(v, (list, tuple, dict)):
            v = json.dumps(v)
        out[f"{prefix}{k}"] = v
    return out


def run(X, Y, slug, dataset, rep):
    """Return (result, tuning): one row with the test MSE, the times and the chosen
    setting, and one row for each setting that the tuning scored."""
    module = estimators.load(slug)
    n, p = X.shape
    Xtr, Xte, Ytr, Yte = split(X, Y, rep)
    seed = SEED + rep
    if slug in COMPILED:
        m = min(WARM_ROWS, len(Ytr))
        module.learner(folds(m, seed), seed, p).fit(Xtr[:m], Ytr[:m]).predict(Xte[:5])
    t0 = time.perf_counter()
    fitted = module.learner(folds(len(Ytr), seed), seed, p).fit(Xtr, Ytr)
    t_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = fitted.predict(Xte)
    t_pred = time.perf_counter() - t0
    base = {"data": dataset, "n": n, "d": p, "method": slug, "learner": module.NAME, "rep": rep}
    result = pd.DataFrame([{
        **base, "n_train": len(Ytr), "n_test": len(Yte),
        "mse": float(np.mean((pred - Yte) ** 2)), "time fitting": t_fit, "time predicting": t_pred,
        "cpu": cpu(), **_flat(module.chosen(fitted), "chosen_"),
    }])
    rows = [{**base, **row} for row in module.tuning(fitted)]
    tuning = pd.DataFrame(rows) if rows else pd.DataFrame(columns=list(base))
    return result, tuning
