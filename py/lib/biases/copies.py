"""The dimension check of the Section 4.3 wave (the paper's table
tab:bias-dimension): the wave along two covariates copied onto every pair of p
covariates.

The DGP "copies<p>" has p uniform covariates (p even) and the raw truth

    w(x_1, x_2) + w(x_3, x_4) + ... + w(x_{p-1}, x_p),

where w is the Section 4.3 wave along two covariates, sin(2 pi m t_2) with
m = dgps.WAVE_PERIODS. The copies are functions of disjoint sets of independent
covariates, so the raw truth has mean 0 and variance p/2 times that of w, and
the scaled truth has variance 1 at every p.

The HAR norm and the mixed Sobolev norm are sums over sections, and the copies
live on disjoint sections, so the squared norm of the scaled truth is the
squared norm of w over its variance, the same at every p. Every covariate
carries signal, so a radial basis kernel cannot drop any of them.

Within a repetition the DGPs are nested: the first 4 covariates and the noise
are those of the Section 4.3 main run (dgps.draws), and the other covariates
come from a second stream of the same repetition, of which copies<p> takes the
first p - 4 columns. So the comparisons across p, like those between
estimators, are paired.
"""
import time

import numpy as np
import pandas as pd

import estimators
from estimators.folds import folds

from . import design, dgps
from .cell import DROP, _plain

P_VALUES = [4, 8, 12, 16, 24]
P_MAX = max(P_VALUES)
DGPS = {f"copies{p}": p for p in P_VALUES}

K = 2   # the wave along two covariates


def raw_truth(X):
    w = dgps.wave(K, dgps.WAVE_PERIODS)
    return sum(w(X[:, j:j + K]) for j in range(0, X.shape[1], K))


def truth(name, X):
    """The scaled truth: mean 0 and variance 1 under uniform covariates."""
    p = DGPS[name]
    _, sd = dgps.wave_moments(K, dgps.WAVE_PERIODS)
    return raw_truth(X) / (np.sqrt(p // K) * sd)


def draw(name, rep):
    """One dataset of copies<p>: (X, Y) for the training draw and (X, f(X)) for
    the test draw."""
    p = DGPS[name]
    d = dgps.draws(rep)
    rng = np.random.default_rng([design.SEED, rep, 1])
    extra = {"train": rng.random((design.N, P_MAX - design.P)),
             "test": rng.random((design.N_TEST, P_MAX - design.P))}
    X = np.hstack([d["U_train"], extra["train"][:, :p - design.P]])
    Xte = np.hstack([d["U_test"], extra["test"][:, :p - design.P]])
    return {"train": (X, truth(name, X) + design.SIGMA * d["e_train"]),
            "test": (Xte, truth(name, Xte))}


def run(dgp, slug, reps):
    """cell.run for a DGP of this module: one row per repetition with the test
    MSE against the truth and the chosen setting, and one row for each setting
    that the tuning scored, with its CV risk."""
    module = estimators.load(slug)
    results, tunings = [], []
    for rep in reps:
        data = draw(dgp, rep)
        (X, Y), (Xte, fte) = data["train"], data["test"]
        seed = design.SEED + rep
        start = time.perf_counter()
        fitted = module.learner(folds(len(Y), seed), seed, X.shape[1]).fit(X, Y)
        pred = fitted.predict(Xte)
        seconds = time.perf_counter() - start
        row = {"dgp": dgp, "method": slug, "rep": rep, "n": len(Y), "n_test": len(fte),
               "p": X.shape[1], "sigma": design.SIGMA,
               "mse": float(np.mean((pred - fte) ** 2)), "seconds": seconds}
        row.update({f"chosen_{k}": _plain(v) for k, v in module.chosen(fitted).items()
                    if k not in DROP})
        results.append(row)
        for t in module.tuning(fitted):
            tunings.append({"dgp": dgp, "method": slug, "rep": rep,
                            **{k: _plain(v) for k, v in dict(t).items()}})
    return pd.DataFrame(results), pd.DataFrame(tunings)
