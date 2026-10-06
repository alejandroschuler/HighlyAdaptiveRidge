"""One cell of each simulation. Each returns a tidy frame for results/."""
import numpy as np
import pandas as pd

from . import design
from .dgps import DIMENSION_DGPS, draw_convergence
from .learners import dimension_learners, har_learner


def convergence(sigma, n):
    """Figure 2 and the noise sweep: HAR's test MSE at one noise level and
    sample size, for each repetition. The seed leaves out sigma, so the noise
    levels are compared on the same datasets."""
    rows = []
    for rep in range(design.CONVERGENCE_REPS):
        rng = np.random.default_rng([design.SEED, n, rep])
        X, Y = draw_convergence(n + design.N_TEST, sigma, rng)
        learner = har_learner()
        learner.fit(X[:n], Y[:n])
        mse = float(np.mean((learner.predict(X[n:]) - Y[n:]) ** 2))
        rows.append({"sigma": sigma, "n": n, "d": design.P, "rep": rep, "mse": mse})
    return pd.DataFrame(rows)


def dimension(dgp, p, n_jobs=-1):
    """The dimension sweep: each method's test MSE on one DGP at one p."""
    number, draw = DIMENSION_DGPS[dgp]
    n = design.DIMENSION_N
    rows = []
    for rep in range(design.DIMENSION_REPS):
        rng = np.random.default_rng([design.SEED, number, p, rep])
        X, Y = draw(n + design.N_TEST, p, rng)
        for name, learner in dimension_learners(rep, n_jobs).items():
            learner.fit(X[:n], Y[:n])
            mse = float(np.mean((learner.predict(X[n:]) - Y[n:]) ** 2))
            rows.append({"dgp": dgp, "p": p, "rep": rep, "learner": name, "mse": mse})
    return pd.DataFrame(rows)
