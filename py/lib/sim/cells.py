"""One cell of each simulation. Each returns a tidy frame for results/."""
import numpy as np
import pandas as pd

from . import design
from .dgps import DIMENSION_DGPS, demo_truth, draw_convergence, draw_demo
from .learners import demo_learners, dimension_learners, har_learner


def fits(rep, n_jobs=-1):
    """Figure 1, one repetition: each method's predictions on a grid."""
    rng = np.random.default_rng(design.SEED + rep)
    X, Y = draw_demo(design.DEMO_N, design.DEMO_SIGMA, rng)
    grid = np.linspace(-1, 1, design.DEMO_GRID)
    truth = demo_truth(grid)
    frames = []
    for name, learner in demo_learners(rep, n_jobs).items():
        learner.fit(X, Y)
        frames.append(pd.DataFrame({
            "rep": rep, "learner": name, "x": grid, "truth": truth,
            "prediction": learner.predict(grid.reshape(-1, 1)),
        }))
    return pd.concat(frames, ignore_index=True)


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
