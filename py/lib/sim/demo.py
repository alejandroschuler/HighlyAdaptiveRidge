"""Figure 1, one estimator: its fits to the one-dimensional demonstration data,
in each repetition, tuned as in Table 1 over 5 folds of the 50 points."""
import numpy as np
import pandas as pd

import estimators
from estimators.folds import folds

from . import design
from .dgps import demo_truth, draw_demo


def fits(slug):
    """One row for each repetition and point of the grid: the truth and the prediction."""
    module = estimators.load(slug)
    grid = np.linspace(-1, 1, design.DEMO_GRID)
    truth = demo_truth(grid)
    frames = []
    for rep in range(design.DEMO_REPS):
        seed = design.SEED + rep
        X, Y = draw_demo(design.DEMO_N, design.DEMO_SIGMA, np.random.default_rng(seed))
        fitted = module.learner(folds(len(Y), seed), seed, X.shape[1]).fit(X, Y)
        frames.append(pd.DataFrame({
            "rep": rep, "method": slug, "learner": module.NAME, "n": len(Y), "sigma": design.DEMO_SIGMA,
            "x": grid, "truth": truth, "prediction": fitted.predict(grid.reshape(-1, 1)),
        }))
    return pd.concat(frames, ignore_index=True)
