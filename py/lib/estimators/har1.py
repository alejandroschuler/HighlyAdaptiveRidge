"""First-order HAR, tuned as HAR (har.py) is, with the first-order kernel.

The first-order basis depends on the scale of each covariate and is anchored
at the origin, as on the unit cube of the paper, so each covariate is scaled
to [0, 1] by its minimum and maximum on the training rows of each fit, and
new points are clipped to [0, 1].

The first-order kernel costs O(p D) operations for each knot and pair of
points at depth D < p, against O(p) at the full depth, which has a closed form.
So the walk has a time budget of BUDGET seconds, with the cost of a depth D < p
taken as D, and the cost of the full depth as 1.
"""
from kernel_ridge.kernel_ridge import ClippedMinMaxScaler
from kernel_ridge.kernels import HighlyAdaptiveRidge

from .depths import PATIENCE, depth_path
from .kernel_path import KernelPathCV, path_chosen, path_tuning

NAME = "1st-order HAR"
ORDER = 1
BUDGET = 20 * 60


def _cost(p):
    """The relative cost of a first-order kernel for p covariates, by its depth."""
    return lambda kernel: 1.0 if kernel.depth < 0 or kernel.depth >= p else float(max(kernel.depth, 1))


def learner(folds, seed, p):
    return KernelPathCV(
        [HighlyAdaptiveRidge(depth=d, order=ORDER) for d in depth_path(p)],
        folds, patience=PATIENCE, scaler=ClippedMinMaxScaler, budget=BUDGET, cost=_cost(p),
    )


def _depths(fitted):
    return [k.depth for k in fitted.kernels]


def chosen(fitted):
    return path_chosen(fitted, "depth", _depths(fitted))


def tuning(fitted):
    return path_tuning(fitted, "depth", _depths(fitted))
