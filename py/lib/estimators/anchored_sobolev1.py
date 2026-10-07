"""First-order anchored mixed Sobolev KRR: kernel ridge with the limit of the
first-order HAR kernel for uniform covariates, with HAR's depth weights
(sobolev_kernels.AnchoredSobolev, order 1), tuned as first-order HAR (har1.py):
the depth along the depth path by 5-fold CV with early stopping, the penalty by
LOOCV inside each fold, covariates scaled to [0, 1] by their minimum and
maximum on the training rows of each fit with new points clipped to [0, 1],
and the same time budget for the walk, with the same relative cost of a depth.

The budget is there only to match first-order HAR. This kernel has no sum over
knots, so a depth costs about n/p as much as it does for first-order HAR.
"""
from kernel_ridge.kernel_ridge import ClippedMinMaxScaler

from .depths import PATIENCE, depth_path
from .kernel_path import KernelPathCV, path_chosen, path_tuning
from .sobolev_kernels import AnchoredSobolev

NAME = "1st-order anchored mixed Sobolev KRR"
ORDER = 1
BUDGET = 20 * 60  # first-order HAR's (har1.py)


def _cost(p):
    """The relative cost of a kernel for p covariates, by its depth, as in har1.py."""
    return lambda kernel: 1.0 if kernel.depth < 0 or kernel.depth >= p else float(max(kernel.depth, 1))


def learner(folds, seed, p):
    return KernelPathCV(
        [AnchoredSobolev(order=ORDER, depth=d) for d in depth_path(p)],
        folds, patience=PATIENCE, scaler=ClippedMinMaxScaler, budget=BUDGET, cost=_cost(p),
    )


def _depths(fitted):
    return [k.depth for k in fitted.kernels]


def chosen(fitted):
    return path_chosen(fitted, "depth", _depths(fitted))


def tuning(fitted):
    return path_tuning(fitted, "depth", _depths(fitted))
