"""First-order mixed Sobolev KRR with depth: kernel ridge with the first-order
mixed Sobolev kernel of the usual norm (as in mixed_sobolev1.py), split into
the components of the classical functional ANOVA, with HAR's depth weights
(sobolev_depth.UsualSobolevDepth, order 1). It is tuned as first-order
anchored mixed Sobolev KRR (anchored_sobolev1.py) and first-order HAR are: the
depth along the depth path by 5-fold CV with early stopping, the penalty by
LOOCV inside each fold, covariates scaled to [0, 1] by their minimum and
maximum on the training rows of each fit with new points clipped to [0, 1],
and the time budget of first-order HAR, with the same relative cost of a depth.
"""
from kernel_ridge.kernel_ridge import ClippedMinMaxScaler

from .depths import PATIENCE, depth_path
from .kernel_path import KernelPathCV, path_chosen, path_tuning
from .sobolev_depth import UsualSobolevDepth

NAME = "1st-order mixed Sobolev KRR with depth"
ORDER = 1
BUDGET = 20 * 60  # first-order HAR's (har1.py)


def _cost(p):
    """The relative cost of a kernel for p covariates, by its depth, as in har1.py."""
    return lambda kernel: 1.0 if kernel.depth < 0 or kernel.depth >= p else float(max(kernel.depth, 1))


def learner(folds, seed, p):
    return KernelPathCV(
        [UsualSobolevDepth(order=ORDER, depth=d) for d in depth_path(p)],
        folds, patience=PATIENCE, scaler=ClippedMinMaxScaler, budget=BUDGET, cost=_cost(p),
    )


def _depths(fitted):
    return [k.depth for k in fitted.kernels]


def chosen(fitted):
    return path_chosen(fitted, "depth", _depths(fitted))


def tuning(fitted):
    return path_tuning(fitted, "depth", _depths(fitted))
