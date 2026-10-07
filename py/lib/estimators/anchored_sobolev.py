"""Anchored mixed Sobolev KRR: kernel ridge with the limit of the zero-order HAR
kernel for uniform covariates, with HAR's depth weights
(sobolev_kernels.AnchoredSobolev, order 0), tuned as HAR (har.py): the depth
along the depth path (depths.py) by 5-fold CV with early stopping, and the
penalty by LOOCV inside each fold (kernel_path.py).

The kernel is defined on the unit cube, so each covariate is scaled to [0, 1]
by its minimum and maximum on the training rows of each fit, and new points
are clipped to [0, 1], as for mixed Sobolev KRR (mixed_sobolev.py).
"""
from kernel_ridge.kernel_ridge import ClippedMinMaxScaler

from .depths import PATIENCE, depth_path
from .kernel_path import KernelPathCV, path_chosen, path_tuning
from .sobolev_kernels import AnchoredSobolev

NAME = "Anchored mixed Sobolev KRR"
ORDER = 0


def learner(folds, seed, p):
    return KernelPathCV([AnchoredSobolev(order=ORDER, depth=d) for d in depth_path(p)],
                        folds, patience=PATIENCE, scaler=ClippedMinMaxScaler)


def _depths(fitted):
    return [k.depth for k in fitted.kernels]


def chosen(fitted):
    return path_chosen(fitted, "depth", _depths(fitted))


def tuning(fitted):
    return path_tuning(fitted, "depth", _depths(fitted))
