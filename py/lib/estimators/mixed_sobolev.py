"""Kernel ridge with the mixed Sobolev kernel, the penalty chosen by LOOCV on all
the training rows over the same kind of grid as HAR's (kernel_path).

The kernel is defined on the unit cube, so each covariate is scaled to [0, 1]
by its minimum and maximum on the training rows, and new points are clipped to
[0, 1].
"""
from kernel_ridge.kernel_ridge import ClippedMinMaxScaler
from kernel_ridge.kernels import MixedSobolev

from .kernel_path import KernelPathCV, path_chosen

NAME = "Mixed Sobolev KRR"


def learner(folds, seed, p):
    return KernelPathCV([MixedSobolev()], folds, scaler=ClippedMinMaxScaler)


def chosen(fitted):
    return path_chosen(fitted, "kernel", ["mixed Sobolev"])


def tuning(fitted):
    """No setting is tuned but the penalty, whose grid position chosen() records."""
    return []
