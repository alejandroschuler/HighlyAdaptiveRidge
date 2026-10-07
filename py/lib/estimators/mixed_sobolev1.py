"""First-order mixed Sobolev KRR: kernel ridge with the reproducing kernel of the
mixed Sobolev class of order 2 under the usual norm
(sobolev_kernels.UsualSobolev, order 1), the first-order analogue of the
paper's mixed Sobolev KRR (mixed_sobolev.py), tuned as it is: the penalty
chosen by LOOCV on all the training rows over the same kind of grid as HAR's
(kernel_path), with each covariate scaled to [0, 1] by its minimum and maximum
on the training rows and new points clipped to [0, 1].
"""
from kernel_ridge.kernel_ridge import ClippedMinMaxScaler

from .kernel_path import KernelPathCV, path_chosen
from .sobolev_kernels import UsualSobolev

NAME = "1st-order mixed Sobolev KRR"
ORDER = 1


def learner(folds, seed, p):
    return KernelPathCV([UsualSobolev(order=ORDER)], folds, scaler=ClippedMinMaxScaler)


def chosen(fitted):
    return path_chosen(fitted, "kernel", ["mixed Sobolev, order 2"])


def tuning(fitted):
    """No setting is tuned but the penalty, whose grid position chosen() records."""
    return []
