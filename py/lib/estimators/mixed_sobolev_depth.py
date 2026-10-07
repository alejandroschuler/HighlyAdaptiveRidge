"""Mixed Sobolev KRR with depth: kernel ridge with the paper's mixed Sobolev
kernel of the usual norm, split into the components of the classical
functional ANOVA, with HAR's depth weights (sobolev_depth.UsualSobolevDepth,
order 0). It is tuned as anchored mixed Sobolev KRR (anchored_sobolev.py) and
HAR are: the depth along the depth path (depths.py) by 5-fold CV with early
stopping, the penalty by LOOCV inside each fold (kernel_path.py), and each
covariate scaled to [0, 1] by its minimum and maximum on the training rows of
each fit, with new points clipped to [0, 1].
"""
from kernel_ridge.kernel_ridge import ClippedMinMaxScaler

from .depths import PATIENCE, depth_path
from .kernel_path import KernelPathCV, path_chosen, path_tuning
from .sobolev_depth import UsualSobolevDepth

NAME = "Mixed Sobolev KRR with depth"
ORDER = 0


def learner(folds, seed, p):
    return KernelPathCV([UsualSobolevDepth(order=ORDER, depth=d) for d in depth_path(p)],
                        folds, patience=PATIENCE, scaler=ClippedMinMaxScaler)


def _depths(fitted):
    return [k.depth for k in fitted.kernels]


def chosen(fitted):
    return path_chosen(fitted, "depth", _depths(fitted))


def tuning(fitted):
    return path_tuning(fitted, "depth", _depths(fitted))
