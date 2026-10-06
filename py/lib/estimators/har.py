"""HAR: zero-order highly adaptive ridge, with its depth tuned along the depth
path (depths.py) by 5-fold CV, and the penalty by LOOCV inside each fold
(kernel_path.py).

The zero-order kernel depends on each covariate only through comparisons with
the knots, so the covariates are not scaled.
"""
from kernel_ridge.kernels import HighlyAdaptiveRidge

from .depths import PATIENCE, depth_path
from .kernel_path import KernelPathCV, path_chosen, path_tuning

NAME = "HAR"
ORDER = 0


def learner(folds, seed, p):
    return KernelPathCV([HighlyAdaptiveRidge(depth=d, order=ORDER) for d in depth_path(p)],
                        folds, patience=PATIENCE)


def _depths(fitted):
    return [k.depth for k in fitted.kernels]


def chosen(fitted):
    return path_chosen(fitted, "depth", _depths(fitted))


def tuning(fitted):
    return path_tuning(fitted, "depth", _depths(fitted))
