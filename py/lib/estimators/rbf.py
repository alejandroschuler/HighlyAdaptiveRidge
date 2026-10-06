"""Kernel ridge with the radial basis kernel exp(-gamma ||x - x'||^2), tuned as
HAR is (kernel_path), with the bandwidth gamma in place of the depth.

The covariates are standardized to mean 0 and variance 1 on the training rows
of each fit. Two independent standardized points are then at squared distance
2p on average, so the grid is gamma = c / p for the values c of SCALES, which
mean about the same on every dataset. The path runs from wide kernels (small
gamma) to narrow ones, and the whole grid is searched.
"""
from sklearn.preprocessing import StandardScaler

from kernel_ridge.kernels import RadialBasis

from .kernel_path import KernelPathCV, path_chosen, path_tuning

NAME = "Radial Basis KRR"
SCALES = [2.0 ** k for k in range(-6, 6)]


def learner(folds, seed, p):
    return KernelPathCV([RadialBasis(gamma=c / p) for c in SCALES], folds, scaler=StandardScaler)


def _scales(fitted):
    p = fitted.X_fit_.shape[1]
    return [k.gamma * p for k in fitted.kernels]


def chosen(fitted):
    return path_chosen(fitted, "scale", _scales(fitted))


def tuning(fitted):
    return path_tuning(fitted, "scale", _scales(fitted))
