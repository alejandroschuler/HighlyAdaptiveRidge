"""The edge checks of the Section 4.3 simulation (appendix C): two estimators of
Section 4.1 with one grid extended past the edge that the pilot found them
choosing, each run on the DGPs where they chose it.

- rbf_wide: radial basis KRR, whose bandwidth grid gamma = c / p runs over
  c = 2^-6, ..., 2^5. On the wave along one covariate it chose the widest
  kernel, c = 2^-6, in nearly every fit of the pilot. Here c runs from 2^-14.
- gbt_deep: gradient boosted trees, whose depth grid ends at 8. On the waves
  along three and four covariates it chose depth 8 in part of the fits. Here
  the grid adds the depths 12 and 16.

Every other setting is that of Section 4.1. Each variant names its rows by its
own slug, so its results sit next to those of the estimator it extends.
"""
from estimators import gbt, rbf

from . import cell

WIDE_SCALES = [2.0 ** k for k in range(-14, 6)]
DEEP_DEPTHS = gbt.DEPTHS + [12, 16]


def _rbf_wide(folds, seed, p):
    return rbf.KernelPathCV([rbf.RadialBasis(gamma=c / p) for c in WIDE_SCALES], folds,
                            scaler=rbf.StandardScaler)


def _gbt_deep(folds, seed, p):
    return gbt.TunedXGB(folds, seed, depths=DEEP_DEPTHS)


# variant: (the estimator it extends, its learner)
VARIANTS = {"rbf_wide": ("rbf", _rbf_wide), "gbt_deep": ("gbt", _gbt_deep)}


def run(dgp, variant, reps):
    """cell.run for one variant."""
    slug, learner = VARIANTS[variant]
    return cell.run(dgp, slug, reps, learner=learner, method=variant)
