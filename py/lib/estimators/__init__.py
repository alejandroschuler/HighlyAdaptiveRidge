"""The estimators of the paper's Section 4 and appendix C, one module for each.

Each module defines NAME, the estimator's name (the Snakefile's PAPER_NAMES
gives the paper's name where it differs), and

    learner(folds, seed, p)  a fresh, unfitted estimator for p covariates,
                             tuned by cross-validation over the given folds of
                             its training data;
    chosen(fitted)           a dict of the setting that the tuning chose;
    tuning(fitted)           a frame with one row for each setting that the
                             tuning scored, with its cross-validated risk
                             (cv_risk).

Every estimator of a cell gets the same folds, so their tuning is comparable.
This file imports none of the modules, so that a rule that fits one estimator
reads only that estimator's code, and an edit to one estimator reruns only its
cells. load(slug) imports one module by its slug. The mars module imports
pymars, which only the environment of envs/mars has.
"""
import importlib

# The slug of each estimator, in the order of the paper's full tables
# (PAPER_ALL in the Snakefile).
SLUGS = ["har", "mixed_sobolev_depth", "hal", "gbt", "rbf", "mars", "rf", "mlp", "enet"]


def load(slug):
    """The module of one estimator."""
    if slug not in SLUGS:
        raise ValueError(f"no estimator {slug!r}; the estimators are {SLUGS}")
    return importlib.import_module(f"estimators.{slug}")
