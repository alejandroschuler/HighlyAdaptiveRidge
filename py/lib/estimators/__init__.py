"""The estimators of the paper's Section 4, one module for each.

Each module defines NAME, the name the paper uses, and

    learner(folds, seed)  a fresh, unfitted estimator, tuned by cross-validation
                          over the given folds of its training data;
    chosen(fitted)        a dict of the setting that the tuning chose;
    tuning(fitted)        a frame with one row for each setting that the tuning
                          scored, with its cross-validated risk (cv_risk).

Every estimator of a cell gets the same folds, so their tuning is comparable.
This file imports none of the modules, so that a rule that fits one estimator
reads only that estimator's code, and an edit to one estimator reruns only its
cells. load(slug) imports one module by its slug.
"""
import importlib

# The slug of each estimator, in the order of the paper's Section 4.
SLUGS = ["har", "har1", "mixed_sobolev", "rbf", "hal", "rf", "gbt", "mlp", "enet"]


def load(slug):
    """The module of one estimator."""
    if slug not in SLUGS:
        raise ValueError(f"no estimator {slug!r}; the estimators are {SLUGS}")
    return importlib.import_module(f"estimators.{slug}")
