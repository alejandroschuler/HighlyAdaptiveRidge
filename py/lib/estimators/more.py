"""The estimators that notes/sobolev-mars.tex adds to the nine of Section 4.

They are not in estimators.SLUGS, the paper's list, because every fit of
Table 1 reads estimators/__init__.py, so an edit there would rerun all of
Table 1. The scripts that fit the added estimators call register() first,
which lets estimators.load find them.
"""

# The slugs of the added estimators.
SLUGS = ["anchored_sobolev", "mixed_sobolev1", "anchored_sobolev1", "mars"]

# The added estimators whose code numba compiles. The Table 1 fits give them an
# untimed warm-up fit first, as they do the paper's (table1/fit.py).
COMPILED = {"anchored_sobolev", "mixed_sobolev1", "anchored_sobolev1"}


def register():
    """Add the added estimators to estimators.SLUGS, for estimators.load."""
    import estimators

    for slug in SLUGS:
        if slug not in estimators.SLUGS:
            estimators.SLUGS.append(slug)
