"""The estimators that notes/sobolev-mars.tex added second, at the user's
request: mixed Sobolev KRR of the usual norm with HAR's depth weights, of
orders 0 and 1, so that the comparison of the anchored and the usual-norm
kernels is not also a comparison of depth tuning.

They have a registry of their own, which py/added_fits.py reads, because
more.py and py/more_fits.py are inputs of the fits of the four estimators
added first, and an edit there would rerun those fits.
"""

# The slugs of the estimators added second.
SLUGS = ["mixed_sobolev_depth", "mixed_sobolev1_depth"]

# Their code is compiled by numba, so the Table 1 fits give them an untimed
# warm-up fit first, as they do the paper's kernel methods (table1/fit.py).
COMPILED = set(SLUGS)


def register():
    """Add these estimators to estimators.SLUGS, for estimators.load."""
    import estimators

    for slug in SLUGS:
        if slug not in estimators.SLUGS:
            estimators.SLUGS.append(slug)
