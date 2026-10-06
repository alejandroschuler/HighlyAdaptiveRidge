"""The depth path of HAR and first-order HAR.

The section weights are w_s = 1(|s| <= D) for a depth D (decay 1). The path
walks DEPTHS, cut below p and ending at the full depth p: steps of one at first,
where the error changes most, then about 1.5 times the previous depth
(notes/har-weights.tex chose this path). The walk stops after PATIENCE depths
in a row that do not lower the smallest cross-validated risk. A kernel costs
about the same to build at every depth, so a patience of 2 guards against a
stop on noise for little time.
"""
DEPTHS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
PATIENCE = 2


def depth_path(p):
    """The depths of the path for p covariates, in increasing order, ending at p."""
    return [d for d in DEPTHS if d < p] + [p]
