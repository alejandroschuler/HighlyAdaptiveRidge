"""The Table 1 benchmark (tab:empirical) and its fit times (tab:runtime): the nine
estimators of Section 4 on eleven UCI datasets.

A fit is one estimator on one dataset and one repetition, and it is one job of
the pipeline, on one thread. The Snakefile lists the fits and declares, for
each, only the code of its estimator, so that an edit to one estimator reruns
only its fits.
"""
