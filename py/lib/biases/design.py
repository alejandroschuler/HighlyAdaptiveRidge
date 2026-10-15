"""Settings of the Section 4.3 simulation.

The Snakefile lists which cells exist (DGPs, estimators, blocks of
repetitions). The settings inside a cell are here, so that a change reruns the
cells that read them.
"""

SEED = 20261015

# Each dataset is a training draw of N points, on which an estimator tunes
# itself by the 5-fold CV of Section 4.1 and fits, and a test draw of N_TEST
# points, on which its fit is compared with the truth.
N = 500
N_TEST = 5000

# Every truth is scaled to mean 0 and variance 1 under its covariate
# distribution, so the noise sd SIGMA makes the truth explain 1 / (1 + SIGMA^2)
# of the variance of Y, 80%, in every DGP.
SIGMA = 0.5

# The number of covariates of every DGP.
P = 4
