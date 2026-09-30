"""Settings of the simulations. The Snakefile lists which cells exist; the
settings inside a cell are here."""

SEED = 20240101

# Figure 1 (fig:fits): the paper's demonstration, n = 50 and sigma = 0.3,
# repeated three times, with predictions on a grid over [-1, 1].
DEMO_N = 50
DEMO_SIGMA = 0.3
DEMO_REPS = 3
DEMO_GRID = 500

# Figure 2 (fig:convergence) and the noise sweep: X uniform on [0, 1]^10, with
# a cliff of width CLIFF_EPS placed so that about half the data fall on each
# side. Each fit is scored on N_TEST fresh points.
P = 10
CLIFF_EPS = 0.05
CLIFF_X0 = 1 - 2 ** (-1 / 5) - CLIFF_EPS
N_TEST = 1000
CONVERGENCE_REPS = 10

# The dimension sweep: n = 400 at every p, three repetitions.
DIMENSION_N = 400
DIMENSION_REPS = 3
