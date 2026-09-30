"""The data-generating processes of the simulations."""
import numpy as np

from .design import CLIFF_EPS, CLIFF_X0, P


# ---- Figure 1

def demo_truth(x):
    """-x for x <= 0, and sin(2 pi x) for x > 0."""
    return np.sin(2 * np.pi * x) * (x > 0) + (-x) * (x <= 0)


def draw_demo(n, sigma, rng):
    X = rng.uniform(-1, 1, size=(n, 1))
    Y = demo_truth(X[:, 0]) + rng.normal(0, sigma, size=n)
    return X, Y


# ---- Figure 2 and the noise sweep

def ramp(x, x0=CLIFF_X0, eps=CLIFF_EPS):
    return np.clip((x - x0) / eps, 0, 1)


def convergence_truth(X):
    """A smooth 5-way interaction of X_1..X_5 minus a 5-way cliff in X_6..X_10."""
    return np.prod(X[:, 0:5], axis=1) - np.prod(ramp(X[:, 5:10]), axis=1)


def draw_convergence(n, sigma, rng):
    """normal(scale=sigma) is sigma times a standard normal draw, so datasets
    drawn from one seed at different noise levels share their covariates and
    their standardized noise."""
    X = rng.uniform(size=(n, P))
    Y = convergence_truth(X) + rng.normal(scale=sigma, size=n)
    return X, Y


# ---- The dimension sweep

def draw_interaction(n, p, rng):
    """A 2-way interaction of X_1 and X_2; the other p - 2 covariates are noise."""
    X = rng.uniform(size=(n, p))
    Y = np.cos(2 * np.pi * X[:, 0]) * np.cos(2 * np.pi * X[:, 1]) + rng.normal(scale=0.1, size=n)
    return X, Y


def draw_additive(n, p, rng):
    """A sparse additive function of X_1, X_2 and X_3."""
    X = rng.uniform(size=(n, p))
    Y = (np.sin(2 * np.pi * X[:, 0]) + (X[:, 1] > 0.5).astype(float) + X[:, 2] ** 2
         + rng.normal(scale=0.1, size=n))
    return X, Y


# Each DGP's number enters its seeds, so it must never change.
DIMENSION_DGPS = {
    "interaction": (1, draw_interaction),
    "additive": (2, draw_additive),
}
