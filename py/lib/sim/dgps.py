"""The data-generating process of the Figure 1 demonstration."""
import numpy as np


def demo_truth(x):
    """-x for x <= 0, and sin(2 pi x) for x > 0."""
    return np.sin(2 * np.pi * x) * (x > 0) + (-x) * (x <= 0)


def draw_demo(n, sigma, rng):
    X = rng.uniform(-1, 1, size=(n, 1))
    Y = demo_truth(X[:, 0]) + rng.normal(0, sigma, size=n)
    return X, Y
