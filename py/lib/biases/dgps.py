"""The data-generating processes (DGPs) of the Section 4.3 simulation.

Every DGP has P = 4 covariates X = Q(U), where U is uniform on [0, 1]^4 and the
quantile function Q acts on each coordinate, and Y = f(X) + SIGMA * noise, where
the truth f is a raw function scaled to mean 0 and variance 1 under the
distribution of X.

Claim 1 turns one sine wave. The wave "wave<k>" varies along the direction
(1, ..., 1, 0, ..., 0) / sqrt(k), which spans the first k covariates, about the
centre of the cube:

    sin(2 pi m (x_1 + ... + x_k - k/2) / sqrt(k)),

with m = WAVE_PERIODS periods per unit length. For k = 1 it is a main effect;
for larger k it is an interaction of k covariates. Every wave<k> is a rotation
of wave1 about the centre of the cube, so the radial basis kernel, which a
rotation leaves unchanged, sees the same function in every direction, up to the
corners of the cube.

Claims 2 and 3 use rule ensembles of two covariates, as in
notes/inductive-bias.tex,

    g(x_1, x_2) = r(x_1; c_1) + r(x_2; c_2) + r(x_1; c_3) r(x_2; c_4),

where r(t; c) rises linearly from 0 to 1 over [c - w/2, c + w/2], and is the
step 1(t >= c) when the width w is 0. The raw function puts one copy on each
pair of covariates, g(x_1, x_2) + g(x_3, x_4).

Within a repetition every DGP uses the same uniform draws and the same noise, so
contrasts between DGPs, like those between estimators, are paired.
"""
from functools import lru_cache

import numpy as np

from .design import N, N_TEST, P, SEED, SIGMA

# ---- covariates: one quantile function for every coordinate


def uniform(u):
    return u


# The skewed covariate puts DENSE_MASS of the data in [0, DENSE_TOP): density 4
# there and 1/4 on [DENSE_TOP, 1].
DENSE_TOP = 0.2
DENSE_MASS = 0.8


def skewed(u):
    return np.where(
        u < DENSE_MASS,
        u * DENSE_TOP / DENSE_MASS,
        DENSE_TOP + (u - DENSE_MASS) * (1 - DENSE_TOP) / (1 - DENSE_MASS),
    )


def skewed_flipped(u):
    """The quantile function of 1 - X for a skewed X, increasing in u: the dense
    part of the range is at the top."""
    return 1.0 - skewed(1.0 - u)


# ---- claim 1: the wave along a direction that spans k covariates

# A pilot of 20 datasets chose the frequency among 1, 1.5, 2 and 3 periods. At
# each of them, every bounded-variation estimator beat radial basis KRR at
# k = 1, and the error of radial basis KRR hardly changed with k. From 1.5 up,
# the bounded-variation estimators learned almost nothing of the wave at k = 4
# (test MSE near 100% of its variance), and from 2 up nothing at k = 3. At 1
# their error rises with k without reaching that ceiling, so the rise shows.
WAVE_PERIODS = 1.0


def wave(k, periods):
    def f(X):
        t = (X[:, :k].sum(axis=1) - k / 2) / np.sqrt(k)
        return np.sin(2 * np.pi * periods * t)

    return f


def wave_moments(k, periods):
    """The exact mean and sd of wave(k, periods) under uniform X. The
    direction's projection t is symmetric about 0, so the mean is 0, and
    E sin^2(a t) = (1 - E cos(2 a t)) / 2, where E cos(2 a t) is the product over
    the k coordinates of E cos(2 a (U - 1/2) / sqrt(k)) = sin(b) / b, with
    a = 2 pi periods and b = a / sqrt(k)."""
    a = 2 * np.pi * periods
    b = a / np.sqrt(k)
    return 0.0, float(np.sqrt((1 - (np.sin(b) / b) ** k) / 2))


# The waves: name -> (k, periods), all at WAVE_PERIODS.
WAVES = {f"wave{k}": (k, WAVE_PERIODS) for k in range(1, P + 1)}


# ---- claims 2 and 3: rule ensembles, one copy on each pair of covariates


def ramp(t, c, w):
    """0 below c - w/2, 1 above c + w/2, linear between; the step 1(t >= c) if w = 0."""
    if w == 0:
        return (t >= c).astype(float)
    return np.clip((t - c) / w + 0.5, 0.0, 1.0)


def rule_ensemble(centres, width):
    """g(x_1, x_2) of the module docstring, as a function of an (n, 2) array."""
    c1, c2, c3, c4 = centres

    def g(Z):
        z1, z2 = Z[:, 0], Z[:, 1]
        return ramp(z1, c1, width) + ramp(z2, c2, width) + ramp(z1, c3, width) * ramp(z2, c4, width)

    return g


def copies(g):
    """g on each pair of covariates, (x_1, x_2) and (x_3, x_4)."""
    def f(X):
        return sum(g(X[:, 2 * i:2 * i + 2]) for i in range(P // 2))

    return f


# Claim 2 gives the same rules sharp steps (rules) or ramps of width RAMP_WIDTH
# (ramps). Claim 3 puts every ramp across the dense part of the skewed
# covariates' range or across the sparse part; the flipped DGP is the dense DGP
# with every covariate x replaced by 1 - x.
RULE_CENTRES = (0.35, 0.55, 0.6, 0.3)
RAMP_WIDTH = 0.5
DENSE = rule_ensemble((DENSE_TOP / 2,) * 4, DENSE_TOP)
SPARSE = rule_ensemble(((1 + DENSE_TOP) / 2,) * 4, 1 - DENSE_TOP)

# name: (quantile function, the pair function g or None, the raw truth f)
DGPS = {
    **{name: (uniform, None, wave(k, m)) for name, (k, m) in WAVES.items()},
    "rules": (uniform, rule_ensemble(RULE_CENTRES, 0.0), None),
    "ramps": (uniform, rule_ensemble(RULE_CENTRES, RAMP_WIDTH), None),
    "dense": (skewed, DENSE, None),
    "sparse": (skewed, SPARSE, None),
    "flipped": (skewed_flipped, lambda Z: DENSE(1.0 - Z), None),
}
for _name, (_q, _g, _f) in list(DGPS.items()):
    if _f is None:
        DGPS[_name] = (_q, _g, copies(_g))


@lru_cache(maxsize=None)
def pair_moments(name, grid=4000):
    """The mean and the sd of the pair function g under the covariate
    distribution: a midpoint rule over (U_1, U_2) on a grid x grid lattice. The
    rules' steps and the skewed covariate's kinks sit on lattice lines, so the
    rule is exact up to rounding."""
    q, g, _ = DGPS[name]
    u = (np.arange(grid) + 0.5) / grid
    total = total_sq = 0.0
    for block in np.array_split(u, 16):     # bounded memory
        U = np.stack(np.meshgrid(block, u, indexing="ij"), axis=-1).reshape(-1, 2)
        v = g(q(U))
        total += v.sum()
        total_sq += (v * v).sum()
    mean = total / grid ** 2
    return mean, float(np.sqrt(total_sq / grid ** 2 - mean ** 2))


@lru_cache(maxsize=None)
def moments(name):
    """The mean and the sd of the raw truth under the covariate distribution.
    The copies of g are functions of disjoint sets of independent covariates, so
    the mean and the variance of their sum are P/2 times those of g."""
    if name in WAVES:
        return wave_moments(*WAVES[name])
    mean, sd = pair_moments(name)
    return (P // 2) * mean, float(np.sqrt(P // 2) * sd)


def truth(name, X):
    """The scaled truth: mean 0 and variance 1 under the covariate distribution."""
    _, _, f = DGPS[name]
    mean, sd = moments(name)
    return (f(X) - mean) / sd


def draws(rep):
    """The uniform draws and the noise of one repetition, shared by every DGP."""
    rng = np.random.default_rng([SEED, rep])
    return {"U_train": rng.random((N, P)), "U_test": rng.random((N_TEST, P)),
            "e_train": rng.standard_normal(N)}


def draw(name, rep):
    """One dataset: (X, Y) for the training draw and (X, f(X)) for the test draw.
    The flipped DGP's dataset is the dense DGP's with every covariate x replaced
    by 1 - x, since 1 - skewed(1 - u) applied to the draw 1 - U gives 1 - skewed(U)."""
    if name == "flipped":
        out = draw("dense", rep)
        return {part: (1.0 - X, v) for part, (X, v) in out.items()}
    q, _, _ = DGPS[name]
    d = draws(rep)
    X = q(d["U_train"])
    Xte = q(d["U_test"])
    return {"train": (X, truth(name, X) + SIGMA * d["e_train"]), "test": (Xte, truth(name, Xte))}
