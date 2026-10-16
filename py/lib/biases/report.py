"""The tables of the Section 4.3 simulation, one for each claim, and the numbers
its text quotes, from the per-repetition results."""
import numpy as np
import pandas as pd

from .summary import contrast, means

# The DGPs as the paper names them, and the DGPs of each claim, whose table is
# the artefact named here.
DGP_NAMES = {"wave1": "wave, $k = 1$", "wave2": "wave, $k = 2$", "wave3": "wave, $k = 3$",
             "wave4": "wave, $k = 4$", "rules": "steps", "ramps": "ramps", "dense": "dense",
             "sparse": "sparse", "flipped": "dense, flipped"}
CLAIM_TABLES = {"waves": ["wave1", "wave2", "wave3", "wave4"],
                "rules": ["rules", "ramps"],
                "density": ["dense", "sparse", "flipped"]}

# The estimators that penalize sectional variation (claim 1), those that also
# penalize squared derivatives (claim 2), and those that only penalize
# sectional variation (claim 2).
BV = ["har", "mixed_sobolev_depth", "hal", "gbt"]
SQUARED = ["har", "mixed_sobolev_depth"]
STEPS = ["hal", "gbt"]
WAVES = ["wave1", "wave2", "wave3", "wave4"]


def pct_cell(x):
    """(Used by notes/bias-dimension.tex on wip/bias-dimension.) A mean test MSE in percent of the truth's variance, to two significant
    digits: 2.3, 0.95, 31, and whole numbers from 100 up."""
    v = 100 * x
    if v >= 10:
        return f"{v:.0f}"
    s = f"{v:#.2g}"     # keeps a trailing zero, as in 4.0
    return s.rstrip(".")


def mse_cell(x):
    """A mean test MSE, which is a fraction of the truth's variance, to two
    significant digits: 0.022, 0.090, 0.35, 1.0."""
    return f"{x:#.2g}".rstrip(".")


def table(results, dgps, methods, header):
    """The table of one claim: one row per DGP, each method's mean test MSE, with
    the smallest of the row, before rounding, in bold."""
    m = means(results)["mean"]
    rows = []
    for d in dgps:
        best = min(methods, key=lambda k: m[(d, k)])
        row = {"Process": DGP_NAMES[d]}
        for k in methods:
            text = mse_cell(m[(d, k)])
            row[header(k)] = rf"\textbf{{{text}}}" if k == best else text
        rows.append(row)
    return pd.DataFrame(rows)


def contrasts(results):
    """Every contrast that a claim promises, as (claim, name, dgp_a, method_a,
    dgp_b, method_b), with the side that the claim expects to win first."""
    out = []
    for m in BV:
        out.append((1, f"wave1 {m}/rbf", "wave1", m, "wave1", "rbf"))
        for d in WAVES[1:]:
            out.append((1, f"{d} rbf/{m}", d, "rbf", d, m))
    for a in STEPS:
        for b in SQUARED:
            out.append((2, f"rules {a}/{b}", "rules", a, "rules", b))
            out.append((2, f"ramps {b}/{a}", "ramps", b, "ramps", a))
    out.append((3, "dense har/ms", "dense", "har", "dense", "mixed_sobolev_depth"))
    out.append((3, "sparse ms/har", "sparse", "mixed_sobolev_depth", "sparse", "har"))
    out.append((3, "flip har dense/flipped", "dense", "har", "flipped", "har"))
    return out


def contrast_frame(results):
    rows = []
    for claim, name, da, ma, db, mb in contrasts(results):
        rows.append({"claim": claim, "name": name, **contrast(results, da, ma, db, mb)})
    return pd.DataFrame(rows)


def ratios(results, pairs):
    """contrast() of each (dgp_a, method_a, dgp_b, method_b) in pairs."""
    return [contrast(results, *p) for p in pairs]


def flip_max_rel(results, method):
    """The largest relative difference, over the paired datasets, between a
    method's test MSE on the dense DGP and on the flipped one."""
    w = results.pivot_table(index=["dgp", "rep"], columns="method", values="mse")
    a, b = w.loc["dense"][method], w.loc["flipped"][method]
    return float(np.max(np.abs(a - b) / b))


def frequency_table(results, periods_names, k_values):
    """The frequency check: one row per frequency and one column per k, each
    cell the smallest and the largest mean test MSE of the bounded-variation
    estimators, and in parentheses that of radial basis KRR. `periods_names`
    maps a frequency to {k: DGP name}."""
    m = means(results)["mean"]
    rows = []
    for periods, by_k in periods_names.items():
        row = {"Periods": f"{periods:g}"}
        for k in k_values:
            d = by_k[k]
            bv = [m[(d, b)] for b in BV]
            row[f"$k = {k}$"] = f"{mse_cell(min(bv))}--{mse_cell(max(bv))} ({mse_cell(m[(d, 'rbf')])})"
        rows.append(row)
    return pd.DataFrame(rows)
