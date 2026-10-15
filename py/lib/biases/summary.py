"""Summaries of the Section 4.3 simulation, from the per-repetition results.

Every contrast compares two mean test MSEs over the same datasets, through
their ratio, with a 95% Monte Carlo interval from the delta method on the log
of the ratio, using the covariance of the two MSEs across the paired datasets.
"""
import numpy as np
import pandas as pd

Z95 = 1.959963984540054


def wide(results):
    """One row per (DGP, repetition), one column per method: the test MSE."""
    return results.pivot_table(index=["dgp", "rep"], columns="method", values="mse")


def means(results):
    """The mean test MSE and its Monte Carlo SE for each DGP and method."""
    g = results.groupby(["dgp", "method"])["mse"]
    out = g.agg(["mean", "std", "count"])
    out["se"] = out["std"] / np.sqrt(out["count"])
    return out


def ratio(a, b):
    """The ratio of the means of two paired samples, with a 95% Monte Carlo
    interval, the paired z of their difference, and the share of pairs in which
    a is smaller. Returns a dict."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(a)
    ma, mb = a.mean(), b.mean()
    c = np.cov(a, b)
    var_log = (c[0, 0] / ma ** 2 + c[1, 1] / mb ** 2 - 2 * c[0, 1] / (ma * mb)) / n
    se_log = np.sqrt(max(var_log, 0.0))
    r = ma / mb
    d = a - b
    sd = d.std(ddof=1)
    return {"ratio": r, "low": r * np.exp(-Z95 * se_log), "high": r * np.exp(Z95 * se_log),
            "z": d.mean() / (sd / np.sqrt(n)) if sd > 0 else np.inf * np.sign(d.mean()),
            "wins": float(np.mean(a < b)), "n": n}


def contrast(results, dgp_a, method_a, dgp_b, method_b):
    """ratio() of method_a on dgp_a against method_b on dgp_b, paired by repetition."""
    w = wide(results)
    a = w.loc[dgp_a][method_a]
    b = w.loc[dgp_b][method_b]
    both = a.index.intersection(b.index)
    return ratio(a.loc[both], b.loc[both])
