"""The table and the summaries of claim 1 of Section 4.3 with more covariates
(biases/copies.py, tab:bias-dimension), from the per-repetition results."""
import pandas as pd

from .report import mse_cell
from .summary import contrast, means


def table(results, methods, header):
    """One row per number of covariates p, each method's mean test MSE, with the
    smallest of the row, before rounding, in bold."""
    m = means(results)["mean"]
    p_of = results.groupby("dgp")["p"].first().sort_values()
    rows = []
    for d, p in p_of.items():
        best = min(methods, key=lambda k: m[(d, k)])
        row = {"$p$": str(p)}
        for k in methods:
            text = mse_cell(m[(d, k)])
            row[header(k)] = rf"\textbf{{{text}}}" if k == best else text
        rows.append(row)
    return pd.DataFrame(rows)


def ratios_to_rbf(results, methods):
    """For each DGP and each method other than radial basis KRR, contrast() of
    radial basis KRR against the method: above 1 when the method has the smaller
    mean test MSE."""
    rows = []
    for d, p in results.groupby("dgp")["p"].first().items():
        for k in methods:
            if k != "rbf":
                rows.append({"dgp": d, "p": p, "method": k, **contrast(results, d, "rbf", d, k)})
    return pd.DataFrame(rows).sort_values(["p", "method"])
