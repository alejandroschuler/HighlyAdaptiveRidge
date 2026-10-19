"""The table of claim 1 of Section 4.3 with more covariates
(biases/copies.py, tab:bias-dimension), from the per-repetition results."""
import pandas as pd

from .report import mse_cell
from .summary import means


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

