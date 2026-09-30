"""Table 1 as the paper prints it."""
import pandas as pd

# Column headers, as the paper sets them.
HEADERS = {
    "HAR": "HAR",
    "HAL": "HAL",
    "Mixed Sobolev KRR": r"\makecell{Mixed\\Sobolev\\KRR}",
    "Radial Basis KRR": r"\makecell{Radial\\Basis\\KRR}",
    "Random Forest": r"\makecell{Random\\Forest}",
    "Ridge Regression": r"\makecell{Ridge\\Regression}",
}


def rmse_cell(x):
    """Three significant digits in the paper's style: 4.05, 8.74e-1, 1.05e+1."""
    mantissa, exponent = f"{x:.2e}".split("e")
    e = int(exponent)
    return mantissa if e == 0 else f"{mantissa}e{e:+d}"


def empirical(tab, methods):
    """The body of Table 1: one row per dataset, the lowest RMSE of each row in
    bold, and --- where a method was not run."""
    rows = []
    for _, r in tab.iterrows():
        present = {m: r[m] for m in methods if pd.notna(r[m])}
        best = min(present, key=present.get)
        row = {"data": str(r["data"]), "$n$": int(r["n"]), "$p$": int(r["d"])}
        for m in methods:
            cell = rmse_cell(r[m]) if m in present else "---"
            row[HEADERS[m]] = rf"\textbf{{{cell}}}" if m == best else cell
        rows.append(row)
    return pd.DataFrame(rows)
