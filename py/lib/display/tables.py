"""Table 1 and the time table as the paper prints them."""
import pandas as pd

# Column headers, as the paper sets them, by estimator slug.
HEADERS = {
    "har": "HAR",
    "har1": r"\makecell{1st-order\\HAR}",
    "mixed_sobolev": r"\makecell{Mixed\\Sobolev\\KRR}",
    "rbf": r"\makecell{Radial\\Basis\\KRR}",
    "hal": "HAL",
    "rf": r"\makecell{Random\\Forest}",
    "gbt": r"\makecell{Gradient\\Boosted\\Trees}",
    "mlp": "MLP",
    "enet": r"\makecell{Elastic\\Net}",
    # the estimators that notes/sobolev-mars.tex adds
    "anchored_sobolev": r"\makecell{Anchored\\Mixed\\Sobolev\\KRR}",
    "mixed_sobolev1": r"\makecell{1st-order\\Mixed\\Sobolev\\KRR}",
    "anchored_sobolev1": r"\makecell{1st-order\\Anchored\\Mixed\\Sobolev\\KRR}",
    "mars": "MARS",
    "mixed_sobolev_depth": r"\makecell{Mixed\\Sobolev\\KRR\\with depth}",
    "mixed_sobolev1_depth": r"\makecell{1st-order\\Mixed\\Sobolev\\KRR\\with depth}",
}


def rmse_cell(x):
    """Three significant digits in the paper's style: 4.05, 8.74e-1, 1.05e+1."""
    mantissa, exponent = f"{x:.2e}".split("e")
    e = int(exponent)
    return mantissa if e == 0 else f"{mantissa}e{e:+d}"


def seconds_cell(x):
    """Two significant digits, and whole seconds from 10 up: 0.031, 0.42, 3.4, 45, 1200."""
    if x >= 10:
        return f"{x:.0f}"
    return f"{float(f'{x:.2g}'):g}"


def _body(tab, methods, cell, bold_min):
    rows = []
    for _, r in tab.iterrows():
        present = {m: r[m] for m in methods if m in r and pd.notna(r[m])}
        best = min(present, key=present.get) if bold_min else None
        row = {"data": str(r["data"]), "$n$": int(r["n"]), "$p$": int(r["d"])}
        for m in methods:
            text = cell(r[m]) if m in present else "---"
            row[HEADERS[m]] = rf"\textbf{{{text}}}" if m == best else text
        rows.append(row)
    return pd.DataFrame(rows)


def empirical(tab, methods):
    """The body of Table 1: one row per dataset, the lowest RMSE of each row in
    bold, and --- where a method was not run."""
    return _body(tab, methods, rmse_cell, bold_min=True)


def runtime(tab, methods):
    """The body of the time table: mean seconds, --- where a method was not run."""
    return _body(tab, methods, seconds_cell, bold_min=False)


def ratios(tab, contrasts, headers):
    """The body of the note's ratio table: for each dataset and contrast (a, b),
    the ratio of a's mean RMSE to b's and, in parentheses, the repetitions in
    which a's RMSE is the smaller; --- where a method was not run. `headers`
    gives each contrast's column header."""
    rows = []
    for _, r in tab.iterrows():
        row = {"data": str(r["data"]), "$n$": int(r["n"]), "$p$": int(r["d"])}
        for (a, b), head in zip(contrasts, headers):
            ratio, wins = r[(a, b, "ratio")], r[(a, b, "wins")]
            row[head] = "---" if pd.isna(ratio) else f"{ratio:.2f} ({int(wins)})"
        rows.append(row)
    return pd.DataFrame(rows)
