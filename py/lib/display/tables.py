"""Table 1 and the time table as the paper prints them."""
import pandas as pd

# Column headers, as the paper sets them, by estimator slug.
HEADERS = {
    "har": "HAR",
    "rbf": r"\makecell{Radial\\Basis\\KRR}",
    "hal": "HAL",
    "rf": r"\makecell{Random\\Forest}",
    "gbt": r"\makecell{Gradient\\Boosted\\Trees}",
    "mlp": "MLP",
    "enet": r"\makecell{Elastic\\Net}",
    "mars": "MARS",
}


def seconds_cell(x):
    """Two significant digits, and whole seconds from 10 up: 0.031, 0.42, 3.4, 45, 1200."""
    if x >= 10:
        return f"{x:.0f}"
    return f"{float(f'{x:.2g}'):g}"


def ratio_cell(x):
    """Three significant digits, and whole numbers from 100 up: 1.00, 1.23, 12.3, 123."""
    if x >= 100:
        return f"{x:.0f}"
    return f"{x:.1f}" if x >= 10 else f"{x:.2f}"


def _header(m, names):
    """A column header: the paper's name for the method if `names` has one, a
    word to a line as in HEADERS, else the header of HEADERS."""
    if m not in (names or {}):
        return HEADERS[m]
    words = names[m].split()
    return r"\makecell{" + r"\\".join(words) + "}" if len(words) > 1 else words[0]


def _body(tab, methods, cell, bold_min, names=None):
    rows = []
    for _, r in tab.iterrows():
        present = {m: r[m] for m in methods if m in r and pd.notna(r[m])}
        best = min(present, key=present.get) if bold_min else None
        row = {"data": str(r["data"]), "rows": int(r["n"]), "$p$": int(r["d"])}
        for m in methods:
            text = cell(r[m]) if m in present else "---"
            row[_header(m, names)] = rf"\textbf{{{text}}}" if m == best else text
        rows.append(row)
    return pd.DataFrame(rows)


def relative(tab, methods, names=None):
    """The body of Table 1: one row per dataset, each method's test MSE as a
    multiple of the row's smallest, which is in bold, and --- where a method was
    not run."""
    return _body(tab, methods, ratio_cell, bold_min=True, names=names)


def runtime(tab, methods, names=None):
    """The body of the time table: mean seconds, --- where a method was not run."""
    return _body(tab, methods, seconds_cell, bold_min=False, names=names)
