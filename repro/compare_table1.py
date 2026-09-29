"""Compare the reproduced Table 1 against the values printed in the manuscript.

The paper had no random seeds, so an exact match is impossible; we check that each cell
agrees within Monte-Carlo error and that the per-row winner (the bolded method) is preserved.
"""
import numpy as np
import pandas as pd

from .table1_assemble import load_all, rmse_table, METHOD_ORDER

# Values transcribed from main.tex Table 1 (tab:empirical). None = "---" in the paper.
PAPER = {
    "power":   {"HAR": 4.05,    "HAL": None,    "Mixed Sobolev KRR": 4.11,    "Radial Basis KRR": 4.28,    "Random Forest": 4.11,    "Ridge Regression": 4.56},
    "yacht":   {"HAR": 0.874,   "HAL": 0.679,   "Mixed Sobolev KRR": 0.418,   "Radial Basis KRR": 0.563,   "Random Forest": 1.01,    "Ridge Regression": 8.66},
    "concrete":{"HAR": 3.65,    "HAL": 3.74,    "Mixed Sobolev KRR": 3.80,    "Radial Basis KRR": 9.23,    "Random Forest": 4.71,    "Ridge Regression": 10.5},
    "energy":  {"HAR": 0.365,   "HAL": 0.439,   "Mixed Sobolev KRR": 0.382,   "Radial Basis KRR": 0.472,   "Random Forest": 0.476,   "Ridge Regression": 2.85},
    "kin8nm":  {"HAR": 0.140,   "HAL": None,    "Mixed Sobolev KRR": 0.129,   "Radial Basis KRR": 0.0922,  "Random Forest": 0.167,   "Ridge Regression": 0.204},
    "protein": {"HAR": 1.88,    "HAL": None,    "Mixed Sobolev KRR": 1.91,    "Radial Basis KRR": 5.85,    "Random Forest": 1.86,    "Ridge Regression": 2.64},
    "wine":    {"HAR": 0.607,   "HAL": None,    "Mixed Sobolev KRR": 0.611,   "Radial Basis KRR": 0.636,   "Random Forest": 0.579,   "Ridge Regression": 0.660},
    "boston":  {"HAR": 3.33,    "HAL": 3.36,    "Mixed Sobolev KRR": 2.54,    "Radial Basis KRR": 4.65,    "Random Forest": 3.03,    "Ridge Regression": 4.51},
    "naval":   {"HAR": 7.66e-4, "HAL": None,    "Mixed Sobolev KRR": 4.16e-4, "Radial Basis KRR": 1.89e-3, "Random Forest": 8.86e-4, "Ridge Regression": 1.32e-3},
    "yearmsd": {"HAR": 11.5,    "HAL": None,    "Mixed Sobolev KRR": 9.07,    "Radial Basis KRR": 11.5,    "Random Forest": 9.46,    "Ridge Regression": 9.88},
    "slice":   {"HAR": 9.00,    "HAL": None,    "Mixed Sobolev KRR": 7.96,    "Radial Basis KRR": 0.131,   "Random Forest": 0.370,   "Ridge Regression": 0.635},
}


def winner(d):
    vals = {k: v for k, v in d.items() if v is not None and not (isinstance(v, float) and np.isnan(v))}
    return min(vals, key=vals.get) if vals else None


def main():
    df = load_all()
    tab = rmse_table(df).set_index("data")
    rel_rows, win_match = [], []
    for dataset, paper in PAPER.items():
        if dataset not in tab.index:
            continue
        repro = {m: tab.loc[dataset, m] for m in METHOD_ORDER}
        rel = {}
        for m in METHOD_ORDER:
            pv, rv = paper[m], repro[m]
            if pv is None or rv is None or (isinstance(rv, float) and np.isnan(rv)):
                rel[m] = np.nan
            else:
                rel[m] = abs(rv - pv) / abs(pv)
        rel_rows.append({"data": dataset, **rel})
        pw, rw = winner(paper), winner(repro)
        win_match.append({"data": dataset, "paper_best": pw, "repro_best": rw, "match": pw == rw})

    rel_df = pd.DataFrame(rel_rows).set_index("data")
    win_df = pd.DataFrame(win_match)
    print("=== relative |reproduced - paper| / |paper| per cell ===")
    print(rel_df.to_string(float_format=lambda x: f"{x:.0%}" if x == x else "  -"))
    print(f"\nmedian relative difference: {np.nanmedian(rel_df.to_numpy()):.0%}")
    print("\n=== per-row best method (bolded in paper) ===")
    print(win_df.to_string(index=False))
    print(f"\nwinner matches: {win_df['match'].sum()}/{len(win_df)}")


if __name__ == "__main__":
    main()
