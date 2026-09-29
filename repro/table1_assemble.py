"""Assemble Table 1 from the per-(dataset, rep) benchmark checkpoints.

Reads every results/repro/data/table1/{dataset}_{rep}.csv, averages MSE over the reps,
takes the square root, pivots to the paper's method columns, and emits both a readable
table and the LaTeX body for tab:empirical.
"""
import numpy as np
import pandas as pd

from .config import DATASETS, DATASET_P
from .table1_benchmark import TABLE1_DIR

METHOD_ORDER = [
    "HAR", "HAL", "Mixed Sobolev KRR", "Radial Basis KRR", "Random Forest", "Ridge Regression",
]


def load_all():
    frames = []
    for dataset in DATASETS:
        for path in sorted(TABLE1_DIR.glob(f"{dataset}_*.csv")):
            frames.append(pd.read_csv(path))
    if not frames:
        raise SystemExit(f"no checkpoints found in {TABLE1_DIR}")
    return pd.concat(frames, ignore_index=True)


def rmse_table(df):
    # Paper text: "took the average of the test-set RMSEs" -> mean over reps of per-rep RMSE
    # (= mean of sqrt(mse)), not sqrt of the mean MSE. The difference is small but this
    # matches the paper's stated procedure.
    agg = (
        df.assign(rmse=lambda d: np.sqrt(d["mse"]))
        .groupby(["data", "n", "d", "learner"], as_index=False)["rmse"].mean()
    )
    tab = (
        agg.pivot_table(index=["data", "n", "d"], columns="learner", values="rmse")
        .reindex(columns=METHOD_ORDER)
        .reset_index()
        .sort_values("d")
    )
    # Order datasets as in the paper (by p).
    tab["data"] = pd.Categorical(tab["data"], categories=DATASETS, ordered=True)
    return tab.sort_values(["d", "data"]).reset_index(drop=True)


def report(df):
    n_reps = df.groupby(["data", "learner"])["rep"].nunique().max()
    tab = rmse_table(df)
    print(f"reps per cell (max): {n_reps}")
    print(tab.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    print("\n--- LaTeX body ---")
    print(tab.to_latex(index=False, float_format="%.2e", na_rep="---"))
    return tab


def main():
    df = load_all()
    report(df)


if __name__ == "__main__":
    main()
