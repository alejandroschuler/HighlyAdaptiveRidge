"""The summary behind each figure and table, from the tidy results files."""
import numpy as np
import pandas as pd


def table1(df, datasets, methods, value):
    """Table 1 and the time table: the mean over repetitions of `value` for each
    dataset and method. One row per dataset, in the order of `datasets`, with
    its rows used (n) and covariates (d), and one column per method slug, in the
    order of `methods`; NaN where a method was not run."""
    agg = df.groupby(["data", "n", "d", "method"], as_index=False)[value].mean()
    tab = (
        agg.pivot_table(index=["data", "n", "d"], columns="method", values=value)
        .reindex(columns=methods)
        .reset_index()
    )
    tab["data"] = pd.Categorical(tab["data"], categories=datasets, ordered=True)
    return tab.sort_values("data").reset_index(drop=True)


def table1_rmse(df, datasets, methods):
    """Table 1: the mean over repetitions of each fit's test RMSE. The paper
    averages the RMSEs, not the MSEs."""
    return table1(df.assign(rmse=np.sqrt(df["mse"])), datasets, methods, "rmse")


def table1_time(df, datasets, methods):
    """The time table: the mean over repetitions of the seconds to tune, fit and
    predict."""
    return table1(df.assign(seconds=df["time fitting"] + df["time predicting"]), datasets, methods, "seconds")


def convergence(df):
    """Figure 2 and the noise sweep: the root of the mean test MSE over
    repetitions, divided by the rate n^(-1/3) (log n)^(2(p-1)/3)."""
    agg = df.groupby(["sigma", "n", "d"], as_index=False)["mse"].mean()
    agg["rmse"] = np.sqrt(agg["mse"])
    agg["rate"] = agg["n"] ** (-1 / 3) * np.log(agg["n"]) ** (2 * (agg["d"] - 1) / 3)
    agg["relative_rmse"] = agg["rmse"] / agg["rate"]
    return agg.sort_values(["sigma", "n"]).reset_index(drop=True)


def dimension(df):
    """The dimension sweep: the root of the mean test MSE over repetitions."""
    agg = df.groupby(["dgp", "p", "learner"], as_index=False)["mse"].mean()
    agg["rmse"] = np.sqrt(agg["mse"])
    return agg


def table1_ratios(df, datasets, contrasts):
    """For each dataset and each contrast (a, b) of two method slugs: the ratio of
    a's mean test RMSE over the repetitions to b's, and the number of
    repetitions in which a's RMSE is the smaller. One row per dataset, in the
    order of `datasets`, with columns (a, b, "ratio") and (a, b, "wins"); NaN
    where a method was not run."""
    rmse = df.assign(rmse=np.sqrt(df["mse"])).pivot_table(index=["data", "n", "d", "rep"], columns="method",
                                                          values="rmse")
    rows = []
    for (data, n, d), g in rmse.groupby(level=["data", "n", "d"]):
        row = {"data": data, "n": n, "d": d}
        for a, b in contrasts:
            if a in g and b in g and g[a].notna().all() and g[b].notna().all():
                row[(a, b, "ratio")] = g[a].mean() / g[b].mean()
                row[(a, b, "wins")] = int((g[a] < g[b]).sum())
            else:
                row[(a, b, "ratio")] = np.nan
                row[(a, b, "wins")] = np.nan
        rows.append(row)
    tab = pd.DataFrame(rows)
    tab["data"] = pd.Categorical(tab["data"], categories=datasets, ordered=True)
    return tab.sort_values("data").reset_index(drop=True)
