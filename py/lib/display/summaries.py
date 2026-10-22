"""The summary behind each figure and table, from the tidy results files."""
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


def table1_relative_mse(df, datasets, methods):
    """Table 1: the mean over repetitions of each fit's test MSE, divided by the
    smallest such mean among `methods` on the same dataset, so that the best
    method of each row has the value 1."""
    tab = table1(df, datasets, methods, "mse")
    tab[methods] = tab[methods].div(tab[methods].min(axis=1), axis=0)
    return tab


def depth_curves(df, datasets):
    """The depth figure: the mean over repetitions of the cross-validated risk of
    each method at each depth, divided by the smallest such mean on the same
    dataset over all the methods and depths, so that the best point of each
    dataset has the value 1. One row per dataset, method and depth, with `full`
    true at the full depth p."""
    agg = df.groupby(["data", "n", "d", "method", "depth"], as_index=False)["cv_risk"].mean()
    agg["relative"] = agg["cv_risk"] / agg.groupby("data")["cv_risk"].transform("min")
    agg["full"] = agg["depth"] >= agg["d"]
    agg["data"] = pd.Categorical(agg["data"], categories=datasets, ordered=True)
    return agg.sort_values(["data", "method", "depth"]).reset_index(drop=True)


def table1_time(df, datasets, methods):
    """The time table: the mean over repetitions of the seconds to tune, fit and
    predict."""
    return table1(df.assign(seconds=df["time fitting"] + df["time predicting"]), datasets, methods, "seconds")
