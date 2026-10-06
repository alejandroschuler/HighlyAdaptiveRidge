"""Table 1 (tab:empirical) and the time table (tab:runtime): each estimator's test
RMSE and its time to tune, fit and predict, on each dataset, averaged over the
repetitions. The numbers file holds the settings of the estimators and of the
benchmark, read from the stored fits."""
import json
import math

import numpy as np
import pandas as pd

import artefacts as art
from display import summaries, tables

results = pd.concat([pd.read_csv(art.track_read(p)) for p in snakemake.input.results], ignore_index=True)
for p in snakemake.input.tunings:
    art.track_read(p)
methods = list(snakemake.params.methods)
datasets = list(snakemake.params.datasets)
column_format = "lrr|" + "l" * len(methods)

art.save_table(snakemake.output.rmse, tables.empirical(summaries.table1_rmse(results, datasets, methods), methods),
               label="tab:empirical", escape=False, column_format=column_format)
art.save_table(snakemake.output.runtime, tables.runtime(summaries.table1_time(results, datasets, methods), methods),
               label="tab:runtime", escape=False, column_format=column_format)


def one(method, column):
    """The single value of a setting that every fit of an estimator recorded."""
    values = results.loc[results["method"] == method, f"chosen_{column}"].dropna().unique()
    if len(values) != 1:
        raise ValueError(f"{method}: {column} took the values {list(values)}, not one")
    return values[0]


def one_list(method, column):
    return json.loads(one(method, column))


def _exponent10(x):
    k = round(math.log10(x))
    if not math.isclose(x, 10.0 ** k, rel_tol=1e-9):
        raise ValueError(f"{x} is not a power of ten")
    return k


def power10(x):
    """\\num{e-12}, which siunitx prints as 10^-12, for an exact power of ten."""
    return rf"\num{{e{_exponent10(x)}}}"


def power10_list(xs):
    """\\numlist{e-5;e-3}, which siunitx prints as 10^-5 and 10^-3."""
    return r"\numlist{" + ";".join(f"e{_exponent10(x)}" for x in xs) + "}"


def minutes(seconds):
    return art.int(round(float(seconds) / 60))


# The depth path of the dataset with the most covariates, without its last
# depth, which is p.
widest = results.loc[(results["method"] == "har"), "d"].max()
har_path = json.loads(results.loc[(results["method"] == "har") & (results["d"] == widest), "chosen_path"].iloc[0])
scales = one_list("rbf", "path")
folds = results["chosen_folds"].dropna().unique()
if len(folds) != 1:
    raise ValueError(f"the estimators used different numbers of folds: {list(folds)}")

art.emit_numbers(
    snakemake.output.numbers,
    # the benchmark
    mthTableReps=art.int(results["rep"].nunique()),
    mthTableRows=art.int(results["n"].max()),
    mthTableTrainPct=art.pct(100 * float((results["n_train"] / results["n"]).mean()), 0),
    mthTableTestPct=art.pct(100 * float((results["n_test"] / results["n"]).mean()), 0),
    mthFolds=art.int(folds[0]),
    mthCpu=art.text(results["cpu"].iloc[0]),
    # the kernel methods' penalty grid and LOOCV
    mthKrrNAlphas=art.int(one("har", "n_alphas")),
    mthKrrEps=power10(one("har", "eps")),
    mthKrrFloor=power10(round(float(np.median(results.loc[results["method"] == "har", "chosen_alpha_floor_ratio"])), 15)),
    # HAR and first-order HAR
    mthHarDepths=art.numlist([int(d) for d in har_path[:-1]]),
    mthDepthPatience=art.int(one("har", "patience")),
    mthHarOneBudget=minutes(one("har1", "budget")),
    # radial basis KRR: gamma = 2^k / p
    mthRbfScaleMin=art.int(round(math.log2(min(scales)))),
    mthRbfScaleMax=art.int(round(math.log2(max(scales)))),
    # HAL
    mthHalNAlphas=art.int(one("hal", "n_alphas")),
    mthHalEps=power10(one("hal", "eps")),
    mthHalPatience=art.int(one("hal", "patience")),
    mthHalBudget=minutes(one("hal", "budget")),
    # random forest
    mthRfTrees=art.int(one("rf", "n_estimators")),
    mthSklearn=art.pkg_version("scikit-learn"),
    # gradient-boosted trees
    mthGbtDepths=art.numlist(one_list("gbt", "depths")),
    mthGbtRates=art.numlist(one_list("gbt", "rates")),
    mthGbtMaxTrees=art.int(one("gbt", "max_trees")),
    mthGbtPatience=art.int(one("gbt", "patience")),
    mthXgboost=art.pkg_version("xgboost"),
    # the MLP
    mthMlpLayers=art.int(one("mlp", "layers")),
    mthMlpWidths=art.numlist(one_list("mlp", "widths")),
    mthMlpDecays=power10_list(one_list("mlp", "decays")),
    mthMlpRate=power10(one("mlp", "learning_rate")),
    mthMlpBatch=art.int(one("mlp", "batch")),
    mthMlpMaxEpochs=art.int(one("mlp", "max_epochs")),
    mthMlpPatience=art.int(one("mlp", "patience")),
    # the elastic net
    mthEnetMixtures=art.numlist(one_list("enet", "mixtures")),
    mthEnetNAlphas=art.int(one("enet", "n_alphas")),
    mthEnetEps=power10(one("enet", "eps")),
)
