"""The note's Table 1, time table and ratio table (notes/sobolev-mars.tex): the
nine estimators of Section 4 and the four that the note adds, on the eleven
UCI datasets. The numbers file holds the settings that the note prints, read
from the stored fits, and the counts that its text reports."""
import json

import numpy as np
import pandas as pd

import artefacts as art
from display import summaries, tables

results = pd.concat([pd.read_csv(art.track_read(p)) for p in snakemake.input.results], ignore_index=True)
for p in snakemake.input.tunings:
    art.track_read(p)
methods = list(snakemake.params.methods)
datasets = list(snakemake.params.datasets)
# Rules between the zero-order kernels, the first-order kernels and the rest.
column_format = "lrr|llll|llll|" + "l" * (len(methods) - 8)

art.save_table(snakemake.output.rmse, tables.empirical(summaries.table1_rmse(results, datasets, methods), methods),
               label="tab:sm-rmse", escape=False, column_format=column_format)
art.save_table(snakemake.output.runtime, tables.runtime(summaries.table1_time(results, datasets, methods), methods),
               label="tab:sm-runtime", escape=False, column_format=column_format)

# The contrasts that the added kernels were chosen for, for each order: each
# limit kernel against its HAR (the knots), the usual-norm kernel with depth
# against the one without (the depth tuning), and the limit kernel against the
# usual-norm kernel with depth (the norm, with the same tuning).
CONTRASTS = [("anchored_sobolev", "har"), ("mixed_sobolev_depth", "mixed_sobolev"),
             ("anchored_sobolev", "mixed_sobolev_depth"),
             ("anchored_sobolev1", "har1"), ("mixed_sobolev1_depth", "mixed_sobolev1"),
             ("anchored_sobolev1", "mixed_sobolev1_depth")]
SHORT = {"har": "HAR", "har1": "HAR1", "anchored_sobolev": "AMS", "anchored_sobolev1": "AMS1",
         "mixed_sobolev": "MS", "mixed_sobolev1": "MS1",
         "mixed_sobolev_depth": "MSD", "mixed_sobolev1_depth": "MSD1"}
STEM = {"har": "Har", "har1": "HarOne", "anchored_sobolev": "Ams", "anchored_sobolev1": "AmsOne",
        "mixed_sobolev": "Ms", "mixed_sobolev1": "MsOne",
        "mixed_sobolev_depth": "Msd", "mixed_sobolev1_depth": "MsdOne"}  # for the macro names
headers = [f"{SHORT[a]} / {SHORT[b]}" for a, b in CONTRASTS]
ratios = summaries.table1_ratios(results, datasets, CONTRASTS)
art.save_table(snakemake.output.ratios, tables.ratios(ratios, CONTRASTS, headers), label="tab:sm-ratios",
               escape=False, column_format="lrr|rrr|rrr")


def of(method):
    return results[results["method"] == method]


def one(method, column):
    """The single value of a setting that every fit of an estimator recorded."""
    values = of(method)[f"chosen_{column}"].dropna().unique()
    if len(values) != 1:
        raise ValueError(f"{method}: {column} took the values {list(values)}, not one")
    return values[0]


folds = results["chosen_folds"].dropna().unique()
if len(folds) != 1:
    raise ValueError(f"the estimators used different numbers of folds: {list(folds)}")
mars = of("mars")


def wins(a, b):
    """The datasets on which a's mean RMSE is below b's, and those with both."""
    r = ratios[(a, b, "ratio")].dropna()
    return int((r < 1).sum()), int(r.size)


def ratio_range(a, b):
    r = ratios[(a, b, "ratio")].dropna()
    return art.num(float(r.min()), 2), art.num(float(r.max()), 2)


# A ratio within CLOSE of one counts as close, for the summary of the note.
CLOSE = 0.10
emitted = {"mthSmClose": art.pct(100 * CLOSE, 0)}
by_data = ratios.set_index("data")
for a, b in CONTRASTS:
    name = STEM[a] + STEM[b]
    w, k = wins(a, b)
    lo, hi = ratio_range(a, b)
    emitted[f"wipSm{name}Wins"] = art.int(w)
    emitted[f"wipSm{name}Sets"] = art.int(k)
    emitted[f"wipSm{name}Min"] = lo
    emitted[f"wipSm{name}Max"] = hi
    r = ratios[(a, b, "ratio")].dropna()
    emitted[f"wipSm{name}Close"] = art.int(int(((r - 1).abs() <= CLOSE).sum()))
    # slice, the dataset with the most covariates, where the kernels differ most
    on_slice = by_data[(a, b, "ratio")]["slice"]
    if pd.notna(on_slice):
        emitted[f"wipSmSlice{name}"] = art.num(float(on_slice), 2)

# The estimator with the lowest mean RMSE on each dataset, among all of them.
rmse = summaries.table1_rmse(results, datasets, methods).set_index("data")[methods]
best = rmse.idxmin(axis=1)
added = list(snakemake.params.added)
added_best = [str(d) for d, m in best.items() if m in added]
mars_to_best = rmse["mars"] / rmse.min(axis=1)

# Ratios of mean times on the datasets with both: first-order HAR over its limit
# kernel, the anchored kernels over the usual-norm kernels with depth (the same
# tuning), and the usual-norm kernels with depth over those without.
seconds = summaries.table1_time(results, datasets, methods).set_index("data")
time_ratio = (seconds["har1"] / seconds["anchored_sobolev1"]).dropna()
time_ratios = {
    "AmsMsd": seconds["anchored_sobolev"] / seconds["mixed_sobolev_depth"],
    "AmsOneMsdOne": seconds["anchored_sobolev1"] / seconds["mixed_sobolev1_depth"],
    "MsdMs": seconds["mixed_sobolev_depth"] / seconds["mixed_sobolev"],
    "MsdOneMsOne": seconds["mixed_sobolev1_depth"] / seconds["mixed_sobolev1"],
}
for key, r in time_ratios.items():
    r = r.dropna()
    emitted[f"wipSmTime{key}Min"] = art.num(float(r.min()), 1)
    emitted[f"wipSmTime{key}Max"] = art.num(float(r.max()), 1)
mars_slowest = seconds["mars"].idxmax()

# The refits of the added kernel methods whose penalty is the bottom of its grid.
kernels = results[results["method"].isin(["anchored_sobolev", "mixed_sobolev1", "anchored_sobolev1",
                                          "mixed_sobolev_depth", "mixed_sobolev1_depth"])]
budgeted = results[results["method"].isin(["anchored_sobolev1", "mixed_sobolev1_depth"])]

art.emit_numbers(
    snakemake.output.numbers,
    # the benchmark, as for the paper's Table 1
    mthSmReps=art.int(results["rep"].nunique()),
    mthSmRows=art.int(results["n"].max()),
    mthSmTrainPct=art.pct(100 * float((results["n_train"] / results["n"]).mean()), 0),
    mthSmFolds=art.int(folds[0]),
    mthSmDatasets=art.int(results["data"].nunique()),
    # first-order anchored mixed Sobolev KRR: the budget of first-order HAR
    mthSmBudget=art.int(round(float(one("anchored_sobolev1", "budget")) / 60)),
    wipSmBudgetStops=art.int(int((budgeted["chosen_stop"] == "budget").sum())),
    # MARS: the code, its environment, and what its fits did
    mthSmMarsCommit=art.text(str(one("mars", "pymars_commit"))[:7]),
    mthSmMarsSklearn=art.text(one("mars", "sklearn")),
    mthSmMainSklearn=art.pkg_version("scikit-learn"),
    mthSmMarsMaxTerms=art.int(int(one("mars", "max_terms"))),
    wipSmMarsDegreeMin=art.int(int(mars["chosen_degree"].min())),
    wipSmMarsDegreeMax=art.int(int(mars["chosen_degree"].max())),
    wipSmMarsLimitStops=art.int(int((mars["chosen_forward_stop"] == "TERM_LIMIT").sum())),
    wipSmMarsFits=art.int(len(mars)),
    wipSmMarsBest=art.int(int((best == "mars").sum())),
    wipSmMarsToBestMin=art.num(float(mars_to_best.min()), 2),
    wipSmMarsToBestMax=art.num(float(mars_to_best.max()), 2),
    wipSmMarsSlowest=art.text(mars_slowest),
    wipSmMarsSlowestMinutes=art.int(round(float(seconds.loc[mars_slowest, "mars"]) / 60)),
    # the datasets on which an added estimator has the lowest mean RMSE
    wipSmAddedBest=art.int(len(added_best)),
    wipSmAddedBestSets=art.words(added_best) if added_best else art.text("none"),
    # times
    wipSmTimeRatioMin=art.int(round(float(time_ratio.min()))),
    wipSmTimeRatioMax=art.int(round(float(time_ratio.max()))),
    # the penalty grid of the added kernel methods
    wipSmFloorPicks=art.int(int((kernels["chosen_alpha_index"] == 0).sum())),
    wipSmTopPicks=art.int(int((kernels["chosen_alpha_index"] == kernels["chosen_n_alphas"] - 1).sum())),
    wipSmKernelFits=art.int(len(kernels)),
    **emitted,
)
