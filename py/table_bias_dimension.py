"""Section 4.3, claim 1 with more covariates: the table of each estimator's mean
test MSE on the wave along two covariates copied onto every pair of p
covariates (tab:bias-dimension), and the numbers file with the ratios that the
text quotes and the settings that ran, read from the stored results."""
import numpy as np
import pandas as pd

import artefacts as art
from biases import copies, copies_display as show, design, report, summary
from display import tables

results = pd.concat([pd.read_csv(art.track_read(p)) for p in snakemake.input.results], ignore_index=True)
methods = list(snakemake.params.methods)
names = dict(snakemake.params.names)

# ---- the table
art.save_table(snakemake.output.table, show.table(results, methods, lambda m: tables._header(m, names)),
               label="tab:bias-dimension", escape=False, column_format="r|" + "r" * len(methods))

# ---- the numbers
m = summary.means(results)
r = show.ratios_to_rbf(results, methods)
ps = sorted(results["p"].unique())
dgp_of = {p: d for d, p in copies.DGPS.items()}


def mse(x):
    """A mean test MSE, rounded as in the table."""
    return rf"\num{{{report.mse_cell(x)}}}"


def rng(values, digits=2):
    """The smallest and the largest of some values, as two macros' values."""
    return art.num(min(values), digits), art.num(max(values), digits)


numbers = {}
# The fewest covariates: each estimator's mean test MSE over that of radial
# basis KRR, so above 1 when radial basis KRR is better.
first = r[r["p"] == ps[0]]
numbers["resBiasDimMinLow"], numbers["resBiasDimMinHigh"] = rng(1 / first["ratio"])
# The smallest p at which every estimator beats radial basis KRR, its 95%
# interval of the ratio above 1, and the ratios there, radial basis KRR over
# the estimator; and the smallest mean test MSE there.
clear = r.assign(clear=r["low"] > 1).groupby("p")["clear"].all()
cross = int(clear[clear].index.min())
numbers["resBiasDimPCross"] = art.int(cross)
numbers["resBiasDimCrossLow"], numbers["resBiasDimCrossHigh"] = rng(r.loc[r["p"] == cross, "ratio"])
numbers["resBiasDimBestCross"] = mse(m.loc[dgp_of[cross], "mean"].min())
# The most covariates: the range of the mean test MSEs.
last = m.loc[dgp_of[ps[-1]], "mean"]
numbers["resBiasDimMaxLow"], numbers["resBiasDimMaxHigh"] = mse(last.min()), mse(last.max())
# Monte Carlo error: the largest SE of a mean in the table relative to the mean.
numbers["resBiasDimMaxRelSe"] = art.pct(100 * float((m["se"] / m["mean"]).max()), 1)


# ---- the settings that ran. The text takes n, the noise and the test draw
# from the setup of Section 4.3, so they must be those of its main run.
def single(column):
    values = results[column].unique()
    if len(values) != 1:
        raise ValueError(f"{column} took the values {list(values)}, not one")
    return values[0]


for column, value in [("n", design.N), ("n_test", design.N_TEST), ("sigma", design.SIGMA)]:
    if not np.isclose(single(column), value):
        raise ValueError(f"{column} is {single(column)}, not {value} as in the main run of Section 4.3")
reps = results.groupby(["dgp", "method"])["rep"].nunique()
if reps.nunique() != 1:
    raise ValueError(f"the cells have different numbers of repetitions: {reps.to_dict()}")
numbers.update(
    mthBiasDimPs=art.numset(ps), mthBiasDimPMin=art.int(ps[0]), mthBiasDimPMax=art.int(ps[-1]),
    mthBiasDimReps=art.int(reps.iloc[0]),
)
art.emit_numbers(snakemake.output.numbers, **numbers)
