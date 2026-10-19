"""Section 4.3, claim 1 with more covariates: the table of each estimator's mean
test MSE on the wave along two covariates copied onto every pair of p
covariates (tab:bias-dimension), and the numbers file with the Monte Carlo
error of the table and the settings that ran, read from the stored results."""
import numpy as np
import pandas as pd

import artefacts as art
from biases import copies_display as show, design, summary
from display import tables

results = pd.concat([pd.read_csv(art.track_read(p)) for p in snakemake.input.results], ignore_index=True)
methods = list(snakemake.params.methods)
names = dict(snakemake.params.names)

# ---- the table
art.save_table(snakemake.output.table, show.table(results, methods, lambda m: tables._header(m, names)),
               label="tab:bias-dimension", escape=False, column_format="r|" + "r" * len(methods))

# ---- the numbers
# Monte Carlo error: the largest SE of a mean in the table relative to the mean.
m = summary.means(results)
numbers = {"resBiasDimMaxRelSe": art.pct(100 * float((m["se"] / m["mean"]).max()), 1)}


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
numbers.update(mthBiasDimPs=art.numset(sorted(results["p"].unique())),
               mthBiasDimReps=art.int(reps.iloc[0]))
art.emit_numbers(snakemake.output.numbers, **numbers)
