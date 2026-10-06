"""One Table 1 fit: one estimator on one dataset and repetition, on one thread."""
from table1 import data, fit

rep = int(snakemake.wildcards.rep)
X, Y = data.load(snakemake.input.data)
result, tuning = fit.run(X, Y, snakemake.wildcards.method, snakemake.wildcards.dataset, rep)
result.to_csv(snakemake.output.result, index=False)
tuning.to_csv(snakemake.output.tuning, index=False)
