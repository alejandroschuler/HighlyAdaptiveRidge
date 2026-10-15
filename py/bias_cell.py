"""Section 4.3, one cell: one estimator on one DGP over a block of repetitions,
on one thread."""
from biases import cell

w = snakemake.wildcards
results, tuning = cell.run(w.dgp, w.method, snakemake.params.reps)
results.to_csv(snakemake.output.result, index=False)
tuning.to_csv(snakemake.output.tuning, index=False)
