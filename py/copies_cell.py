"""notes/bias-dimension.tex, one cell: one estimator on one copies<p> DGP
(biases/copies.py) over a block of repetitions, on one thread."""
from biases import copies

w = snakemake.wildcards
results, tuning = copies.run(w.dgp, w.method, snakemake.params.reps)
results.to_csv(snakemake.output.result, index=False)
tuning.to_csv(snakemake.output.tuning, index=False)
