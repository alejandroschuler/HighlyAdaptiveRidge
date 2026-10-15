"""Section 4.3, one edge-check cell: one variant (biases/variants.py) on one DGP
over a block of repetitions, on one thread."""
from biases import variants

w = snakemake.wildcards
results, tuning = variants.run(w.dgp, w.variant, snakemake.params.reps)
results.to_csv(snakemake.output.result, index=False)
tuning.to_csv(snakemake.output.tuning, index=False)
