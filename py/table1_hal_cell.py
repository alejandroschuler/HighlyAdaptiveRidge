"""One Table 1 cell for HAL, on one of the small datasets."""
import numba

from table1 import cell, data
from table1.hal import hal_learner

# The thread count changes no result, only the running time.
n_jobs = min(snakemake.threads, numba.config.NUMBA_NUM_THREADS)
numba.set_num_threads(n_jobs)

rep = int(snakemake.wildcards.rep)
X, Y = data.load(snakemake.input.data)
out = cell.run(X, Y, {"HAL": hal_learner(n_jobs)}, snakemake.wildcards.dataset, rep)
out.to_csv(snakemake.output[0], index=False)
