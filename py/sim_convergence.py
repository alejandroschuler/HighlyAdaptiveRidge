"""One cell of the convergence study: HAR at one noise level and sample size."""
import numba

from sim import cells

# The thread count changes no result, only the running time.
numba.set_num_threads(min(snakemake.threads, numba.config.NUMBA_NUM_THREADS))

out = cells.convergence(float(snakemake.wildcards.sigma), int(snakemake.wildcards.n))
out.to_csv(snakemake.output[0], index=False)
