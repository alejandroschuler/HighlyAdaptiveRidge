"""One cell of the dimension sweep: every method on one DGP at one p."""
import numba

from sim import cells

# The thread count changes no result, only the running time.
n_jobs = min(snakemake.threads, numba.config.NUMBA_NUM_THREADS)
numba.set_num_threads(n_jobs)

out = cells.dimension(snakemake.wildcards.dgp, int(snakemake.wildcards.p), n_jobs)
out.to_csv(snakemake.output[0], index=False)
