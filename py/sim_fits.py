"""Figure 1: the predictions of each method, in each repetition."""
import numba
import pandas as pd

from sim import cells, design

# The thread count changes no result, only the running time.
n_jobs = min(snakemake.threads, numba.config.NUMBA_NUM_THREADS)
numba.set_num_threads(n_jobs)

out = pd.concat([cells.fits(rep, n_jobs) for rep in range(design.DEMO_REPS)], ignore_index=True)
out.to_csv(snakemake.output[0], index=False)
