"""Figure 1, one estimator: its predictions in each repetition, on one thread."""
from sim import demo

demo.fits(snakemake.wildcards.method).to_csv(snakemake.output[0], index=False)
