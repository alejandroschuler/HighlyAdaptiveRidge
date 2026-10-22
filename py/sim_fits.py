"""Figure 1, one estimator: its predictions in each repetition, on one thread.

    sim_fits.py <method> <output csv>

The Snakefile runs this script from a shell command, as it does py/table1_fit.py.
"""
import sys

from sim import demo

method, out = sys.argv[1:]
demo.fits(method).to_csv(out, index=False)
