"""One Table 1 fit: one estimator on one dataset and repetition, on one thread.

    table1_fit.py <data csv> <dataset> <method> <rep> <result csv> <tuning csv>

The Snakefile runs this script from a shell command, so that the MARS fits can
run under the interpreter of envs/mars.
"""
import sys

from table1 import data, fit

path, dataset, method, rep, result_path, tuning_path = sys.argv[1:]
X, Y = data.load(path)
result, tuning = fit.run(X, Y, method, dataset, int(rep))
result.to_csv(result_path, index=False)
tuning.to_csv(tuning_path, index=False)
