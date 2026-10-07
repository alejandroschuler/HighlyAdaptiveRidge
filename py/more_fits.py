"""The fits of the estimators that notes/sobolev-mars.tex adds to Section 4:
one Table 1 fit, or the Figure 1 fits of one estimator, on one thread.

    more_fits.py table1 <data csv> <dataset> <method> <rep> <result csv> <tuning csv>
    more_fits.py fits <method> <output csv>

The Snakefile runs this script from a shell command, so that the MARS fits can
run under the interpreter of envs/mars. It first lets estimators.load find the
added estimators, which estimators.SLUGS, the paper's list, leaves out
(estimators/more.py), and then fits them with the code of the paper's fits.
"""
import sys

from estimators import more

more.register()

task, args = sys.argv[1], sys.argv[2:]
if task == "table1":
    from table1 import data, fit

    path, dataset, method, rep, result_path, tuning_path = args
    fit.COMPILED.update(more.COMPILED)
    X, Y = data.load(path)
    result, tuning = fit.run(X, Y, method, dataset, int(rep))
    result.to_csv(result_path, index=False)
    tuning.to_csv(tuning_path, index=False)
elif task == "fits":
    from sim import demo

    method, out = args
    demo.fits(method).to_csv(out, index=False)
else:
    sys.exit(f"unknown task {task!r}; the tasks are table1 and fits")
