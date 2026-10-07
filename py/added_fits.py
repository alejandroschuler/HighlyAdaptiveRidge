"""The fits of estimators added to Section 4 after the first four of
notes/sobolev-mars.tex: one Table 1 fit, or the Figure 1 fits of one
estimator, on one thread.

    added_fits.py <registry> table1 <data csv> <dataset> <method> <rep> <result csv> <tuning csv>
    added_fits.py <registry> fits <method> <output csv>

<registry> names a module of py/lib/estimators that lists the estimators in
SLUGS and COMPILED and has register(), such as more_depth. py/more_fits.py does
the same for the first four added estimators with estimators/more.py. This
script takes the registry as an argument, so that a later addition needs a new
registry and no edit of a file that earlier fits read.
"""
import importlib
import sys

registry = importlib.import_module(f"estimators.{sys.argv[1]}")
registry.register()

task, args = sys.argv[2], sys.argv[3:]
if task == "table1":
    from table1 import data, fit

    path, dataset, method, rep, result_path, tuning_path = args
    fit.COMPILED.update(registry.COMPILED)
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
