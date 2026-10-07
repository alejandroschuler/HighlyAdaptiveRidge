"""One curve of the depth figure: the cross-validated risk of HAR or mixed
Sobolev KRR at every depth of the depth path, on one dataset and repetition,
on one thread.

The fit is the estimator's own Table 1 fit (its learner, the training part of
the repetition and the shared folds), with the early stopping removed, so the
walk visits every depth up to the full depth p. Its refit at the best depth is
not used.
"""
import pandas as pd

import estimators
from estimators import more_depth
from estimators.folds import folds
from table1 import data
from table1.design import SEED

more_depth.register()   # lets estimators.load find mixed_sobolev_depth

w = snakemake.wildcards
rep = int(w.rep)
X, Y = data.load(snakemake.input.data)
Xtr, _, Ytr, _ = data.split(X, Y, rep)
seed = SEED + rep
n, p = X.shape
fitted = estimators.load(w.method).learner(folds(len(Ytr), seed), seed, p).set_params(patience=None).fit(Xtr, Ytr)

rows = []
for k, kernel in enumerate(fitted.kernels):
    rows.append({
        "data": w.dataset, "n": n, "d": p, "method": w.method, "rep": rep,
        "depth": kernel.depth, "cv_risk": fitted.cv_risk_[k], "seconds": fitted.seconds_[k],
        **{f"fold{i}_mse": e for i, e in enumerate(fitted.fold_mse_[k])},
    })
pd.DataFrame(rows).to_csv(snakemake.output[0], index=False)
