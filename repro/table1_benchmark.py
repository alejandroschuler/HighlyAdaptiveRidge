"""Table 1 (tab:empirical): test RMSE of six methods on 11 UCI datasets.

Setup (main.tex 287-292): first 2000 rows of each dataset, random 80/20 train/test split,
5 repetitions, averaged test RMSE. HAL only on the four small datasets.

This rewrite fixes the original's cumulative-snapshot double counting: each (dataset, rep)
writes its own tidy file with exactly one row per learner, so the assemble step's
group-mean is a clean unweighted average over the 5 reps.

Resumable: a (dataset, rep) whose checkpoint file already exists is skipped.

Usage:
    python -m repro.table1_benchmark                 # all datasets, 5 reps, sequential
    python -m repro.table1_benchmark --datasets yacht energy --reps 2
    python -m repro.table1_benchmark --n-jobs 4      # parallel over (dataset, rep) cells
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import SEED, DATASETS, HAL_DATASETS, TEST_FRAC, N_REPS, DATA_OUT, ensure_dirs
from . import datasets as ds


TABLE1_DIR = DATA_OUT / "table1"


def cell_path(dataset, rep, phase="krr"):
    suffix = "_hal" if phase == "hal" else ""
    return TABLE1_DIR / f"{dataset}_{rep}{suffix}.csv"


def _learners(phase, rs):
    from .learners import krr_learners, hal_learner  # import here so subprocs pick up env
    if phase == "hal":
        return {"HAL": hal_learner()}
    return krr_learners(random_state=rs)


def run_cell(dataset, rep, phase="krr", overwrite=False):
    """Fit the phase's learners on one (dataset, rep) and checkpoint a tidy CSV."""
    if phase == "hal" and dataset not in HAL_DATASETS:
        return f"skip {dataset} rep{rep} (HAL not run for this dataset)"

    out = cell_path(dataset, rep, phase)
    if out.exists() and not overwrite:
        return f"skip {dataset} rep{rep} {phase} (exists)"

    X, Y = ds.load(dataset)
    n, d = X.shape
    rs = SEED + rep
    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=TEST_FRAC, random_state=rs)

    rows = []
    for name, learner in _learners(phase, rs).items():
        t0 = time.time()
        learner.fit(Xtr, Ytr)
        t_fit = time.time() - t0
        t0 = time.time()
        pred = learner.predict(Xte)
        t_pred = time.time() - t0
        mse = float(np.mean((pred - Yte) ** 2))
        rows.append({
            "data": dataset, "n": n, "d": d, "learner": name, "rep": rep,
            "mse": mse, "time fitting": t_fit, "time predicting": t_pred,
        })
    pd.DataFrame(rows).to_csv(out, index=False)
    total = sum(r["time fitting"] for r in rows)
    return f"done {dataset} rep{rep} {phase} ({n}x{d}) in {total:.1f}s -> {out.name}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--reps", type=int, default=N_REPS)
    ap.add_argument("--phase", choices=["krr", "hal"], default="krr")
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    ensure_dirs()
    TABLE1_DIR.mkdir(parents=True, exist_ok=True)
    cells = [(dataset, rep) for dataset in args.datasets for rep in range(args.reps)]

    if args.n_jobs == 1:
        for dataset, rep in cells:
            print(run_cell(dataset, rep, phase=args.phase, overwrite=args.overwrite), flush=True)
    else:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=args.n_jobs, backend="loky")(
            delayed(run_cell)(dataset, rep, args.phase, args.overwrite) for dataset, rep in cells
        )
        for r in results:
            print(r, flush=True)


if __name__ == "__main__":
    main()
