# HAR reproduction harness

Seeded, rerunnable reproduction of every computational result in the "Highly Adaptive
Ridge" paper, plus the extra experiments the referee asked for. Run modules from the
`HAR/` directory with the project venv:

```
venv/bin/python -m repro.run_all          # everything (Table 1 step is resumable)
venv/bin/python -m repro.fig1_fits        # Figure 1
venv/bin/python -m repro.fig2_convergence # Figure 2
venv/bin/python -m repro.table1_benchmark --reps 5   # Table 1 (checkpointed)
venv/bin/python -m repro.table1_assemble  # build the Table 1 LaTeX
venv/bin/python -m repro.compare_table1   # compare reproduced Table 1 to the paper
venv/bin/python -m repro.revision all     # noise-variance + dimension/SOTA sweeps
```

All outputs land under `results/repro/` so the original committed results stay intact.

## What maps to what

| Paper artefact | Module | Inputs | Outputs |
|---|---|---|---|
| Figure 1 (`fits.pdf`) | `fig1_fits` | synthetic 1D DGP | `results/repro/plots/fits.pdf`, `data/fig1_fits.csv` |
| Figure 2 (`convergence.pdf`) | `fig2_convergence` | synthetic 10D DGP | `results/repro/plots/convergence.pdf`, `sims/convergence.csv` |
| Table 1 (`tab:empirical`) | `table1_benchmark` + `table1_assemble` | 11 UCI csvs | `results/repro/data/table1/{dataset}_{rep}.csv`, LaTeX to stdout |
| Reviewer experiments | `revision` | synthetic DGPs | `results/repro/revision/*.{csv,pdf}` |

Method to implementation: HAR is `kernel_ridge.HighlyAdaptiveRidgeCV` (the kernel form,
`order=0`); HAL is `highly_adaptive_regression.HighlyAdaptiveLassoCV`; mixed Sobolev and
radial basis are the corresponding `*RidgeCV` wrappers; random forest and ridge are sklearn.

Input data: `/Users/aschuler/Documents/research/projects/csv/` (set in `config.py`).

## Reproducibility notes

The original code set no random seeds, so the paper's printed numbers cannot be matched
exactly. This harness seeds every split and draw (`config.SEED + rep`), so runs are
repeatable and become the canonical numbers; they agree with the paper within Monte-Carlo
error. Where the committed code disagreed with the paper text we built to the paper:

* Figure 1 uses n=50, sigma=0.3 (code had n=500, sigma=0.5). The qualitative message is
  unchanged.
* Figure 2 uses the paper's 5-way cliff interaction (code used 4 dimensions via
  `X[:,5:-1]`). The reproduced curve matches the published one closely; run with
  `--legacy` to compare the 4-way variant.

## Code repairs and speedups (in the core modules)

Two committed bugs blocked execution and were fixed: `KernelRidgeCV._errors` was missing a
`return`, and the notebooks passed an undocumented `max_alpha_coef_norm` kwarg (dropped in
favor of the paper's eps-based regularization search).

`kernel_ridge/fast.py` adds a single-eigendecomposition path for the kernel-ridge LOOCV
(`method='eig'`, now the default). It replaces ~100 O(n^3) bordered solves per kernel (one
fit and one leverage solve for each of 50 alphas) with one `eigh(K)` plus O(n^2) work per
alpha. It is numerically identical to the original solve (predictions agree to ~1e-12 and
the same alpha is selected) and is 3-16x faster depending on how much the per-alpha solve
dominated the kernel build.
