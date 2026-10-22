# Highly Adaptive Ridge

Code for the paper *Highly Adaptive Ridge*, by Schuler, Wang, Hagemeister and van der
Laan ([arXiv:2410.02680](https://arxiv.org/pdf/2410.02680)).

## Usage

The methods are in `py/lib`. Put that folder on the Python path, for example
with `PYTHONPATH=py/lib`, then:

```
from kernel_ridge import HighlyAdaptiveRidgeCV, MixedSobolevRidgeCV
from sklearn.model_selection import train_test_split

X, X_, Y, Y_ = train_test_split(X_data, Y_data, test_size=0.2)

har = HighlyAdaptiveRidgeCV() # or MixedSobolevRidgeCV()
har.fit(X,Y)
har.predict(X_)
```

## Layout

| Path | What it holds |
|---|---|
| `py/lib/kernel_ridge/` | HAR, and kernel ridge regression with the mixed Sobolev and radial basis kernels |
| `py/lib/highly_adaptive_regression.py` | HAL, the highly adaptive lasso |
| `py/lib/estimators/` | the nine estimators of Section 4 and appendix C as the pipeline tunes them, one module each, and the code they share |
| `py/lib/table1/` | the benchmark of Section 4: the benchmark table and the time table |
| `py/lib/sim/` | the demonstration of Figure 1 |
| `py/lib/biases/` | the simulation of Section 4.3, and its check with more covariates |
| `py/lib/display/` | the summaries, figures and tables |
| `py/*.py` | the scripts that the rules of the `Snakefile` run |
| `envs/mars/` | the environment of the MARS fits (pymars needs scikit-learn 1.6) |
| `tests/` | the unit tests of the methods (`make test`) |
| `data/` | the UCI data, not in git; see `data/README.md` |

## The HAR basis in the code

The implementation uses the indicators `1(X_ij <= x_j)` on the raw covariates,
which are not scaled. This is the basis `1(0 < X_ij <= x_j)` of Section 2 of the
paper after a strictly increasing map of each covariate into (0, 1), which does
not change the fit (Section 3.3 of the paper).

## Reproducing the paper

Every figure and table in the paper is built by the `Snakefile`, in the
environment that `uv sync` builds from `pyproject.toml` and `uv.lock`. The MARS
fits run in a second environment, which `uv sync` builds when run inside `envs/mars/`. The
`make` targets run each build and check its output:

```
make build    # the paper's figures and tables, on main
make status   # what is current, and what a build would redo
```

`CLAUDE.md` has the rules of the pipeline. Each fit is one job on one thread,
so that the fit times compare. HAL stops its search over depths on a budget of
20 minutes per fit. Its first depth always runs, and on naval that first depth
takes about an hour. The cells of the Section 4.3 simulation take about 3 hours
on 10 cores, and its check with more covariates takes about 2 hours. The
datasets are available from the UCI repository, and `data/README.md` says
where to put them.
