# Highly Adaptive Ridge

Code for the paper *Highly Adaptive Ridge*, by Schuler, Hagemeister and van der
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
| `py/lib/table1/`, `py/lib/sim/` | the experiments: the Table 1 benchmark, and the simulations |
| `py/lib/display/` | the summaries, figures and tables |
| `py/*.py` | one script for each rule of the `Snakefile` |
| `tests/` | the unit tests of the methods (`make test`) |
| `data/` | the UCI data, not in git; see `data/README.md` |
| `legacy/` | the 2024 notebooks that the pipeline replaced |
| `paper/` | the Overleaf clone of the paper, a separate git repo |

## Reproducing the paper

Every figure and table in the paper is built by the `Snakefile`, in the
environment that `uv sync` builds from `pyproject.toml` and `uv.lock`. The
`make` targets run each build and check its output:

```
make build    # the paper's figures and tables, on main
make notes    # the experiments for the referee, on a wip branch
make status   # what is current, and what a build would redo
```

`CLAUDE.md` has the rules of the pipeline. A full build of Table 1 takes about
two hours, nearly all of it in the five fast methods. The 20 HAL cells take about
a minute.
