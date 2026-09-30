# The notebooks before the pipeline

These three notebooks are the 2024 code that made the paper's first results.
They are kept as a record. Their import paths no longer resolve, and the
Snakefile now makes every result.

| Notebook | What it made |
|---|---|
| `notebooks/fits.ipynb` | Figure 1, `fits.pdf` in the paper. The notebook drew n = 500 points with noise sd 0.5. |
| `notebooks/scratch.ipynb` | The Table 1 benchmark, and Figure 2, `convergence.pdf` in the paper. The convergence DGP used `X[:, 5:-1]`, so its cliff has four factors. |
| `notebooks/dimension.ipynb` | An exploratory dimension study (`dim.pdf`). The paper does not use it. |

None of them set a random seed, so their outputs cannot be made again. The
seeded reproduction of June 2026 (commit 2eb4876, the `repro/` harness) agreed
with the paper to within Monte Carlo error. The pipeline keeps its seeds and
settings.
