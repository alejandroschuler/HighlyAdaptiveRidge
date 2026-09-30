"""Table 1 (tab:empirical): each method's test RMSE on each dataset, averaged
over the repetitions."""
import pandas as pd

import artefacts as art
from display import summaries, tables

paths = [*snakemake.input.cells, *snakemake.input.hal]
results = pd.concat([pd.read_csv(art.track_read(p)) for p in paths], ignore_index=True)
methods = list(snakemake.params.methods)
rmse = summaries.table1_rmse(results, list(snakemake.params.datasets), methods)
art.save_table(
    snakemake.output[0], tables.empirical(rmse, methods), label="tab:empirical",
    escape=False, column_format="lrr|llllll",
)
