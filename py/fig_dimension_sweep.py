"""The dimension sweep for the referee: test RMSE against p for each method."""
import pandas as pd

import artefacts as art
from display import figures, summaries

df = pd.concat([pd.read_csv(art.track_read(p)) for p in snakemake.input.cells], ignore_index=True)
art.save_figure(snakemake.output[0], figures.dimension_sweep(summaries.dimension(df)),
                metadata={"CreationDate": None})
