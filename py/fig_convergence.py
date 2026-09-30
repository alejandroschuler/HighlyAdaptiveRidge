"""Figure 2 (fig:convergence): HAR's RMSE relative to the theorized rate."""
import pandas as pd

import artefacts as art
from display import figures, summaries

df = pd.concat([pd.read_csv(art.track_read(p)) for p in snakemake.input.cells], ignore_index=True)
art.save_figure(snakemake.output[0], figures.convergence(summaries.convergence(df)),
                label="fig:convergence", metadata={"CreationDate": None})
