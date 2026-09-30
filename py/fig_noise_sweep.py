"""The noise sweep for the referee: Figure 2 at several noise levels."""
import pandas as pd

import artefacts as art
from display import figures, summaries

df = pd.concat([pd.read_csv(art.track_read(p)) for p in snakemake.input.cells], ignore_index=True)
art.save_figure(snakemake.output[0], figures.noise_sweep(summaries.convergence(df)),
                metadata={"CreationDate": None})
