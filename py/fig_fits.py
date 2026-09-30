"""Figure 1 (fig:fits): fits of HAR and the other methods on one-dimensional data."""
import pandas as pd

import artefacts as art
from display import figures

df = pd.read_csv(art.track_read(snakemake.input.predictions))
# No creation date in the PDF, so a rebuild from the same results gives the same file.
art.save_figure(snakemake.output[0], figures.fits(df), label="fig:fits",
                metadata={"CreationDate": None})
