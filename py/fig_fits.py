"""Figure 1 (fig:fits): the fits of the estimators on one-dimensional data, and
the settings of the demonstration, read from the stored fits."""
import pandas as pd

import artefacts as art
from display import figures

df = pd.concat([pd.read_csv(art.track_read(p)) for p in snakemake.input.predictions], ignore_index=True)
# No creation date in the PDF, so a rebuild from the same results gives the same file.
art.save_figure(snakemake.output.figure, figures.fits(df, list(snakemake.params.methods)), label="fig:fits",
                metadata={"CreationDate": None})
art.emit_numbers(
    snakemake.output.numbers,
    mthDemoN=art.int(df["n"].iloc[0]),
    mthDemoSigma=art.num(df["sigma"].iloc[0], 1),
    mthDemoReps=art.int(df["rep"].nunique()),
)
