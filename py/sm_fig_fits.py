"""The note's Figure 1 (notes/sobolev-mars.tex): the fits of the nine estimators
of Section 4 and the four that the note adds, on the one-dimensional data of
the paper's Figure 1, and the settings of those data, read from the stored fits."""
import pandas as pd

import artefacts as art
from display import figures

df = pd.concat([pd.read_csv(art.track_read(p)) for p in snakemake.input.predictions], ignore_index=True)
# No creation date in the PDF, so a rebuild from the same results gives the same file.
# Four panels to a row, so that each row holds the four kernels of one order
# and the figure fits on a page of the note, with titles that fit a panel.
TITLES = {
    "anchored_sobolev": "Anchored mixed\nSobolev KRR",
    "mixed_sobolev_depth": "Mixed Sobolev KRR\nwith depth",
    "mixed_sobolev": "Mixed Sobolev KRR",
    "anchored_sobolev1": "1st-order anchored\nmixed Sobolev KRR",
    "mixed_sobolev1_depth": "1st-order mixed Sobolev\nKRR with depth",
    "mixed_sobolev1": "1st-order mixed\nSobolev KRR",
}
art.save_figure(snakemake.output.figure,
                figures.fits(df, list(snakemake.params.methods), cols=4, panel_height=2.4, titles=TITLES),
                label="fig:sm-fits", metadata={"CreationDate": None})
art.emit_numbers(
    snakemake.output.numbers,
    mthSmDemoN=art.int(df["n"].iloc[0]),
    mthSmDemoSigma=art.num(df["sigma"].iloc[0], 1),
    mthSmDemoReps=art.int(df["rep"].nunique()),
)
