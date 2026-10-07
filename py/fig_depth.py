"""The depth figure (fig:depth): the cross-validated error of HAR and mixed
Sobolev KRR at each depth of the depth path, on each dataset of Table 1,
averaged over the repetitions and divided by the dataset's best. The numbers
file holds the repetitions it averages and what the text says about it: on how
many datasets each method's best depth is below p, and how far above its best
each method's error is at the full depth p, at most."""
import pandas as pd

import artefacts as art
from display import figures, summaries

df = pd.concat([pd.read_csv(art.track_read(p)) for p in snakemake.input.curves], ignore_index=True)
datasets = list(snakemake.params.datasets)
methods = list(snakemake.params.methods)
agg = summaries.depth_curves(df, datasets)
# No creation date in the PDF, so a rebuild from the same results gives the same file.
art.save_figure(snakemake.output.figure, figures.depth_curves(agg, datasets, methods, dict(snakemake.params.names)),
                label="fig:depth", metadata={"CreationDate": None})


def summary(method):
    """For one method: the number of datasets whose best depth is below p, and the
    largest ratio, over the datasets, of the error at the full depth to the error
    at the best depth, with that dataset."""
    below, worst, worst_data = 0, 0.0, None
    for data, g in agg[agg["method"] == method].groupby("data", observed=True):
        best = g.loc[g["cv_risk"].idxmin()]
        below += int(not best["full"])
        ratio = float(g.loc[g["full"], "cv_risk"].iloc[0] / best["cv_risk"])
        if ratio > worst:
            worst, worst_data = ratio, str(data)
    return below, worst, worst_data


har_below, har_worst, har_worst_data = summary("har")
ms_below, ms_worst, ms_worst_data = summary("mixed_sobolev_depth")
art.emit_numbers(
    snakemake.output.numbers,
    mthDepthReps=art.int(df["rep"].nunique()),
    resDepthNData=art.int(df["data"].nunique()),
    resDepthBelowHar=art.int(har_below),
    resDepthBelowMs=art.int(ms_below),
    resDepthFullHar=art.num(har_worst, 1),
    resDepthFullHarData=art.text(har_worst_data),
    resDepthFullMs=art.num(ms_worst, 1),
    resDepthFullMsData=art.text(ms_worst_data),
)
