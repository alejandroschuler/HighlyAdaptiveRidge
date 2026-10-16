"""Section 4.3: the truths of the inductive-bias simulation on the square of the
first two covariates, one figure for each claim (fig:bias-truths-waves,
fig:bias-truths-rules, fig:bias-truths-density), with points of one training
draw in the figure of claim 3."""
import artefacts as art
from biases import figures

POINTS = 300
for key, fig in [("waves", figures.waves()), ("rules", figures.rules()),
                 ("density", figures.density(POINTS))]:
    art.save_figure(snakemake.output[key], fig, label=f"fig:bias-truths-{key}",
                    metadata={"CreationDate": None})
art.emit_numbers(snakemake.output.numbers, mthBiasTruthPoints=art.int(POINTS))
