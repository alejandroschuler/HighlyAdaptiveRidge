"""Section 4.3: the truths of the inductive-bias simulation on the square of the
first two covariates (fig:bias-truths), with points of one training draw."""
import artefacts as art
from biases import figures

POINTS = 300
art.save_figure(snakemake.output.figure, figures.truths(POINTS), label="fig:bias-truths",
                metadata={"CreationDate": None})
art.emit_numbers(snakemake.output.numbers, mthBiasTruthPoints=art.int(POINTS))
