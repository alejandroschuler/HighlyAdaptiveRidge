"""The scale of the plain HAR kernel matrix (Section "Sectional Weights").

One draw of N points from Unif([0,1]^P), and the order-0 HAR kernel matrix on
them with every section weight one. It emits the largest and the smallest
element, and the largest off-diagonal element. The diagonal holds the largest
elements, because each point is a knot at or below itself in all P coordinates.
"""
import numpy as np

import artefacts as art
from kernel_ridge.kernels import HighlyAdaptiveRidge

N, P, SEED = 1000, 200, 2026


def sci(value, digits=1):
    """A number in scientific notation, through siunitx: \\num{1.6e60}."""
    mantissa, exponent = f"{value:.{digits}e}".split("e")
    return rf"\num{{{mantissa}e{int(exponent)}}}"


rng = np.random.default_rng(SEED)
X = rng.uniform(size=(N, P))
n, p = X.shape
kernel = HighlyAdaptiveRidge(order=0)
if not np.all(kernel.section_weights(p)[1:] == 1.0):
    raise ValueError("the kernel is not plain HAR: some section weight is not one")
K = kernel(X)
off_diagonal = K[~np.eye(n, dtype=bool)]

art.emit_numbers(
    snakemake.output[0],
    resKernelScaleMax=sci(K.max()),
    resKernelScaleMin=sci(K.min()),
    resKernelScaleMaxOff=sci(off_diagonal.max()),
    mthKernelScaleN=art.int(n),
    mthKernelScaleP=art.int(p),
    mthKernelScaleSeed=art.int(rng.bit_generator.seed_seq.entropy),
)
