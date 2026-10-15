"""The figure of the Section 4.3 simulation: its truths on the square of the
first two covariates. The rule's script saves it through the emit helpers."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from . import dgps  # noqa: E402

INK, MUTED = "#0b0b0b", "#898781"
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#184f95", "#6da7ec", "#f0efec", "#ec8a89", "#a7302f"])

# The panels, one column per claim: the truth of each wave as a function of
# (x_1, x_2), and the pair function g of each rule ensemble.
PANELS = [[("wave1", "Claim 1: wave, $k = 1$"), ("wave2", "Claim 1: wave, $k = 2$")],
          [("rules", "Claim 2: steps"), ("ramps", "Claim 2: ramps")],
          [("dense", "Claim 3: dense"), ("sparse", "Claim 3: sparse")]]


def _surface(name, Z):
    """The function a panel draws at the points Z of the square, centred."""
    if name.startswith("wave"):
        X = np.full((len(Z), dgps.P), 0.5)
        X[:, :2] = Z
        return dgps.truth(name, X)
    _, g, _ = dgps.DGPS[name]
    v = g(Z)
    return v - dgps.pair_moments(name)[0]


def truths(n_points, grid=300):
    """Six panels, one column per claim, with the first n_points points of the
    training draw of repetition 0, projected on (x_1, x_2)."""
    fig, axes = plt.subplots(2, 3, figsize=(7.5, 5.0), constrained_layout=True)
    t = (np.arange(grid) + 0.5) / grid
    Z = np.stack(np.meshgrid(t, t, indexing="xy"), axis=-1).reshape(-1, 2)
    for col, column in enumerate(PANELS):
        for row, (name, title) in enumerate(column):
            ax = axes[row, col]
            F = _surface(name, Z).reshape(grid, grid)
            top = np.abs(F).max()
            im = ax.imshow(F, origin="lower", extent=(0, 1, 0, 1), cmap=DIVERGING,
                           vmin=-top, vmax=top, interpolation="nearest")
            X, _ = dgps.draw(name, 0)["train"]
            ax.scatter(X[:n_points, 0], X[:n_points, 1], s=2, color=INK, alpha=0.6, lw=0)
            ax.set_title(title, fontsize=9)
            ax.set_xticks([0, 0.5, 1])
            ax.set_yticks([0, 0.5, 1])
            ax.tick_params(labelsize=7, colors=MUTED)
            ax.set_xlabel("$x_1$", fontsize=8, labelpad=1)
            ax.set_ylabel("$x_2$", fontsize=8, labelpad=1)
            cb = fig.colorbar(im, ax=ax, shrink=0.8)
            cb.ax.tick_params(labelsize=6, colors=MUTED)
    return fig
