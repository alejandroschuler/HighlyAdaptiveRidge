"""The figures of the Section 4.3 simulation: the truths of each claim on the
square of the first two covariates, one figure for each claim. The rule's
script saves them through the emit helpers."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from . import dgps  # noqa: E402

INK, MUTED = "#0b0b0b", "#898781"
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#184f95", "#6da7ec", "#f0efec", "#ec8a89", "#a7302f"])

# The panels of each figure: (DGP, title). Claims 2 and 3 draw the terms of the
# truth in (x_1, x_2); the terms in (x_3, x_4) are the same function of them.
WAVES = [("wave1", "$k = 1$"), ("wave2", "$k = 2$")]
RULES = [("rules", r"steps, $\theta = 0$"), ("ramps", rf"ramps, $\theta = {dgps.RAMP_WIDTH:g}$")]
DENSITY = [("dense", "dense"), ("sparse", "sparse"), ("flipped", "flipped")]

PANEL = 1.45    # inches of width per panel: the panels match across figures, and three
                # panels with their colour bars fill the 406pt text width at print size


def _surface(name, Z):
    """The truth at the points Z of the square: the whole truth for a wave,
    which depends on (x_1, x_2) alone, and otherwise its terms in (x_1, x_2),
    half of its mean taken off."""
    if name in dgps.WAVES:
        X = np.full((len(Z), dgps.P), 0.5)
        X[:, :2] = Z
        return dgps.truth(name, X)
    _, g, _ = dgps.DGPS[name]
    mean, sd = dgps.moments(name)
    return (g(Z) - mean / (dgps.P // 2)) / sd


def _row(panels, n_points=0, grid=300, decorate=None, shared=True):
    """One row of panels, with one colour scale for the row if shared and one
    for each panel otherwise, and the first n_points points of the training
    draw of repetition 0, projected on (x_1, x_2)."""
    fig, axes = plt.subplots(1, len(panels), constrained_layout=True)
    t = (np.arange(grid) + 0.5) / grid
    Z = np.stack(np.meshgrid(t, t, indexing="xy"), axis=-1).reshape(-1, 2)
    surfaces = [_surface(name, Z).reshape(grid, grid) for name, _ in panels]
    width = PANEL * len(panels) + (0.6 if shared else 0.4 * len(panels))
    fig.set_size_inches(width, 1.95)
    row_top = max(np.abs(F).max() for F in surfaces)
    for ax, (name, title), F in zip(axes, panels, surfaces):
        top = row_top if shared else np.abs(F).max()
        im = ax.imshow(F, origin="lower", extent=(0, 1, 0, 1), cmap=DIVERGING,
                       vmin=-top, vmax=top, interpolation="nearest")
        if not shared:
            cb = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
            cb.ax.tick_params(labelsize=6, colors=MUTED)
        if n_points:
            X, _ = dgps.draw(name, 0)["train"]
            ax.scatter(X[:n_points, 0], X[:n_points, 1], s=2, color=INK, alpha=0.6, lw=0)
        if decorate:
            decorate(ax, name)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.tick_params(labelsize=7, colors=MUTED)
        ax.set_xlabel("$x_1$", fontsize=8, labelpad=1)
        ax.set_ylabel("$x_2$", fontsize=8, labelpad=1)
    if shared:
        cb = fig.colorbar(im, ax=list(axes), shrink=0.75, pad=0.02)
        cb.ax.tick_params(labelsize=6, colors=MUTED)
    return fig


def _direction(ax, name):
    """The centre of the square, about which the waves turn, and an arrow along
    the direction in which the wave varies."""
    k, _ = dgps.WAVES[name]
    v = np.array([1.0, 1.0 if k > 1 else 0.0]) / np.sqrt(min(k, 2))
    ax.annotate("", xy=0.5 + 0.3 * v, xytext=(0.5, 0.5),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2, shrinkA=0, shrinkB=0))
    ax.plot([0.5], [0.5], marker="o", ms=4, color=INK)


def waves():
    """Claim 1: the waves along one and two covariates."""
    return _row(WAVES, decorate=_direction)


def rules():
    """Claim 2: the steps and the ramps."""
    return _row(RULES)


def density(n_points):
    """Claim 3: the dense, sparse and flipped processes, with training points.
    The sparse truth reaches far above the others where data are rare, so each
    panel has its own colour scale."""
    return _row(DENSITY, n_points=n_points, shared=False)
