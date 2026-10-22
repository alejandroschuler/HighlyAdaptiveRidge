"""The figures. Each function returns a matplotlib figure, and the rule's
script saves it through the emit helpers."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402


def fits(df, methods, cols=3, panel_height=3.0, titles=None):
    """Figure 1: each estimator's fits in the repetitions, against the truth, one
    panel per estimator in the order of `methods`, `cols` panels to a row, each
    row `panel_height` inches high. `titles` maps a method to its panel title;
    the others get the estimator's name."""
    rows = -(-len(methods) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(10, panel_height * rows), sharex=True, sharey=True)
    grid = np.sort(df["x"].unique())
    truth = df.drop_duplicates("x").set_index("x").loc[grid, "truth"].to_numpy()
    colors = plt.cm.viridis(np.linspace(0.1, 0.8, df["rep"].nunique()))
    for ax, method in zip(axes.ravel(), methods):
        sub = df[df["method"] == method]
        for rep, c in zip(sorted(sub["rep"].unique()), colors):
            r = sub[sub["rep"] == rep].sort_values("x")
            ax.plot(r["x"], r["prediction"], color=c, lw=1.3, alpha=0.9)
        ax.plot(grid, truth, color="black", lw=2.5, label="truth")
        ax.set_title((titles or {}).get(method, sub["learner"].iloc[0]))
        ax.set_ylim(-1.25, 1.25)
        ax.axhline(0, color="0.85", lw=0.6, zorder=0)
    for ax in axes.ravel()[len(methods):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("x")
    for ax in axes[:, 0]:
        ax.set_ylabel("prediction")
    fig.tight_layout()
    return fig


# The colors and markers of the depth figure's two methods: the first two
# categorical slots of the dataviz palette, with a marker each, so that the
# methods differ by more than color.
DEPTH_STYLES = [("#2a78d6", "o"), ("#eb6834", "s")]


def depth_curves(agg, datasets, methods, names, log_range=10):
    """The depth figure: each method's cross-validated risk at each depth of the
    path, relative to the dataset's best (summaries.depth_curves), one panel per
    dataset, with the depth on a log scale and a dotted line at 1. A panel whose
    largest value exceeds `log_range` has its error axis on a log scale too, and
    its title says so."""
    cols = 4
    rows = -(-(len(datasets) + 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(10, 2.35 * rows))
    for ax, data in zip(axes.ravel(), datasets):
        sub = agg[agg["data"] == data]
        p = int(sub["d"].iloc[0])
        for method, (color, marker) in zip(methods, DEPTH_STYLES):
            g = sub[sub["method"] == method]
            ax.plot(g["depth"], g["relative"], color=color, marker=marker, ms=4, lw=1.6,
                    label=names.get(method, method.upper()))
        ax.axhline(1, color="0.6", lw=0.8, ls=":", zorder=0)
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:g}"))
        log_y = sub["relative"].max() > log_range
        if log_y:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:g}"))
        ax.set_title(f"{data} ($p$ = {p}{', log scale' if log_y else ''})", fontsize=10)
        ax.tick_params(labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(True, which="major", color="0.92", lw=0.6)
    for ax in axes.ravel()[len(datasets):]:
        ax.axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    axes.ravel()[len(datasets)].legend(handles, labels, loc="center", frameon=False, fontsize=10)
    for ax in axes[-1]:
        ax.set_xlabel("depth $D$", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("relative CV error", fontsize=9)
    fig.tight_layout()
    return fig
