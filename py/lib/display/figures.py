"""The figures. Each function returns a matplotlib figure, and the rule's
script saves it through the emit helpers."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

FITS_PANELS = [
    "Ridge Regression", "Random Forest", "Radial Basis KRR",
    "Mixed Sobolev KRR", "HAL", "HAR",
]
DIMENSION_TITLES = {"interaction": "2-way interaction", "additive": "sparse additive"}


def fits(df):
    """Figure 1: each method's fits in the three repetitions, against the truth."""
    fig, axes = plt.subplots(2, 3, figsize=(11, 7), sharex=True, sharey=True)
    grid = np.sort(df["x"].unique())
    truth = df.drop_duplicates("x").set_index("x").loc[grid, "truth"].to_numpy()
    colors = plt.cm.viridis(np.linspace(0.1, 0.8, df["rep"].nunique()))
    for ax, name in zip(axes.ravel(), FITS_PANELS):
        sub = df[df["learner"] == name]
        for rep, c in zip(sorted(sub["rep"].unique()), colors):
            r = sub[sub["rep"] == rep].sort_values("x")
            ax.plot(r["x"], r["prediction"], color=c, lw=1.3, alpha=0.9)
        ax.plot(grid, truth, color="black", lw=2.5, label="truth")
        ax.set_title(name)
        ax.set_ylim(-1.25, 1.25)
        ax.axhline(0, color="0.85", lw=0.6, zorder=0)
    for ax in axes[-1]:
        ax.set_xlabel("x")
    for ax in axes[:, 0]:
        ax.set_ylabel("prediction")
    fig.tight_layout()
    return fig


def convergence(agg):
    """Figure 2: HAR's RMSE divided by the rate, against n."""
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(agg["n"], agg["relative_rmse"], marker="o", color="#3b6ea5")
    ax.set_xlabel("n")
    ax.set_ylabel("Rate-Scaled RMSE")
    ax.set_title("Convergence of HAR relative to theorized rate")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def noise_sweep(agg):
    """The noise sweep: the curve of Figure 2 at each noise level."""
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for sigma, g in agg.groupby("sigma"):
        ax.plot(g["n"], g["relative_rmse"], marker="o", label=f"σ = {sigma}")
    ax.set_xlabel("n")
    ax.set_ylabel("Rate-Scaled RMSE")
    ax.set_title("HAR convergence under increasing noise variance")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def dimension_sweep(agg):
    """The dimension sweep: each method's test RMSE against p, one panel per DGP."""
    dgps = [d for d in DIMENSION_TITLES if d in set(agg["dgp"])]
    fig, axes = plt.subplots(1, len(dgps), figsize=(11, 3.8), squeeze=False)
    for ax, dgp in zip(axes[0], dgps):
        sub = agg[agg["dgp"] == dgp]
        for name, g in sub.groupby("learner"):
            g = g.sort_values("p")
            ax.plot(g["p"], g["rmse"], marker="o", label=name)
        ax.set_title(DIMENSION_TITLES[dgp])
        ax.set_xlabel("p (dimensions)")
        ax.set_ylabel("test RMSE")
        ax.grid(True, alpha=0.3)
    axes[0, -1].legend(fontsize=8)
    fig.tight_layout()
    return fig
