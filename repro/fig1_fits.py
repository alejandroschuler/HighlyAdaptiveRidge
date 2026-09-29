"""Figure 1 (fig:fits): qualitative 1D fits of six methods.

Paper DGP (main.tex 244-250):
    X ~ Unif([-1, 1]),  Y = N(0, 0.3^2) + { -x          if x <= 0
                                           { sin(2 pi x)  if x  > 0
    n = 50, repeated 3 times.

NOTE (flagged): the committed fits.ipynb used n=500 and sigma=0.5. Per the discrepancy
decision we build to the paper text (n=50, sigma=0.3). run() is parameterized by
(n, sigma), so the n=500/sigma=0.5 variant can be produced with run(n=500, sigma=0.5)
for a side-by-side check; the qualitative message (HAR piecewise-constant with more,
smaller jumps than HAL; close to mixed Sobolev) is the same under both.
"""
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import SEED, PLOTS_DIR, DATA_OUT, ensure_dirs
from .learners import demo_learners

N = 50
SIGMA = 0.3
N_REPS = 3
N_GRID = 500
PANEL_ORDER = [
    "Ridge Regression", "Random Forest", "Radial Basis KRR",
    "Mixed Sobolev KRR", "HAL", "HAR",
]


def mu(x):
    return np.sin(2 * np.pi * x) * (x > 0) + (-x) * (x <= 0)


def run(n=N, sigma=SIGMA, n_reps=N_REPS, seed=SEED):
    grid = np.linspace(-1, 1, N_GRID)
    truth = mu(grid)
    rows = []
    for rep in range(n_reps):
        rng = np.random.default_rng(seed + rep)
        X = rng.uniform(-1, 1, size=(n, 1))
        Y = mu(X[:, 0]) + rng.normal(0, sigma, size=n)
        for name, learner in demo_learners(random_state=rep).items():
            learner.fit(X, Y)
            pred = learner.predict(grid.reshape(-1, 1))
            for xi, ti, pi in zip(grid, truth, pred):
                rows.append({"rep": rep, "learner": name, "x": xi, "truth": ti, "prediction": pi})
    return pd.DataFrame(rows)


def plot(df, out_path):
    fig, axes = plt.subplots(2, 3, figsize=(11, 7), sharex=True, sharey=True)
    grid = np.sort(df["x"].unique())
    truth = mu(grid)
    colors = plt.cm.viridis(np.linspace(0.1, 0.8, df["rep"].nunique()))
    for ax, name in zip(axes.ravel(), PANEL_ORDER):
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
    fig.savefig(out_path)
    plt.close(fig)


def main():
    ensure_dirs()
    df = run()
    csv_path = DATA_OUT / "fig1_fits.csv"
    pdf_path = PLOTS_DIR / "fits.pdf"
    df.to_csv(csv_path, index=False)
    plot(df, pdf_path)
    print(f"wrote {csv_path}")
    print(f"wrote {pdf_path}")


if __name__ == "__main__":
    main()
