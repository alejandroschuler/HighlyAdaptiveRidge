"""Reviewer-requested experiments (referee comment 7).

The referee asked for: (a) different noise variances (in particular variance 1, not just
0.1^2), (b) a broader range of p, (c) more than one generative model, and (d) comparisons
with additional state-of-the-art methods.

Two experiments here:
  noise : the convergence DGP run at sigma in {0.1, 0.5, 1.0}; shows HAR still attains
          (and beats) the theoretical rate as noise grows.
  dim   : two generative models over a growing number of nuisance dimensions p, comparing
          HAR and mixed Sobolev KRR against gradient boosting, random forest, a neural net,
          and k-NN.

Outputs land in results/repro/revision/ so nothing overwrites the paper artefacts.

Usage:
    python -m repro.revision noise
    python -m repro.revision dim
    python -m repro.revision all
"""
import sys
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import SEED, REPRO_DIR, ensure_dirs
from .learners import har_only, sota_learners
from .fig2_convergence import ramp, X0, EPS, P, N_TEST

REVISION_DIR = REPRO_DIR / "revision"
N_RANGE_NOISE = [50, 125, 200, 300, 400, 600]
SIGMAS = [0.1, 0.5, 1.0]
NOISE_REPS = 10

DIM_P = [5, 8, 10, 15, 20, 30]
DIM_NTRAIN = 400
DIM_REPS = 3


# ---------------------------------------------------------------- noise sweep
def dgp_convergence(n, rng, sigma):
    X = rng.uniform(size=(n, P))
    Y = np.prod(X[:, 0:5], axis=1) - np.prod(ramp(X[:, 5:10]), axis=1) + rng.normal(scale=sigma, size=n)
    return X, Y


def run_noise(seed=SEED):
    rows = []
    for sigma in SIGMAS:
        for ni, n in enumerate(N_RANGE_NOISE):
            for rep in range(NOISE_REPS):
                rng = np.random.default_rng(seed + 7000 * int(sigma * 10) + 100 * ni + rep)
                X, Y = dgp_convergence(n + N_TEST, rng, sigma)
                Xtr, Ytr, Xte, Yte = X[:n], Y[:n], X[n:], Y[n:]
                m = har_only()["HAR"]
                m.fit(Xtr, Ytr)
                mse = np.mean((m.predict(Xte) - Yte) ** 2)
                rows.append({"sigma": sigma, "n": n, "rep": rep, "mse": mse})
    return pd.DataFrame(rows)


def plot_noise(df, out_path):
    agg = (
        df.groupby(["sigma", "n"], as_index=False)["mse"].mean()
        .assign(
            rmse=lambda d: np.sqrt(d["mse"]),
            rate=lambda d: d["n"] ** (-1 / 3) * np.log(d["n"]) ** (2 * (P - 1) / 3),
        )
        .assign(relative_rmse=lambda d: d["rmse"] / d["rate"])
    )
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for sigma, g in agg.groupby("sigma"):
        ax.plot(g["n"], g["relative_rmse"], marker="o", label=f"σ = {sigma}")
    ax.set_xlabel("n")
    ax.set_ylabel("Rate-Scaled RMSE")
    ax.set_title("HAR convergence under increasing noise variance")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return agg


# --------------------------------------------------------------- dimension sweep
def dgp_interaction(n, p, rng):
    X = rng.uniform(size=(n, p))
    Y = np.cos(2 * np.pi * X[:, 0]) * np.cos(2 * np.pi * X[:, 1]) + rng.normal(scale=0.1, size=n)
    return X, Y


def dgp_additive(n, p, rng):
    X = rng.uniform(size=(n, p))
    Y = np.sin(2 * np.pi * X[:, 0]) + (X[:, 1] > 0.5).astype(float) + X[:, 2] ** 2 + rng.normal(scale=0.1, size=n)
    return X, Y


DGPS = {"2-way interaction": dgp_interaction, "sparse additive": dgp_additive}


def run_dim(seed=SEED):
    rows = []
    for di, (dgp_name, dgp) in enumerate(DGPS.items()):
        for pi, p in enumerate(DIM_P):
            for rep in range(DIM_REPS):
                rng = np.random.default_rng(seed + 9000 + 5000 * di + 137 * pi + rep)
                X, Y = dgp(DIM_NTRAIN + N_TEST, p, rng)
                Xtr, Ytr, Xte, Yte = X[:DIM_NTRAIN], Y[:DIM_NTRAIN], X[DIM_NTRAIN:], Y[DIM_NTRAIN:]
                for name, learner in sota_learners(random_state=rep).items():
                    learner.fit(Xtr, Ytr)
                    mse = np.mean((learner.predict(Xte) - Yte) ** 2)
                    rows.append({"dgp": dgp_name, "p": p, "rep": rep, "learner": name, "mse": mse})
    return pd.DataFrame(rows)


def plot_dim(df, out_path):
    agg = (
        df.groupby(["dgp", "p", "learner"], as_index=False)["mse"].mean()
        .assign(rmse=lambda d: np.sqrt(d["mse"]))
    )
    dgps = list(DGPS)
    fig, axes = plt.subplots(1, len(dgps), figsize=(11, 3.8), sharey=False)
    for ax, dgp_name in zip(np.atleast_1d(axes), dgps):
        sub = agg[agg["dgp"] == dgp_name]
        for name, g in sub.groupby("learner"):
            ax.plot(g["p"], g["rmse"], marker="o", label=name)
        ax.set_title(dgp_name)
        ax.set_xlabel("p (dimensions)")
        ax.set_ylabel("test RMSE")
        ax.grid(True, alpha=0.3)
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return agg


def main():
    ensure_dirs()
    REVISION_DIR.mkdir(parents=True, exist_ok=True)
    which = sys.argv[1] if len(sys.argv) > 1 else "all"

    if which in ("noise", "all"):
        df = run_noise()
        df.to_csv(REVISION_DIR / "noise_sweep.csv", index=False)
        agg = plot_noise(df, REVISION_DIR / "noise_sweep.pdf")
        print("=== noise sweep (rate-scaled RMSE) ===")
        print(agg.pivot_table(index="n", columns="sigma", values="relative_rmse").to_string(float_format=lambda x: f"{x:.2e}"))
        print(f"wrote {REVISION_DIR / 'noise_sweep.pdf'}")

    if which in ("dim", "all"):
        df = run_dim()
        df.to_csv(REVISION_DIR / "dim_sweep.csv", index=False)
        agg = plot_dim(df, REVISION_DIR / "dim_sweep.pdf")
        print("\n=== dimension sweep (test RMSE) ===")
        print(agg.pivot_table(index=["dgp", "p"], columns="learner", values="rmse").to_string(float_format=lambda x: f"{x:.3g}"))
        print(f"wrote {REVISION_DIR / 'dim_sweep.pdf'}")


if __name__ == "__main__":
    main()
