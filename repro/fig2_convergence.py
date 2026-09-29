"""Figure 2 (fig:convergence): HAR's RMSE relative to the theorized rate.

Paper DGP (main.tex 266-275):
    X ~ Unif([0,1]^10)
    Y = prod_{j=1}^{5} X_j  -  prod_{j=6}^{10} ramp(X_j; x0, eps)  +  N(0, 0.1^2)
    ramp(x) = clip((x - x0)/eps, 0, 1),  eps = 0.05,  x0 = 1 - (1/2)^(1/5) - eps

NOTE (flagged): the committed scratch.ipynb used `X[:,5:-1]`, i.e. only 4 of the 5 cliff
dimensions (indices 5..8). Per the discrepancy decision we build the paper's 5-way cliff
(indices 5..9). compare against the 4-way variant is available via --legacy to confirm the
"faster than the advertised rate" message is unchanged.

For each n we draw n train + 1000 test points, fit HAR, and record RMSE. Repeated 10 times
and averaged, then divided by the theoretical rate n^{-1/3}(log n)^{2(p-1)/3} with p=10.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import SEED, PLOTS_DIR, SIMS_OUT, ensure_dirs
from .learners import har_only

P = 10
EPS = 0.05
X0 = 1 - 2 ** (-1 / 5) - EPS
N_TEST = 1000
N_RANGE = [50, 125, 200, 300, 400, 600]
REPS = 10


def ramp(x, x0=X0, eps=EPS):
    return np.clip((x - x0) / eps, 0, 1)


def dgp(n, rng, cliff_dims=5):
    X = rng.uniform(size=(n, P))
    smooth = np.prod(X[:, 0:5], axis=1)
    cliff = np.prod(ramp(X[:, 5:5 + cliff_dims]), axis=1)
    Y = smooth - cliff + rng.normal(scale=0.1, size=n)
    return X, Y


def run(cliff_dims=5, seed=SEED):
    rows = []
    for ni, n in enumerate(N_RANGE):
        for rep in range(REPS):
            rng = np.random.default_rng(seed + 1000 * ni + rep)
            X, Y = dgp(n + N_TEST, rng, cliff_dims=cliff_dims)
            Xtr, Ytr, Xte, Yte = X[:n], Y[:n], X[n:], Y[n:]
            learner = har_only()["HAR"]
            learner.fit(Xtr, Ytr)
            mse = np.mean((learner.predict(Xte) - Yte) ** 2)
            rows.append({"n": n, "d": P, "learner": "HAR", "rep": rep, "mse": mse})
    return pd.DataFrame(rows)


def aggregate(df):
    agg = (
        df.groupby(["n", "d", "learner"], as_index=False)["mse"].mean()
        .assign(
            rmse=lambda d: np.sqrt(d["mse"]),
            rate=lambda d: d["n"] ** (-1 / 3) * np.log(d["n"]) ** (2 * (d["d"] - 1) / 3),
        )
        .assign(relative_rmse=lambda d: d["rmse"] / d["rate"])
        .sort_values("n")
    )
    return agg


def plot(agg, out_path):
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(agg["n"], agg["relative_rmse"], marker="o", color="#3b6ea5")
    ax.set_xlabel("n")
    ax.set_ylabel("Rate-Scaled RMSE")
    ax.set_title("Convergence of HAR relative to theorized rate")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    ensure_dirs()
    cliff_dims = 4 if "--legacy" in sys.argv else 5
    suffix = "_legacy4way" if cliff_dims == 4 else ""
    df = run(cliff_dims=cliff_dims)
    agg = aggregate(df)
    csv_path = SIMS_OUT / f"convergence{suffix}.csv"
    pdf_path = PLOTS_DIR / f"convergence{suffix}.pdf"
    df.to_csv(csv_path, index=False)
    plot(agg, pdf_path)
    print(agg[["n", "rmse", "rate", "relative_rmse"]].to_string(index=False))
    print(f"wrote {csv_path}")
    print(f"wrote {pdf_path}")


if __name__ == "__main__":
    main()
