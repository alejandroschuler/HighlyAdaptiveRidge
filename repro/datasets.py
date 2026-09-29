"""UCI dataset loading, matching the original notebook exactly.

The original benchmark did `pd.read_csv(...); X = df.iloc[:2000,:-1]; Y = df.iloc[:2000,-1]`.
Verified: this reproduces the (n, p) reported in the paper's Table 1 for all 11 datasets
(pandas parses the comma inside concrete's header correctly, so p=8 there as in the paper).
"""
import pandas as pd

from .config import DATA_DIR, MAX_ROWS, DATASET_P


def load(name, max_rows=MAX_ROWS):
    """Return (X, Y) for a named dataset: first `max_rows` rows, features all-but-last col."""
    df = pd.read_csv(DATA_DIR / f"{name}.csv")
    X = df.iloc[:max_rows, :-1].to_numpy(dtype=float)
    Y = df.iloc[:max_rows, -1].to_numpy(dtype=float)
    expected = DATASET_P.get(name)
    if expected is not None and X.shape[1] != expected:
        raise AssertionError(
            f"{name}: loaded p={X.shape[1]} features but paper Table 1 expects p={expected}"
        )
    return X, Y
