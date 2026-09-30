"""Load a UCI dataset and split it for one Table 1 cell."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .design import DATASET_P, MAX_ROWS, SEED, TEST_FRAC


def load(path):
    """Return (X, Y) from the first MAX_ROWS rows: the features are every
    column but the last, and the target is the last column."""
    df = pd.read_csv(path)
    X = df.iloc[:MAX_ROWS, :-1].to_numpy(dtype=float)
    Y = df.iloc[:MAX_ROWS, -1].to_numpy(dtype=float)
    name = Path(path).stem
    expected = DATASET_P.get(name)
    if expected is not None and X.shape[1] != expected:
        raise AssertionError(
            f"{name}: loaded p={X.shape[1]} features, but Table 1 has p={expected}"
        )
    return X, Y


def split(X, Y, rep):
    """The random train/test split of repetition `rep`."""
    return train_test_split(X, Y, test_size=TEST_FRAC, random_state=SEED + rep)
