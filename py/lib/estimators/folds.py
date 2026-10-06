"""The cross-validation folds that every estimator of a cell shares."""
from sklearn.model_selection import KFold

# Every estimator tunes by FOLDS-fold cross-validation on the training part,
# over the same folds.
FOLDS = 5


def folds(n, seed):
    """FOLDS shuffled folds of n rows, as a list of (train, validation) index arrays."""
    return list(KFold(FOLDS, shuffle=True, random_state=seed).split(range(n)))
