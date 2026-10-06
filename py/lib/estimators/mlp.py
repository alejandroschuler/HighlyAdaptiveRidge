"""A multilayer perceptron with LAYERS hidden layers of equal width, ReLU
activations and Adam, from scikit-learn's MLPRegressor.

The covariates are standardized, and so is the outcome, on the rows each
network trains on. The first of the shared folds is the validation set: each
setting of the width and the weight decay trains on the other rows, one epoch
at a time, and its validation mean squared error is recorded after every
epoch. Training stops when that error has not improved for PATIENCE epochs, or
after MAX_EPOCHS. The setting and the number of epochs with the smallest
validation error are chosen, and a network with that setting is trained for
that many epochs on all the training rows.

The weight decay is scikit-learn's alpha. Each batch of B rows minimizes half
its mean squared error plus alpha ||W||^2 / (2 B), with W the weights.
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

NAME = "MLP"
LAYERS = 2
WIDTHS = [32, 128, 512]
DECAYS = [1e-5, 1e-3, 1e-1, 1e1]
LEARNING_RATE = 1e-3
BATCH = 64
MAX_EPOCHS = 1000
PATIENCE = 50


class TunedMLP(BaseEstimator, RegressorMixin):
    """The MLP of this module, tuned on the first of `folds`.

    After fit, cv_ holds one row for each setting: its best number of epochs,
    its validation error there (in the units of the outcome, squared), and the
    number of epochs trained before the stop.
    """

    def __init__(self, folds, seed, widths=WIDTHS, decays=DECAYS):
        self.folds = folds
        self.seed = seed
        self.widths = widths
        self.decays = decays

    def _net(self, width, decay):
        return MLPRegressor(hidden_layer_sizes=(width,) * LAYERS, activation="relu", solver="adam",
                            alpha=decay, learning_rate_init=LEARNING_RATE, batch_size=BATCH,
                            shuffle=True, random_state=self.seed)

    @staticmethod
    def _standardize(X, Y):
        sx = StandardScaler().fit(X)
        return sx, float(np.mean(Y)), float(np.std(Y))

    def fit(self, X, Y):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        train, val = self.folds[0]
        sx, my, sy = self._standardize(X[train], Y[train])
        Xt, Xv = sx.transform(X[train]), sx.transform(X[val])
        yt, yv = (Y[train] - my) / sy, (Y[val] - my) / sy
        self.cv_ = []
        for width in self.widths:
            for decay in self.decays:
                net = self._net(width, decay)
                best, best_epoch, epoch = np.inf, 0, 0
                for epoch in range(1, MAX_EPOCHS + 1):
                    net.partial_fit(Xt, yt)
                    mse = float(np.mean((net.predict(Xv) - yv) ** 2))
                    if mse < best:
                        best, best_epoch = mse, epoch
                    elif epoch - best_epoch >= PATIENCE:
                        break
                self.cv_.append({"width": width, "decay": decay, "epochs": best_epoch,
                                 "cv_risk": best * sy ** 2, "epochs_run": epoch})
        pick = min(self.cv_, key=lambda r: r["cv_risk"])
        self.width_, self.decay_, self.epochs_ = pick["width"], pick["decay"], pick["epochs"]
        self.scaler_, self.y_mean_, self.y_sd_ = self._standardize(X, Y)
        Xs, ys = self.scaler_.transform(X), (Y - self.y_mean_) / self.y_sd_
        self.net_ = self._net(self.width_, self.decay_)
        for _ in range(self.epochs_):
            self.net_.partial_fit(Xs, ys)
        return self

    def predict(self, X):
        return self.y_mean_ + self.y_sd_ * self.net_.predict(self.scaler_.transform(np.asarray(X, dtype=float)))


def learner(folds, seed, p):
    return TunedMLP(folds, seed)


def chosen(fitted):
    return {"width": fitted.width_, "decay": fitted.decay_, "epochs": fitted.epochs_, "layers": LAYERS,
            "widths": list(fitted.widths), "decays": list(fitted.decays),
            "learning_rate": LEARNING_RATE, "batch": BATCH, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
            "validation_rows": len(fitted.folds[0][1])}


def tuning(fitted):
    return fitted.cv_
