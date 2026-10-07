"""A multilayer perceptron with LAYERS hidden layers of equal width, ReLU
activations and Adam, from scikit-learn's MLPRegressor, tuned by 5-fold CV
over the shared folds, as the other estimators are.

The covariates are standardized, and so is the outcome, on the rows each
network trains on. For each setting of the width and the weight decay, one
network trains in each fold on the fold's training rows, one epoch at a time,
all the folds in step. After each epoch, the CV risk of that number of epochs
is the mean over the folds of the validation mean squared error. Training
stops when the CV risk has not improved for PATIENCE epochs, or after
MAX_EPOCHS. The setting and the number of epochs with the smallest CV risk are
chosen, and a network with that setting is trained for that many epochs on all
the training rows.

The weight decay is scikit-learn's alpha, its L2 penalty on the weights.

The grid is the one that a tuning on one validation fold used before (widths
32, 128 and 512, decays 1e-5 to 10), less the width 32 and the decay 10. That
tuning chose width 32 in 8 of the 55 Table 1 fits, always within a few percent
of a wider network, and never chose the decay 10, whose validation error was
1.1 to 100 times the best. The smaller grid keeps the five folds within a few
minutes on one core: an epoch of width 512 costs about ten times one of width
128.
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

NAME = "MLP"
LAYERS = 2
WIDTHS = [128, 512]
DECAYS = [1e-5, 1e-3, 1e-1]
LEARNING_RATE = 1e-3
BATCH = 64
MAX_EPOCHS = 1000
PATIENCE = 50


class TunedMLP(BaseEstimator, RegressorMixin):
    """The MLP of this module, tuned by CV over `folds`.

    After fit, cv_ holds one row for each setting: its best number of epochs,
    its CV risk there (in the units of the outcome, squared), and the number
    of epochs trained before the stop.
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

    def _fold_data(self, X, Y):
        """Each fold's standardized training and validation rows, and the scale of its outcome."""
        out = []
        for train, val in self.folds:
            sx, my, sy = self._standardize(X[train], Y[train])
            out.append((sx.transform(X[train]), (Y[train] - my) / sy, sx.transform(X[val]), (Y[val] - my) / sy, sy))
        return out

    def fit(self, X, Y):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        data = self._fold_data(X, Y)
        self.cv_ = []
        for width in self.widths:
            for decay in self.decays:
                nets = [self._net(width, decay) for _ in data]
                best, best_epoch, epoch = np.inf, 0, 0
                for epoch in range(1, MAX_EPOCHS + 1):
                    errors = []
                    for net, (Xt, yt, Xv, yv, sy) in zip(nets, data):
                        net.partial_fit(Xt, yt)
                        errors.append(float(np.mean((net.predict(Xv) - yv) ** 2)) * sy ** 2)
                    risk = float(np.mean(errors))
                    if risk < best:
                        best, best_epoch = risk, epoch
                    elif epoch - best_epoch >= PATIENCE:
                        break
                self.cv_.append({"width": width, "decay": decay, "epochs": best_epoch,
                                 "cv_risk": best, "epochs_run": epoch})
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
            "folds": len(fitted.folds)}


def tuning(fitted):
    return fitted.cv_
