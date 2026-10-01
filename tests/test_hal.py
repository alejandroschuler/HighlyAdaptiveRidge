"""HAL against the explicit basis and sklearn's lasso and elastic net."""
import warnings
from itertools import combinations

import numpy as np
import pytest
from scipy.sparse import csc_matrix
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Lasso, LassoCV
from sklearn.model_selection import KFold

from highly_adaptive_regression import (
    HighlyAdaptiveBaseCV,
    HighlyAdaptiveLassoCV,
    _Basis,
    _predict,
    _Problem,
)


def draw(n, p, seed, discrete=False):
    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(n, p))
    if discrete:
        X = np.round(3 * X) / 3
    y = np.sin(4 * X[:, 0]) + (X[:, -1] > 0.5) + rng.normal(scale=0.3, size=n)
    return X, y


def explicit_basis(knots, X):
    """Every basis column, through the code that the HAR tests check the kernel against."""
    h = HighlyAdaptiveBaseCV()
    h.knots = knots
    return h._bases(X).astype(float)


def basis_column(X, i, smask, Xnew):
    s = [j for j in range(X.shape[1]) if (smask >> j) & 1]
    return np.all(X[i, s] <= Xnew[:, s], axis=1)


@pytest.mark.parametrize("n, p, discrete", [
    (30, 3, False), (30, 3, True), (40, 5, False), (25, 6, True), (64, 3, True), (150, 3, False),
])
def test_groups_are_the_distinct_basis_columns(n, p, discrete):
    X, _ = draw(n, p, 0, discrete)
    basis = _Basis(X, p)
    rows = np.arange(n)
    # Each column's group stores that column, and different groups store different columns.
    for col in np.random.default_rng(1).choice(basis.n_columns, size=200):
        i, smask = col % n, basis.smask[col // n]
        g = basis.group[col]
        assert np.array_equal(basis.values([g], rows)[0], basis_column(X, i, smask, X))
    values = basis.values(np.arange(basis.size.shape[0]), rows)
    assert np.unique(values, axis=0).shape[0] == values.shape[0]
    # The same distinct columns, with the same multiplicities, as the explicit basis.
    B = explicit_basis(X, X)
    cols, counts = np.unique(B.T, axis=0, return_counts=True)
    order = np.lexsort(values.T[::-1])
    assert np.array_equal(cols, values[order])
    assert np.array_equal(counts, basis.size[order])


@pytest.mark.parametrize("p", [1, 4])
def test_terms_are_the_basis_functions_at_new_points(p):
    X, _ = draw(30, p, 2)
    Xnew, _ = draw(50, p, 3)
    basis = _Basis(X, p)
    for col in np.random.default_rng(4).choice(basis.n_columns, size=50):
        i, smask = col % 30, basis.smask[col // 30]
        f = _predict(X, Xnew, np.array([i]), np.array([smask]), np.ones(1), 0.0)
        assert np.array_equal(f.astype(bool), basis_column(X, i, smask, Xnew))


@pytest.mark.parametrize("discrete", [False, True])
def test_lasso_path_has_sklearns_objective_and_fit(discrete):
    """ridge = 0: the solution may not be unique, but its objective and fitted values are."""
    X, y = draw(40, 3, 5, discrete)
    basis = _Basis(X, 3)
    prob = _Problem(basis, np.arange(40))
    B = explicit_basis(X, X)
    alpha_max = np.max(np.abs((B - B.mean(0)).T @ (y - y.mean()))) / 40
    alphas = np.geomspace(alpha_max, alpha_max * 1e-3, 20)
    cols, coefs, b, _, _, exact = prob.path(y, alphas, 0.0, 1e-12, 100_000, 10)
    for l in range(0, 20, 4):
        fit = b[l] + prob.predict(basis, cols, coefs[l:l + 1], np.zeros(1), np.arange(40))[0]
        obj = 0.5 * np.mean((y - fit) ** 2) + alphas[l] * np.abs(coefs[l]).sum()
        sk = Lasso(alpha=alphas[l], tol=1e-14, max_iter=1_000_000).fit(B, y)
        sk_fit = sk.predict(B)
        sk_obj = 0.5 * np.mean((y - sk_fit) ** 2) + alphas[l] * np.abs(sk.coef_).sum()
        assert obj <= sk_obj * (1 + 1e-10)
        assert np.max(np.abs(fit - sk_fit)) < 1e-6
    assert exact.mean() > 0.9


@pytest.mark.parametrize("n, seed", [(30, 0), (30, 1), (130, 2)])
def test_ridge_is_sklearns_elastic_net_on_every_column(n, seed):
    """The groups with their shares of the ridge give the elastic net on the explicit basis,
    whose equal columns share their coefficient, also at new points."""
    X, y = draw(n, 3, seed, discrete=True)
    Xnew, _ = draw(40, 3, seed + 10)
    alpha, ridge = 0.01, 0.05
    hal = HighlyAdaptiveLassoCV(alphas=[alpha], ridge=ridge, tol=1e-14).fit(X, y)
    B = csc_matrix(explicit_basis(X, X))
    en = ElasticNet(alpha=alpha + ridge, l1_ratio=alpha / (alpha + ridge), tol=1e-14,
                    max_iter=1_000_000).fit(B, y)
    assert np.max(np.abs(hal.predict(Xnew) - en.predict(explicit_basis(X, Xnew)))) < 1e-6


def test_fit_does_not_depend_on_the_order_of_the_rows():
    """The least-norm solution is unique, so a permutation of the training points (and so of
    the basis columns) changes nothing."""
    X, y = draw(60, 4, 6, discrete=True)
    Xnew, _ = draw(100, 4, 7)
    alpha = 0.002
    f = HighlyAdaptiveLassoCV(alphas=[alpha]).fit(X, y).predict(Xnew)
    perm = np.random.default_rng(8).permutation(60)
    g = HighlyAdaptiveLassoCV(alphas=[alpha]).fit(X[perm], y[perm]).predict(Xnew)
    assert np.max(np.abs(f - g)) < 1e-5 * np.std(y)


def test_grid_is_lassocvs():
    X, y = draw(40, 3, 9)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        sk = LassoCV().fit(explicit_basis(X, X), y)
    assert np.allclose(HighlyAdaptiveLassoCV().fit(X, y).alphas_, sk.alphas_, rtol=1e-12)


def test_cv_errors_are_the_elastic_nets_in_each_fold():
    """A fold's error at each penalty is that of sklearn's elastic net on the explicit basis
    at the fold's training rows, with knots at every training point. Columns that are equal
    at those rows (a knot held out between two others, say) share their coefficient."""
    X, y = draw(40, 2, 13, discrete=True)
    ridge = 0.05
    alphas = HighlyAdaptiveLassoCV().fit(X, y).alphas_[::20]
    hal = HighlyAdaptiveLassoCV(alphas=alphas, ridge=ridge, tol=1e-14).fit(X, y)
    B = explicit_basis(X, X)
    ref = np.empty_like(hal.mse_path_)
    for k, (train, test) in enumerate(KFold(5).split(X)):
        for l, a in enumerate(hal.alphas_):
            en = ElasticNet(alpha=a + ridge, l1_ratio=a / (a + ridge), tol=1e-14,
                            max_iter=1_000_000).fit(B[train], y[train])
            ref[l, k] = np.mean((en.predict(B[test]) - y[test]) ** 2)
    assert np.allclose(hal.mse_path_, ref, rtol=1e-6)
    assert hal.alpha_ == hal.alphas_[np.argmin(ref.mean(axis=1))]


def test_max_degree_keeps_the_small_sections():
    X, _ = draw(20, 4, 11)
    basis = _Basis(X, 2)
    sizes = [bin(int(s)).count("1") for s in basis.smask]
    assert max(sizes) == 2
    assert len(basis.smask) == sum(1 for k in (1, 2) for _ in combinations(range(4), k))
    hal = HighlyAdaptiveLassoCV(max_degree=1).fit(*draw(40, 3, 12))
    assert all(bin(int(s)).count("1") == 1 for s in hal.term_section_)
