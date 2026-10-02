"""Fast multi-lambda kernel-ridge LOOCV via a single eigendecomposition.

The original KernelRidgeCV solves the bordered system

    [[K + a*I, 1],
     [1^T    , 0]] @ [c; b] = [Y; 0]

once per regularization value `a` (an O(n^3) solve), and then inverts the bordered
matrix A inside loocv() to get the diagonal of A^-1. With a 50-point grid that is
~100 O(n^3) solves per kernel.

This module computes the identical quantities for every `a` from a single symmetric
eigendecomposition, then O(n^2) work per `a`.

Numerical conditioning. The HAR kernel scales like 2^p, so for high-dimensional data
(yearmsd p=90, slice p=384) its entries reach ~1e27-1e45 and a raw eigendecomposition is
ill-conditioned. We rescale K to unit mean-diagonal first: replacing K by K/c and a by a/c
leaves the fitted values and the leave-one-out residuals exactly invariant (the
kernel-ridge solution depends only on the ratio), so the rescaling changes nothing
statistically while making eigh well-behaved. Numerical-negative eigenvalues of the PSD
kernel are clamped to 0.

Equivalence (derived analytically and checked numerically against the original solve):
  with B = (K + a I)^-1, s = B 1, denom = 1^T s and G = B - s s^T / denom,
    b   = (s^T Y) / denom,   c = B (Y - b 1) = G Y,   Yhat = K c + b 1 = Y - a c.
  G is the top-left n x n block of A^-1, so the smoother Y -> Yhat is I - a G, its
  leverage is h_i = 1 - a G_ii, and the leave-one-out residual is
    R_i = (Y_i - Yhat_i) / (1 - h_i) = c_i / G_ii.
  The middle form divides two differences that both go to 0 with a, so in double
  precision its relative error grows like 1/a: on yacht's HAR kernel (rep 2) it reads
  0.8% low at a = 1e-12 mean(diag K). The last form has no such cancellation. At a = 0
  it is the leave-one-out residual of the interpolant, when K has full rank. In the
  eigenbasis, G_ii = sum_j Q_ij^2 / (lam_j + a) - s_i^2 / denom. Over the alpha grids of
  the Table 1 kernels (rep 0), the second term is at most 11% of the first, so the
  subtraction loses at most 0.05 digits.
"""
import numpy as np


def _prep(K, Y):
    """Rescale to unit mean-diagonal and eigendecompose.

    Returns (cache, min_eig, max_eig). The cache feeds loocv_path and coef_at.
    min_eig and max_eig are the smallest (before the clamp) and the largest
    eigenvalue of K, for Kernel.alpha_grid.
    """
    n = K.shape[0]
    c = float(np.mean(np.diag(K)))
    if not np.isfinite(c) or c <= 0:
        c = 1.0
    lam, Q = np.linalg.eigh(K / c)      # K/c has O(1) entries -> well-conditioned
    min_eig, max_eig = float(np.min(lam)) * c, float(np.max(lam)) * c
    lam = np.maximum(lam, 0.0)          # the kernel is PSD; clamp numerical negatives
    g = Q.T @ np.ones(n)
    w = Q.T @ Y
    return (lam, Q, g, w, c), min_eig, max_eig


def loocv_path(K, Y, alphas, cache=None):
    """LOOCV MSE for each alpha: the mean of R_i^2, with R_i = c_i / G_ii as in
    KernelRidge.loocv.

    Non-finite entries (alpha = 0 with a singular kernel, for example) are returned as inf so
    the caller selects among the finite ones, matching the original argmin-over-errors.
    A cache from _prep(K, Y) skips the eigendecomposition.
    Returns (mses, cache); cache feeds coef_at.
    """
    if cache is None:
        cache, _, _ = _prep(K, Y)
    lam, Q, g, w, c = cache
    Q2 = Q * Q
    mses = np.full(len(alphas), np.inf, dtype=float)
    for t, a in enumerate(alphas):
        a_s = a / c
        inv = 1.0 / (lam + a_s)
        s_eig = g * inv
        denom = g @ s_eig
        if not np.isfinite(denom) or denom == 0:
            continue
        s = Q @ s_eig
        b = (s_eig @ w) / denom
        cs = Q @ ((w - b * g) * inv)        # the kernel coefficients, times the scale c
        G_diag = Q2 @ inv - s * s / denom   # the diagonal of G, times the scale c
        R = cs / G_diag                     # the scales cancel
        m = np.mean(R * R)
        if np.isfinite(m):
            mses[t] = m
    return mses, (lam, Q, g, w, c)


def coef_at(cache, alpha):
    """Return [c; b] (length n+1) for the ORIGINAL (unscaled) kernel, matching
    KernelRidge.coef so prediction is k(x).c + b with the original kernel."""
    lam, Q, g, w, c = cache
    a_s = alpha / c
    inv = 1.0 / (lam + a_s)
    s_eig = g * inv
    denom = g @ s_eig
    b = (s_eig @ w) / denom
    c_coef = (Q @ ((w - b * g) * inv)) / c   # divide by c -> coefficients for K, not K/c
    return np.hstack([c_coef, b])
