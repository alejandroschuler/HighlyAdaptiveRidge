"""Fast multi-lambda kernel-ridge LOOCV via a single eigendecomposition.

The original KernelRidgeCV solves the bordered system

    [[K + a*I, 1],
     [1^T    , 0]] @ [c; b] = [Y; 0]

once per regularization value `a` (an O(n^3) solve), and then solves it a *second*
time inside loocv() to get the leverage. With a 50-point grid that is ~100 O(n^3)
solves per kernel.

This module computes the identical quantities for every `a` from a single symmetric
eigendecomposition, then O(n^2) work per `a`.

Numerical conditioning. The HAR kernel scales like 2^p, so for high-dimensional data
(yearmsd p=90, slice p=384) its entries reach ~1e27-1e45 and a raw eigendecomposition is
ill-conditioned. We rescale K to unit mean-diagonal first: replacing K by K/c and a by a/c
leaves the fitted values and the LOOCV leverage exactly invariant (the kernel-ridge
solution depends only on the ratio), so the rescaling changes nothing statistically while
making eigh well-behaved. Numerical-negative eigenvalues of the PSD kernel are clamped to 0.

Equivalence (derived analytically and checked numerically against the original solve):
  with B = (K + a I)^-1, s = B 1, denom = 1^T s,
    b   = (s^T Y) / denom,   c = B (Y - b 1),   Yhat = K c + b 1,
  and the original's loocv leverage diag(A^-1 [K; 1^T]) equals the i-th diagonal of the
  smoother Y -> Yhat, namely  h_i = (K B)_ii - s_i (K s)_i / denom + s_i / denom.
"""
import numpy as np


def _prep(K, Y):
    """Rescale to unit mean-diagonal and eigendecompose. Returns a cache."""
    n = K.shape[0]
    c = float(np.mean(np.diag(K)))
    if not np.isfinite(c) or c <= 0:
        c = 1.0
    lam, Q = np.linalg.eigh(K / c)      # K/c has O(1) entries -> well-conditioned
    lam = np.maximum(lam, 0.0)          # the kernel is PSD; clamp numerical negatives
    g = Q.T @ np.ones(n)
    w = Q.T @ Y
    return lam, Q, g, w, c


def loocv_path(K, Y, alphas):
    """LOOCV MSE for each alpha, reproducing KernelRidge.loocv exactly.

    Non-finite entries (very small alpha can drive a leverage to 1) are returned as inf so
    the caller selects among the finite ones, matching the original argmin-over-errors.
    Returns (mses, cache); cache feeds coef_at.
    """
    lam, Q, g, w, c = _prep(K, Y)
    Ks = K / c
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
        cs = Q @ ((w - b * g) * inv)        # scaled kernel coefficients
        Yhat = Ks @ cs + b
        KB_diag = Q2 @ (lam * inv)
        Ks_vec = Q @ (lam * s_eig)
        h = KB_diag - s * Ks_vec / denom + s / denom
        R = (Y - Yhat) / (1.0 - h)
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
