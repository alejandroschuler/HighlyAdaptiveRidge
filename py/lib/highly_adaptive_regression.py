"""The highly adaptive lasso (HAL) of order 0.

HAL is the lasso, with an unpenalized intercept, on the basis functions

    h_{i,s}(x) = prod_{j in s} 1(X_ij <= x_j),

one for each training point i (the knot) and each nonempty section s of the p coordinates,
(2^p - 1) n of them. HighlyAdaptiveLassoCV chooses the penalty by K-fold cross-validation over
the grid of sklearn's LassoCV, without forming the basis matrix.

The lasso on this basis has many solutions. Most basis columns are equal to another one at
every training point (83 to 98 percent of them on the Table 1 datasets), and distinct columns
are often linearly dependent there too. All the solutions have the same fitted values and the
same 1-norm, but they differ off the training points, and a solver lands on one of them by
accident of its path. HighlyAdaptiveLassoCV takes the solution of least 2-norm, which is the
limit of the elastic net as its ridge penalty goes to 0, by adding the penalty
(ridge / 2) ||beta||^2 with a tiny ridge. Equal columns then share their coefficient equally.
The same rule holds in each CV fold, among the columns that are equal at the fold's training
points.

The computation:
1. The comparisons 1(X_ij <= X_aj) are packed into bits, 64 training points to a word, so a
   basis column is an AND of at most p words for every 64 points.
2. The columns are hashed and grouped into the distinct ones. A group of S equal columns
   becomes one column with the ridge penalty ridge / S, which has the same solution.
3. Each CV fold groups the distinct columns again, by their values at its training rows, and
   drops the columns that are constant there.
4. Pathwise coordinate descent with the sequential strong rule (Tibshirani et al. 2012) solves
   each problem on a working set. Once the support holds, feature-sign search (Lee et al. 2007)
   solves the stationarity equations on the support exactly. Each penalty ends with a check
   of the optimality conditions over every column.
"""
import math
import os
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations

import numpy as np
from numba import njit, prange
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold
from sklearn.utils.validation import check_array, check_is_fitted, column_or_1d


class HighlyAdaptiveBaseCV:
    """Highly Adaptive Base (HABase) class. Implements the Highly Adaptive Lasso/Ridge algorithms."""

    @classmethod
    def _basis_products(cls, arr, index=0, current=None, result=None):
        """
        Recursive helper function for computing basis products.

        Args:
            arr (ndarray): Array of boolean values.
            index (int): Current index for recursion (default: 0).
            current (ndarray): Current basis product (default: None).
            result (list): List to store the computed basis products (default: None).

        Returns:
            list: List of computed basis products.
        """

        if result is None:
            result = []
        if current is None:
            current = np.ones_like(arr[0], dtype=bool)

        if index == len(arr):
            result.append(current)
        else:
            cls._basis_products(arr, index + 1, current & arr[index], result)
            cls._basis_products(arr, index + 1, current, result)

        return result

    def _bases(self, X):
        """
        Computes the basis functions for the given knots and input data.
        Args:
            X (ndarray): Input data.
        Returns:
            ndarray: Array of computed basis functions.
        """
        one_way_bases = np.stack([
            np.less_equal.outer(self.knots[:,j], X[:,j])
            for j in range(self.knots.shape[1])
        ])
        bases = self._basis_products(one_way_bases)

        return np.concatenate(bases[:-1]).T

    def _pre_fit(self, X,Y):
        pass

    def fit(self, X, Y):
        self._pre_fit(X,Y)
        self.knots = X
        self.regression.fit(self._bases(X), Y)

    def predict(self, X):
        return self.regression.predict(self._bases(X))


# ---------------------------------------------------------------------------
# Bits. uint64 constants only: numba promotes a mix of uint64 and int64 to float64.

_M1 = np.uint64(0x5555555555555555)
_M2 = np.uint64(0x3333333333333333)
_M4 = np.uint64(0x0F0F0F0F0F0F0F0F)
_H01 = np.uint64(0x0101010101010101)
_S1, _S2, _S4, _S33, _S56 = np.uint64(1), np.uint64(2), np.uint64(4), np.uint64(33), np.uint64(56)
_F1 = np.uint64(0xFF51AFD7ED558CCD)
_F2 = np.uint64(0xC4CEB9FE1A85EC53)
_SEED = np.uint64(0x9E3779B97F4A7C15)
_ONE = np.uint64(1)

# Above this many basis columns the grouping needs gigabytes; max_degree cuts the count.
_MAX_COLUMNS = 10**8
# A zero coefficient violates the optimality conditions when its column's correlation with
# the residual exceeds the penalty by more than this factor, which allows for rounding.
_KKT = 1.0 + 1e-9


@njit(inline="always")
def _popcount64(x):
    """The number of 1 bits in a uint64."""
    x = x - ((x >> _S1) & _M1)
    x = (x & _M2) + ((x >> _S2) & _M2)
    x = (x + (x >> _S4)) & _M4
    return np.int64((x * _H01) >> _S56)


@njit(inline="always")
def _fmix(h):
    """The 64-bit finalizer of MurmurHash3."""
    h ^= h >> _S33
    h *= _F1
    h ^= h >> _S33
    h *= _F2
    h ^= h >> _S33
    return h


@njit(parallel=True, cache=True)
def _one_way_bits(X):
    """O[i, j] holds the bits 1(X[i, j] <= X[a, j]) of the points a, bit a % 64 of word a // 64."""
    n, p = X.shape
    W = (n + 63) // 64
    O = np.zeros((n, p, W), dtype=np.uint64)
    for i in prange(n):
        for j in range(p):
            xij = X[i, j]
            for a in range(n):
                if xij <= X[a, j]:
                    O[i, j, a >> 6] |= _ONE << np.uint64(a & 63)
    return O


@njit(inline="always")
def _column_bits(O, i, smask, p, out):
    """The basis column of knot i and the section with bit mask smask, at the training points."""
    W = out.shape[0]
    first = True
    for j in range(p):
        if (smask >> j) & 1:
            if first:
                for t in range(W):
                    out[t] = O[i, j, t]
                first = False
            else:
                for t in range(W):
                    out[t] &= O[i, j, t]


@njit(parallel=True, cache=True)
def _hash_columns(O, parent, top):
    """Hash the column of section k and knot i into H[k * n + i]. A section is its parent,
    the section without its largest coordinate top[k], plus that coordinate."""
    n, p, W = O.shape
    nsec = parent.shape[0]
    H = np.empty(nsec * n, dtype=np.uint64)
    for i in prange(n):
        buf = np.empty((nsec, W), dtype=np.uint64)
        for k in range(nsec):
            j = top[k]
            pk = parent[k]
            h = _SEED
            for t in range(W):
                v = O[i, j, t] if pk < 0 else buf[pk, t] & O[i, j, t]
                buf[k, t] = v
                h = _fmix(h ^ v)
            H[k * n + i] = h
    return H


@njit(cache=True)
def _group_columns(H, order, O, smask, n, p):
    """rep[col] is the smallest column equal to column col at the training points. order
    sorts H, with ties in increasing column order, so equal columns are in one run of H."""
    M = H.shape[0]
    W = O.shape[2]
    rep = np.empty(M, dtype=np.int64)
    reps = np.empty(M, dtype=np.int64)
    a = np.empty(W, dtype=np.uint64)
    b = np.empty(W, dtype=np.uint64)
    start = 0
    while start < M:
        h = H[order[start]]
        end = start + 1
        while end < M and H[order[end]] == h:
            end += 1
        nr = 0
        for q in range(start, end):
            col = order[q]
            if end - start > 1:
                _column_bits(O, col % n, smask[col // n], p, a)
            found = -1
            for r in range(nr):
                other = reps[r]
                _column_bits(O, other % n, smask[other // n], p, b)
                eq = True
                for t in range(W):
                    if a[t] != b[t]:
                        eq = False
                        break
                if eq:
                    found = other
                    break
            if found < 0:
                reps[nr] = col
                nr += 1
                rep[col] = col
            else:
                rep[col] = found
        start = end
    return rep


@njit(parallel=True, cache=True)
def _distinct_bits(O, knot, smask, p):
    B = np.empty((knot.shape[0], O.shape[2]), dtype=np.uint64)
    for k in prange(knot.shape[0]):
        _column_bits(O, knot[k], smask[k], p, B[k])
    return B


@njit(parallel=True, cache=True)
def _masked_hash_count(B, mask):
    """The hash and the number of 1 bits of each row of B & mask."""
    K, W = B.shape
    H = np.empty(K, dtype=np.uint64)
    C = np.empty(K, dtype=np.int64)
    for k in prange(K):
        h = _SEED
        c = 0
        for t in range(W):
            v = B[k, t] & mask[t]
            h = _fmix(h ^ v)
            c += _popcount64(v)
        H[k] = h
        C[k] = c
    return H, C


@njit(cache=True)
def _group_masked(H, order, B, mask, rep):
    """rep[k] is the smallest row equal to row k of B & mask, for the rows k in order, which
    sorts them by H with ties in increasing row order."""
    M = order.shape[0]
    W = B.shape[1]
    reps = np.empty(M, dtype=np.int64)
    start = 0
    while start < M:
        h = H[order[start]]
        end = start + 1
        while end < M and H[order[end]] == h:
            end += 1
        nr = 0
        for q in range(start, end):
            k = order[q]
            found = -1
            for r in range(nr):
                other = reps[r]
                eq = True
                for t in range(W):
                    if (B[k, t] & mask[t]) != (B[other, t] & mask[t]):
                        eq = False
                        break
                if eq:
                    found = other
                    break
            if found < 0:
                reps[nr] = k
                nr += 1
                rep[k] = k
            else:
                rep[k] = found
        start = end


@njit(parallel=True, cache=True)
def _csc_rows(B, cols, mask, newrow, indptr):
    """The row indices, renumbered by newrow, of the bits of B[cols[q]] & mask."""
    W = B.shape[1]
    indices = np.empty(indptr[-1], dtype=np.int32)
    for q in prange(cols.shape[0]):
        pos = indptr[q]
        k = cols[q]
        for t in range(W):
            v = B[k, t] & mask[t]
            b = 0
            while v != 0:
                if v & _ONE:
                    indices[pos] = newrow[t * 64 + b]
                    pos += 1
                v >>= _ONE
                b += 1
    return indices


# ---------------------------------------------------------------------------
# The lasso path. X is a 0/1 matrix in CSC form whose data are all ones. The solver works
# with the centered columns, so the intercept is not penalized, and keeps the residual as
# rho + c: an update of column j then costs one pass over its rows.

@njit(nogil=True, cache=True)
def _full_grad(indptr, indices, rho, c, cnt, G):
    """G[j] = x_j'(rho + c), which equals the centered x_j'r because r sums to 0."""
    for j in range(indptr.shape[0] - 1):
        g = 0.0
        for q in range(indptr[j], indptr[j + 1]):
            g += rho[indices[q]]
        G[j] = g + c * cnt[j]


@njit(nogil=True, cache=True)
def _sweep(cols, k, indptr, indices, rho, c, cnt, mean, s, w, na, rw):
    """One cyclic pass of coordinate descent over cols[:k]. Returns the largest
    s_j d_j^2 of the changes d_j, and the new c."""
    maxd = 0.0
    for q in range(k):
        j = cols[q]
        g = 0.0
        for r in range(indptr[j], indptr[j + 1]):
            g += rho[indices[r]]
        z = g + c * cnt[j] + s[j] * w[j]
        if z > na:
            wn = (z - na) / (s[j] + rw[j])
        elif z < -na:
            wn = (z + na) / (s[j] + rw[j])
        else:
            wn = 0.0
        d = wn - w[j]
        if d != 0.0:
            for r in range(indptr[j], indptr[j + 1]):
                rho[indices[r]] -= d
            c += d * mean[j]
            w[j] = wn
            if s[j] * d * d > maxd:
                maxd = s[j] * d * d
    return maxd, c


@njit(nogil=True, cache=True)
def _gram(A, k, bits, cnt, n, rw):
    """The centered Gram matrix of the columns A[:k], with the ridge added to its diagonal."""
    Wd = bits.shape[1]
    Gm = np.empty((k, k))
    for u in range(k):
        ju = A[u]
        for v in range(u, k):
            jv = A[v]
            cc = 0
            for t in range(Wd):
                cc += _popcount64(bits[ju, t] & bits[jv, t])
            g = cc - cnt[ju] * cnt[jv] / n
            Gm[u, v] = g
            Gm[v, u] = g
        Gm[u, u] += rw[ju]
    return Gm


@njit(nogil=True, cache=True)
def _chol_solve(R, k, b, z, x):
    """x solves R'R x = b, for the upper triangular R[:k, :k]."""
    for i in range(k):
        v = b[i]
        for j in range(i):
            v -= R[j, i] * z[j]
        z[i] = v / R[i, i]
    for i in range(k - 1, -1, -1):
        v = z[i]
        for j in range(i + 1, k):
            v -= R[i, j] * x[j]
        x[i] = v / R[i, i]


@njit(nogil=True, cache=True)
def _chol_delete(R, k, u):
    """Given R'R = G for R[:k, :k], make R[:k - 1, :k - 1] the factor of G without row and
    column u: drop column u of R, and rotate the rows back to triangular form."""
    for i in range(k):
        for j in range(u, k - 1):
            R[i, j] = R[i, j + 1]
    for j in range(u, k - 1):
        a = R[j, j]
        b = R[j + 1, j]
        r = math.hypot(a, b)
        cos, sin = a / r, b / r
        R[j, j] = r
        R[j + 1, j] = 0.0
        for l in range(j + 1, k - 1):
            t1 = R[j, l]
            t2 = R[j + 1, l]
            R[j, l] = cos * t1 + sin * t2
            R[j + 1, l] = cos * t2 - sin * t1


@njit(nogil=True, cache=True)
def _polish(A, k, bits, cnt, n, Xty, w, na, indptr, indices, rho, c, mean, rw):
    """Feature-sign search on the support A[:k]. Solve the stationarity equations with the
    current signs; if a coefficient would change sign, move to the first change of sign,
    drop that coordinate and solve again. Each step lowers the objective. The Cholesky
    factor of the support's Gram matrix is updated as coordinates drop, and LU solves stand
    in when the factorization fails. Returns (solved, c), where solved is False if a system
    was singular."""
    if k == 0:
        return True, c
    rhs = np.empty(k)
    for u in range(k):
        rhs[u] = Xty[A[u]] - (na if w[A[u]] > 0 else -na)
    chol = True
    try:
        R = np.ascontiguousarray(np.linalg.cholesky(_gram(A, k, bits, cnt, n, rw)).T)
    except Exception:
        chol = False
        R = np.empty((1, 1))
    x = np.empty(k)
    z = np.empty(k)
    while k > 0:
        if chol:
            _chol_solve(R, k, rhs, z, x)
            for u in range(k):
                if not np.isfinite(x[u]):
                    chol = False
        if not chol:
            try:
                x[:k] = np.linalg.solve(_gram(A, k, bits, cnt, n, rw), rhs[:k])
            except Exception:
                return False, c
        tstar = 1.0
        drop = -1
        for u in range(k):
            wu = w[A[u]]
            if x[u] * wu <= 0.0:
                t = wu / (wu - x[u])
                if t < tstar:
                    tstar = t
                    drop = u
        for u in range(k):
            j = A[u]
            if drop < 0:
                target = x[u]
            elif u == drop:
                target = 0.0
            else:
                target = w[j] + tstar * (x[u] - w[j])
            d = target - w[j]
            if d != 0.0:
                for r in range(indptr[j], indptr[j + 1]):
                    rho[indices[r]] -= d
                c += d * mean[j]
                w[j] = target
        if drop < 0:
            return True, c
        # Drop every coordinate now at 0 (ties included), from the last, with its equation.
        for u in range(k - 1, -1, -1):
            if w[A[u]] == 0.0:
                if chol:
                    _chol_delete(R, k, u)
                for v in range(u, k - 1):
                    A[v] = A[v + 1]
                    rhs[v] = rhs[v + 1]
                k -= 1
    return True, c


@njit(nogil=True, cache=True)
def _check(indptr, indices, rho, yc, cnt, s, w, rw, na, G, inW, Wl, nW):
    """The optimality conditions over every column, at the residual rho. A zero column that
    violates them joins the working set Wl[:nW]. Returns (violated, gap, nW), where gap is
    sklearn's duality gap, for the elastic net with a ridge on each column."""
    _full_grad(indptr, indices, rho, 0.0, cnt, G)
    violated = False
    dual = 0.0
    for j in range(s.shape[0]):
        if s[j] > 0:
            if w[j] == 0.0 and abs(G[j]) > na * _KKT:
                violated = True
                if not inW[j]:
                    inW[j] = True
                    Wl[nW] = j
                    nW += 1
            if abs(G[j] - rw[j] * w[j]) > dual:
                dual = abs(G[j] - rw[j] * w[j])
    const = na / dual if dual > na else 1.0
    l1 = 0.0
    l2 = 0.0
    for q in range(nW):
        l1 += abs(w[Wl[q]])
        l2 += rw[Wl[q]] * w[Wl[q]] ** 2
    gap = 0.5 * (1 + const * const) * (np.dot(rho, rho) + l2) + na * l1 - const * np.dot(rho, yc)
    return violated, gap, nW


@njit(nogil=True, cache=True)
def _lasso_path(indptr, indices, bits, y, alphas, rw, tol, max_sweeps, polish_after):
    """The path of

        (1 / 2n) ||y - b - X w||^2 + alpha ||w||_1 + (1 / 2n) sum_j rw_j w_j^2

    over the alphas, from the largest, with the intercept b unpenalized. bits holds the
    columns of X as bits of the rows, for the exact solves on the support.

    Returns the columns that were ever nonzero, in the order they became so, their
    coefficients at each alpha, the intercepts, and for each alpha the passes over columns,
    the duality gap relative to ||y - mean(y)||^2, and whether the solution is exact: it
    solves the stationarity equations on its support and no other column violates the
    optimality conditions.
    """
    n = y.shape[0]
    m = indptr.shape[0] - 1
    L = alphas.shape[0]
    ybar = y.mean()
    yc = y - ybar
    ynorm2 = np.dot(yc, yc)
    cnt = np.empty(m)
    for j in range(m):
        cnt[j] = indptr[j + 1] - indptr[j]
    mean = cnt / n
    s = cnt - cnt * cnt / n                              # centered x_j'x_j
    w = np.zeros(m)
    rho = yc.copy()
    c = 0.0
    G = np.empty(m)
    _full_grad(indptr, indices, rho, c, cnt, G)
    Xty = G.copy()
    inE = np.zeros(m, dtype=np.bool_)
    Elist = np.empty(m, dtype=np.int64)
    nE = 0
    inW = np.zeros(m, dtype=np.bool_)
    Wl = np.empty(m, dtype=np.int64)
    A = np.empty(m, dtype=np.int64)
    cap = 64
    path = np.zeros((L, cap))
    b_path = np.empty(L)
    sweeps = np.zeros(L, dtype=np.int64)
    gaps = np.empty(L)
    exact = np.zeros(L, dtype=np.bool_)
    a_prev = alphas[0]
    for j in range(m):
        a_prev = max(a_prev, abs(G[j]) / n)
    for l in range(L):
        na = n * alphas[l]
        # The working set: the columns that were ever nonzero, and the strong rule's.
        thr = n * (2 * alphas[l] - a_prev)
        nW = 0
        for q in range(nE):
            inW[Elist[q]] = True
            Wl[nW] = Elist[q]
            nW += 1
        for j in range(m):
            if not inW[j] and s[j] > 0 and abs(G[j]) >= thr:
                inW[j] = True
                Wl[nW] = j
                nW += 1
        order = np.sort(Wl[:nW])
        gap = np.inf
        # On a fine grid the support often holds from one alpha to the next, and then an
        # exact solve on it is the solution.
        k = 0
        for q in range(nE):
            if w[Elist[q]] != 0.0:
                A[k] = Elist[q]
                k += 1
        if l > 0 and exact[l - 1] and k > 0:
            ok, c = _polish(A, k, bits, cnt, n, Xty, w, na, indptr, indices, rho, c, mean, rw)
            rho += c
            c = 0.0
            violated, gap, nW = _check(indptr, indices, rho, yc, cnt, s, w, rw, na, G, inW, Wl, nW)
            exact[l] = ok and not violated
            order = np.sort(Wl[:nW])
        eps = tol
        for rnd in range(0 if exact[l] else 50):
            # Coordinate descent on the working set, cycling on the support.
            wait = polish_after
            while sweeps[l] < max_sweeps:
                maxd, c = _sweep(order, nW, indptr, indices, rho, c, cnt, mean, s, w, na, rw)
                sweeps[l] += 1
                if maxd <= eps * ynorm2:
                    break
                k = 0
                for q in range(nW):
                    if w[order[q]] != 0.0:
                        A[k] = order[q]
                        k += 1
                stable = 0
                while sweeps[l] < max_sweeps:
                    maxd, c = _sweep(A, k, indptr, indices, rho, c, cnt, mean, s, w, na, rw)
                    sweeps[l] += 1
                    if maxd <= eps * ynorm2:
                        break
                    k2 = 0
                    for u in range(k):
                        if w[A[u]] != 0.0:
                            A[k2] = A[u]
                            k2 += 1
                    stable = stable + 1 if k2 == k else 0
                    k = k2
                    if stable >= wait and k > 0:
                        ok, c = _polish(A, k, bits, cnt, n, Xty, w, na, indptr, indices, rho, c, mean, rw)
                        if ok:
                            break
                        wait *= 2
                        stable = 0
            # The exact solve on the support, then the optimality conditions everywhere.
            k = 0
            for q in range(nW):
                if w[order[q]] != 0.0:
                    A[k] = order[q]
                    k += 1
            ok = True
            if k > 0:
                ok, c = _polish(A, k, bits, cnt, n, Xty, w, na, indptr, indices, rho, c, mean, rw)
            rho += c
            c = 0.0
            violated, gap, nW = _check(indptr, indices, rho, yc, cnt, s, w, rw, na, G, inW, Wl, nW)
            if violated and sweeps[l] < max_sweeps:
                order = np.sort(Wl[:nW])
                continue
            if ok and not violated:
                exact[l] = True
                break
            if gap <= tol * ynorm2 or sweeps[l] >= max_sweeps:
                break
            eps *= 0.1
        gaps[l] = gap / ynorm2
        for q in range(nW):
            j = Wl[q]
            inW[j] = False
            if w[j] != 0.0 and not inE[j]:
                inE[j] = True
                Elist[nE] = j
                nE += 1
        if nE > cap:
            while cap < nE:
                cap *= 2
            grown = np.zeros((L, cap))
            grown[:, : path.shape[1]] = path
            path = grown
        b = ybar
        for q in range(nE):
            path[l, q] = w[Elist[q]]
            b -= mean[Elist[q]] * w[Elist[q]]
        b_path[l] = b
        a_prev = alphas[l]
    return Elist[:nE].copy(), path[:, :nE].copy(), b_path, sweeps, gaps, exact


@njit(parallel=True, cache=True)
def _predict(knots, X, tknot, tsmask, tw, b):
    """b + sum_q tw[q] h(x), where h is the basis function of knot tknot[q] and the
    section with bit mask tsmask[q]. The terms are sorted by knot."""
    nt, p = X.shape
    out = np.empty(nt)
    for a in prange(nt):
        acc = b
        last = -1
        below = 0                                      # bit j: the knot is <= x in coordinate j
        for q in range(tknot.shape[0]):
            i = tknot[q]
            if i != last:
                below = 0
                for j in range(p):
                    if knots[i, j] <= X[a, j]:
                        below |= 1 << j
                last = i
            if (tsmask[q] & ~below) == 0:
                acc += tw[q]
        out[a] = acc
    return out


# ---------------------------------------------------------------------------

def _sections(p, max_degree):
    """The sections of at most max_degree coordinates, as bit masks, ordered by size. Each
    comes after its parent, the section without its largest coordinate top."""
    smask, parent, top = [], [], []
    index = {}
    for size in range(1, max_degree + 1):
        for s in combinations(range(p), size):
            mask = sum(1 << j for j in s)
            index[mask] = len(smask)
            smask.append(mask)
            top.append(s[-1])
            parent.append(index[mask ^ (1 << s[-1])] if size > 1 else -1)
    return (np.array(smask, dtype=np.int64), np.array(parent, dtype=np.int64),
            np.array(top, dtype=np.int64))


class _Basis:
    """The basis columns of the training points X, grouped into the distinct ones.

    Column k * n + i is the basis function of knot i and section smask[k]. group[col] is the
    group of each column, and size[g] the number of columns in group g. B[g] holds the
    bits of group g at the training points, and (knot[g], gsmask[g]) is its first column.
    """

    def __init__(self, X, max_degree):
        n, p = X.shape
        self.n = n
        self.smask, parent, top = _sections(p, max_degree)
        M = self.smask.shape[0] * n
        if M > _MAX_COLUMNS:
            raise ValueError(f"{M:,} basis columns is too many; use a smaller max_degree")
        O = _one_way_bits(X)
        H = _hash_columns(O, parent, top)
        rep = _group_columns(H, np.argsort(H, kind="stable"), O, self.smask, n, p)
        del H
        first = np.flatnonzero(rep == np.arange(M))
        gid = np.empty(M, dtype=np.int64)
        gid[first] = np.arange(first.shape[0])
        self.group = gid[rep]
        self.size = np.bincount(self.group, minlength=first.shape[0])
        self.knot = first % n
        self.gsmask = self.smask[first // n]
        self.B = _distinct_bits(O, self.knot, self.gsmask, p)

    @property
    def n_columns(self):
        return self.group.shape[0]

    def values(self, groups, rows):
        """The values of the groups' columns at the training points rows."""
        word = rows >> 6
        bit = (rows & 63).astype(np.uint64)
        return ((self.B[np.asarray(groups)[:, None], word[None, :]] >> bit[None, :]) & _ONE).astype(float)

    def members(self, groups):
        """(knot, section mask, group) of every column of the given groups."""
        cols = np.flatnonzero(np.isin(self.group, groups))
        return cols % self.n, self.smask[cols // self.n], self.group[cols]


class _Problem:
    """The penalized regression on the training rows `rows`: one column for each set of
    groups that are equal at those rows, without the columns that are constant there.

    member_groups[ptr[f]:ptr[f + 1]] are the groups of problem column f, and size[f] the
    number of basis columns in them.
    """

    def __init__(self, basis, rows):
        rows = np.sort(np.asarray(rows))
        nr = rows.shape[0]
        mask = np.zeros(basis.B.shape[1], dtype=np.uint64)
        np.bitwise_or.at(mask, rows >> 6, _ONE << (rows & 63).astype(np.uint64))
        H, C = _masked_hash_count(basis.B, mask)
        live = np.flatnonzero((C > 0) & (C < nr))
        rep = np.full(basis.B.shape[0], -1, dtype=np.int64)
        _group_masked(H, live[np.argsort(H[live], kind="stable")], basis.B, mask, rep)
        first = np.flatnonzero(rep == np.arange(rep.shape[0]))
        fid = np.full(rep.shape[0], -1, dtype=np.int64)
        fid[first] = np.arange(first.shape[0])
        of = np.where(rep >= 0, fid[np.maximum(rep, 0)], -1)    # problem column of each group
        newrow = np.full(basis.n, -1, dtype=np.int64)
        newrow[rows] = np.arange(nr)
        self.indptr = np.concatenate([[0], np.cumsum(C[first])]).astype(np.int64)
        self.indices = _csc_rows(basis.B, first, mask, newrow, self.indptr)
        self.bits = basis.B[first] & mask[None, :]
        grouped = np.flatnonzero(of >= 0)
        self.member_groups = grouped[np.argsort(of[grouped], kind="stable")]
        self.ptr = np.searchsorted(of[self.member_groups], np.arange(first.shape[0] + 1))
        self.size = np.add.reduceat(basis.size[self.member_groups], self.ptr[:-1])

    def path(self, y, alphas, ridge, tol, max_sweeps, polish_after):
        # (ridge / 2) ||beta||^2 over the basis columns, whose coefficients are equal within
        # a problem column, is ridge / (2 size) times the problem column's squared coefficient.
        rw = y.shape[0] * ridge / self.size
        return _lasso_path(self.indptr, self.indices, self.bits, y, alphas, rw, tol,
                           max_sweeps, polish_after)

    def predict(self, basis, cols, coefs, b, rows):
        """Predictions at the training points rows, for each row of coefs. A problem column's
        coefficient is shared among its basis columns equally, as the ridge shares it."""
        pred = np.tile(b[:, None], (1, rows.shape[0]))
        for q, f in enumerate(cols):
            groups = self.member_groups[self.ptr[f]:self.ptr[f + 1]]
            share = basis.size[groups] / self.size[f]
            pred += coefs[:, q][:, None] * (share @ basis.values(groups, rows))[None, :]
        return pred


class HighlyAdaptiveLassoCV(BaseEstimator, RegressorMixin):
    """HAL of order 0, with the penalty chosen by cv-fold cross-validation.

    The grid and the folds are those of sklearn's LassoCV: n_alphas penalties from the
    smallest that sets every coefficient to 0 down to eps times it, evenly spaced on the log
    scale, and unshuffled folds. The CV error of a penalty is the mean over the folds of each
    fold's mean squared error, and the fit at the penalty with the smallest one is refit on
    all the data. The basis has knots at every training point, also in the folds.

    alphas: a grid to use instead, as in LassoCV.
    cv: the number of unshuffled folds, as in LassoCV, or the folds themselves: a
        scikit-learn splitter, or a list of (train, test) index arrays.
    ridge: the tiny ridge penalty that picks the least-norm lasso solution (see the module
        docstring). 0 gives the lasso solution that the solver happens to land on.
    tol: the duality gap, relative to ||y - mean(y)||^2, below which coordinate descent may
        stop at a penalty whose exact solve failed. Exact solves need no tolerance.
    max_degree: the largest section size, None for p (HAL as defined).
    n_jobs: the number of folds fit at once, in threads; -1 for one per core.

    After fit, mse_path_ holds the CV error of each penalty in each fold, as in LassoCV, and
    exact_ the share of the fits along the paths that were solved exactly.
    """

    def __init__(self, n_alphas=100, eps=1e-3, alphas=None, cv=5, ridge=1e-8, tol=1e-8,
                 max_sweeps=100_000, max_degree=None, n_jobs=None):
        self.n_alphas = n_alphas
        self.eps = eps
        self.alphas = alphas
        self.cv = cv
        self.ridge = ridge
        self.tol = tol
        self.max_sweeps = max_sweeps
        self.max_degree = max_degree
        self.n_jobs = n_jobs

    def _path(self, prob, y, alphas):
        return prob.path(y, alphas, self.ridge, self.tol, self.max_sweeps, polish_after=10)

    def fit(self, X, y):
        X = np.ascontiguousarray(check_array(X, dtype=np.float64))
        y = column_or_1d(y).astype(np.float64)
        n, p = X.shape
        degree = p if self.max_degree is None else min(int(self.max_degree), p)
        basis = _Basis(X, degree)

        # LassoCV's grid, from the smallest penalty that sets every coefficient to 0
        full = _Problem(basis, np.arange(n))
        yc = y - y.mean()
        Xty = np.add.reduceat(yc[full.indices], full.indptr[:-1]) if full.indices.size else np.zeros(1)
        alpha_max = np.max(np.abs(Xty)) / n
        if self.alphas is not None:
            alphas = np.sort(np.asarray(self.alphas, dtype=np.float64))[::-1]
        elif alpha_max <= np.finfo(float).resolution:
            alphas = np.full(self.n_alphas, np.finfo(float).resolution)
        else:
            alphas = np.geomspace(alpha_max, alpha_max * self.eps, num=self.n_alphas)

        if isinstance(self.cv, (int, np.integer)):
            folds = list(KFold(self.cv).split(X))
        elif hasattr(self.cv, "split"):
            folds = list(self.cv.split(X))
        else:
            folds = [(np.asarray(train), np.asarray(test)) for train, test in self.cv]
        problems = [_Problem(basis, train) for train, _ in folds]

        def cv_error(k):
            train, test = folds[k]
            cols, coefs, b, _, _, exact = self._path(problems[k], y[train], alphas)
            pred = problems[k].predict(basis, cols, coefs, b, test)
            return np.mean((pred - y[test][None, :]) ** 2, axis=1), exact

        threads = os.cpu_count() if self.n_jobs == -1 else (self.n_jobs or 1)
        with ThreadPoolExecutor(max(1, min(threads, len(folds)))) as pool:
            out = list(pool.map(cv_error, range(len(folds))))
        self.mse_path_ = np.stack([o[0] for o in out], axis=1)
        best = int(np.argmin(self.mse_path_.mean(axis=1)))
        self.alphas_ = alphas
        self.alpha_ = float(alphas[best])

        cols, coefs, b, _, _, exact = self._path(full, y, alphas[: best + 1])
        self.exact_ = float(np.mean(np.concatenate([o[1] for o in out] + [exact])))
        self.intercept_ = float(b[-1])
        nonzero = coefs[-1] != 0
        # Each problem column of the full data is one group, because the groups are distinct
        # at the training points. Its coefficient is shared among the group's columns.
        groups = full.member_groups[full.ptr[cols[nonzero]]]
        beta = np.zeros(basis.size.shape[0])
        beta[groups] = coefs[-1][nonzero]
        knot, smask, group = basis.members(groups)
        order = np.argsort(knot, kind="stable")
        self.knots_ = X.copy()
        self.term_knot_ = knot[order]
        self.term_section_ = smask[order]
        self.term_coef_ = (beta[group] / basis.size[group])[order]
        self.n_columns_ = basis.n_columns
        self.n_distinct_ = basis.size.shape[0]
        self.n_nonzero_ = len(groups)
        return self

    def predict(self, X):
        check_is_fitted(self, "intercept_")
        X = np.ascontiguousarray(check_array(X, dtype=np.float64))
        return _predict(self.knots_, X, self.term_knot_, self.term_section_,
                        self.term_coef_, self.intercept_)
