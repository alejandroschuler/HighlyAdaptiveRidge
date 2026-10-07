"""The usual-norm mixed Sobolev kernels with HAR's depth weights, for
notes/sobolev-mars.tex: the counterpart of sobolev_kernels.AnchoredSobolev for
the kernels of the usual norm (sobolev_kernels.UsualSobolev).

Under the usual inner product of the Sobolev class of order m = t + 1 on
[0, 1], sum_{k <= m} int f^(k) g^(k), the constant function 1 has norm one and
its inner product with f is int f. So the one-dimensional kernel G splits as
G = 1 + G_0, where G_0 = G - 1 is the reproducing kernel of the functions with
integral zero. The product kernel then splits by section into the components
of the classical functional ANOVA under the uniform distribution, and HAR's
depth weights on these components give

    K(x, x') = sum_{1 <= |s| <= D} prod_{j in s} G_0(x_j, x'_j),

AnchoredSobolev with G_0 in place of c_t. Like the HAR kernels, it leaves out
the empty section, the constant, whose function is the intercept that the fit
leaves unpenalized, so at the full depth D = p it is prod_j G(x_j, x'_j) - 1,
the kernel of UsualSobolev less one.

The kernel has its own module so that the fits that read sobolev_kernels.py
do not rerun.
"""
from dataclasses import dataclass

import numpy as np
from numba import njit, prange

from kernel_ridge.kernels import Kernel

from .sobolev_kernels import usual_halves


@njit(parallel=True)
def _usual_depth_kernel(X_test, X, U_test, V_test, U, V, equal, depth):
    """K[te, tr] = sum_{1 <= k <= depth} e_k(G_0(X_test[te], X[tr])), with e_k
    the elementary symmetric polynomial of degree k of the p factors
    G_0 = G - 1, and G = U[lo] . V[hi] in each coordinate, where lo is the
    point with the smaller coordinate and hi the other. A depth below 1 or at
    least p keeps every section, through the closed form prod_j (1 + G_0) - 1,
    computed as q <- q + G_0 (1 + q), which has no subtraction."""
    n, p = X.shape
    n_test = X_test.shape[0]
    m = U.shape[2]
    full = depth < 1 or depth >= p
    D = p if full else depth
    K = np.empty((n_test, n), dtype=np.float64)
    for tr in prange(n):
        e = np.empty(D + 1, dtype=np.float64)
        max_index = tr + 1 if equal else n_test
        for te in range(max_index):
            if not full:
                e[0] = 1.0
                for k in range(1, D + 1):
                    e[k] = 0.0
            q = 0.0
            for j in range(p):
                g = 0.0
                if X_test[te, j] <= X[tr, j]:
                    for c in range(m):
                        g += U_test[te, j, c] * V[tr, j, c]
                else:
                    for c in range(m):
                        g += U[tr, j, c] * V_test[te, j, c]
                g -= 1.0
                if full:
                    q += g * (1.0 + q)
                else:
                    for k in range(min(j + 1, D), 0, -1):
                        e[k] += g * e[k - 1]
            if full:
                value = q
            else:
                value = 0.0
                for k in range(1, D + 1):
                    value += e[k]
            K[te, tr] = value
            if equal:
                K[tr, te] = value
    return K


@dataclass
class UsualSobolevDepth(Kernel):
    """The usual-norm mixed Sobolev kernel of order t + 1, t = `order`, on [0, 1]^p,
    split into the components of the classical functional ANOVA, with the
    sections of more than `depth` coordinates and the empty section left out. A
    negative depth, or one of at least p, keeps every nonempty section."""
    order: int = 0
    depth: int = -1

    def __post_init__(self):
        if self.order != int(self.order) or self.order < 0:
            raise ValueError(f"order must be an integer >= 0, not {self.order!r}")
        if self.depth == 0 or self.depth != int(self.depth):
            raise ValueError(f"depth must be a nonzero integer, not {self.depth!r}")

    def _halves(self, X):
        n, p = X.shape
        u, v = usual_halves(X.ravel(), int(self.order) + 1)
        m = u.shape[1]
        return np.ascontiguousarray(u.reshape(n, p, m)), np.ascontiguousarray(v.reshape(n, p, m))

    def __call__(self, X, X_test=None):
        """The kernel matrix, with a row for each point of X_test and a column for each point of X."""
        X = np.ascontiguousarray(X, dtype=np.float64)
        U, V = self._halves(X)
        if X_test is None:
            return _usual_depth_kernel(X, X, U, V, U, V, True, int(self.depth))
        X_test = np.ascontiguousarray(X_test, dtype=np.float64)
        U_test, V_test = self._halves(X_test)
        return _usual_depth_kernel(X_test, X, U_test, V_test, U, V, False, int(self.depth))
