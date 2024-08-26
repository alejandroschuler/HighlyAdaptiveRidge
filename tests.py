import numpy as np
import pytest
from highly_adaptive_regression import HighlyAdaptiveRidgeCV as HARCV
from kernel_ridge import HighlyAdaptiveRidge as kHAR, HighlyAdaptiveRidgeCV as kHARCV
from kernel_ridge.kernels import HighlyAdaptiveRidgeKernel

@pytest.fixture
def data(request):
    n, n_, d, seed = request.param
    np.random.seed(seed)
    X = np.random.rand(n, d)
    Y = np.random.rand(n)

    X_ = np.random.rand(n_, d)
    Y_ = np.random.rand(n_)
    return X, X_, Y, Y_

# Define the parameter sets to test with
#     n  n_ d  
param_sets = [
    (10, 1, 3),
    (10, 1, 8),
    (20, 5, 5),
    (25, 10, 2)
]

# Combine with seeds to run each set multiple times
param_sets_with_seeds = [(n, n_, d, seed) for (n, n_, d) in param_sets for seed in range(5)]

@pytest.mark.parametrize("data", param_sets_with_seeds, indirect=True)
def test_kernel_function(data):
    X, X_, _, _ = data

    har = HARCV(alphas=1)
    har.knots = X
    H = har._bases(X).astype(int)
    H_ = har._bases(X_).astype(int)
    K_ = (H @ H.T)
    Kk_ = (H_ @ H.T)

    K = HighlyAdaptiveRidgeKernel()(X, X)
    Kk = HighlyAdaptiveRidgeKernel()(X, X_)

    assert np.all(Kk == Kk_), "Kernel results for new data points do not match."
    assert np.all(K == K_), "Kernel results for the same data points do not match."

@pytest.mark.parametrize("data", param_sets_with_seeds, indirect=True)
def test_HAR_vs_kernel_HAR(data):
    X, X_, Y, Y_ = data

    har = HARCV()
    har.fit(X, Y)
    Y_har = har.predict(X_)
    rmse_har = np.sqrt(np.mean((Y_har - Y_) ** 2))

    khar = kHAR(alpha=har.regression.alpha_)
    khar.fit(X, Y)
    Y_khar = khar.predict(X_)
    rmse_khar = np.sqrt(np.mean((Y_khar - Y_) ** 2))

    rmse_diff = np.sqrt(np.mean((Y_har - Y_khar) ** 2))
    
    assert rmse_diff / np.std(Y) < 0.1, "RMSE difference is too large between HAR and Kernel HAR."
    assert np.abs((rmse_har - rmse_khar) / rmse_har) < 1e-2, "RMSEs between HAR and Kernel HAR are not sufficiently close."

@pytest.mark.parametrize("data", param_sets_with_seeds, indirect=True)
def test_LOOCV_for_HAR_vs_kernel_HAR(data):
    X, X_, Y, Y_ = data

    har = HARCV(n_alphas = 5)
    har.fit(X,Y)
    Y_har = har.predict(X_)
    rmse_har = np.sqrt(np.mean((Y_har - Y_)**2))

    khar = kHARCV(n_alphas = 5)
    khar.fit(X,Y)
    Y_khar = khar.predict(X_)
    rmse_khar = np.sqrt(np.mean((Y_khar - Y_)**2))

    rmse_diff = np.sqrt(np.mean((Y_har - Y_khar)**2))
    assert rmse_diff / np.std(Y) < 0.1, "RMSE difference is too large between HAR and Kernel HAR."
    assert np.abs((rmse_har - rmse_khar) / rmse_har) < 1e-2, "RMSEs between HAR and Kernel HAR are not sufficiently close."

# @pytest.mark.parametrize("data", param_sets_with_seeds, indirect=True)
# def test_krr_rbf(data):
#     X, X_, Y, Y_ = data
#     KRR(kernel=GaussianKernel).fit(X,Y)