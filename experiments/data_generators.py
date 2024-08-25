import numpy as np

class DGP:
    def sample(self, n):
        X = self.sample_X(n)
        Y = self.sample_Y(X)
        return X, Y

class UnifIndX(DGP):
    def __init__(self, d):
        self.d = d

    def sample_X(self, n):
        return np.random.uniform(-1, 1, size=(n, self.d))

class Smooth(UnifIndX):
    def __init__(self, d):
        super().__init__(d)
        
        # draw a random subset of 1:d, more likely to be low-dimensional
        self.subsets = [np.random.choice(
            np.arange(d), 
            size=min([1 + np.random.poisson(1), d]),
            replace=False
        ) for k in range(d)]
        self.coefs = np.random.uniform(-2, 2, size=d)

        def f(X):
            return np.sum(
                coef * np.sin(np.sum(X[:, subset], axis=1)) 
                for (coef, subset) in zip(self.coefs, self.subsets)
            )
        
        self.sample_Y = f

# def f_general(X, d):
# """
# Computes the smooth target function value for higher dimensions.

# Parameters:
# - X: Input data of shape (n, d), where n is the number of samples.
# - d: The dimension of the data.

# Returns:
# - The computed target function values for the given dimension.
# """
# Y = np.zeros(X.shape[0])
# for i in range(d):
#     if i % 2 == 0:
#         Y += 0.1 * X[:, i] - 0.2 * X[:, i] ** 2
#     else:
#         Y += 0.05 * X[:, i]
# return Y

# def f_general(X, d): # also some interactions
# """
# Computes the sinusoidal target function value for higher dimensions.

# Parameters:
# - X: Input data of shape (n, d), where n is the number of samples.
# - d: The dimension of the data.

# Returns:
# - The computed target function values for the given dimension.
# """
# # use seed for pseudo-randomness
# np.random.seed(42)
# Y = np.zeros(X.shape[0])
# for term in range(d):
#     # draw a random subset of 1:d, more likely to be low-dimensional
#     subset = np.random.choice(
#         np.arange(d), 
#         max([1 + np.random.poisson(1), d]),
#         replace=False
#     )
#     term = np.prod(X[:, subset], axis=1)
    
#     coef_sin = np.random.uniform(-2, 2)
#     coef_cos = np.random.uniform(-2, 2)
#     # Y += fun(term, coef_sin=..., coef_cos=...)
#     Y += coef_sin * np.sin(np.pi * np.abs(term) / 2) + coef_cos * np.cos(np.pi * np.abs(term) / 2)

# # # Randomly generate coefficients for each dimension
# # coeffs_sin = np.random.uniform(-2, 2, d)
# # coeffs_cos = np.random.uniform(-2, 2, d)
# # for i in range(d):
# #     Y += coeffs_sin[i] * np.sin(np.pi * np.abs(X[:, i]) / 2) + coeffs_cos[i] * np.cos(np.pi * np.abs(X[:, i]) / 2)
# return Y
