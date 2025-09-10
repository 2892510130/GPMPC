import numpy as np
from GPy.kern import RBF

class ExactGP:
    """
    Sparse Gaussian Process with recursive Bayesian regression updates.

    Parameters
    ----------
    X : np.ndarray, shape (N, D)
        Training inputs.
    Y : np.ndarray, shape (N, 1)
        Training outputs.
    kernel : object
        Kernel with method K(X1, X2=None) -> covariance matrix.
    variance : float, optional
        Observation noise variance (sigma^2).
    """

    def __init__(self, 
                 X: np.ndarray, 
                 Y: np.ndarray, 
                 kernel, 
                 variance: float = 1e-6):
        
        self.X = np.atleast_2d(X)
        self.Y = np.atleast_2d(Y)
        self.kernel = kernel
        self.variance = variance

        self.Kff = self.kernel.K(self.X) + (1e-6 + self.variance) * np.eye(len(self.X))
        self.L = np.linalg.cholesky(self.Kff)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, Y))

    def predict(self, Xs):
        """
        Predict mean and variance at new test points.

        Parameters
        ----------
        Xs : np.ndarray, shape (N*, D)
            Test inputs.

        Returns
        -------
        mu : np.ndarray, shape (N*,)
            Predictive mean.
        var : np.ndarray, shape (N*,)
            Predictive variance.
        """
        Ks = self.kernel.K(self.X, Xs)
        Kss = self.kernel.K(Xs)
        mu = Ks.T @ self.alpha
        v = np.linalg.solve(self.L, Ks)
        var = Kss - v.T @ v
        return mu, np.diag(var)