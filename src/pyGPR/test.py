import matplotlib.pyplot as plt

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

class SparseGP:
    """
    Sparse Gaussian Process with recursive Bayesian regression updates.

    Parameters
    ----------
    X : np.ndarray, shape (N, D)
        Training inputs.
    Y : np.ndarray, shape (N, 1)
        Training outputs.
    U : np.ndarray, shape (M, D)
        Inducing points.
    kernel : object
        Kernel with method K(X1, X2=None) -> covariance matrix.
    variance : float, optional
        Observation noise variance (sigma^2).
    lambda_ : float, optional
        Forgetting factor (default=1.0, no forgetting).
    """

    def __init__(self, 
                 X: np.ndarray, 
                 Y: np.ndarray, 
                 U: np.ndarray, 
                 kernel, 
                 variance: float = 1e-6,
                 lambda_: float = 1.0):

        self.X = np.atleast_2d(X)
        self.Y = np.atleast_2d(Y)
        self.U = np.atleast_2d(U)
        self.kernel = kernel
        self.variance = variance
        self.lambda_ = lambda_

        # Core kernel matrices
        self.Kuu = self.kernel.K(self.U)
        self.Kuf = self.kernel.K(self.U, self.X)

        # Precompute inverses (with jitter for stability)
        jitter = 1e-6 * np.eye(len(self.U))
        self.Lu = np.linalg.cholesky(self.Kuu + jitter)
        self.Kuu_inv = np.linalg.solve(
            self.Lu.T, np.linalg.solve(self.Lu, np.eye(len(self.U)))
        )

        # Posterior covariance Su
        L = np.linalg.cholesky(self.Kuu + (1.0 / variance) * self.Kuf @ self.Kuf.T)
        self.Su = self.Kuu @ np.linalg.solve(L.T, np.linalg.solve(L, self.Kuu))

        # Posterior mean mu
        self.mu = (1.0 / self.variance) * self.Su @ self.Kuu_inv @ self.Kuf @ self.Y

    # ------------------------------------------------------------------

    def predict(self, Xs: np.ndarray):
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
        Xs = np.atleast_2d(Xs)

        Kss = self.kernel.K(Xs)
        Ksu = self.kernel.K(Xs, self.U)
        
        mu = Ksu @ self.Kuu_inv @ self.mu
        var = Kss - Ksu @ (self.Kuu_inv - self.Kuu_inv @ self.Su @ self.Kuu_inv) @ Ksu.T

        return mu, np.diag(var)

    # ------------------------------------------------------------------

    def update(self, X_new: np.ndarray, Y_new: np.ndarray):
        """
        Online update with a single new data point.
        Not good for the new data that are far from the inducing points as the inducing points are not updated.

        Parameters
        ----------
        X_new : np.ndarray, shape (1, D)
            New input.
        Y_new : np.ndarray, shape (1,) or (1,1)
            New output.
        """
        X_new = np.atleast_2d(X_new)
        Y_new = np.atleast_1d(Y_new).reshape(-1, 1)

        # Φ_k = K(z, U) Kuu^{-1}, shape (M, 1)
        K_zU = self.kernel.K(X_new, self.U)        # shape (1, M)
        Phi_k = (K_zU @ self.Kuu_inv).T            # (M, 1)

        # Residual: r_k = y - Phi_k^T mu
        r_k = Y_new - Phi_k.T @ self.mu            # scalar

        # G_k = Phi_k^T S_u Phi_k + sigma^2
        G_k = Phi_k.T @ self.Su @ Phi_k + self.variance + 1 - self.lambda_  # scalar

        # L_k = S_u Phi_k / G_k
        L_k = self.Su @ Phi_k / G_k                # (M, 1)

        # Update mean and covariance
        self.mu = self.mu + L_k * r_k
        self.Su = (1.0 / self.lambda_) * (self.Su - L_k @ (Phi_k.T @ self.Su))

    def update_batch(self, X_new, Y_new):
        """
        Online update with a batch of new data points.
        Not good for the new data that are far from the inducing points as the inducing points are not updated.

        Parameters
        ----------
        X_new : np.ndarray, shape (B, D)
            New inputs.
        Y_new : np.ndarray, shape (B, 1)
            New outputs.
        """
        X_new = np.atleast_2d(X_new)
        Y_new = np.atleast_2d(Y_new)

        # Phi_b = K(X_new, U) Kuu^{-1}, shape (B, M)
        K_XU = self.kernel.K(X_new, self.U)      # (B, M)
        Phi_b = K_XU @ self.Kuu_inv              # (B, M)

        # Residuals: R_b = Y_new - Phi_b @ mu
        R_b = Y_new - Phi_b @ self.mu            # (B, 1)

        # G_b = Phi_b S_u Phi_b^T + sigma^2 I
        G_b = Phi_b @ self.Su @ Phi_b.T + (self.variance + 1 - self.lambda_) * np.eye(len(X_new))  # (B,B)

        # L_b = S_u Phi_b^T G_b^{-1}
        L_b = self.Su @ Phi_b.T @ np.linalg.inv(G_b)  # (M,B)

        # Update mean and covariance
        self.mu = self.mu + L_b @ R_b
        self.Su = (1.0 / self.lambda_) * (self.Su - L_b @ Phi_b @ self.Su)

def testExactGP(noise_std, X, Y):
    gpModel = ExactGP(X, Y, RBF(input_dim=1, variance=1., lengthscale=1.), variance=noise_std**2)

    xPredict = np.linspace(-15, 25, 200)[:, None]
    mu, var = gpModel.predict(xPredict)
    std = np.sqrt(var)
    print(mu.shape, std.shape)
    plt.figure(figsize=(10, 6))
    plt.plot(X, Y, 'kx', label='Training Data')
    plt.plot(xPredict, mu, 'b', label='Predictive Mean')
    plt.fill_between(xPredict.flatten(), 
                     (mu.flatten() - 2 * std), 
                     (mu.flatten() + 2 * std), 
                     color='blue', alpha=0.2, label='Confidence Interval')
    plt.title('Gaussian Process Regression')
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.legend()
    plt.show()

def testSparseGP(noise_std, X, Y, U):
    gpModel = SparseGP(X, Y, U, RBF(input_dim=1, variance=1., lengthscale=1.), variance=noise_std**2)

    xPredict = np.linspace(-15, 25, 200)[:, None]
    mu, var = gpModel.predict(xPredict)
    std = np.sqrt(var)
    print(mu.shape, std.shape)
    plt.figure(figsize=(10, 6))
    plt.plot(X, Y, 'kx', label='Training Data')
    plt.plot(xPredict, mu, 'b', label='Predictive Mean')
    plt.fill_between(xPredict.flatten(), 
                     (mu.flatten() - 2 * std), 
                     (mu.flatten() + 2 * std), 
                     color='blue', alpha=0.2, label='Confidence Interval')
    plt.title('Gaussian Process Regression')
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.legend()
    plt.show()

def testCompareExactSparse(noise_std, X, Y, U):
    gpExact = ExactGP(X, Y, RBF(input_dim=1, variance=1., lengthscale=1.), variance=noise_std**2)
    gpSparse = SparseGP(X, Y, U, RBF(input_dim=1, variance=1., lengthscale=1.), variance=noise_std**2)

    xPredict = np.linspace(-15, 25, 200)[:, None]
    mu_exact, var_exact = gpExact.predict(xPredict)
    mu_sparse, var_sparse = gpSparse.predict(xPredict)
    std_exact = np.sqrt(var_exact)
    std_sparse = np.sqrt(var_sparse)

    xOrigin = np.linspace(0, 10, 200)[:, None]
    mu_exact_origin, var_exact_origin = gpExact.predict(xOrigin)
    mu_sparse_origin, var_sparse_origin = gpSparse.predict(xOrigin)
    error_exact = np.sum(np.abs((mu_exact_origin - np.sin(xOrigin))))
    error_sparse = np.sum(np.abs((mu_sparse_origin - np.sin(xOrigin))))
    print(f"L1 Error (Exact GP):  {error_exact:.4f}")
    print(f"L1 Error (Sparse GP): {error_sparse:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(X, Y, 'kx', label='Training Data')
    plt.plot(xPredict, mu_exact, 'b', label='Exact GP Mean')
    plt.fill_between(xPredict.flatten(), 
                     (mu_exact.flatten() - 2 * std_exact), 
                     (mu_exact.flatten() + 2 * std_exact), 
                     color='blue', alpha=0.2, label='Exact GP Confidence Interval')
    plt.plot(xPredict, mu_sparse, 'r', label='Sparse GP Mean')
    plt.fill_between(xPredict.flatten(), 
                     (mu_sparse.flatten() - 2 * std_sparse), 
                     (mu_sparse.flatten() + 2 * std_sparse), 
                     color='red', alpha=0.2, label='Sparse GP Confidence Interval')
    plt.title('Comparison of Exact and Sparse Gaussian Process Regression')
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.legend()
    plt.show()

def testUpdateSparse(noise_std, X, Y, U):
    # Initialize Sparse GP
    gpSparse = SparseGP(
        X, Y, U,
        RBF(input_dim=1, variance=1., lengthscale=1.),
        variance=noise_std**2,
        lambda_=1.0
    )

    # Prediction grid
    xPredict = np.linspace(-15, 25, 200)[:, None]

    # Predict before update
    mu_sparse, var_sparse = gpSparse.predict(xPredict)   # shapes (200,1)
    std_sparse = np.sqrt(var_sparse)                     # (200,1)

    # Plot before update
    plt.figure(figsize=(10, 6))
    plt.plot(X, Y, 'kx', label='Training Data')
    plt.plot(xPredict.flatten(), mu_sparse.flatten(), 'r', label='Sparse GP Mean Before Update')
    plt.fill_between(
        xPredict.flatten(),
        (mu_sparse.flatten() - 2 * std_sparse),
        (mu_sparse.flatten() + 2 * std_sparse),
        color='red', alpha=0.2,
        label='Sparse GP Confidence Interval Before Update'
    )

    # New data point
    newN = 10
    X_new = np.linspace(-5, 0, newN)[:, None]
    Y_new = np.sin(X_new) + noise_std * np.random.randn(newN, 1)
    plt.plot(X_new, Y_new, 'cx', label='New Data Point')

    # Update GP
    gpSparse.update_batch(X_new, Y_new)
    # for xk, yk in zip(X_new, Y_new):
    #     gpSparse.update(xk, yk)

    # Predict after update
    mu_sparse_updated, var_sparse_updated = gpSparse.predict(xPredict)
    std_sparse_updated = np.sqrt(var_sparse_updated)

    # Plot after update
    plt.plot(xPredict.flatten(), mu_sparse_updated.flatten(), 'g', label='Sparse GP Mean After Update')
    plt.fill_between(
        xPredict.flatten(),
        (mu_sparse_updated.flatten() - 2 * std_sparse_updated),
        (mu_sparse_updated.flatten() + 2 * std_sparse_updated),
        color='green', alpha=0.2,
        label='Sparse GP Confidence Interval After Update'
    )

    # Final touches
    plt.title('Sparse Gaussian Process Regression with Online Update')
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.legend()
    plt.show()
    

if __name__ == "__main__":
    # np.random.seed(0)
    noise_std = 0.03
    X = np.linspace(0, 10, 100)[:, None]
    Y = np.sin(X) + noise_std * np.random.randn(100, 1)
    U = np.linspace(0, 10, 10)[:, None]

    # testExactGP(noise_std, X, Y)
    # testSparseGP(noise_std, X, Y, U)
    # testCompareExactSparse(noise_std, X, Y, U)
    testUpdateSparse(noise_std, X, Y, U)

