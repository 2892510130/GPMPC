import numpy as np
from GPy.kern import RBF

from utils import block_matrix_inversion, cholesky_add_row

class SparseFITCGP:
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

        self.M = len(self.U)
        self.N = len(self.X)

        # Core kernel matrices
        self.Kuu = self.kernel.K(self.U)
        self.Kuu = (self.Kuu + self.Kuu.T) / 2  # ensure symmetry
        self.Kuf = self.kernel.K(self.U, self.X)

        # Precompute inverses (with jitter for stability)
        jitter = 1e-6 * np.eye(self.M)
        self.Luu = np.linalg.cholesky(self.Kuu + jitter)

        # --- Factorized form ---
        # W = Luu^{-1} Kuf
        self.W = np.linalg.solve(self.Luu, self.Kuf)

        self.Kff_diag = self.kernel.Kdiag(self.X)
        # self.Qff_diag = np.diag(self.Kuf.T @ self.Kuu_inv @ self.Kuf)
        self.Qff_diag = np.power(self.W.T, 2).sum(axis=-1)
        self.diag = self.Kff_diag - self.Qff_diag + self.variance
        self.diag_inv = 1.0 / self.diag

        # K = I + W D^{-1} W^T
        # WD_half = self.W @ np.sqrt(self.D)
        self.K = (self.W * self.diag_inv) @ self.W.T
        self.K = (self.K + self.K.T) / 2 + np.eye(self.M) # ensure symmetry

        # Cholesky of K
        self.L = np.linalg.cholesky(self.K)

        self.L_updated = True

        self.update_mu_su()

    # ------------------------------------------------------------------

    def update_mu_su(self):
        self.update_L()

        # self.Kuu = self.kernel.K(self.U)
        # self.Kuu = (self.Kuu + self.Kuu.T) / 2  # ensure symmetry

        # self.Kuu_inv = np.linalg.solve(
        #     self.Luu.T, np.linalg.solve(self.Luu, np.eye(self.M))
        # )

        # Posterior covariance Su
        # Sigma = inv[Kuu + Kuf @ inv(D) @ Kfu]
        #   = inv(Luu).T @ inv[I + inv(Luu)@ Kuf @ inv(D)@ Kfu @ inv(Luu).T] @ inv(Luu)
        #   = inv(Luu).T @ inv[I + W @ inv(D) @ W.T] @ inv(Luu)
        #   = inv(Luu).T @ inv(K) @ inv(Luu)
        #   = Luu^{-T} L^{-T} L^{-1} Luu^{-1}
        # Su = Kuu @ Sigma @ Kuu = Luu L^{-T} L^{-1} Luu.T = tmp.T @ tmp
        tmp = np.linalg.solve(self.L, self.Luu.T)
        # tmp = np.linalg.solve(self.L, np.linalg.solve(self.Luu, self.Kuu))
        self.Su = tmp.T @ tmp  # symmetric MxM

        # Posterior mean mu
        # mu = Su @ Kuu^{-1} Kuf D^{-1} Y
        # rhs = (self.Kuf * self.diag_inv) @ self.Y
        # self.mu = self.Su @ (self.Kuu_inv @ rhs)
        rhs = np.linalg.solve(self.Luu, (self.Kuf * self.diag_inv) @ self.Y)
        self.mu = np.linalg.solve(self.Luu, self.Su.T).T @ rhs

    # ------------------------------------------------------------------

    def predict(self, Xs: np.ndarray, method="cholesky"):
        if method == "cholesky":
            return self.predict_cholesky(Xs)
        elif method == "mu_su":
            return self.predict_mu_su(Xs)

    # ------------------------------------------------------------------

    def predict_mu_su(self, Xs: np.ndarray):
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

        self.update_L()

        Kss = self.kernel.K(Xs)
        Ksu = self.kernel.K(Xs, self.U)
        Ws = np.linalg.solve(self.Luu, Ksu.T)
        tmp = np.linalg.solve(self.L, np.linalg.solve(self.Luu, Ksu.T))
        
        mu = Ws.T @ np.linalg.solve(self.Luu, self.mu)
        # var = Kss - Ksu @ (self.Kuu_inv - self.Kuu_inv @ self.Su @ self.Kuu_inv) @ Ksu.T
        var = Kss - Ws.T @ Ws + tmp.T @ tmp

        # whether we can only use Luu to compute the mean and var, and modify the update function too, without Kuu_inv
        # Ws = np.linalg.solve(self.Luu, Ksu.T)
        # Luu_inv_mu = np.linalg.solve(self.Luu, self.mu)
        # mu = Ws.T @ Luu_inv_mu
        # var = kss - Ws.T @ Ws + 

        return mu, np.diag(var)
    
    # ------------------------------------------------------------------

    def predict_cholesky(self, Xs: np.ndarray):
        """
        Predict mean and variance at new test points using Cholesky factors.
        W = inv(Luu) @ Kuf
        Ws = inv(Luu) @ Kus
        D as in self.model()
        K = I + W @ inv(D) @ W.T = L @ L.T
        S = inv[Kuu + Kuf @ inv(D) @ Kfu]
          = inv(Luu).T @ inv[I + inv(Luu)@ Kuf @ inv(D)@ Kfu @ inv(Luu).T] @ inv(Luu)
          = inv(Luu).T @ inv[I + W @ inv(D) @ W.T] @ inv(Luu)
          = inv(Luu).T @ inv(K) @ inv(Luu)
          = inv(Luu).T @ inv(L).T @ inv(L) @ inv(Luu)
        loc = Ksu @ S @ Kuf @ inv(D) @ y = Ws.T @ inv(L).T @ inv(L) @ W @ inv(D) @ y
        cov = Kss - Ksu @ inv(Kuu) @ Kus + Ksu @ S @ Kus
            = kss - Ksu @ inv(Kuu) @ Kus + Ws.T @ inv(L).T @ inv(L) @ Ws

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
        Ws = np.linalg.solve(self.Luu, Ksu.T)
        W_d_inv_y = (self.W * self.diag_inv) @ self.Y

        self.update_L()

        L_inv_Ws = np.linalg.solve(self.L, Ws)
        L_inv_W_d_inv_y = np.linalg.solve(self.L, W_d_inv_y)

        mu = L_inv_Ws.T @ L_inv_W_d_inv_y
        var = Kss - Ws.T @ Ws + L_inv_Ws.T @ L_inv_Ws
        return mu, np.diag(var)

    # ------------------------------------------------------------------

    def update(self, X_new: np.ndarray, Y_new: np.ndarray, Ksu=None):
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

        self.update_L()

        if Ksu is None:
            Ksu = self.kernel.K(X_new, self.U)
        # When the new data is far from the original data, Ktt_diag - Qtt_diag will close to 1, make the update not useful
        Ktt_diag = self.kernel.Kdiag(X_new)
        # Qtt_diag = np.diag(Ksu @ self.Kuu_inv @ Ksu.T)
        Ws = np.linalg.solve(self.Luu, Ksu.T)
        Qtt_diag = np.power(Ws.T, 2).sum(axis=-1)

        # Φ_k = K(z, U) Kuu^{-1}, shape (M, 1)
        K_zU = self.kernel.K(X_new, self.U)        # shape (1, M)
        # Phi_k = (K_zU @ self.Kuu_inv).T            # (M, 1)
        Phi_k = np.linalg.solve(self.Luu.T, np.linalg.solve(self.Luu, K_zU.T))            # (M, 1)

        # Residual: r_k = y - Phi_k^T mu
        r_k = Y_new - Phi_k.T @ self.mu            # scalar

        # G_k = Phi_k^T S_u Phi_k + sigma^2
        G_k = Phi_k.T @ self.Su @ Phi_k + self.variance + Ktt_diag - Qtt_diag + 1 - self.lambda_  # scalar

        # L_k = S_u Phi_k / G_k
        L_k = self.Su @ Phi_k / G_k                # (M, 1)

        # Update mean and covariance
        self.mu = self.mu + L_k * r_k
        self.Su = (1.0 / self.lambda_) * (self.Su - L_k @ (Phi_k.T @ self.Su))

    def update_batch(self, X_new, Y_new, Ksu=None):
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

        self.update_L()

        if Ksu is None:
            Ksu = self.kernel.K(X_new, self.U)
        # When the new data is far from the original data, Ktt_diag - Qtt_diag will close to 1, make the update not useful
        Ktt_diag = self.kernel.Kdiag(X_new)
        # Qtt_diag = np.diag(Ksu @ self.Kuu_inv @ Ksu.T)
        Ws = np.linalg.solve(self.Luu, Ksu.T)
        Qtt_diag = np.power(Ws.T, 2).sum(axis=-1)

        # Phi_b = K(X_new, U) Kuu^{-1}, shape (B, M)
        K_XU = self.kernel.K(X_new, self.U)      # (B, M)
        # Phi_b = K_XU @ self.Kuu_inv              # (B, M)
        Phi_b = np.linalg.solve(self.Luu.T, np.linalg.solve(self.Luu, K_XU.T)).T

        # Residuals: R_b = Y_new - Phi_b @ mu
        R_b = Y_new - Phi_b @ self.mu            # (B, 1)

        # G_b = Phi_b S_u Phi_b^T + sigma^2 I
        G_b = Phi_b @ self.Su @ Phi_b.T + (self.variance + 1 - self.lambda_) * np.eye(len(X_new)) + np.diag(Ktt_diag - Qtt_diag)  # (B,B)

        # L_b = S_u Phi_b^T G_b^{-1}
        L_b = self.Su @ Phi_b.T @ np.linalg.inv(G_b)  # (M,B)

        # Update mean and covariance
        self.mu = self.mu + L_b @ R_b
        self.Su = (1.0 / self.lambda_) * (self.Su - L_b @ Phi_b @ self.Su)

    def update_L(self):
        if not self.L_updated:
            self.K = (self.W * self.diag_inv) @ self.W.T
            self.K = (self.K + self.K.T) / 2 + np.eye(self.M)
            self.L = np.linalg.cholesky(self.K)
            self.L_updated = True

    def add_new_data(self, X_new, Y_new):
        self.X = np.vstack((self.X, X_new))
        self.Y = np.vstack((self.Y, Y_new))

        self.N = self.X.shape[0]

        Kuf_new = self.kernel.K(self.U, X_new)
        W_new = np.linalg.solve(self.Luu, Kuf_new)
        self.Kuf = np.hstack((self.Kuf, Kuf_new))
        self.W = np.hstack((self.W, W_new))

        Kff_diag_new = self.kernel.Kdiag(X_new)
        Qff_diag_new = np.power(W_new.T, 2).sum(axis=-1)
        self.diag = np.hstack((self.diag, Kff_diag_new - Qff_diag_new + self.variance))
        self.diag_inv = 1.0 / self.diag
        
        self.L_updated = False

    def remove_data(self, remove_number = None, reserve_indices = None):
        if remove_number is None and reserve_indices is None:
            raise ValueError("Either remove_number or reserve_indices must be provided.")

        self.X = self.X[remove_number:]
        self.Y = self.Y[remove_number:]
        self.N = self.X.shape[0]

        self.Kuf = self.Kuf[:, remove_number:]
        self.W = self.W[:, remove_number:]
        self.diag = self.diag[remove_number:]
        self.diag_inv = 1.0 / self.diag

        # self.L_updated = False # Whether this is needed?

    def add_new_inducing(self, U_new):
        self.U = np.vstack((self.U, U_new))
        self.M = self.U.shape[0]

        a = self.kernel.K(self.U[:-1], U_new)
        aa = self.kernel.K(U_new) + 1e-6  # jitter for numerical stability
        self.Luu = cholesky_add_row(self.Luu, a, aa)

        Ku_new_f = self.kernel.K(U_new, self.X)
        self.Kuf = np.vstack((self.Kuf, Ku_new_f))

        W_new = (Ku_new_f - self.Luu[-1:, :-1] @ self.W) / self.Luu[-1, -1]
        self.W = np.vstack((self.W, W_new))

        Qff_diag_new = np.power(self.W.T[:,-1:], 2).sum(axis=-1)
        self.diag -= Qff_diag_new
        self.diag_inv = 1.0 / self.diag

        self.L_updated = False

    def remove_inducing(self, remove_number = None, reserve_indices = None):
        pass