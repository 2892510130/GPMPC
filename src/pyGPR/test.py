import matplotlib.pyplot as plt

import numpy as np
from GPy.kern import RBF

from exact_gp import ExactGP
from sparse_dtc_gp import SparseDTCGP
from sparse_fitc_gp import SparseFITCGP
from utils import block_matrix_inversion, cholesky_add_row


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

def testSparseDTCGP(noise_std, X, Y, U):
    gpModel = SparseDTCGP(X, Y, U, RBF(input_dim=1, variance=1., lengthscale=1.), variance=noise_std**2)

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
    gpSparse = SparseFITCGP(X, Y, U, RBF(input_dim=1, variance=1., lengthscale=1.), variance=noise_std**2)

    predict_method = "mu_su" # mu_su cholesky

    xPredict = np.linspace(-15, 25, 200)[:, None]
    mu_exact, var_exact = gpExact.predict(xPredict)
    mu_sparse, var_sparse = gpSparse.predict(xPredict, method=predict_method)
    std_exact = np.sqrt(var_exact)
    std_sparse = np.sqrt(var_sparse)

    xOrigin = np.linspace(0, 10, 200)[:, None]
    mu_exact_origin, var_exact_origin = gpExact.predict(xOrigin)
    mu_sparse_origin, var_sparse_origin = gpSparse.predict(xOrigin, method=predict_method)
    error_exact = np.sum(np.abs((mu_exact_origin - np.sin(xOrigin))))
    error_sparse = np.sum(np.abs((mu_sparse_origin - np.sin(xOrigin))))
    print(f"L1 Error (Exact GP):  {error_exact:.4f}")
    print(f"L1 Error (Sparse GP): {error_sparse:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(X, Y, 'kx', label='Training Data')
    plt.plot(U, np.sin(U), 'o', label='Inducing Data')
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

def testCompareDTCFITC(noise_std, X, Y, U):
    gpDTC = SparseDTCGP(X, Y, U, RBF(input_dim=1, variance=1., lengthscale=1.), variance=noise_std**2)
    gpFITC = SparseFITCGP(X, Y, U, RBF(input_dim=1, variance=1., lengthscale=1.), variance=noise_std**2)

    xPredict = np.linspace(-15, 25, 200)[:, None]
    muDTC, varDTC = gpDTC.predict(xPredict)
    muFITC, varFITC = gpFITC.predict(xPredict)
    stdDTC = np.sqrt(varDTC)
    stdFITC = np.sqrt(varFITC)

    xOrigin = np.linspace(0, 10, 200)[:, None]
    muDTC_origin, varDTC_origin = gpDTC.predict(xOrigin)
    muFITC_origin, varFITC_origin = gpFITC.predict(xOrigin)
    errorDTC = np.sum(np.abs((muDTC_origin - np.sin(xOrigin))))
    errorFITC = np.sum(np.abs((muFITC_origin - np.sin(xOrigin))))
    print(f"L1 Error (DTC GP):  {errorDTC:.4f}")
    print(f"L1 Error (FITC GP): {errorFITC:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(X, Y, 'kx', label='Training Data')
    plt.plot(U, np.sin(U), 'o', label='Inducing Data')
    plt.plot(xPredict, muDTC, 'b', label='DTC GP Mean')
    plt.fill_between(xPredict.flatten(), 
                     (muDTC.flatten() - 2 * stdDTC), 
                     (muDTC.flatten() + 2 * stdDTC), 
                     color='blue', alpha=0.2, label='DTC GP Confidence Interval')
    plt.plot(xPredict, muFITC, 'r', label='FITC GP Mean')
    plt.fill_between(xPredict.flatten(), 
                     (muFITC.flatten() - 2 * stdFITC), 
                     (muFITC.flatten() + 2 * stdFITC), 
                     color='red', alpha=0.2, label='FITC GP Confidence Interval')
    plt.title('Comparison of DTC and FITC Gaussian Process Regression')
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.legend()
    plt.show()

def testUpdateSparse(noise_std, X, Y, U):
    # Initialize Sparse GP
    gpSparse = SparseDTCGP( # SparseDTCGP SparseFITCGP
        X, Y, U,
        RBF(input_dim=1, variance=1., lengthscale=1.),
        variance=noise_std**2,
        lambda_=1.0
    )

    # Prediction grid
    xPredict = np.linspace(-15, 25, 200)[:, None]
    predict_method = "mu_su"

    # Predict before update
    mu_sparse, var_sparse = gpSparse.predict(xPredict, method=predict_method)   # shapes (200,1)
    std_sparse = np.sqrt(var_sparse)                     # (200,1)

    # Plot before update
    plt.figure(figsize=(10, 6))
    plt.plot(X, Y, 'kx', label='Training Data')
    plt.plot(U, np.sin(U), 'o', label='Inducing Data')
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
    X_new = np.linspace(10, 12, newN)[:, None]
    Y_new = np.sin(X_new) + noise_std * np.random.randn(newN, 1)
    plt.plot(X_new, Y_new, 'cx', label='New Data Point')

    # Update GP
    gpSparse.update_batch(X_new, Y_new)
    # for xk, yk in zip(X_new, Y_new):
    #     gpSparse.update(xk, yk)

    # Predict after update
    mu_sparse_updated, var_sparse_updated = gpSparse.predict(xPredict, method=predict_method)
    std_sparse_updated = np.sqrt(var_sparse_updated)

    print(np.sum(mu_sparse - mu_sparse_updated), np.sum(var_sparse - var_sparse_updated))
    print(np.sum(var_sparse), np.sum(var_sparse_updated))

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

def test_block_inversion():
    random_matrix = np.random.randn(5, 5)
    sub_matrix = random_matrix[:4, :4].copy()

    inv_matrix = np.linalg.inv(random_matrix)
    inv_sub_matrix = np.linalg.inv(sub_matrix)

    A, B, C, D = sub_matrix, random_matrix[:4, 4:5], random_matrix[4:5, :4], random_matrix[4:5, 4:5]
    block_inv = block_matrix_inversion(A, B, C, D, A_inv=inv_sub_matrix)

    print(block_inv @ random_matrix)

def test_cholesky_add_row():
    random_matrix = np.random.randn(5, 5)
    random_matrix = (random_matrix + random_matrix.T) / 2 + 10 * np.eye(5)
    sub_matrix = random_matrix[:4, :4].copy()

    jitter = 1e-6 * np.eye(len(sub_matrix))
    Ls = np.linalg.cholesky(sub_matrix + jitter)

    jitter = 1e-6 * np.eye(len(random_matrix))
    L = np.linalg.cholesky(random_matrix + jitter)

    A, B, C, D = sub_matrix, random_matrix[:4, 4:5], random_matrix[4:5, :4], random_matrix[4:5, 4:5]

    Lnew = cholesky_add_row(Ls, B, D)
    print(Lnew - L)

def test_add_data(noise_std, X, Y, U):
    # Initialize Sparse GP
    gpSparse = SparseDTCGP( # SparseDTCGP SparseFITCGP
        X, Y, U,
        RBF(input_dim=1, variance=1., lengthscale=1.),
        variance=noise_std**2,
        lambda_=1.0
    )

    # Prediction grid
    xPredict = np.linspace(-15, 25, 200)[:, None]
    predict_method = "mu_su" # cholesky mu_su

    # Predict before update
    mu_sparse, var_sparse = gpSparse.predict(xPredict, method=predict_method)   # shapes (200,1)
    std_sparse = np.sqrt(var_sparse)                     # (200,1)

    # Plot before update
    plt.figure(figsize=(10, 6))
    plt.plot(X, Y, 'kx', label='Training Data')
    plt.plot(U, np.sin(U), 'o', label='Inducing Data')
    plt.plot(xPredict.flatten(), mu_sparse.flatten(), 'r', label='Sparse GP Mean Before Update')
    plt.fill_between(
        xPredict.flatten(),
        (mu_sparse.flatten() - 2 * std_sparse),
        (mu_sparse.flatten() + 2 * std_sparse),
        color='red', alpha=0.2,
        label='Sparse GP Confidence Interval Before Update'
    )

    # New data point
    newN = 20
    X_new = np.linspace(12, 15, newN)[:, None]
    Y_new = 0.5 * np.sin(X_new) + noise_std * np.random.randn(newN, 1)

    new_inducing_x, new_inducing_y = [], []

    # Update GP
    predict_method = "mu_su" # cholesky mu_su

    # add new data
    # gpSparse.add_new_data(X_new, Y_new)
    # plt.plot(X_new, Y_new, 'cx', label='New Data Point')

    # remove data
    # gpSparse.remove_data(remove_number=10)

    # add new inducing points
    # gpSparse.add_new_inducing(X_new[0].reshape(1, -1))
    for i, (xk, yk) in enumerate(zip(X_new, Y_new)):
        if i % 5 == 0:
            gpSparse.add_new_inducing(xk.reshape(1, -1))
            new_inducing_x.append(xk)
            new_inducing_y.append(yk)
    plt.plot(new_inducing_x, new_inducing_y, 'o', label='New Inducing Point')

    gpSparse.update_mu_su(from_scratch=True)
    # gpSparse.update_batch(X_new, Y_new)
    for xk, yk in zip(new_inducing_x, new_inducing_y):
        gpSparse.update(xk, yk)
    # gpSparse.update_mu_su()

    print(gpSparse.mu.T)

    # Predict after update
    mu_sparse_updated, var_sparse_updated = gpSparse.predict(xPredict, method=predict_method)
    std_sparse_updated = np.sqrt(var_sparse_updated)

    print(np.sum(mu_sparse - mu_sparse_updated), np.sum(var_sparse - var_sparse_updated))

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

def quick_test(noise_std, X, Y, U):
    random_matrix = np.random.randn(5, 5)
    random_matrix = (random_matrix + random_matrix.T) / 2 + 10 * np.eye(5)
    sub_matrix = random_matrix[:4, :4].copy()

    jitter = 1e-6 * np.eye(len(sub_matrix))
    Ls = np.linalg.cholesky(sub_matrix + jitter)

    jitter = 1e-6 * np.eye(len(random_matrix))
    L = np.linalg.cholesky(random_matrix + jitter)

    A, B, C, D = sub_matrix, random_matrix[:4, 4:5], random_matrix[4:5, :4], random_matrix[4:5, 4:5]

    Lnew = cholesky_add_row(Ls, B, D)
    print(Lnew - L)

if __name__ == "__main__":
    np.random.seed(0)
    noise_std = 0.03
    X = np.linspace(0, 10, 100)[:, None]
    Y = np.sin(X) + noise_std * np.random.randn(100, 1)
    U = np.linspace(0, 10, 10)[:, None]

    # testExactGP(noise_std, X, Y)
    # testSparseDTCGP(noise_std, X, Y, U)
    # testCompareExactSparse(noise_std, X, Y, U)
    # testCompareDTCFITC(noise_std, X, Y, U)
    # testUpdateSparse(noise_std, X, Y, U)
    test_add_data(noise_std, X, Y, U)

    # quick_test(noise_std, X, Y, U)

