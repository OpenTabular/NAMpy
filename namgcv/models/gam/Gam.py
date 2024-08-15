import numpy as np
from scipy.linalg import sqrtm, eigh
from scipy.stats import mstats

class GAM:
    def __init__(self):
        pass

    def rk(x, z):
    """R(x, z) for cubic spline on [0,1]"""
    return ((z - 0.5) ** 2 - 1 / 12) * ((x - 0.5) ** 2 - 1 / 12) / 4 - \
           ((np.abs(x - z) - 0.5) ** 4 - (np.abs(x - z) - 0.5) ** 2 / 2 + 7 / 240) / 24

def _mat_sqrt(S):
    """A simple matrix square root"""
    eigenvalues, eigenvectors = eigh(S)
    sqrt_eigenvalues = np.diag(np.sqrt(eigenvalues))
    return eigenvectors @ sqrt_eigenvalues @ eigenvectors.T

def _spl_X(x, xk):
    """Set up model matrix for cubic penalized regression spline"""
    q = len(xk) + 2  # number of parameters
    n = len(x)  # number of data points
    X = np.ones((n, q))  # initialized model matrix
    X[:, 1] = x  # set second column to x
    for i in range(2, q):
        X[:, i] = rk(x, xk[i-2])  # and remaining columns to R(x, xk)
    return X

def _spl_S(xk):
    """Set up the penalized regression spline penalty matrix, given knot sequence xk"""
    q = len(xk) + 2
    S = np.zeros((q, q))  # initialize matrix to 0
    for i in range(2, q):
        for j in range(2, q):
            S[i, j] = rk(xk[i-2], xk[j-2])  # fill in non-zero part
    return S

def _setup(x, z, q=10):
    """Get X, S_1 and S_2 for a simple 2 term AM"""
    # choose knots
    xk = mstats.mquantiles(np.unique(x), prob=np.linspace(1 / (q - 1), 1 - 1 / (q - 1), q - 2))
    zk = mstats.mquantiles(np.unique(z), prob=np.linspace(1 / (q - 1), 1 - 1 / (q - 1), q - 2))

    # get penalty matrices
    S = [np.zeros((2 * q - 1, 2 * q - 1)), np.zeros((2 * q - 1, 2 * q - 1))]
    S[0][1:q, 1:q] = spl_S(xk)[1:, 1:]
    S[1][q:(2 * q - 1), q:(2 * q - 1)] = spl_S(zk)[1:, 1:]

    # get model matrix
    n = len(x)
    X = np.ones((n, 2 * q - 1))
    X[:, 1:q] = spl_X(x, xk)[:, 1:]  # 1st smooth
    X[:, q:(2 * q - 1)] = spl_X(z, zk)[:, 1:]  # 2nd smooth

    return {"X": X, "S": S}

def fit(y, X, S, sp):
    """Function to fit simple 2 term generalized additive model
       with Gamma errors and log link"""
    # get sqrt of combined penalty matrix
    rS = mat_sqrt(sp[0] * S[0] + sp[1] * S[1])
    q = X.shape[1]  # number of params
    n = X.shape[0]  # number of data points
    X1 = np.vstack([X, rS])  # augmented model matrix

    b = np.zeros(q)
    b[0] = 1  # initialize parameters
    norm = 0
    old_norm = 1  # initialize convergence control

    while np.abs(norm - old_norm) > 1e-4 * norm:  # repeat until converged
        eta = X1 @ b  # linear predictor
        mu = np.exp(eta[:n])  # fitted values
        z = (y - mu) / mu + eta[:n]  # pseudodata
        z = np.hstack([z, np.zeros(q)])  # augmented pseudodata
        b = np.linalg.lstsq(X1, z, rcond=None)[0]  # fit penalized working model
        hat_matrix_diag = np.linalg.lstsq(X1.T @ X1, X1.T, rcond=None)[0].diagonal()[:n]
        trA = np.sum(hat_matrix_diag)  # tr(A)
        old_norm = norm  # store for convergence test
        norm = np.sum((z[:n] - X1[:n] @ b) ** 2)  # RSS of working model

    return {"model": b, "gcv": norm * n / (n - trA) ** 2, "sp": sp}

def apply_gcv(y, am_setup_output):
    """Function to apply GCV over a grid of smoothing parameters and find the best one."""
    best = None
    best_sp = [0, 0]  # initialize smoothing parameters
    
    for i in range(1, 31):  # loop over s.p. grid
        for j in range(1, 31):
            sp = [1e-5 * 2 ** (i - 1), 1e-5 * 2 ** (j - 1)]
            b = fit_gamG(y, am_setup_output["X"], am_setup_output["S"], sp)  # fit using s.p.s
            
            if best is None or b["gcv"] < best["gcv"]:
                best = b  # store best model
                best_sp = sp  # store best smoothing parameters

    return best_sp, best  # return the best smoothing parameters and the model
