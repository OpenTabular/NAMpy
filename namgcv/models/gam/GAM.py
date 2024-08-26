import numpy as np
from scipy.linalg import sqrtm, eigh
import pandas as pd
from Splines import CubicSplines

class GAM:
    def __init__(self, 
                formula=None, 
                family='gaussian', 
                data=pd.DataFrame(), 
                weights=None, 
                subset=None, 
                na_action='drop', 
                offset=None, 
                method="GCV.Cp", 
                optimizer=("outer", "newton"), 
                control=None, 
                scale=0, 
                select=False, 
                knots=None, 
                sp=None, 
                min_sp=None, 
                H=None, 
                gamma=1.0, 
                fit=True, 
                paraPen=None, 
                G=None, 
                in_out=None, 
                drop_unused_levels=True, 
                drop_intercept=None, 
                nei=None, 
                discrete=False, 
                **kwargs):
        
        self.formula = formula
        self.family = family
        self.data = data
        self.weights = weights
        self.subset = subset
        self.na_action = na_action
        self.offset = offset
        self.method = method
        self.optimizer = optimizer
        self.control = control if control is not None else {}
        self.scale = scale
        self.select = select
        self.knots = knots
        self.sp = sp
        self.min_sp = min_sp
        self.H = H
        self.gamma = gamma
        self.fit = fit
        self.paraPen = paraPen
        self.G = G
        self.in_out = in_out
        self.drop_unused_levels = drop_unused_levels
        self.drop_intercept = drop_intercept
        self.nei = nei
        self.discrete = discrete
        self.additional_args = kwargs

        self.data = data
        self.X = self.data.iloc[:, 1:]
        self.y = self.data.iloc[:, 0]
        

    def fit(self):
        """Function to fit simple 2 term generalized additive model with Gamma errors and log link"""
        # Assuming sp, S are defined
        sp = self.sp
        S = [self._spl_S(self.X.iloc[:, i]) for i in range(self.X.shape[1])]
        
        rS = self._mat_sqrt(sp[0] * S[0] + sp[1] * S[1])
        q = self.X.shape[1]  # number of params
        n = self.X.shape[0]  # number of data points
        X1 = np.vstack([self.X, rS])  # augmented model matrix

        b = np.zeros(q)
        b[0] = 1  # initialize parameters
        norm = 0
        old_norm = 1  # initialize convergence control

        while np.abs(norm - old_norm) > 1e-4 * norm:  # repeat until converged
            eta = X1 @ b  # linear predictor
            mu = np.exp(eta[:n])  # fitted values
            z = (self.y - mu) / mu + eta[:n]  # pseudodata
            z = np.hstack([z, np.zeros(q)])  # augmented pseudodata
            b = np.linalg.lstsq(X1, z, rcond=None)[0]  # fit penalized working model
            hat_matrix_diag = np.linalg.lstsq(X1.T @ X1, X1.T, rcond=None)[0].diagonal()[:n]
            trA = np.sum(hat_matrix_diag)  # tr(A)
            old_norm = norm  # store for convergence test
            norm = np.sum((z[:n] - X1[:n] @ b) ** 2)  # RSS of working model

        return {"model": b, "gcv": norm * n / (n - trA) ** 2, "sp": sp}

    def rk(self, x, z):
        return ((z - 0.5) ** 2 - 1 / 12) * ((x - 0.5) ** 2 - 1 / 12) / 4 - \
            ((np.abs(x - z) - 0.5) ** 4 - (np.abs(x - z) - 0.5) ** 2 / 2 + 7 / 240) / 24
    
    def _mat_sqrt(self, S):
        """A simple matrix square root"""
        eigenvalues, eigenvectors = eigh(S)
        sqrt_eigenvalues = np.diag(np.sqrt(eigenvalues))
        return eigenvectors @ sqrt_eigenvalues @ eigenvectors.T

    def _spl_X(self, x, xk):
        """Set up model matrix for cubic penalized regression spline"""
        q = len(xk) + 2  # number of parameters
        n = len(x)  # number of data points
        X = np.ones((n, q))  # initialized model matrix
        X[:, 1] = x  # set second column to x
        for i in range(2, q):
            X[:, i] = self.rk(x, xk[i-2])  # and remaining columns to R(x, xk)
        return X

    def _spl_S(self, xk):
        """Set up the penalized regression spline penalty matrix, given knot sequence xk"""
        q = len(xk) + 2
        S = np.zeros((q, q))  # initialize matrix to 0
        for i in range(2, q):
            for j in range(2, q):
                S[i, j] = self.rk(xk[i-2], xk[j-2])  # fill in non-zero part
        return S

    def apply_gcv(self, y, am_setup_output):
        """Function to apply GCV over a grid of smoothing parameters and find the best one."""
        best = None
        best_sp = [0, 0]  # initialize smoothing parameters
        
        for i in range(1, 31):  # loop over s.p. grid
            for j in range(1, 31):
                sp = [1e-5 * 2 ** (i - 1), 1e-5 * 2 ** (j - 1)]
                b = self.fit_gamG(y, am_setup_output["X"], am_setup_output["S"], sp)  # fit using s.p.s
                
                if best is None or b["gcv"] < best["gcv"]:
                    best = b  # store best model
                    best_sp = sp  # store best smoothing parameters

        return best_sp, best  # return the best smoothing parameters and the model
