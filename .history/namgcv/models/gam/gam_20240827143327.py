import numpy as np
from scipy.linalg import sqrtm, eigh
import pandas as pd
from Splines import CubicSplines
import itertools


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
        #self.fit = fit
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
        
        # Initialize smoothing parameter (s.p.) array with a specified size

        sp_size = self.X.shape[1]  # for example, sp can have 3 elements
        sp = [0] * sp_size
        best = None
        Ss = [CubicSplines(self.X.iloc[:, i], k=12).S for i in range(self.X.shape[1])]
        bases = [CubicSplines(self.X.iloc[:, i], k=12).S for i in range(self.X.shape[1])]

        # Generate the range for the grid search
        sp_ranges = [range(1, 31)] * sp_size

        # Loop over s.p. grid
        for indices in itertools.product(*sp_ranges):
            for k in range(sp_size):
                sp[k] = 1e-5 * 2**(indices[k]-1)  # update each element of sp
            b = self.fit_GAM(self.y, bases, Ss, sp)  # fit using s.p.s
            if best is None or b['gcv'] < best['gcv']:
                best = b  # store best model
        print(b)


    def fit_GAM(self, y, X, S, sp):
        """Function to fit generalized additive model with Gamma errors and log link, allowing for variable number of smoothing terms."""

        # Initialize rS as a weighted sum of the smoothing matrices in S
        rS = sum(sp[i] * S[i] for i in range(len(S)))

        rS = self._mat_sqrt(rS)
        q = X.shape[1]  # number of parameters
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

