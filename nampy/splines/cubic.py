import numpy as np

from .constraints import identconst
from .cubic_basis import cr_spl, cr_spl_predict
from .penalty_scaling import scale_penalty


def _compute_np_transform(X_raw, k, knots, F, x):
    """
    Compute the mgcv np=TRUE conditioning transform XP for raw (unscaled) marginals.

    XP = V * D^{-1} * U^T  where U*D*V^T = SVD of the prediction matrix at k
    equispaced points.  Applied as X_new = X_raw @ XP in te/ti constructions.
    Returns None if the condition number is too poor.
    """
    x_eval = np.linspace(float(np.min(x)), float(np.max(x)), k).reshape(-1, 1)
    X_eval = cr_spl_predict(x_eval, knots=knots, F=F)
    try:
        U, d, Vt = np.linalg.svd(X_eval, full_matrices=False)
        if d.size == 0 or float(d[0]) <= 0.0:
            return None
        eps_crit = float(np.finfo(np.float64).eps) ** 0.66
        if float(d[-1]) / float(d[0]) < eps_crit:
            return None
        return Vt.T @ np.diag(1.0 / d) @ U.T
    except Exception:
        return None


def _compute_np_transform_centered(X_raw, center_mat, k, knots, F, x):
    """
    Conditioning transform for centered (constraint-absorbed) marginals in ti with mc=TRUE.

    Uses PredictMat of the centered basis at k equispaced points.
    """
    x_eval = np.linspace(float(np.min(x)), float(np.max(x)), k).reshape(-1, 1)
    X_eval_raw = cr_spl_predict(x_eval, knots=knots, F=F)
    X_eval_c = X_eval_raw @ center_mat
    try:
        U, d, Vt = np.linalg.svd(X_eval_c, full_matrices=False)
        if d.size == 0 or float(d[0]) <= 0.0:
            return None
        eps_crit = float(np.finfo(np.float64).eps) ** 0.66
        if float(d[-1]) / float(d[0]) < eps_crit:
            return None
        return Vt.T @ np.diag(1.0 / d) @ U.T
    except Exception:
        return None


class CubicSplines:
    """
    Cubic regression spline basis with both raw and constrained representations.
    """

    def __init__(self, x, k, knots=None):
        X_raw, S_raw_unscaled, knots, F = cr_spl(x, n_knots=k, knots=knots)

        # Store the penalty before scale_penalty — needed by te/ti for eigenvalue
        # normalization of marginal penalties before building the tensor product.
        self.raw_penalty_unscaled = np.asarray(S_raw_unscaled, dtype=np.float64)

        S_raw = scale_penalty(X_raw, S_raw_unscaled)
        X_centered, S_centered, center_mat = identconst(X_raw, S_raw)

        self.raw_basis = X_raw
        self.raw_penalty = S_raw

        self.basis = X_centered
        self.penalty = S_centered
        self.center_mat = center_mat

        self.knots = knots
        self.gammas = None
        self.deltas = None
        self.x_plot = np.linspace(np.min(x), np.max(x), 1000).reshape(1000, 1)
        self.dim_basis = X_centered.shape[1]
        self.F = F

        # cr smooths set noterp=TRUE in mgcv, meaning the np=TRUE reparameterization
        # is skipped for cr marginals in te/ti tensor products.  The XP transforms
        # are therefore never applied to cr spline marginals.
        self._np_transform = None
        self._np_transform_centered = None

    def uncenter(self):
        self.uncentered_gammas = self.center_mat @ self.gammas

    def transform_new_raw(self, x_new):
        return cr_spl_predict(x_new, knots=self.knots, F=self.F)

    def transform_new_centered(self, x_new):
        return self.transform_new_raw(x_new) @ self.center_mat

    def transform_new(self, x_new):
        return self.transform_new_raw(x_new)

    def plot(
        self,
        ax=None,
        intercept=0,
        plot_analytical=False,
        col="b",
        alpha=1,
        col_analytical="r",
    ):
        import matplotlib.pyplot as plt

        basis = self.transform_new_raw(self.x_plot)
        y_fitted = intercept + basis @ self.uncentered_gammas

        if ax is None:
            if plot_analytical:
                y_plot = intercept + basis @ self.center_mat @ self.analytical_gammas
                plt.plot(self.x_plot, y_plot, col_analytical)
            plt.plot(self.x_plot, y_fitted, alpha=alpha)
        else:
            if plot_analytical:
                y_plot = intercept + basis @ self.center_mat @ self.analytical_gammas
                ax.plot(self.x_plot, y_plot, col_analytical)
            ax.plot(self.x_plot, y_fitted, col, alpha=alpha)
