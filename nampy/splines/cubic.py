import matplotlib.pyplot as plt
import numpy as np

from .constraints import identconst
from .cubic_basis import cr_spl, cr_spl_predict
from .penalty_scaling import scale_penalty


class CubicSplines:
    """
    Cubic regression spline basis with both raw and constrained representations.
    """

    def __init__(self, x, k, knots=None):
        X_raw, S_raw, knots, F = cr_spl(x, n_knots=k, knots=knots)

        S_raw = scale_penalty(X_raw, S_raw)
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


