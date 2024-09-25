import numpy as np
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from .spline_utils import (
    scale_penalty,
    identconst,
)


class TPSplines:
    """ """

    def __init__(self, x, k, pen_order):
        """
        Create the model matrices for Thin-Plate Splines based on the data, rank and penalty order
        :param x:
        :param k:
        :param pen_order:
        :param y:
        """
        n = x.shape[0]
        if len(x.shape) == 1:
            x = x.values.reshape(n, 1)
        d = x.shape[1]
        M = np.math.comb(pen_order + d - 1, d)

        # model matrices
        X, S, UZ, map_idx = tp_spline(x, k, pen_order, n, d, M)

        # rescale penalty, this is done because it is done in mgcv. Trying to create smoothers that perform
        # comparably and that are affected by smoothing parameters comparably.
        S = scale_penalty(X, S)
        # center model matrices
        X_centered, S_centered, center_mat = identconst(X, S)

        # following the mgcv-implemetation I subtract the mean from the original data (also during creation of the model
        # matrices) for d = 2 this created some weird issues (probably just my own mistakes) so I don't do it there.
        # Doesn't change the model so can be ignored
        if d == 1:
            self.x_mean = x.mean(0)
            self.x = x - self.x_mean
        if d >= 2:
            self.x = x

        # save all attributes that will be useful later
        self.UZ = UZ
        self.matrices = center_mat
        self.basis = X_centered
        self.penalty = S_centered
        self.k = k
        self.M = M
        self.pen_order = pen_order
        self.dim_basis = X_centered.shape[1]
        self.center_mat = center_mat
        self.gammas = None
        self.uncentered_gammas = None
        self.map_idx = map_idx

        # self.x_plot: for plot-method
        if d == 1:
            self.x_plot = np.linspace(self.x.min(), self.x.max(), 1000).reshape(1000, 1)

        # if two-dimensional, prepare for contour-plot
        if d == 2:
            x_plot = np.linspace(x.iloc[:, 0].min(), x.iloc[:, 0].max())
            z_plot = np.linspace(x.iloc[:, 1].min(), x.iloc[:, 1].max())
            mesh = np.meshgrid(x_plot, z_plot)
            self.x_plot = np.column_stack(
                [mesh[0].reshape(2500, 1), mesh[1].reshape(2500, 1)]
            )

    def transform_new(self, x_new):
        """
        evaluate spline for new points:
        Simply create the full model matrix based on the new points
        """
        m = self.pen_order
        M = self.M
        x_orig = np.unique(self.x, axis=0)
        x_new = x_new - self.x_mean[0]
        n = x_new.shape[0]
        if len(x_new.shape) == 1:
            x_new = x_new.values.reshape(n, 1)
        d = x_orig.shape[1]
        # matrix of euclidean distances needed for eta
        E = distance_matrix(x_new, x_orig)
        E = eta(E, m, d)
        T = tp_T(x_new, M, m, d)
        ET = np.column_stack([E, T])
        return ET

    def plot(
        self,
        ax=None,
        intercept=0,
        plot_analytical=False,
        col="b",
        col_analytical="r",
        alpha=1,
    ):
        """
        plot-method with some parameters that I mainly created for the plots in my thesis

        """

        m = self.pen_order
        M = self.M
        x_plot = self.x_plot
        x_un = np.unique(self.x, axis=0)
        n = x_plot.shape[0]
        d = x_un.shape[1]
        if self.uncentered_gammas is None:
            self.uncenter()
        E = distance_matrix(x_plot, x_un)
        E = eta(E, m, d)
        T = tp_T(x_plot, M, m, d)
        ET = np.column_stack([E, T])
        y_fitted = intercept + ET @ self.uncentered_gammas
        if d == 1:
            # if statement to see whether the plot method is called from GAM or the smoother:
            # if it is from GAM, there is subplot in which to plot, if there is no subplot,
            # just do plt.plot
            # same for every smoother
            if ax is None:
                if plot_analytical:
                    y_plot = (
                        intercept
                        + ET @ self.UZ @ self.center_mat @ self.analytical_gammas
                    )
                    plt.plot(x_plot + self.x_mean[0], y_plot, col_analytical)
                plt.plot(x_plot, y_fitted, col, alpha=alpha)

            else:
                if plot_analytical:
                    y_plot = (
                        intercept
                        + ET @ self.UZ @ self.center_mat @ self.analytical_gammas
                    )
                    ax.plot(x_plot + self.x_mean[0], y_plot, col_analytical)
                ax.plot(x_plot + self.x_mean[0], y_fitted, col, alpha=alpha)

        if d == 2:
            x_mesh = x_plot[:, 0].reshape(50, 50)
            z_mesh = x_plot[:, 1].reshape(50, 50)
            y_fitted = np.array(y_fitted).reshape(50, 50)

            if ax is None:
                plt.contour(x_mesh, z_mesh, y_fitted)
            else:
                cs = ax.contour(x_mesh, z_mesh, y_fitted)
                ax.clabel(cs, inline=True, fontsize=10)

        if d > 2:
            print("No plot :(")

    def uncenter(self):

        # multiply gammas with center mat to "uncenter"
        gammas = self.center_mat @ self.gammas
        # this part is TP-spline specific, additional to "uncentering", I also evaluate the full delta-coefficients
        self.uncentered_gammas = self.UZ @ gammas
