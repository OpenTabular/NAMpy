import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from .spline_utils import (
    scale_penalty,
    identconst,
    cr_spl,
    cr_spl_predict,
    mrf_design,
    pol2nb,
    color_fader,
    color_bounds,
)


class CubicSplines:
    def __init__(self, x, k):

        X, S, knots, F = cr_spl(x, n_knots=k)

        # rescale penalty. Contributes to getting pretty much identical penalty matrix as in mgcv
        S = scale_penalty(X, S)
        # center
        X_centered, S_centered, center_mat = identconst(X, S)

        self.basis = X_centered
        self.penalty = S_centered
        self.knots = knots
        self.center_mat = center_mat
        self.gammas = None
        self.deltas = None
        self.x_plot = np.linspace(x.min(), x.max(), 1000).reshape(1000, 1)
        self.dim_basis = X_centered.shape[1]
        self.F = F

    def uncenter(self):
        self.uncentered_gammas = self.center_mat @ self.gammas

    def transform_new(self, x_new):
        # given the knots and F, evaluate new data
        basis = cr_spl_predict(x_new, knots=self.knots, F=self.F)
        return basis

    def plot(
        self,
        ax=None,
        intercept=0,
        plot_analytical=False,
        col="b",
        alpha=1,
        col_analytical="r",
    ):

        # evaluate x_plot
        basis = cr_spl_predict(self.x_plot, knots=self.knots, F=self.F)
        # create fitted_values
        y_fitted = intercept + basis @ self.uncentered_gammas

        # plot into given subplot or simply new plot
        if ax is None:
            # if least squares solution should be plotted ...
            if plot_analytical:
                y_plot = intercept + basis @ self.center_mat @ self.analytical_gammas
                plt.plot(self.x_plot, y_plot, col_analytical)
            plt.plot(self.x_plot, y_fitted, alpha=alpha)

        else:
            if plot_analytical:
                y_plot = intercept + basis @ self.center_mat @ self.analytical_gammas
                ax.plot(self.x_plot, y_plot, col_analytical)
            ax.plot(self.x_plot, y_fitted, col, alpha=alpha)


class MRFSmooth:
    def __init__(self, x, polygons=None, penalty=None):
        self.polygons = polygons
        basis = mrf_design(regions=x, pc=polygons)
        penalty = pol2nb(pc=polygons.copy())
        penalty = scale_penalty(basis, penalty)
        basis, penalty, center_mat = identconst(basis, penalty)
        self.basis = basis
        self.penalty = penalty
        self.dim_basis = basis.shape[1]
        self.center_mat = center_mat

    def uncenter(self):
        self.uncentered_gammas = self.center_mat @ self.gammas

    def plot(
        self, col1="blue", col2="red", intercept=None, plot_analytical=None, ax=None
    ):
        pols = self.polygons
        if self.polygons is None:
            print("Need map")
        else:
            if self.uncentered_gammas is None:
                self.uncenter()

            # weird behaviour of tensorflow objects requires me to do this (or I am missing something)
            full_gammas = self.uncentered_gammas.numpy()

            full_gammas = (full_gammas - min(full_gammas)) / (
                max(full_gammas) - min(full_gammas)
            )
            mix_dict = {k: v for k, v in zip(pols, full_gammas)}

            # for colorbar
            # values from 0 to 1
            mix = np.linspace(0, 1, 100)
            # retrieve color codes from col1 to col2
            col_list = color_fader(col1, col2, mix)
            cmap = mpl.colors.ListedColormap(col_list)
            mapped_colors = color_bounds(self.uncentered_gammas.numpy())
            norm = mpl.colors.BoundaryNorm(mapped_colors, cmap.N)

            if ax is None:
                for i in pols.keys():
                    plt.fill(
                        pols[i][:, 0],
                        pols[i][:, 1],
                        color=color_fader(col1, col2, mix=mix_dict[i][0] / 1),
                    )
                plt.axis("off")

            else:
                for i in pols.keys():
                    ax.fill(
                        pols[i][:, 0],
                        pols[i][:, 1],
                        color=color_fader(col1, col2, mix=mix_dict[i][0] / 1),
                    )
                plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
                ax.axis("off")

    def transform_new(self, x_new):
        return mrf_design(regions=x_new, pc=self.polygons)
