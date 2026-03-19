#splines/cubic.py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from .spline_utils import (
    color_bounds,
    color_fader,
    cr_spl,
    cr_spl_predict,
    identconst,
    mrf_design,
    pol2nb,
    scale_penalty,
)


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

            full_gammas = self.uncentered_gammas.numpy()
            full_gammas = (full_gammas - min(full_gammas)) / (
                max(full_gammas) - min(full_gammas)
            )
            mix_dict = dict(zip(pols, full_gammas))

            mix = np.linspace(0, 1, 100)
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