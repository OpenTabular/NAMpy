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


    def plot(self, ax=None, intercept=0, plot_analytical=False, col='b', alpha=1, col_analytical='r'):

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
            plt.plot(self.x_plot, y_fitted, col=col, alpha=alpha)

        else:
            if plot_analytical:
                y_plot = intercept + basis @ self.center_mat @ self.analytical_gammas
                ax.plot(self.x_plot, y_plot, col_analytical)
            ax.plot(self.x_plot, y_fitted, col, alpha=alpha)
