from ..splines.cubic import CubicSplines
import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import minimize
from scipy.stats import chi2
from scipy.stats import norm


class GAM:
    """
    Generalized Additive Model (GAM) using cubic splines.

    This class fits a GAM model with cubic splines as basis functions, with the option to apply
    regularization using smoothing parameters. The smoothing parameters can be optimized using
    the Generalized Cross-Validation (GCV) criterion.

    Parameters
    ----------
    X : np.ndarray
        The input matrix of shape (n_samples, n_features), where each column corresponds to a feature.
    k : int, optional
        The number of basis functions for each cubic spline, by default 10.
    s : np.ndarray, optional
        Initial smoothing parameters for each feature, by default None. If None, all smoothing parameters are initialized to 1.
    include_intercept : bool, optional
        Whether to include an unpenalized intercept term, by default True.

    Attributes
    ----------
    X : np.ndarray
        The input matrix of shape (n_samples, n_features).
    splines : list
        List of CubicSplines objects for each feature.
    Z : np.ndarray
        The matrix of basis functions (n_samples, total_basis_functions).
    penalties : list
        List of penalty matrices for each feature's spline.
    smoothing_params : np.ndarray
        Smoothing parameters for each feature. These control the trade-off between goodness-of-fit and smoothness.
    beta : list
        List of coefficient vectors for each spline after fitting.
    cov_matrix : np.ndarray
        Covariance matrix of the fitted coefficients.

    Methods
    -------
    fit_without_optimization(y)
        Fit the model using current smoothing parameters without optimizing them.
    gcv_score(y, log_smoothing_params)
        Compute the Generalized Cross-Validation (GCV) score for a given set of smoothing parameters.
    confidence_intervals(alpha=0.05)
        Compute confidence intervals for the coefficients of the model.
    wald_test_spline(spline_index)
        Perform a Wald test to assess the significance of a particular spline.
    summary(y)
        Print a summary of the fitted model, including significance tests for each spline.
    optimize_smoothing_params(y, initial_smoothing_params=None)
        Optimize the smoothing parameters by minimizing the GCV score.
    fit(y)
        Fit the GAM model and optimize the smoothing parameters.
    """

    def __init__(self, X, k=10, s=None, include_intercept=True):
        """
        Initialize the GAM model.

        Parameters
        ----------
        X : np.ndarray
            The input matrix of shape (n_samples, n_features).
        k : int, optional
            The number of basis functions for each cubic spline, by default 10.
        s : np.ndarray, optional
            Initial smoothing parameters for each feature, by default None.
        include_intercept : bool, optional
            Whether to include an unpenalized intercept term, by default True.
        """
        self.X = X
        self.splines = [CubicSplines(X[:, i], k) for i in range(X.shape[1])]
        self.include_intercept = include_intercept
        spline_bases = np.column_stack([s.basis for s in self.splines])
        if self.include_intercept:
            self.Z = np.column_stack([np.ones(X.shape[0]), spline_bases])
        else:
            self.Z = spline_bases
        assert self.Z.shape[0] == X.shape[0]

        # Initialize penalty matrices and smoothing parameters
        self.penalties = [s.penalty for s in self.splines]
        self._basis_sizes = [s.basis.shape[1] for s in self.splines]
        if s is not None:
            self.smoothing_params = s
        else:
            self.smoothing_params = np.ones(X.shape[1])  # smoothing parameters

    def fit_without_optimization(self, y):
        """
        Fit the model using the current smoothing parameters without optimization.

        Parameters
        ----------
        y : np.ndarray
            The target vector of shape (n_samples,).
        """
        penalties = [
            self.smoothing_params[i] * self.penalties[i] for i in range(self.X.shape[1])
        ]
        if self.include_intercept:
            penalties = [np.zeros((1, 1))] + penalties
        penalties_block = block_diag(*penalties)

        # Compute beta using penalized least squares
        ZTZ_plus_penalties = self.Z.T @ self.Z + penalties_block
        full_beta = np.linalg.solve(ZTZ_plus_penalties, self.Z.T @ y)

        # Split beta into the coefficients for each spline
        start = 1 if self.include_intercept else 0
        self.beta = []
        if self.include_intercept:
            self.intercept_ = full_beta[0]
        else:
            self.intercept_ = None
        for num_basis in self._basis_sizes:
            self.beta.append(full_beta[start : start + num_basis])
            start += num_basis

        # Save the covariance matrix for confidence intervals
        self.cov_matrix = np.linalg.inv(ZTZ_plus_penalties)

    def gcv_score(self, y, log_smoothing_params):
        """
        Compute the Generalized Cross-Validation (GCV) score.

        Parameters
        ----------
        y : np.ndarray
            The target vector of shape (n_samples,).
        log_smoothing_params : np.ndarray
            Logarithm of the smoothing parameters to ensure positivity during optimization.

        Returns
        -------
        float
            The GCV score.
        """
        # Transform smoothing parameters back to the original space (positive)
        smoothing_params = np.exp(log_smoothing_params)
        self.smoothing_params = smoothing_params
        self.fit_without_optimization(y)  # Fit the model with new smoothing parameters

        # Calculate fitted values using the hat matrix
        penalties = [
            self.smoothing_params[i] * self.penalties[i] for i in range(self.X.shape[1])
        ]
        if self.include_intercept:
            penalties = [np.zeros((1, 1))] + penalties
        penalties_block = block_diag(*penalties)

        # Hat matrix and fitted values
        ZTZ_plus_penalties = self.Z.T @ self.Z + penalties_block
        hat_matrix = self.Z @ np.linalg.solve(ZTZ_plus_penalties, self.Z.T)
        y_hat = hat_matrix @ y

        # Compute RSS (Residual Sum of Squares)
        residuals = y - y_hat
        rss = np.sum(residuals**2)

        # Effective degrees of freedom (trace of the hat matrix)
        trace_S = np.trace(hat_matrix)

        # Compute GCV score
        n = len(y)
        gcv = (rss / n) / (1 - trace_S / n) ** 2

        return gcv

    def confidence_intervals(self, alpha=0.05):
        """
        Compute confidence intervals for each coefficient in the model.

        Parameters
        ----------
        alpha : float, optional
            Significance level for the confidence intervals, by default 0.05.

        Returns
        -------
        list of tuple
            List of confidence intervals for each coefficient, represented as (lower_bound, upper_bound).
        """
        critical_value = norm.ppf(
            1 - alpha / 2
        )  # Z value for two-sided interval (e.g., 1.96 for 95% CI)
        std_errors = np.sqrt(
            np.diag(self.cov_matrix)
        )  # Standard errors from the diagonal of the covariance matrix
        confidence_intervals = []

        # Ensure we create confidence intervals for each coefficient
        if self.include_intercept:
            all_beta = np.concatenate(([self.intercept_], *self.beta))
        else:
            all_beta = np.concatenate(self.beta)
        for beta, se in zip(all_beta, std_errors[: len(all_beta)]):
            lower = beta - critical_value * se
            upper = beta + critical_value * se
            confidence_intervals.append(
                (lower, upper)
            )  # Return as tuple (lower, upper)

        return confidence_intervals

    # TODO: Double check implementation to align with mgcv
    def wald_test_spline(self, spline_index):
        """
        Perform the Wald test for a specific spline to assess its significance.

        Parameters
        ----------
        spline_index : int
            The index of the spline for which to perform the Wald test.

        Returns
        -------
        float
            The Wald statistic.
        float
            The p-value of the Wald test.
        """
        # Extract coefficients and covariance matrix for the given spline
        num_basis = self.splines[spline_index].basis.shape[1]
        start_idx = sum(self._basis_sizes[:spline_index])
        if self.include_intercept:
            start_idx += 1
            full_beta = np.concatenate(([self.intercept_], *self.beta))
        else:
            full_beta = np.concatenate(self.beta)
        spline_coefficients = full_beta[start_idx : start_idx + num_basis]
        spline_cov_matrix = self.cov_matrix[
            start_idx : start_idx + num_basis, start_idx : start_idx + num_basis
        ]

        # Compute the Wald statistic
        wald_stat = (
            spline_coefficients.T
            @ np.linalg.inv(spline_cov_matrix)
            @ spline_coefficients
        )

        # Compute the p-value using the chi-squared distribution
        p_value = 1 - chi2.cdf(wald_stat, num_basis)

        return wald_stat, p_value

    def summary(self, y):
        """
        Generate a summary of the model, including the significance of each spline.

        Parameters
        ----------
        y : np.ndarray
            The target vector of shape (n_samples,).
        """
        print("GAM Model Summary")
        print("=" * 40)

        # Residuals and GCV score
        if self.include_intercept:
            full_beta = np.concatenate(([self.intercept_], *self.beta))
        else:
            full_beta = np.concatenate(self.beta)
        residuals = y - self.Z @ full_beta
        rss = np.sum(residuals**2)
        gcv_score = self.gcv_score(y, np.log(self.smoothing_params))

        # Effective degrees of freedom and other global stats
        n = len(y)
        tss = np.sum((y - np.mean(y)) ** 2)

        # Compute edf and chi-square for each smooth term
        print("Approximate significance of smooth terms:")
        print("                 edf Ref.df   ChiSq p-value    ")

        penalties = [
            self.smoothing_params[i] * self.penalties[i]
            for i in range(self.X.shape[1])
        ]
        if self.include_intercept:
            penalties = [np.zeros((1, 1))] + penalties
        penalties_block = block_diag(*penalties)
        ZTZ_plus_penalties = self.Z.T @ self.Z + penalties_block
        hat_matrix = self.Z @ np.linalg.solve(ZTZ_plus_penalties, self.Z.T)
        edf_total = np.trace(hat_matrix)
        r_squared_adj = 1 - (rss / (n - edf_total)) / (tss / (n - 1))
        deviance_explained = 1 - rss / tss
        Q = np.linalg.solve(ZTZ_plus_penalties, self.Z.T)

        start_idx = 1 if self.include_intercept else 0
        for i, num_basis in enumerate(self._basis_sizes):
            end_idx = start_idx + num_basis

            # Calculate edf (effective degrees of freedom) for this spline
            Z_i = self.Z[:, start_idx:end_idx]
            Q_i = Q[start_idx:end_idx, :]
            edf = np.trace(Z_i @ Q_i)
            ref_df = num_basis  # Ref.df (reference degrees of freedom)

            wald_stat, p_value = self.wald_test_spline(i)

            # Format the p-value and significance
            p_value_str = f"{p_value:.4e}"
            if p_value < 0.001:
                sig = "***"
            elif p_value < 0.01:
                sig = "**"
            elif p_value < 0.05:
                sig = "*"
            elif p_value < 0.1:
                sig = "."
            else:
                sig = " "

            # Print row for each smooth term
            if hasattr(self.X, "columns"):
                feature_name = self.X.columns[i]
            else:
                feature_name = f"x{i}"
            print(
                f"s({feature_name})   {edf:.3f} {ref_df:.3f} {wald_stat:.3f} {p_value_str} {sig}"
            )
            start_idx = end_idx

        # Print the final global model statistics
        print("---")
        print(
            f"R-sq.(adj) = {r_squared_adj}   Deviance explained = {deviance_explained:.1%}"
        )
        print(f"GCV = {gcv_score}  Scale est. = {rss / n}   n = {n}")

    def optimize_smoothing_params(self, y, initial_smoothing_params=None):
        """Optimize smoothing parameters by minimizing the GCV score."""
        if initial_smoothing_params is None:
            initial_smoothing_params = np.log(
                self.smoothing_params
            )  # Start with log smoothing parameters

        # Minimize GCV score using L-BFGS-B in the log space
        result = minimize(
            lambda log_s: self.gcv_score(y, log_s),
            initial_smoothing_params,
            method="L-BFGS-B",
        )

        # Update smoothing parameters with the optimized values (in log space, so apply exp)
        self.smoothing_params = np.exp(result.x)

    def fit(self, y):
        """Fit the GAM model and optimize smoothing parameters."""
        # Optimize the smoothing parameters using GCV
        self.optimize_smoothing_params(y)

        # Fit the model with the optimized smoothing parameters
        self.fit_without_optimization(y)

        return self

    def predict(self, X=None):
        """
        Predict target values for the given input matrix.

        Parameters
        ----------
        X : np.ndarray, optional
            Input matrix of shape (n_samples, n_features). If None, uses the
            training data.

        Returns
        -------
        np.ndarray
            Predicted values of shape (n_samples,).
        """
        if X is None:
            Z = self.Z
        else:
            if hasattr(X, "to_numpy"):
                X = X.to_numpy()
            X = np.asarray(X)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            if X.shape[1] != len(self.splines):
                raise ValueError(
                    f"Expected X with {len(self.splines)} feature(s), got {X.shape[1]}"
                )
            # Build design matrix for new data
            # transform_new returns raw basis, apply center_mat to match training
            centered_bases = []
            for i in range(X.shape[1]):
                raw_basis = self.splines[i].transform_new(X[:, i].reshape(-1, 1))
                centered_basis = raw_basis @ self.splines[i].center_mat
                centered_bases.append(centered_basis)
            spline_bases = np.column_stack(centered_bases)
            if self.include_intercept:
                Z = np.column_stack([np.ones(X.shape[0]), spline_bases])
            else:
                Z = spline_bases

        # Reconstruct full beta including intercept
        if self.include_intercept:
            full_beta = np.concatenate(([self.intercept_], *self.beta))
        else:
            full_beta = np.concatenate(self.beta)

        return Z @ full_beta
