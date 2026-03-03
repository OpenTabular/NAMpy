"""Classical GAM (Generalized Additive Model) with an sklearn-compatible API.

This module wraps the low-level cubic-spline GAM engine in
``nampy.basemodels.gam`` behind a familiar ``fit`` / ``predict`` / ``score``
interface.  No PyTorch, no Lightning — just penalised least-squares with
GCV / ML / REML smoothing, on top of the existing spline utilities.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from ..basemodels.gam import GAM


class GAMRegressor(BaseEstimator, RegressorMixin):
    """Scikit-learn compatible Generalized Additive Model.

    Uses cubic regression splines with penalised least-squares estimation
    and automatic smoothing-parameter selection (GCV, ML, REML, or LAML).
    Numerical features only.

    Parameters
    ----------
    n_splines : int, default=10
        Number of basis functions (knots) per feature.  Must be >= 3.
    smoothing_params : float, array-like, or None, default=None
        Initial smoothing parameters.  ``None`` → 1.0 per feature.
        A scalar is broadcast to all features.
    method : {'GCV', 'ML', 'REML', 'LAML'}, default='GCV'
        Smoothing-parameter selection criterion.  ``'LAML'`` is a
        Laplace-approximate marginal likelihood; for Gaussian models it
        coincides with ``'REML'`` up to constants.
    optimizer : {'lbfgsb', 'outer_newton'}, default='lbfgsb'
        ``'lbfgsb'`` — L-BFGS-B on the criterion (general purpose).
        ``'outer_newton'`` — Wood-style outer Newton with analytic
        gradient / Hessian (requires ``method`` in ``{'REML', 'LAML'}``).

    Attributes
    ----------
    intercept_ : float
        Global intercept (from the core GAM, i.e. mean(y)).
    feature_names_ : list of str
        Feature names seen during ``fit``.
    n_features_in_ : int
        Number of features seen during ``fit``.

    Examples
    --------
    >>> import numpy as np
    >>> from nampy.models.gam import GAMRegressor
    >>> X = np.column_stack([np.linspace(-3, 3, 200)] * 2)
    >>> y = np.sin(X[:, 0]) + X[:, 1] ** 2
    >>> model = GAMRegressor(n_splines=12).fit(X, y)
    >>> model.score(X, y) > 0.95
    True
    """

    def __init__(
        self, n_splines=10, smoothing_params=None, method="GCV", optimizer="lbfgsb"
    ):
        self.n_splines = n_splines
        self.smoothing_params = smoothing_params
        self.method = method
        self.optimizer = optimizer

    # ------------------------------------------------------------------
    # fit / predict / score
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """Fit the GAM via penalised least-squares + automatic smoothing.

        Parameters
        ----------
        X : array-like or DataFrame, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)

        Returns
        -------
        self
        """
        X_array, feature_names = self._validate_X(X, fitting=True)
        y_array = np.asarray(y, dtype=np.float64).ravel()

        if X_array.shape[0] != y_array.shape[0]:
            raise ValueError(
                f"X has {X_array.shape[0]} samples but y has {y_array.shape[0]}"
            )

        self.feature_names_ = feature_names
        self.n_features_in_ = X_array.shape[1]

        s = self._resolve_smoothing_params(self.n_features_in_)

        self._gam = GAM(
            X_array,
            k=self.n_splines,
            s=s,
            feature_names=self.feature_names_,
        )
        self._gam.fit(y_array, optimize=True, method=self.method, optimizer=self.optimizer)

        self.intercept_ = self._gam.intercept_
        self._y_train = y_array.copy()
        return self

    def predict(self, X):
        """Predict target values for new data.

        Parameters
        ----------
        X : array-like or DataFrame, shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray, shape (n_samples,)
        """
        check_is_fitted(self, ["_gam"])
        X_array, _ = self._validate_X(X, fitting=False)
        return self._gam.predict(X_array)

    def predict_se(self, X, cov="bayes"):
        """Predict with standard errors.

        Parameters
        ----------
        X : array-like or DataFrame
        cov : {'bayes', 'freq', 'kass_steffey', 'wood'}

        Returns
        -------
        mu : ndarray, shape (n_samples,)
        se : ndarray, shape (n_samples,)
        """
        check_is_fitted(self, ["_gam"])
        X_array, _ = self._validate_X(X, fitting=False)
        return self._gam.predict(X_array, return_se=True, cov=cov)

    # ------------------------------------------------------------------
    # Summary / diagnostics
    # ------------------------------------------------------------------

    def summary(self):
        """Print a model summary (EDF per term, R-sq, GCV)."""
        check_is_fitted(self, ["_gam"])
        self._gam.summary()

    def aic_conditional(self, scale="ml", cov="bayes"):
        """Conventional conditional AIC.

        Parameters
        ----------
        scale : {'ml', 'working'}
            Scale estimate for the Gaussian log-likelihood.
        cov : {'bayes', 'freq'}
            Which conditional covariance to use.

        Returns
        -------
        dict with keys ``aic``, ``loglik``, ``edf_aic``, ``scale``.
        """
        check_is_fitted(self, ["_gam"])
        return self._gam.aic_conditional(scale=scale, cov=cov)

    def aic_corrected(
        self,
        scale="ml",
        covariance_kind="wood_full",
        sp_uncertainty_regularization="pinv",
        sp_uncertainty_ridge=1e-6,
    ):
        """Wood-style corrected conditional AIC.

        Parameters
        ----------
        scale : {'ml', 'working'}
            Scale estimate for the Gaussian log-likelihood.
        covariance_kind : {'kass_steffey', 'wood_full'}
            Which unconditional covariance approximation to use.
        sp_uncertainty_regularization : {'pinv', 'ridge'}
            How to invert the criterion Hessian.
        sp_uncertainty_ridge : float, default=1e-6
            Ridge constant (when ``sp_uncertainty_regularization='ridge'``).

        Returns
        -------
        dict with keys ``aic``, ``loglik``, ``edf_aic``, ``scale``,
        ``covariance_kind``.
        """
        check_is_fitted(self, ["_gam"])
        return self._gam.aic_corrected(
            scale=scale,
            covariance_kind=covariance_kind,
            sp_uncertainty_regularization=sp_uncertainty_regularization,
            sp_uncertainty_ridge=sp_uncertainty_ridge,
        )

    def confidence_intervals(self, alpha=0.05, cov="bayes", include_intercept=False):
        """Wald-type CIs for spline coefficients.

        Parameters
        ----------
        alpha : float, default=0.05
        cov : {'bayes', 'freq', 'kass_steffey', 'wood'}
            Covariance matrix to use for the standard errors.
            ``'kass_steffey'`` and ``'wood'`` require a prior call to
            :meth:`compute_unconditional_covariance`.
        include_intercept : bool

        Returns
        -------
        list of (float, float)
        """
        check_is_fitted(self, ["_gam"])
        return self._gam.confidence_intervals(
            alpha=alpha, cov=cov, include_intercept=include_intercept,
        )

    def term_drop_test(self, term_index=0):
        """Drop-one-term F-test (refits reduced model)."""
        check_is_fitted(self, ["_gam"])
        if not (0 <= term_index < self.n_features_in_):
            raise IndexError(
                f"term_index must be in [0, {self.n_features_in_ - 1}], got {term_index}"
            )
        return self._gam.term_drop_test(term_index=term_index, method=self.method)

    def concurvity(self, full=True, include_intercept=False):
        """Wood/mgcv-style concurvity diagnostics.

        Parameters
        ----------
        full : bool, default=True
            If ``True``, measure each term against the whole rest of the model
            (mgcv ``full=TRUE``).  If ``False``, return pairwise matrices.
        include_intercept : bool, default=False
            Include the intercept column as a parametric term.

        Returns
        -------
        dict
            See :meth:`nampy.basemodels.gam.GAM.concurvity` for the full
            return-value specification.
        """
        check_is_fitted(self, ["_gam"])
        return self._gam.concurvity(full=full, include_intercept=include_intercept)

    def k_check(self, subsample=5000, n_rep=400, random_state=None):
        """Wood/mgcv-style basis-dimension check (k-index simulation test).

        Parameters
        ----------
        subsample : int, default=5000
            Maximum observations used (random subsample when ``n > subsample``).
        n_rep : int, default=400
            Number of residual reshuffles for the simulation p-value.
        random_state : int, np.random.Generator, or None
            Seed / generator for reproducibility.

        Returns
        -------
        dict
            Keys: ``labels``, ``table`` (shape (*m*, 4), columns k', edf,
            k-index, p-value), ``columns``, ``subsample_n``, ``n_rep``.
        """
        check_is_fitted(self, ["_gam"])
        return self._gam.k_check(
            subsample=subsample, n_rep=n_rep, random_state=random_state
        )

    def k_refit_check(self, factor=2):
        """Refit-based basis-dimension sensitivity check.

        Doubles (or scales by *factor*) the basis dimension, refits with
        fresh smoothing optimisation, and compares total EDF and criterion.
        Useful as a follow-up when :meth:`k_check` flags a concern.

        Parameters
        ----------
        factor : int or float, default=2
            ``k_new = max(k + 1, factor * k)``.

        Returns
        -------
        dict
            Keys: ``k_old``, ``k_new``, ``edf_old``, ``edf_new``,
            ``criterion_old``, ``criterion_new``.
        """
        check_is_fitted(self, ["_gam"])
        return self._gam.k_refit_check(factor=factor)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_X(self, X, fitting=True):
        if isinstance(X, pd.DataFrame):
            if not fitting and hasattr(self, "feature_names_"):
                missing = [c for c in self.feature_names_ if c not in X.columns]
                if missing:
                    raise ValueError(f"Missing columns for prediction: {missing}")
                X = X[self.feature_names_]
                feature_names = list(self.feature_names_)
            else:
                feature_names = list(X.columns)
            X_array = X.values.astype(np.float64)
        else:
            X_array = np.asarray(X, dtype=np.float64)
            if X_array.ndim == 1:
                X_array = X_array.reshape(-1, 1)
            if fitting:
                feature_names = [f"x{i}" for i in range(X_array.shape[1])]
            else:
                feature_names = getattr(
                    self, "feature_names_",
                    [f"x{i}" for i in range(X_array.shape[1])],
                )

        if X_array.ndim != 2:
            raise ValueError("X must be 2-D")
        if not np.isfinite(X_array).all():
            raise ValueError("X contains NaN / Inf")

        if (
            not fitting
            and hasattr(self, "n_features_in_")
            and X_array.shape[1] != self.n_features_in_
        ):
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X_array.shape[1]}"
            )
        return X_array, feature_names

    def _resolve_smoothing_params(self, n_features):
        if self.smoothing_params is None:
            return None
        s = np.asarray(self.smoothing_params, dtype=np.float64)
        if s.ndim == 0:
            s = np.full(n_features, s.item())
        if len(s) != n_features:
            raise ValueError(
                f"smoothing_params has length {len(s)}, expected {n_features}"
            )
        return s
