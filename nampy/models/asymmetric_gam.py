"""Formula-based GAMs for conditional expectiles and quantiles."""

import numpy as np

from nampy.gam.extensions.objectives import fit_expectile, fit_quantile

from ._loss_gam import _LossGAMBase


class ExpectileGAM(_LossGAMBase):
    """Fixed-smoothing additive expectile regression using LAWS.

    Parameters
    ----------
    formula : str or None, default None
        NAMpy formula, e.g. ``'y ~ s(x, k=8) + z'``. With no formula,
        numeric input columns receive ordinary smooths.
    level : float, default 0.5
        Expectile level in (0,1); 0.5 targets the conditional mean.
    smoothing_params : sequence of float or None, default None
        Nonnegative fixed penalties, one per compiled smoothing parameter.
        None uses ones. Selection by Gaussian REML is deliberately unavailable.
    k : int, default 10
        Default basis dimension for implicit smooths.
    basis : str, default 'tp'
        Default ordinary smooth basis. Shape constraints are unsupported.
    fit_intercept : bool, default True
        Include a separate unpenalized intercept.
    max_iter : int, default 200
        Maximum LAWS iterations.
    tol : float, default 1e-9
        Objective-gradient stopping tolerance.

    Attributes
    ----------
    coef_ : ndarray of float, shape (p,)
        All coefficients, including intercept in NAMpy's full basis coordinates.
    smoothing_params_ : ndarray of float, shape (s,)
        Resolved fixed smoothing parameters.
    penalty_ : ndarray of float, shape (p,p)
        Total penalty S in ``sum(loss) + beta.T @ S @ beta / 2``.
    fit_result_ : LossFit
        Objective, convergence, iterations, optimality and fitted values.
        Its fitted values are basis contributions B@coef_ excluding known
        offsets; predict(X) includes offsets.
        This record does not contain Gaussian covariance or likelihood EDF.
    """

    def _solve(self, design, y, penalty, weights):
        return fit_expectile(
            design,
            y,
            level=self.level,
            penalty=penalty,
            weights=weights,
            max_iter=self.max_iter,
            tol=self.tol,
        )

    def _loss(self, residual):
        return np.where(residual >= 0, self.level, 1 - self.level) * residual**2


class QuantileGAM(_LossGAMBase):
    """Fixed-smoothing spline quantile regression, exact pinball or smooth ELF.

    Parameters and attributes follow ExpectileGAM. ``level=0.5`` targets the
    conditional median. ``bandwidth`` is a float >=0 in response units,
    default 0 (exact convex pinball QP). A positive bandwidth uses smooth ELF
    loss with fixed learning rate 1. This is not calibrated qgam inference.
    ``max_iter`` defaults to 10000; other defaults are identical to ExpectileGAM.
    Fits at separate levels can cross. No inferential intervals are provided.
    """

    def __init__(
        self,
        formula=None,
        *,
        level=0.5,
        smoothing_params=None,
        k=10,
        basis="tp",
        fit_intercept=True,
        bandwidth=0.0,
        max_iter=10000,
        tol=1e-9,
    ):
        super().__init__(
            formula,
            level=level,
            smoothing_params=smoothing_params,
            k=k,
            basis=basis,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
        )
        self.bandwidth = bandwidth

    def _solve(self, design, y, penalty, weights):
        return fit_quantile(
            design,
            y,
            level=self.level,
            penalty=penalty,
            weights=weights,
            bandwidth=self.bandwidth,
            max_iter=self.max_iter,
            tol=self.tol,
        )

    def _loss(self, residual):
        return np.maximum(self.level * residual, (self.level - 1) * residual)
