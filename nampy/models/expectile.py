"""Public additive expectile estimator."""

from nampy.gam.extensions.objectives import _expectile_loss, fit_expectile

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
        return _expectile_loss(residual, self.level)


__all__ = ["ExpectileGAM"]
