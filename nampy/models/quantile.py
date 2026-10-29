"""Public additive quantile estimator."""

from nampy.gam.losses.objectives import _pinball_loss, fit_quantile

from ._loss_gam import _LossGAMBase


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
        return _pinball_loss(residual, self.level)


__all__ = ["QuantileGAM"]
