"""Calibrated Gibbs inference for additive quantile functions.

References
----------
- Fasiolo, M., Wood, S. N., Zaffran, M., Nedellec, R., and Goude, Y. (2021). Fast
  Calibrated Additive Quantile Regression.
  https://doi.org/10.1080/01621459.2020.1725521
- Fasiolo, M., Wood, S. N., Zaffran, M., Nedellec, R., and Goude, Y. (2021). qgam:
  Bayesian Nonparametric Quantile Regression Modeling in R.
  https://doi.org/10.18637/jss.v100.i09
"""

from dataclasses import replace

from ..gam.inference.quantile import quantile_function_interval
from ..gam.losses.objectives import _pinball_loss
from ..gam.losses.quantile_calibration import fit_calibrated_quantile
from ._loss_gam import _LossGAMBase


class CalibratedQuantileGAM(_LossGAMBase):
    """qgam calFast calibration with supplied or adaptive bandwidth and penalties.

    Inference targets the estimated conditional quantile, not a new response.
    Identity link, scalar constant loss scale and unweighted independent rows
    are supported. bandwidth=None estimates the qgam SHASH bandwidth;
    optimize_smoothing selects REML penalties at each calibration scale.
    fit_levels shares initial bandwidth preparation across independent fits.
    Bootstrap qgam calibration, heteroscedastic loss scales, noncrossing multiple quantiles,
    simultaneous bands and classical likelihood/AIC are unsupported.
    """

    def capabilities(self):
        """Declare calibrated quantile-function inference separately from loss fitting."""
        from ..gam.capabilities import estimator_capabilities

        return estimator_capabilities(self, task="calibrated_quantile")

    def __init__(
        self,
        formula=None,
        *,
        level=0.5,
        bandwidth=0.1,
        smoothing_params=None,
        k=10,
        basis="tp",
        fit_intercept=True,
        log_scale_bounds=(-4.0, 4.0),
        max_iter=1000,
        tol=1e-9,
        calibration_tol=1e-6,
        optimize_smoothing=False,
        smoothing_optimizer="bfgs",
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
        self.log_scale_bounds = log_scale_bounds
        self.calibration_tol = calibration_tol
        self.optimize_smoothing = optimize_smoothing
        self.smoothing_optimizer = smoothing_optimizer

    def fit(self, X=None, y=None, *, data=None, sample_weight=None, offset=None):
        staged = type(self)(**self.get_params(deep=False))
        staged._fit_calibrated(
            X, y, data=data, sample_weight=sample_weight, offset=offset
        )
        self.__dict__.clear()
        self.__dict__.update(staged.__dict__)
        return self

    def _fit_calibrated(
        self, X, y, *, data, sample_weight, offset, bandwidth_reference=None
    ):
        from .._offsets import predictor_offset
        from ..gam.design import compile_design
        from ..gam.losses.quantile_calibration import fit_calibrated_quantile_auto

        if offset is not None:
            offset = predictor_offset(offset, len(data if X is None else X), 1)
        design = compile_design(
            X,
            y,
            data=data,
            formula=self.formula,
            k=self.k,
            basis=self.basis,
            fit_intercept=self.fit_intercept,
            smoothing_params=self.smoothing_params,
            sample_weight=sample_weight,
            offset=offset,
        )
        if design.response is None:
            raise ValueError("Calibration requires a scalar response.")
        if not (design.sample_weight == 1).all():
            raise NotImplementedError(
                "Calibrated quantile inference currently requires unweighted observations."
            )
        frame = data if X is None else X
        self.predictor_names_ = ("predictor",)
        self.bandwidth_ = self.bandwidth
        if self.bandwidth is None:
            from ..gam.losses.quantile_bandwidth import estimate_quantile_bandwidth

            if bandwidth_reference is None:
                bandwidth_reference = self._prepare_bandwidth(
                    frame, design.response, offset
                )
            self.bandwidth_, self.bandwidth_error_, self.bandwidth_shash_parameters_ = (
                estimate_quantile_bandwidth(level=self.level, **bandwidth_reference)
            )
        if self.optimize_smoothing:
            result, design, gam = fit_calibrated_quantile_auto(
                frame,
                design.response,
                formula=self.formula,
                level=self.level,
                bandwidth=self.bandwidth_,
                log_scale_bounds=self.log_scale_bounds,
                k=self.k,
                basis=self.basis,
                fit_intercept=self.fit_intercept,
                smoothing_params=self.smoothing_params,
                smoothing_optimizer=self.smoothing_optimizer,
                max_iter=self.max_iter,
                tol=self.tol,
                calibration_tol=self.calibration_tol,
                offset=offset,
            )
            self._publish_calibration(result)
            self.design_, self.coef_, self.penalty_ = (
                design,
                result.solution.coefficients.copy(),
                design.penalty,
            )
            self.smoothing_params_, self.fit_result_, self.gam_ = (
                design.smoothing_params.copy(),
                result.solution,
                gam,
            )
            self.n_features_in_ = len(design.feature_names)
        else:
            _LossGAMBase.fit(
                self, X, y, data=data, sample_weight=sample_weight, offset=offset
            )

    def _prepare_bandwidth(self, frame, response, offset):
        from ..gam import GAM
        from ..gam.losses.quantile_bandwidth import (
            fit_residual_shash,
            initial_quantile_dimension,
        )

        preliminary = GAM(
            formula=self.formula,
            k=self.k,
            basis=self.basis,
            fit_intercept=self.fit_intercept,
            optimize_smoothing=True,
            smoothing_method="reml",
        ).fit(frame, response, offset=offset)
        variance = preliminary.fit_result().scale
        residuals = (
            response - preliminary.predict(frame, offset=offset)
        ) / variance**0.5
        return {
            "residuals": residuals,
            "variance": variance,
            "effective_dimension": initial_quantile_dimension(preliminary),
            "parameters": fit_residual_shash(residuals),
        }

    def fit_levels(
        self, X=None, y=None, *, levels, data=None, sample_weight=None, offset=None
    ):
        """Return independently calibrated models, sharing bandwidth preparation.

        The configured/fitted caller is unchanged. Gaussian initialization and
        residual SHASH fitting are shared across levels when bandwidth=None.
        Quantiles may cross; no sorting, joint/noncrossing constraint or
        simultaneous inference is implied.
        """
        import numpy as np

        from ..gam.design import compile_design

        levels = np.asarray(levels, dtype=float)
        if (
            levels.ndim != 1
            or not len(levels)
            or not np.isfinite(levels).all()
            or np.any((levels <= 0) | (levels >= 1))
            or len(np.unique(levels)) != len(levels)
        ):
            raise ValueError("levels must be distinct quantiles strictly in (0,1).")
        reference = None
        if self.bandwidth is None:
            design = compile_design(
                X,
                y,
                data=data,
                formula=self.formula,
                k=self.k,
                basis=self.basis,
                fit_intercept=self.fit_intercept,
                smoothing_params=self.smoothing_params,
                sample_weight=sample_weight,
                offset=offset,
            )
            if not np.all(design.sample_weight == 1):
                raise NotImplementedError(
                    "Calibrated quantiles require unweighted observations."
                )
            if design.response is None:
                raise ValueError("Calibrated quantiles require a response.")
            reference = self._prepare_bandwidth(
                data if X is None else X, design.response, offset
            )
        models = {}
        for level in levels:
            model = type(self)(
                **(self.get_params(deep=False) | {"level": float(level)})
            )
            model._fit_calibrated(
                X,
                y,
                data=data,
                sample_weight=sample_weight,
                offset=offset,
                bandwidth_reference=reference,
            )
            models[float(level)] = model
        return models

    def _publish_calibration(self, result):
        self.fit_diagnostics_ = {
            "coefficients": {
                "converged": result.solution.converged,
                "iterations": result.solution.iterations,
                "function_evaluations": result.solution.function_evaluations,
                "optimality": result.solution.optimality,
            },
            "smoothing": result.smoothing_diagnostics,
            "calibration": result.calibration_diagnostics,
        }
        self.loss_scale_ = result.loss_scale
        self.posterior_covariance_ = result.posterior_covariance
        self.sandwich_covariance_ = result.sandwich_covariance
        self.calibration_loss_ = result.calibration_loss
        self.calibration_profile_ = result.profile

    def _solve(self, design, y, penalty, weights):
        result = fit_calibrated_quantile(
            design,
            y,
            penalty,
            weights,
            level=self.level,
            bandwidth=self.bandwidth_,
            log_scale_bounds=self.log_scale_bounds,
            max_iter=self.max_iter,
            tol=self.tol,
            calibration_tol=self.calibration_tol,
        )
        result = replace(result, solution=replace(
            result.solution,
            objective=result.solution.objective / result.loss_scale,
            optimality=result.solution.optimality / result.loss_scale,
        ))
        self._publish_calibration(result)
        return result.solution

    def _loss(self, residual):
        return _pinball_loss(residual, self.level)

    def set_params(self, **params):
        if not params:
            return self
        current = self.get_params(deep=False)
        if params.keys() - current.keys():
            raise ValueError(
                f"Unknown calibrated-quantile parameters: {sorted(params.keys() - current.keys())}"
            )
        replacement = type(self)(**(current | params))
        self.__dict__.clear()
        self.__dict__.update(replacement.__dict__)
        return self

    def predict_interval(self, X, *, level=0.95, offset=None):
        """Pointwise approximate intervals for the fitted conditional quantile."""
        estimate = self.predict(X, offset=offset)
        interval = quantile_function_interval(
            estimate,
            self.lpmatrix(X),
            self.posterior_covariance_,
            level=level,
            quantile_level=self.level,
        )

        return (
            replace(
                interval,
                coverage="approximate_pointwise_conditional_on_selected_smoothing",
            )
            if self.fit_diagnostics_["smoothing"] is not None
            else interval
        )
