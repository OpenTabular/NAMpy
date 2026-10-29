"""qgam's standardized-residual SHASH bandwidth calculation.

This is separate from spline smoothing and scalar Gibbs loss-scale calibration.
It supports one constant Gaussian variance and independent unweighted rows.
"""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import ndtr, ndtri

from ..fit.solvers.optim_bfgs import optim_bfgs


def _shash_value_gradient(parameter, response):
    mu, tau, epsilon, phi = parameter
    sigma, delta = np.exp(tau), np.exp(phi)
    z = (response - mu) / (sigma * delta)
    a = np.arcsinh(z)
    transformed = delta * a - epsilon
    with np.errstate(over="ignore", invalid="ignore"):
        sinh = np.sinh(transformed)
        logcosh = np.logaddexp(transformed, -transformed) - np.log(2.0)
        log_density = (
            -tau
            - 0.5 * np.log(2 * np.pi)
            + logcosh
            - 0.5 * np.log1p(z * z)
            - 0.5 * sinh * sinh
        )
        g = -transformed
        de = np.tanh(g) - 0.5 * np.sinh(2 * g)
        root = np.sqrt(1 + z * z)
        dm = (delta * de + z / root) / (delta * sigma * root)
        dt = (response - mu) * dm - 1
        dp = dt + 1 - delta * a * de
    return -float(log_density.sum()), -np.array(
        [dm.sum(), dt.sum(), de.sum(), dp.sum()]
    )


def _shash_quantile(probability, parameter):
    mu, tau, epsilon, phi = parameter
    delta = np.exp(phi)
    return mu + delta * np.exp(tau) * np.sinh(
        (np.arcsinh(ndtri(probability)) + epsilon) / delta
    )


def estimate_quantile_bandwidth(
    residuals, *, level, variance, effective_dimension, parameters=None
):
    """Return (bandwidth, error parameter, fitted SHASH parameters).

    residuals are standardized using a separately fitted Gaussian GAM. The
    mode-neighborhood adjustment and error cap follow qgam .getErrParam;
    neither supplies a universal calibration guarantee.
    """
    residuals = np.asarray(residuals, dtype=float)
    if (
        residuals.ndim != 1
        or len(residuals) < 5
        or not np.isfinite(residuals).all()
        or not 0 < level < 1
        or not np.isfinite([variance, effective_dimension]).all()
        or variance <= 0
        or effective_dimension <= 0
    ):
        raise ValueError(
            "Bandwidth selection needs finite standardized residuals, positive variance/dimension and a valid quantile."
        )
    parameter = (
        fit_residual_shash(residuals)
        if parameters is None
        else np.asarray(parameters, dtype=float)
    )
    if parameter.shape != (4,) or not np.isfinite(parameter).all():
        raise ValueError("Residual SHASH parameters must be a finite four-vector.")
    bounds = tuple(_shash_quantile(np.array([0.001, 0.999]), parameter))
    mode = minimize_scalar(
        lambda x: _shash_value_gradient(parameter, np.array([x]))[0],
        bounds=bounds,
        method="bounded",
        options={"xatol": np.finfo(float).eps ** 0.25},
    )
    if not mode.success:
        raise RuntimeError("SHASH density mode optimization failed.")
    mu, tau, epsilon, phi = parameter
    delta, sigma = np.exp(phi), np.exp(tau)
    mode_probability = ndtr(
        np.sinh(delta * np.arcsinh((mode.x - mu) / (delta * sigma)) - epsilon)
    )
    quantile = level
    if abs(quantile - mode_probability) < 0.05:
        quantile = np.clip(
            mode_probability + np.sign(quantile - mode_probability) * 0.05, 0.01, 0.99
        )
    q = _shash_quantile(quantile, parameter)
    lf0 = -_shash_value_gradient(parameter, np.array([q]))[0]
    # The response density derivative is minus the location derivative.
    dlog_density = _shash_value_gradient(parameter, np.array([q]))[1][0]
    if dlog_density == 0 or not np.isfinite(dlog_density):
        raise FloatingPointError(
            "Density derivative does not identify the requested bandwidth."
        )
    log_abs_density_derivative = np.log(abs(dlog_density)) + lf0
    h = (effective_dimension * 9 / (len(residuals) * np.pi**4)) ** (1 / 3) * np.exp(
        (lf0 - 2 * log_abs_density_derivative) / 3
    )
    error = min(h * 2 * np.log(2) / np.sqrt(2 * np.pi), 1.0)
    bandwidth = error * np.sqrt(2 * np.pi * variance) / (2 * np.log(2))
    return float(bandwidth), float(error), parameter.copy()


def initial_quantile_dimension(gaussian_model):
    """Source .getErrParam dimension: parametric count plus unique pen.edf.

    Total model EDF is not substituted: pen.edf counts coefficients touched by
    each penalty, excluding exact unpenalized directions. Values are calculated
    in the fitted compiler coordinates and kept separate from testing EDF.
    """
    compiled = gaussian_model.gam_result_.require_compiled_model()
    mapping = compiled.coef_reduced_to_full_idx
    coefficient_edf = np.diag(gaussian_model.fit_result().core.H_coef)
    dimension = int(bool(gaussian_model.fit_intercept))
    for term in compiled.compiled_terms:
        if term.term_type == "parametric":
            dimension += len(mapping[term.coef_slice])
    penalty_edf = []
    for penalty in compiled.compiled_penalties:
        active = np.any(penalty.matrix != 0, axis=1)
        penalty_edf.append(coefficient_edf[mapping[penalty.coef_slice][active]].sum())
    return float(dimension + np.unique(penalty_edf).sum())


def fit_residual_shash(residuals):
    """Fit source nuisance density once, reusable across quantile levels."""
    residuals = np.asarray(residuals, dtype=float)
    sd = np.std(residuals, ddof=1)
    if sd <= 0:
        raise ValueError("Constant standardized residuals cannot identify bandwidth.")
    fitted = optim_bfgs(
        lambda p: _shash_value_gradient(p, residuals)[0],
        lambda p: _shash_value_gradient(p, residuals)[1],
        [residuals.mean(), np.log(sd), 0.0, 0.0],
    )
    if not fitted.success or not np.isfinite(fitted.x).all():
        raise RuntimeError(f"Residual SHASH bandwidth fitting failed: {fitted.message}")
    return fitted.x.copy()
