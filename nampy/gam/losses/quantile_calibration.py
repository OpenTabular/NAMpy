"""Fixed-bandwidth/fixed-penalty qgam calFast loss-scale calibration.

Reference: qgam 2.0.1 .llkGrads, .biasedCov, .sandwichLoss and
.invert_psdef_matrix. These are quantile-function Gibbs posterior calculations,
not likelihood coverage or conformal prediction. Automatic bandwidth/smoothing,
bootstrap calibration and covariate-dependent loss scales are separate routes.

References
----------
- Fasiolo, M., Wood, S. N., Zaffran, M., Nedellec, R., and Goude, Y. (2021). Fast
  Calibrated Additive Quantile Regression.
  https://doi.org/10.1080/01621459.2020.1725521
- Fasiolo, M., Wood, S. N., Zaffran, M., Nedellec, R., and Goude, Y. (2021). qgam:
  Bayesian Nonparametric Quantile Regression Modeling in R.
  https://doi.org/10.18637/jss.v100.i09
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar, root
from scipy.special import expit

from .objectives import LossFit, _inputs


def _precision_inverse(matrix, rank):
    """qgam's diagonal-scaled spectral inverse and explicit 1e-6 rank cutoff."""
    diagonal = np.diag(matrix)
    if not np.isfinite(matrix).all() or np.any(diagonal <= 0):
        raise ValueError("qgam calibration needs finite positive diagonal precision.")
    scale = 1 / np.sqrt(diagonal)
    values, vectors = np.linalg.eigh(scale[:, None] * matrix * scale[None, :])
    order = np.argsort(-values)
    values, vectors = values[order], vectors[:, order]
    keep = min(int(np.sum(values > values[0] * 1e-6)), rank)
    if not keep:
        raise ValueError("qgam calibration has no identifiable precision directions.")
    root = (vectors[:, :keep] / np.sqrt(values[:keep])).T * scale
    return root.T @ root


@dataclass(frozen=True)
class QuantileCalibrationFit:
    solution: LossFit
    loss_scale: float
    posterior_covariance: np.ndarray
    sandwich_covariance: np.ndarray
    calibration_loss: float
    profile: np.ndarray
    calibration_diagnostics: dict
    smoothing_diagnostics: dict | None


def _optimizer_diagnostics(result):
    """Keep reported optimizer evidence; a missing derivative is not zero."""
    if result is None:
        return None
    jac = getattr(result, "jac", None)
    return {
        "converged": bool(result.success),
        "iterations": getattr(result, "nit", None),
        "function_evaluations": getattr(result, "nfev", None),
        "optimality": None if jac is None else float(np.max(np.abs(jac))),
        "message": str(result.message),
    }


def _fit_elf_root(matrix, response, penalty, *, level, bandwidth, max_iter, tol):
    """Canonical coefficient score solve for this new calibration route.

    Solving the exact monotone score avoids using cancellation-prone changes
    in the ELF objective as a convergence test. Existing QuantileGAM fitting
    is unchanged. No alternate-solver retry is performed.
    """
    X, y, S, w = _inputs(matrix, response, penalty, None, level)

    def score(beta):
        probability = expit((y - X @ beta) / bandwidth)
        return S @ beta - X.T @ (w * (level - 1 + probability))

    def curvature(beta):
        probability = expit((y - X @ beta) / bandwidth)
        return S + X.T @ (
            (w * probability * (1 - probability) / bandwidth)[:, None] * X
        )

    initial = np.linalg.lstsq(X, y, rcond=None)[0]
    result = root(
        score,
        initial,
        jac=curvature,
        method="hybr",
        options={"xtol": tol, "maxfev": max_iter},
    )
    optimality = float(np.max(np.abs(score(result.x))))
    # Certify parameter accuracy by the exact Newton score step, rather than
    # a gradient norm whose units change with basis scaling. Positive curvature
    # certifies a convex minimum; no alternate solver/restart is performed.
    H = curvature(result.x)
    np.linalg.cholesky(H)
    step = np.linalg.solve(H, score(result.x))
    converged = bool(
        np.isfinite(result.x).all()
        and np.max(np.abs(step)) <= tol * (1 + np.max(np.abs(result.x)))
    )
    residual = y - X @ result.x
    objective = float(
        w @ ((level - 1) * residual + bandwidth * np.logaddexp(0, residual / bandwidth))
        + 0.5 * result.x @ S @ result.x
    )
    return LossFit(
        result.x, X @ result.x, objective, None, converged, optimality,
        function_evaluations=int(result.nfev),
    )


def quantile_calibration_at_scale(
    matrix, response, penalty, *, level, bandwidth, log_scale, max_iter, tol
):
    """Fit one loss scale and evaluate the exact calFast variance criterion."""
    with np.errstate(over="raise", under="raise", invalid="raise"):
        sigma = float(np.exp(log_scale))
    fitted = _fit_elf_root(
        matrix,
        response,
        sigma * penalty,
        level=level,
        bandwidth=bandwidth,
        max_iter=max_iter,
        tol=tol,
    )
    if not fitted.converged:
        raise RuntimeError(
            f"ELF coefficient fitting did not converge during calibration: sigma={sigma}, score_inf={fitted.optimality}."
        )
    posterior, sandwich, loss = _calibration_covariances(
        matrix, response, penalty, fitted, level=level, bandwidth=bandwidth, sigma=sigma
    )
    return fitted, sigma, posterior, sandwich, loss


def _calibration_covariances(
    matrix, response, penalty, fitted, *, level, bandwidth, sigma
):
    n, p = matrix.shape
    probability = expit((response - fitted.fitted) / bandwidth)
    score = (probability - 1 + level) / sigma
    information = matrix.T @ (
        (probability * (1 - probability) / (bandwidth * sigma))[:, None] * matrix
    )
    posterior = np.linalg.solve(information + penalty, np.eye(p))
    gradients = score[:, None] * matrix
    empirical = np.atleast_2d(np.cov(gradients, rowvar=False, ddof=1))
    effective = np.abs(score).sum() ** 2 / np.sum(score**2)
    alpha = min(effective / (p * p), 1.0)
    mean_x = matrix.mean(axis=0)
    independence = (matrix.T @ matrix / n) * np.mean(score**2) - np.outer(
        mean_x, mean_x
    ) * np.mean(score) ** 2
    variance = n * (alpha * empirical + (1 - alpha) * independence)
    sandwich_precision = (
        information @ _precision_inverse(variance, p) @ information + penalty
    )
    sandwich = _precision_inverse(sandwich_precision, p)
    ordinary = np.einsum("np,pq,nq->n", matrix, posterior, matrix)
    robust = np.einsum("np,pq,nq->n", matrix, sandwich, matrix)
    valid = ~((ordinary == 0) & (robust == 0))
    if not valid.any() or np.any(ordinary[valid] <= 0) or np.any(robust[valid] <= 0):
        raise ValueError(
            "Calibration needs identifiable positive quantile-function variances."
        )
    ratio = robust[valid] / ordinary[valid]
    loss = float(np.mean(np.sqrt(ratio - np.log(ratio))))
    if not np.isfinite(loss) or not np.isfinite(posterior).all():
        raise FloatingPointError(
            "Quantile calibration criterion/covariance is nonfinite."
        )
    return posterior, sandwich, loss


def fit_calibrated_quantile(
    matrix,
    response,
    penalty,
    weights,
    *,
    level,
    bandwidth,
    log_scale_bounds,
    max_iter,
    tol,
    calibration_tol,
):
    """Bounded scalar calibration with fixed bandwidth and smoothing penalties.

    Bounds are supplied explicitly; optima at a declared bound raise rather
    than trigger unvalidated bracket expansion or loss-smoothness rescue.
    """
    if not np.all(weights == 1):
        raise NotImplementedError(
            "Calibrated quantile inference currently requires unweighted observations."
        )
    if len(response) < 3 or not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError(
            "Calibration requires n>=3 and a finite positive fixed bandwidth."
        )
    bounds = np.asarray(log_scale_bounds, dtype=float)
    if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
        raise ValueError("log_scale_bounds must be two increasing finite values.")
    if not np.isfinite(calibration_tol) or calibration_tol <= 0:
        raise ValueError("calibration_tol must be finite and positive.")
    cache = {}

    def evaluate(log_sigma):
        if log_sigma not in cache:
            cache[log_sigma] = quantile_calibration_at_scale(
                matrix,
                response,
                penalty,
                level=level,
                bandwidth=bandwidth,
                log_scale=log_sigma,
                max_iter=max_iter,
                tol=tol,
            )
        return cache[log_sigma][-1]

    optimized = minimize_scalar(
        evaluate,
        bounds=tuple(bounds),
        method="bounded",
        options={"xatol": calibration_tol},
    )
    if not optimized.success:
        raise RuntimeError("qgam calFast loss-scale calibration did not converge.")
    if min(optimized.x - bounds[0], bounds[1] - optimized.x) <= 2 * calibration_tol:
        raise NotImplementedError(
            "Calibration optimum reached a declared bound; expand bounds explicitly."
        )
    fitted, sigma, posterior, sandwich, loss = cache[optimized.x]
    profile = np.array([[x, value[-1]] for x, value in cache.items()])
    return QuantileCalibrationFit(
        fitted, sigma, posterior, sandwich, loss, profile,
        _optimizer_diagnostics(optimized), None,
    )


def fit_calibrated_quantile_auto(
    X,
    response,
    *,
    formula,
    level,
    bandwidth,
    log_scale_bounds,
    k,
    basis,
    fit_intercept,
    smoothing_params,
    smoothing_optimizer,
    max_iter,
    tol,
    calibration_tol,
    offset=None,
):
    """Select smoothing at each calibrated loss scale using the shared engine."""
    from ..control import gam_control
    from ..design import CompiledDesign
    from ..diagnostics.fit_status import distribution_fit_status
    from ..families.elf import ELFFamily
    from ..model import GAM

    bounds = np.asarray(log_scale_bounds, dtype=float)
    if (
        bounds.shape != (2,)
        or not np.isfinite(bounds).all()
        or bounds[0] >= bounds[1]
        or not np.isfinite(calibration_tol)
        or calibration_tol <= 0
    ):
        raise ValueError(
            "Calibration needs increasing finite log-scale bounds and positive tolerance."
        )
    cache = {}

    def evaluate(log_sigma):
        if log_sigma not in cache:
            with np.errstate(over="raise", under="raise", invalid="raise"):
                sigma = float(np.exp(log_sigma))
            family = ELFFamily(level=level, bandwidth=bandwidth, loss_scale=sigma)
            gam = GAM(
                formula=formula,
                family=family,
                k=k,
                basis=basis,
                fit_intercept=fit_intercept,
                scale=1.0,
                smoothing_params=smoothing_params,
                optimize_smoothing=True,
                smoothing_method="reml",
                smoothing_optimizer=smoothing_optimizer,
                control=gam_control(maxit=max_iter, epsilon=tol),
            ).fit(X, response, offset=offset)
            status = distribution_fit_status(gam)
            if not status.eligible:
                raise RuntimeError(
                    f"ELF automatic smoothing failed at log scale {log_sigma}: {status.message}"
                )
            design = CompiledDesign.from_fitted_gam(gam, X, offset=offset)
            coef = gam.fit_result().coef_full
            core = gam.fit_result().core
            fitted_values = design.matrix @ coef
            residual = np.asarray(response) - design.training_offset - fitted_values
            score = design.penalty @ coef - design.matrix.T @ (
                (level - 1 + expit(residual / bandwidth)) / sigma
            )
            fitted = LossFit(
                coef, fitted_values, float(-gam.loglik()), core.iter,
                bool(core.converged and not core.failed_step),
                float(np.max(np.abs(score))),
            )
            V, VS, loss = _calibration_covariances(
                design.matrix,
                np.asarray(response) - design.training_offset,
                design.penalty,
                fitted,
                level=level,
                bandwidth=bandwidth,
                sigma=sigma,
            )
            cache[log_sigma] = (fitted, sigma, V, VS, loss, design, gam)
        return cache[log_sigma][4]

    optimized = minimize_scalar(
        evaluate,
        bounds=tuple(bounds),
        method="bounded",
        options={"xatol": calibration_tol},
    )
    if not optimized.success:
        raise RuntimeError("Automatic-smoothing qgam calibration did not converge.")
    if min(optimized.x - bounds[0], bounds[1] - optimized.x) <= 2 * calibration_tol:
        raise NotImplementedError(
            "Calibration optimum reached a declared bound; expand bounds explicitly."
        )
    fitted, sigma, V, VS, loss, design, gam = cache[optimized.x]
    profile = np.array([[x, value[4]] for x, value in cache.items()])
    return QuantileCalibrationFit(
        fitted, sigma, V, VS, loss, profile,
        _optimizer_diagnostics(optimized),
        _optimizer_diagnostics(gam._optim_result),
    ), design, gam
