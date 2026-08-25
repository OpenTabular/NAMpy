"""SCAM's GCV/UBRE BFGS smoothing optimizer.

Direct port of ``scam/R/bfgs.r::bfgs_gcv.ubre`` for the unbounded log-SP
coordinates used by shape-constrained models.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import OptimizeResult


def _current_scale_estimate(objective) -> float:
    """Return ``gcv.ubre_grad()$scale.est`` from the current fixed fit."""
    model = getattr(objective, "model", None)
    if model is None:
        return np.nan
    workspace = getattr(model, "_ws", None)
    if workspace is None:
        return np.nan
    state = workspace.get("transformed_gcv_ubre_state", None)
    if not isinstance(state, dict):
        state = workspace.get("shape_gcv_ubre_state", None)
    if not isinstance(state, dict):
        return np.nan
    solution = state.get("solution", {})
    denominator = float(model.n_samples_) - float(solution.get("trace_H", np.nan))
    if not np.isfinite(denominator) or denominator == 0.0:
        return np.nan
    return float(solution.get("deviance", np.nan)) / denominator


def optimize_shape_bfgs(
    objective, x0, *, control=None, max_steps=200
) -> OptimizeResult:
    rho = np.asarray(x0, dtype=np.float64).reshape(-1).copy()
    control = dict(control or {})
    n_penalties = rho.size
    typical_x = np.ones(n_penalties, dtype=np.float64)
    scaling = 1.0 / typical_x
    max_newton_step = float(control.get("max_n_step", 5.0))
    gradient_tolerance = float(control.get("gradtol_bfgs", 1e-6))
    step_tolerance = float(control.get("steptol_bfgs", 1e-7))
    max_halves = int(control.get("max_half", 30))
    max_steps = int(max_steps)
    c1 = 1e-4
    c2 = 0.9

    score = float(objective.fun(rho))
    old_score = score
    gradient = np.asarray(objective.jac(rho), dtype=np.float64)
    score_history = [score]

    # Upstream uses a one-sided difference of the exact gradient only to seed
    # the inverse Hessian, then forces that seed positive definite.
    finite_step = float(control.get("del", 1e-4))
    hessian = np.zeros((n_penalties, n_penalties), dtype=np.float64)
    for index in range(n_penalties):
        shifted = rho.copy()
        shifted[index] += finite_step
        shifted_gradient = np.asarray(objective.jac(shifted), dtype=np.float64)
        hessian[:, index] = (shifted_gradient - gradient) / finite_step
    hessian = 0.5 * (hessian + hessian.T)
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    eigenvalues = np.abs(eigenvalues)
    threshold = float(np.max(eigenvalues)) * 1e-4
    if threshold == 0.0:
        threshold = np.finfo(np.float64).eps
    eigenvalues[eigenvalues < threshold] = threshold
    inverse_hessian = (eigenvectors / eigenvalues[None, :]) @ eigenvectors.T

    scale_estimate = _current_scale_estimate(objective)
    score_scale = abs(scale_estimate) + abs(score)
    unconverged = np.abs(gradient) > score_scale * gradient_tolerance
    if not np.any(unconverged):
        unconverged[:] = True
    consecutive_maximum_steps = 0
    term_code = 4
    old_rho = rho.copy()

    for iteration in range(1, max_steps + 1):
        newton_step = np.zeros_like(gradient)
        active_inverse = inverse_hessian[np.ix_(unconverged, unconverged)]
        newton_step[unconverged] = -active_inverse @ gradient[unconverged]
        if float(newton_step @ gradient) >= 0.0:
            newton_step = -np.diag(inverse_hessian) * gradient
            newton_step[~unconverged] = 0.0

        scaled_step = scaling * newton_step
        newton_length = float(np.sqrt(np.sum(scaled_step**2)))
        if newton_length > max_newton_step:
            newton_step *= max_newton_step / newton_length
            newton_length = max_newton_step

        maximum_taken = False
        return_code = 2
        initial_slope = float(newton_step @ gradient)
        relative_length = float(
            np.max(np.abs(newton_step) / np.maximum(np.abs(rho), 1.0 / scaling))
        )
        alpha_min = (
            np.inf if relative_length == 0.0 else step_tolerance / relative_length
        )
        alpha_max = np.inf if newton_length == 0.0 else max_newton_step / newton_length
        max_component = float(np.max(np.abs(newton_step)))
        if max_component - max_newton_step > np.finfo(np.float64).eps ** 0.9:
            alpha = max_newton_step / max_component
            alpha_max = alpha * 1.05
        else:
            alpha = 1.0
            alpha_max = min(
                2.0,
                np.inf if max_component == 0.0 else max_newton_step / max_component,
            )

        halvings = 0
        curvature_condition = True
        old_alpha = alpha
        old_trial_score = score
        trial_gradient = gradient.copy()
        trial_score = score
        trial_rho = rho.copy()

        while True:
            trial_rho = rho + alpha * newton_step
            trial_score = float(objective.fun(trial_rho))
            if trial_score <= score + c1 * alpha * initial_slope:
                trial_gradient = np.asarray(objective.jac(trial_rho), dtype=np.float64)
                new_slope = float(trial_gradient @ newton_step)
                curvature_condition = True
                if new_slope < c2 * initial_slope:
                    if alpha == 1.0 and newton_length < max_newton_step:
                        for _ in range(40):
                            old_alpha = alpha
                            old_trial_score = trial_score
                            alpha = min(2.0 * alpha, alpha_max)
                            trial_rho = rho + alpha * newton_step
                            trial_score = float(objective.fun(trial_rho))
                            if trial_score <= score + c1 * alpha * initial_slope:
                                trial_gradient = np.asarray(
                                    objective.jac(trial_rho), dtype=np.float64
                                )
                                new_slope = float(trial_gradient @ newton_step)
                            if (
                                trial_score > score + c1 * alpha * initial_slope
                                or new_slope >= c2 * initial_slope
                                or alpha >= alpha_max
                            ):
                                break

                    needs_interpolation = alpha != 1.0 and (
                        alpha < 1.0
                        or (
                            alpha > 1.0
                            and trial_score > score + c1 * alpha * initial_slope
                        )
                    )
                    if needs_interpolation:
                        alpha_low = min(alpha, old_alpha)
                        alpha_difference = abs(old_alpha - alpha)
                        if alpha < old_alpha:
                            score_low, score_high = trial_score, old_trial_score
                        else:
                            score_low, score_high = old_trial_score, trial_score
                        for _ in range(40):
                            denominator = 2.0 * (
                                score_high - (score_low + new_slope * alpha_difference)
                            )
                            increment = (
                                -new_slope * alpha_difference**2 / denominator
                                if denominator != 0.0
                                else 0.2 * alpha_difference
                            )
                            increment = max(increment, 0.2 * alpha_difference)
                            alpha = alpha_low + increment
                            trial_rho = rho + alpha * newton_step
                            trial_score = float(objective.fun(trial_rho))
                            if trial_score > score + c1 * alpha * initial_slope:
                                alpha_difference = increment
                                score_high = trial_score
                            else:
                                trial_gradient = np.asarray(
                                    objective.jac(trial_rho), dtype=np.float64
                                )
                                new_slope = float(trial_gradient @ newton_step)
                                if new_slope < c2 * initial_slope:
                                    alpha_low = alpha
                                    alpha_difference -= increment
                                    score_low = trial_score
                            if (
                                abs(new_slope) <= -c2 * initial_slope
                                or alpha_difference < alpha_min
                            ):
                                break
                        if new_slope < c2 * initial_slope:
                            curvature_condition = False
                            trial_score = score_low
                            trial_rho = rho + alpha_low * newton_step
                            trial_gradient = np.asarray(
                                objective.jac(trial_rho), dtype=np.float64
                            )
                            alpha = alpha_low
                return_code = 0
                if new_slope < c2 * initial_slope:
                    curvature_condition = False
                if alpha * newton_length > 0.99 * max_newton_step:
                    maximum_taken = True
            elif alpha < alpha_min:
                return_code = 1
                trial_rho = rho.copy()
                trial_score = float(objective.fun(trial_rho))
                trial_gradient = np.asarray(objective.jac(trial_rho), dtype=np.float64)
            else:
                halvings += 1
                if alpha == 1.0:
                    denominator = trial_score - score - initial_slope
                    alpha_temp = (
                        -initial_slope / denominator / 2.0
                        if denominator != 0.0
                        else 0.5 * alpha
                    )
                else:
                    matrix = np.array(
                        [
                            [1.0 / alpha**2, -1.0 / old_alpha**2],
                            [-old_alpha / alpha**2, alpha / old_alpha**2],
                        ]
                    )
                    rhs = np.array(
                        [
                            trial_score - score - alpha * initial_slope,
                            old_trial_score - score - old_alpha * initial_slope,
                        ]
                    )
                    cubic, quadratic = (matrix @ rhs) / (alpha - old_alpha)
                    discriminant = quadratic**2 - 3.0 * cubic * initial_slope
                    if cubic == 0.0:
                        alpha_temp = -initial_slope / quadratic / 2.0
                    else:
                        alpha_temp = (-quadratic + np.sqrt(max(discriminant, 0.0))) / (
                            3.0 * cubic
                        )
                    alpha_temp = min(alpha_temp, 0.5 * alpha)
                old_alpha = alpha
                old_trial_score = trial_score
                alpha = max(alpha_temp, 0.1 * alpha)
            if halvings == max_halves or return_code < 2:
                break

        step = alpha * newton_step
        old_score = score
        old_rho = rho.copy()
        old_gradient = gradient.copy()
        rho = trial_rho.copy()
        score = float(trial_score)
        gradient = trial_gradient.copy()
        score_history.append(score)

        gradient_change = gradient - old_gradient
        skip_update = True
        for index in range(n_penalties):
            closeness = step[index] - inverse_hessian[index, :] @ gradient_change
            if abs(closeness) >= gradient_tolerance * max(
                abs(gradient[index]), abs(old_gradient[index])
            ):
                skip_update = False
        if not curvature_condition:
            skip_update = True
        if not skip_update:
            if iteration == 1:
                inverse_hessian *= alpha
            curvature = float(gradient_change @ step)
            if curvature != 0.0:
                reciprocal = 1.0 / curvature
                # Preserve the two sequential assignments in bfgs_gcv.ubre:
                # its second B %*% yg uses the already-updated B.
                inverse_hessian = inverse_hessian - reciprocal * np.outer(
                    step, gradient_change @ inverse_hessian
                )
                inverse_hessian = (
                    inverse_hessian
                    - reciprocal * np.outer(inverse_hessian @ gradient_change, step)
                    + reciprocal * np.outer(step, step)
                )

        relative_gradient = float(
            np.max(
                np.abs(gradient)
                * np.maximum(np.abs(rho), 1.0 / scaling)
                / max(abs(score), 1.0)
            )
        )
        if return_code == 1:
            term_code = 1 if relative_gradient <= gradient_tolerance * 6.0554 else 3
        elif relative_gradient <= gradient_tolerance * 6.0554:
            term_code = 1
        elif (
            float(
                np.max(np.abs(rho - old_rho) / np.maximum(np.abs(rho), 1.0 / scaling))
            )
            <= step_tolerance
        ):
            term_code = 2
        elif iteration == max_steps:
            term_code = 4
        elif maximum_taken:
            consecutive_maximum_steps += 1
            term_code = 5 if consecutive_maximum_steps == 5 else 0
        else:
            consecutive_maximum_steps = 0
            term_code = 0
        if term_code > 0:
            break

        scale_estimate = _current_scale_estimate(objective)
        score_scale = abs(scale_estimate) + abs(score)
        unconverged = np.abs(gradient) > score_scale * gradient_tolerance
        apparently_converged = not np.any(unconverged)
        if abs(old_score - score) > score_scale * gradient_tolerance:
            if apparently_converged:
                unconverged[:] = True

    messages = {
        1: "Full convergence",
        2: "Successive iterates within tolerance, current iterate is probably solution",
        3: "Last step failed to locate a lower point than the current log-SP",
        4: "Iteration limit reached",
        5: "Five consecutive steps of maximum length were taken",
    }
    result = OptimizeResult(
        x=rho,
        fun=score,
        jac=gradient,
        hess_inv=inverse_hessian,
        nit=iteration,
        nfev=int(getattr(objective, "n_fun", 0)),
        njev=int(getattr(objective, "n_jac", 0)),
        success=term_code in {1, 2},
        status=term_code,
        message=messages.get(term_code, "SCAM BFGS terminated"),
    )
    result.score_hist = np.asarray(score_history, dtype=np.float64)
    result.outer_info = {
        "optimizer": "bfgs",
        "termcode": term_code,
        "conv": str(result.message),
        "iterations": iteration,
        "score_hist": result.score_hist.copy(),
        "grad": gradient.copy(),
    }
    result.strict_shape_bfgs = True
    return result


def optimize_transformed_bfgs(
    objective, x0, *, control=None, max_steps=200
) -> OptimizeResult:
    """Generic name for the exact transformed GCV/UBRE BFGS policy."""
    return optimize_shape_bfgs(objective, x0, control=control, max_steps=max_steps)


__all__ = ["optimize_shape_bfgs", "optimize_transformed_bfgs"]
