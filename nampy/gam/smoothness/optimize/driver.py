"""Entry points: supports_*, expand_*, and optimize_smoothing_params."""
import warnings

import numpy as np
from scipy.optimize import OptimizeResult, minimize

from ..criteria import (
    _static_penalty_null_dim,
    criterion_gradient_ml_reml_gaussian_dynamic_joint,
    resolve_ml_reml_scoring_backend,
)
from .basics import (
    _initial_smoothing_params_from_design_balance,
    supports_criterion_gradient,
    supports_criterion_hessian,
)
from .objectives import (
    _approx_derivative,
    _CriterionObjective,
    _design_has_mrf_smooth,
    _JointGaussianRemlObjective,
)
from .outer import _optimize_outer_newton, _optimize_outer_newton_indefinite_hessian
from .postprocess import (
    _accept_flat_boundary_result,
    _accept_tiny_step_line_search_result,
    _collapse_near_zero_smoothing_params,
    _coordinate_refine_smoothing_params,
    _refine_null_space_smoothing_params,
    _rollback_working_infinite_smoothing_params,
    _stabilize_flat_smoothing_params,
)

def supports_smoothing_method(model, method):
    method = str(method).lower()
    attr_map = {
        "fixed": None,
        "gcv": "supports_gcv",
        "ubre": "supports_ubre",
        "aic": "supports_ubre",
        "ubreaic": "supports_ubre",
        "ml": "supports_ml",
        "reml": "supports_reml",
        "laml": "supports_laml",
    }
    if method not in attr_map:
        raise ValueError(
            "method must be one of "
            "{'fixed', 'gcv', 'ubre', 'aic', 'ubreaic', 'ml', 'reml', 'laml'}"
        )

    attr = attr_map[method]
    if attr is None:
        return True

    base_ok = bool(getattr(model.family, attr, False))
    if not base_ok:
        return False

    if method in {"ml", "reml", "laml"}:
        return resolve_ml_reml_scoring_backend(model, method=method) is not None

    return True


def resolve_smoothing_method(model, method):
    method = "auto" if method is None else str(method).lower()
    if method != "auto":
        return method

    if (
        model.family.supports_reml
        and resolve_ml_reml_scoring_backend(model, method="reml") is not None
    ):
        return "reml"

    if model.family.known_scale is not None and getattr(model.family, "supports_ubre", False):
        return "ubreaic"

    if getattr(model.family, "supports_gcv", False):
        return "gcv"

    return "fixed"


def n_free_smoothing_params(model):
    if model.smoothing_fixed_mask_ is None:
        return int(model.n_smoothing_params_ or 0)
    return int(np.sum(~model.smoothing_fixed_mask_))


def expand_smoothing_params_from_log(model, log_free_sp):
    if model.n_smoothing_params_ is None:
        raise RuntimeError("Design has not been compiled yet.")

    fixed_mask = (
        np.zeros(model.n_smoothing_params_, dtype=bool)
        if model.smoothing_fixed_mask_ is None
        else np.asarray(model.smoothing_fixed_mask_, dtype=bool)
    )

    log_free_sp = np.asarray(log_free_sp, dtype=np.float64).ravel()
    n_free = int(np.sum(~fixed_mask))
    if log_free_sp.shape != (n_free,):
        raise ValueError(
            f"Expected {n_free} free log smoothing parameters, got shape {log_free_sp.shape}."
        )

    sp = np.asarray(model.smoothing_params, dtype=np.float64).copy()
    if n_free > 0:
        sp[~fixed_mask] = np.exp(log_free_sp)

    if model.min_sp_ is not None:
        sp = np.maximum(sp, np.asarray(model.min_sp_, dtype=np.float64))
    return sp


def optimize_smoothing_params(model, y, initial_smoothing_params=None, method="gcv", optimizer="lbfgsb"):
    method = model._resolve_smoothing_method(method)
    optimizer = str(optimizer).lower()
    exact_gaussian = str(getattr(model.family, "name", "")).lower() == "gaussian"

    if method not in {"gcv", "ubre", "aic", "ubreaic", "ml", "reml", "laml"}:
        raise ValueError(
            "method must be one of "
            "{'gcv', 'ubre', 'aic', 'ubreaic', 'ml', 'reml', 'laml'}"
        )
    if not model._supports_smoothing_method(method):
        if method in {"ml", "reml", "laml"}:
            model._raise_ml_reml_backend_error(method)
        raise NotImplementedError(
            f"Automatic smoothing selection with method={method!r} is not "
            f"supported for family={model.family.name!r}."
        )
    if optimizer not in {"lbfgsb", "outer_newton"}:
        raise NotImplementedError(
            "Current core supports smoothing_optimizer in {'lbfgsb', 'outer_newton'} only."
        )

    use_gradient = supports_criterion_gradient(model, method)
    use_hessian = optimizer == "outer_newton" and supports_criterion_hessian(model, method)

    fixed_mask = (
        np.zeros(model.n_smoothing_params_, dtype=bool)
        if model.smoothing_fixed_mask_ is None
        else np.asarray(model.smoothing_fixed_mask_, dtype=bool)
    )
    free_mask = ~fixed_mask
    n_free = int(np.sum(free_mask))
    ml_reml_backend = (
        resolve_ml_reml_scoring_backend(model, method=method)
        if method in {"ml", "reml", "laml"}
        else None
    )

    if n_free == 0:
        model._optim_method = method
        model._optim_result = None
        model._optim_trace = []
        model._optim_used_gradient = False
        model._optim_used_hessian = False
        model.smoothing_score_ = float(
            model._criterion(y, np.empty((0,), dtype=np.float64), method=method)
        )
        return model

    if initial_smoothing_params is None:
        user_sp = getattr(getattr(model, "hparams", {}), "get", None)
        if callable(user_sp):
            user_sp = model.hparams.get("smoothing_params", None)
        else:
            user_sp = None

        use_design_balance_init = (
            user_sp is None
            and (
                (not bool(getattr(model.family, "supports_closed_form_solve", False)))
                or ml_reml_backend == "gaussian_dynamic"
            )
        )
        if use_design_balance_init:
            init = _initial_smoothing_params_from_design_balance(model, y)
            if init is None:
                init_free = np.asarray(model.smoothing_params[free_mask], dtype=np.float64)
            else:
                init_free = np.asarray(init[free_mask], dtype=np.float64)
        else:
            init_free = np.asarray(model.smoothing_params[free_mask], dtype=np.float64)
    else:
        init = np.asarray(initial_smoothing_params, dtype=np.float64)
        if init.shape == (model.n_smoothing_params_,):
            init_free = np.asarray(init[free_mask], dtype=np.float64)
        elif init.shape == (n_free,):
            init_free = init.copy()
        else:
            raise ValueError(
                f"Expected initial smoothing params of shape "
                f"({model.n_smoothing_params_},) or ({n_free},), got {init.shape}."
            )

    if np.any(~np.isfinite(init_free)) or np.any(init_free <= 0):
        raise ValueError(
            "Initial free smoothing parameters must be finite and > 0."
        )

    min_sp = (
        np.zeros(model.n_smoothing_params_, dtype=np.float64)
        if model.min_sp_ is None
        else np.asarray(model.min_sp_, dtype=np.float64)
    )

    init_free = np.maximum(init_free, min_sp[free_mask])
    x0 = np.log(np.maximum(init_free, 1e-300))

    bounds = []
    for lower_sp in min_sp[free_mask]:
        if lower_sp > 0:
            lo = max(float(model.sp_log_bounds[0]), float(np.log(lower_sp)))
        else:
            lo = float(model.sp_log_bounds[0])
        bounds.append((lo, float(model.sp_log_bounds[1])))

    model._gaussian_reml_sigma2_opt_ = None
    # Gaussian REML/LAML uses a joint (log sp, log sigma^2) outer loop. The dynamic
    # Gaussian backend always uses that geometry. The exact backend matches reference
    # software for most smooths when profiling sigma^2; MRF-like cases need the same
    # joint loop on the exact mixed-model path.
    use_joint_gaussian_reml_scale = (
        exact_gaussian
        and method in {"reml", "laml"}
        and (
            ml_reml_backend == "gaussian_dynamic"
            or (
                ml_reml_backend == "gaussian_exact" and _design_has_mrf_smooth(model)
            )
        )
    )

    if use_joint_gaussian_reml_scale:
        sp0 = expand_smoothing_params_from_log(model, x0)
        sol0 = model._solve_gaussian_given_smoothing(y, sp0)
        F0 = float(sol0["rss"]) + float(sol0["penalty_quadratic"] or 0.0)
        Mp = float(
            _static_penalty_null_dim(model)
            + int(bool(getattr(model, "fit_intercept", False)))
        )
        nu0 = float(model.n_samples_ - Mp)
        if not np.isfinite(nu0) or nu0 <= 0.0:
            log_s2_0 = np.log(1e-300)
        else:
            log_s2_0 = float(np.log(max(F0 / nu0, 1e-300)))
        x_joint0 = np.concatenate(
            [np.asarray(x0, dtype=np.float64).ravel(), np.array([log_s2_0], dtype=np.float64)]
        )
        y_eff = (
            np.asarray(y, dtype=np.float64).ravel()
            if model.offset_train_ is None
            else (np.asarray(y, dtype=np.float64).ravel() - model.offset_train_)
        )
        yv = (
            float(np.var(y_eff))
            if y_eff.size > 1
            else float(np.maximum(np.abs(float(y_eff[0])), 1e-300))
        )
        hi_scale = max(yv * 1e8, max(F0 / max(nu0, 1e-300), 1e-300) * 1e8, 1e-30)
        joint_bounds = list(bounds) + [
            (float(np.log(1e-300)), float(np.log(hi_scale)))
        ]
        branch_m = "LAML" if method == "laml" else "REML"
        j_obj = _JointGaussianRemlObjective(
            model, y, branch_m, str(ml_reml_backend)
        )
        # `gaussian_exact` uses finite-difference gradients in `jac`; omitting `jac`
        # often matches reference software more closely on ill-scaled (log sp, log sigma^2) steps.
        use_jac = str(ml_reml_backend) != "gaussian_exact"
        result_joint = minimize(
            fun=j_obj.fun,
            x0=x_joint0,
            method="L-BFGS-B",
            jac=j_obj.jac if use_jac else None,
            bounds=joint_bounds,
            options={
                "maxfun": 25000,
                "ftol": 1e-11,
                "gtol": 1e-10,
            },
        )
        joint_dim = int(n_free + 1)
        if (not bool(result_joint.success)) and np.isfinite(
            float(getattr(result_joint, "fun", np.nan))
        ):
            msg_u = str(getattr(result_joint, "message", "")).upper()
            if "ABNORMAL" in msg_u and joint_dim <= 32:
                result_joint.success = True
                result_joint.message = (
                    "Accepted L-BFGS-B ABNORMAL termination on joint Gaussian REML outer problem."
                )

        x_sp = np.asarray(result_joint.x[:-1], dtype=np.float64).ravel()
        log_s2_opt = float(result_joint.x[-1])
        model.smoothing_params = np.asarray(model.smoothing_params, dtype=np.float64).copy()
        model.smoothing_params[free_mask] = np.exp(x_sp)
        model.smoothing_params = np.maximum(model.smoothing_params, min_sp)
        model._gaussian_reml_sigma2_opt_ = float(np.exp(log_s2_opt))

        x_full_opt = np.concatenate(
            [np.asarray(x_sp, dtype=np.float64).ravel(), np.array([log_s2_opt], dtype=np.float64)]
        )
        if ml_reml_backend == "gaussian_exact":
            if _approx_derivative is not None:
                g_full = _approx_derivative(
                    j_obj._raw_fun, x_full_opt, method="2-point"
                )
            else:
                g_full = None
        else:
            g_full = criterion_gradient_ml_reml_gaussian_dynamic_joint(
                model, y, x_sp, log_s2_opt, method=branch_m
            )
        jac_sp = (
            np.asarray(g_full[:-1], dtype=np.float64).copy()
            if g_full is not None
            else None
        )
        result = OptimizeResult(
            x=x_sp.copy(),
            fun=float(result_joint.fun),
            jac=jac_sp,
            hess=None,
            success=bool(result_joint.success),
            status=int(result_joint.status),
            message=str(result_joint.message),
            nit=int(getattr(result_joint, "nit", 0)),
            nfev=int(getattr(result_joint, "nfev", j_obj.n_fun)),
            njev=int(getattr(result_joint, "njev", j_obj.n_jac)),
            nhev=0,
        )
        setattr(result, "joint_gaussian_reml_outer", True)
        setattr(result, "joint_log_sigma2", float(log_s2_opt))

        model._optim_method = method
        model._optim_result = result
        trace_grad = None
        if g_full is not None:
            trace_grad = np.asarray(g_full, dtype=np.float64).tolist()
        model._optim_trace = [
            {
                "iter": 0,
                "log_sp": np.asarray(x_sp, dtype=np.float64).tolist(),
                "criterion": float(result_joint.fun),
                "gradient": trace_grad,
                "hessian": None,
                "accepted_step_norm": 0.0,
                "rank_info": {"joint_gaussian_reml_outer": True},
            }
        ]
        model._optim_used_gradient = True
        model._optim_used_hessian = False
        model.smoothing_score_ = float(result_joint.fun)

        if not result_joint.success:
            warnings.warn(
                f"Smoothing optimisation did not converge: {result_joint.message}"
            )
        return model

    objective = _CriterionObjective(model, y, method=method, use_gradient=use_gradient)
    if bool(getattr(model.family, "supports_pirls", False)):
        # Carry P-IRLS coefficient warm-starts between outer criterion evaluations.
        setattr(model, "_pirls_coef_start_", None)
    indefinite_hessian_newton_for_pirls = (
        method in {"ml", "reml", "laml"}
        and bool(getattr(model.family, "supports_pirls", False))
        and not bool(getattr(model.family, "supports_closed_form_solve", False))
    )

    if optimizer == "lbfgsb":
        if indefinite_hessian_newton_for_pirls and supports_criterion_hessian(model, method):
            result = _optimize_outer_newton_indefinite_hessian(
                objective=objective,
                x0=x0,
                bounds=bounds,
            )
            result.indefinite_hessian_outer_newton = True
            if not result.success:
                lbfgsb_retry = minimize(
                    fun=objective.fun,
                    x0=np.asarray(result.x, dtype=np.float64),
                    method="L-BFGS-B",
                    jac=objective.jac if use_gradient else None,
                    bounds=bounds,
                )
                lbfgsb_retry.indefinite_hessian_lbfgsb_fallback = True
                if lbfgsb_retry.success or (
                    np.isfinite(getattr(lbfgsb_retry, "fun", np.inf))
                    and float(lbfgsb_retry.fun) <= float(result.fun)
                ):
                    result = lbfgsb_retry
        else:
            result = minimize(
                fun=objective.fun,
                x0=x0,
                method="L-BFGS-B",
                jac=objective.jac if use_gradient else None,
                bounds=bounds,
            )
        if not result.success and supports_criterion_hessian(model, method):
            outer_newton_result = _optimize_outer_newton(
                objective=objective,
                x0=x0,
                bounds=bounds,
            )
            outer_newton_result.lbfgsb_fallback = True
            outer_newton_result.lbfgsb_message = str(result.message)
            if outer_newton_result.success or (
                np.isfinite(getattr(outer_newton_result, "fun", np.inf))
                and (
                    not np.isfinite(getattr(result, "fun", np.inf))
                    or float(outer_newton_result.fun) <= float(result.fun)
                )
            ):
                result = outer_newton_result
    else:
        result = _optimize_outer_newton(
            objective=objective,
            x0=x0,
            bounds=bounds,
        )
        if not result.success:
            lbfgsb_result = minimize(
                fun=objective.fun,
                x0=np.asarray(result.x, dtype=np.float64),
                method="L-BFGS-B",
                jac=objective.jac if use_gradient else None,
                bounds=bounds,
            )
            lbfgsb_result.outer_newton_fallback = True
            lbfgsb_result.outer_newton_message = str(result.message)
            result = lbfgsb_result

    has_null_space_penalty = any(
        bool(getattr(pb, "is_null_space_penalty", False))
        for pb in (getattr(model, "penalty_blocks_", None) or [])
    )
    apply_post_heuristics = (
        method in {"ml", "reml", "laml"}
        and (
            has_null_space_penalty
            or (
                (not exact_gaussian)
                and (not bool(getattr(result, "indefinite_hessian_outer_newton", False)))
            )
        )
    )
    if apply_post_heuristics:
        result = _rollback_working_infinite_smoothing_params(
            objective=objective,
            result=result,
            x0=x0,
            bounds=bounds,
            method=method,
        )
        result = _stabilize_flat_smoothing_params(
            objective=objective,
            result=result,
            x0=x0,
            bounds=bounds,
            method=method,
        )
        result = _collapse_near_zero_smoothing_params(
            objective=objective,
            result=result,
            bounds=bounds,
            method=method,
        )
        result = _accept_flat_boundary_result(
            objective=objective,
            result=result,
            method=method,
        )
    if has_null_space_penalty:
        result = _refine_null_space_smoothing_params(
            objective=objective,
            result=result,
            bounds=bounds,
        )
    if ml_reml_backend == "gaussian_dynamic":
        result = _coordinate_refine_smoothing_params(
            objective=objective,
            result=result,
            bounds=bounds,
        )
    result = _accept_tiny_step_line_search_result(
        objective=objective,
        result=result,
    )

    if not result.success:
        warnings.warn(f"Smoothing optimisation did not converge: {result.message}")

    model.smoothing_params = np.asarray(model.smoothing_params, dtype=np.float64).copy()
    model.smoothing_params[free_mask] = np.exp(result.x)
    model.smoothing_params = np.maximum(model.smoothing_params, min_sp)

    model._optim_method = method
    model._optim_result = result
    if getattr(objective, "trace", None) is not None:
        trace_rows = []
        prev_x = None
        for i, row in enumerate(objective.trace):
            x_row = np.asarray(row["x"], dtype=np.float64)
            step_norm = (
                0.0 if prev_x is None else float(np.linalg.norm(x_row - prev_x, ord=2))
            )
            trace_rows.append(
                {
                    "iter": int(i),
                    "log_sp": x_row.tolist(),
                    "criterion": None if row["fun"] is None else float(row["fun"]),
                    "gradient": None
                    if row["grad"] is None
                    else np.asarray(row["grad"], dtype=np.float64).tolist(),
                    "hessian": None
                    if row["hess"] is None
                    else np.asarray(row["hess"], dtype=np.float64).tolist(),
                    "accepted_step_norm": step_norm,
                    "n_fun": int(row.get("n_fun", 0)),
                    "n_jac": int(row.get("n_jac", 0)),
                    "n_hess": int(row.get("n_hess", 0)),
                    "rank_info": None,
                }
            )
            prev_x = x_row
        model._optim_trace = trace_rows
        result.optim_trace = trace_rows
    model.smoothing_score_ = float(result.fun)
    model._optim_used_gradient = bool(use_gradient)
    model._optim_used_hessian = bool(use_hessian)
    return model
