"""Entry points: supports_*, expand_*, and optimize_smoothing_params."""

import warnings

import numpy as np
from scipy.optimize import OptimizeResult, minimize

from ..._mgcv_constants import LOG_GUARD_MIN
from ..._model_state import (
    _compiled_model,
    _n_smoothing_params,
    _term_blocks_seq,
)
from ...fit.model_ops import (
    criterion_value,
    raise_ml_reml_backend_error,
)
from ..criteria import (
    _gaussian_dynamic_reml_derivative_terms,
    criterion_hessian_ml_reml_pirls_exact,
    resolve_ml_reml_scoring_backend,
)
from .basics import (
    _initial_smoothing_params_from_design_balance,
    _initial_smoothing_params_mgcv_style,
    supports_criterion_gradient,
    supports_criterion_hessian,
)
from .bfgs_mgcv import _optimize_outer_bfgs_mgcv
from .efs_mgcv import _optimize_outer_efs_mgcv
from .objectives import (
    _CriterionObjective,
    _GaussianRemlJointObjective,
    _GaussianRemlProfiledObjective,
    _JointGammaPirlsRemlObjective,
    _JointNegbinPirlsRemlObjective,
)
from .outer import _optimize_outer_newton, _optimize_outer_newton_indefinite_hessian


def _optimize_negbin_reml_joint_native(model, y, x0, free_mask, method, sp_bounds):
    """Native joint negbin REML/LAML outer optimization over (log sp, log theta)."""
    if str(method).lower() not in {"reml", "laml"}:
        return None
    if (
        model.offset_train_ is not None
        or getattr(model, "prior_weights_", None) is not None
    ):
        return None
    if not isinstance(getattr(model, "formula", None), str):
        return None
    if str(getattr(model.family, "name", "")).lower() != "negbin" or not bool(
        getattr(model.family, "estimate_theta", False)
    ):
        return None

    x0 = np.asarray(x0, dtype=np.float64).ravel()
    free_mask = np.asarray(free_mask, dtype=bool)
    free_count = int(np.sum(free_mask))
    if free_count <= 0 or x0.size != free_count:
        return None

    theta0 = float(max(float(getattr(model.family, "theta", 1.0)), LOG_GUARD_MIN))
    log_theta0 = float(np.log(theta0))
    x_joint0 = np.concatenate([x0, np.array([log_theta0], dtype=np.float64)])
    joint_bounds = list(sp_bounds)
    joint_bounds.append((float(np.log(LOG_GUARD_MIN)), np.inf))

    branch_m = "LAML" if str(method).lower() == "laml" else "REML"
    j_obj = _JointNegbinPirlsRemlObjective(model, y, branch_m)
    result_joint = _optimize_outer_newton_indefinite_hessian(
        objective=j_obj,
        x0=x_joint0,
        bounds=joint_bounds,
        conv_tol=1e-6,
    )
    result = OptimizeResult()
    x_joint = np.asarray(result_joint.x, dtype=np.float64).ravel()
    if x_joint.size != x_joint0.size:
        x_joint = x_joint0.copy()
    log_theta_opt = float(x_joint[-1]) if x_joint.size else log_theta0
    if np.isfinite(log_theta_opt):
        theta_opt = float(np.exp(log_theta_opt))
    else:
        theta_opt = float(theta0)
    if not np.isfinite(theta_opt) or theta_opt <= 0.0:
        theta_opt = float(theta0)
        log_theta_opt = float(log_theta0)

    x_sp_opt = x_joint[:-1]
    result.x = x_sp_opt.copy()
    result.fun = float(
        result_joint.fun if np.isfinite(float(result_joint.fun)) else np.nan
    )
    result.nit = int(getattr(result_joint, "nit", 0))
    result.nfev = int(getattr(result_joint, "nfev", j_obj.n_fun))
    result.njev = int(getattr(result_joint, "njev", j_obj.n_jac))
    result.nhev = int(getattr(result_joint, "nhev", j_obj.n_hess))
    result.success = bool(getattr(result_joint, "success", False))
    result.status = int(getattr(result_joint, "status", 0))
    result.message = str(
        getattr(result_joint, "message", "joint negbin REML/LAML solve")
    )
    result.joint_negbin_reml_outer = True
    result.joint_negbin_efs_outer = True
    result.joint_negbin_postprocessed = True
    result.joint_negbin_flat_ridge_stabilized = False
    result.joint_negbin_initial_log_theta = float(log_theta0)
    result.joint_log_theta = float(log_theta_opt)
    result.joint_negbin_message = str(result.message)
    result.joint_negbin_fun = float(result.fun)
    result.joint_negbin_nfev = int(result.nfev)
    result.joint_negbin_njev = int(result.njev)
    result.joint_negbin_selected_x = np.asarray(x_sp_opt, dtype=np.float64).copy()
    result.joint_negbin_selected_full_sp = np.asarray(
        model.smoothing_params, dtype=np.float64
    ).copy()
    result.joint_negbin_selected_full_sp[free_mask] = np.exp(x_sp_opt)
    result.mgcv_selected_full_sp = np.asarray(
        model.smoothing_params, dtype=np.float64
    ).copy()
    result.mgcv_selected_full_sp[free_mask] = np.exp(x_sp_opt)
    result.mgcv_selected_theta = float(theta_opt)

    score_hist = []
    log_theta_hist = []
    log_sp_hist = []
    for row in j_obj.accepted_trace:
        x_row = np.asarray(
            row.get("x", np.array([], dtype=np.float64)), dtype=np.float64
        )
        if x_row.size == 0:
            continue
        log_sp_hist.append(np.asarray(x_row[:-1], dtype=np.float64).tolist())
        log_theta_hist.append(float(x_row[-1]))
        score_hist.append(float(row.get("fun", np.nan)))

    if len(score_hist) == 0:
        score_hist = [float(result.fun)]
        log_theta_hist = [float(log_theta_opt)]
        log_sp_hist = [np.asarray(x_sp_opt, dtype=np.float64).tolist()]

    result.outer_info = {
        "score_hist": score_hist,
        "log_theta_hist": log_theta_hist,
        "log_sp_hist": log_sp_hist,
    }
    result.joint_negbin_reml_outer = True
    return result


def supports_smoothing_method(model, method):
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

    if model.family.known_scale is not None and getattr(
        model.family, "supports_ubre", False
    ):
        return "ubreaic"

    if getattr(model.family, "supports_gcv", False):
        return "gcv"

    return "fixed"


def n_free_smoothing_params(model):
    if model.smoothing_fixed_mask_ is None:
        return int(_n_smoothing_params(model) or 0)
    return int(np.sum(~model.smoothing_fixed_mask_))


def expand_smoothing_params_from_log(model, log_free_sp):
    n_smoothing_params = _n_smoothing_params(model)
    if n_smoothing_params == 0 and _compiled_model(model) is None:
        raise RuntimeError("Design has not been compiled yet.")

    fixed_mask = (
        np.zeros(n_smoothing_params, dtype=bool)
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


def optimize_smoothing_params(
    model, y, initial_smoothing_params=None, method="gcv", optimizer="lbfgsb"
):
    method = resolve_smoothing_method(model, method)
    optimizer = str(optimizer).lower()
    if optimizer == "newton":
        optimizer = "outer_newton"
    exact_gaussian = str(getattr(model.family, "name", "")).lower() == "gaussian"

    if method not in {"gcv", "ubre", "aic", "ubreaic", "ml", "reml", "laml"}:
        raise ValueError(
            "method must be one of "
            "{'gcv', 'ubre', 'aic', 'ubreaic', 'ml', 'reml', 'laml'}"
        )
    if not supports_smoothing_method(model, method):
        if method in {"ml", "reml", "laml"}:
            raise_ml_reml_backend_error(model, method)
        raise NotImplementedError(
            f"Automatic smoothing selection with method={method!r} is not "
            f"supported for family={model.family.name!r}."
        )
    if optimizer not in {"lbfgsb", "outer_newton", "bfgs", "efs"}:
        raise NotImplementedError(
            "Current core supports smoothing_optimizer in "
            "{'lbfgsb', 'outer_newton', 'bfgs', 'efs'} only."
        )
    if (
        optimizer == "lbfgsb"
        and method in {"ml", "reml", "laml"}
        and supports_criterion_hessian(model, method)
    ):
        # mgcv's outer smoothing search for ML/REML/LAML is Newton-shaped when
        # exact first/second derivatives are available. Keep L-BFGS-B only for
        # branches without a full Hessian path.
        optimizer = "outer_newton"

    use_gradient = supports_criterion_gradient(model, method)
    use_hessian = optimizer == "outer_newton" and supports_criterion_hessian(
        model, method
    )
    if (
        optimizer == "outer_newton"
        and method in {"ml", "reml", "laml"}
        and (not use_gradient or not use_hessian)
    ):
        raise NotImplementedError(
            "Strict mgcv-parity ML/REML/LAML smoothing optimisation requires "
            "exact first- and second-derivative support; local fallback paths "
            "have been removed."
        )
    if optimizer == "bfgs" and method in {"ml", "reml", "laml"} and (not use_gradient):
        raise NotImplementedError(
            "Strict mgcv-parity BFGS smoothing optimisation requires an exact "
            "gradient path for this method/family."
        )
    if optimizer == "efs" and method not in {"reml", "laml"}:
        raise NotImplementedError(
            "Strict mgcv-parity EFS smoothing optimisation is currently "
            "implemented only for REML/LAML."
        )

    fixed_mask = (
        np.zeros(_n_smoothing_params(model), dtype=bool)
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
    family_name = str(getattr(model.family, "name", "")).lower()
    use_joint_gamma_reml_scale = (
        family_name == "gamma"
        and method in {"reml", "laml"}
        and ml_reml_backend == "pirls_laplace"
    )
    use_joint_gaussian_reml_scale = exact_gaussian and method == "reml"
    use_joint_negbin_reml_theta = (
        family_name == "negbin"
        and method in {"reml", "laml"}
        and ml_reml_backend == "pirls_laplace"
        and bool(getattr(model.family, "estimate_theta", False))
    )
    model._pirls_disable_theta_efs_ = False

    if n_free == 0:
        model._optim_method = method
        model._optim_result = None
        model._optim_trace = []
        model._optim_used_gradient = False
        model._optim_used_hessian = False
        model.smoothing_score_ = float(
            criterion_value(model, y, np.empty((0,), dtype=np.float64), method=method)
        )
        return model

    if initial_smoothing_params is None:
        user_sp = getattr(getattr(model, "hparams", {}), "get", None)
        if callable(user_sp):
            user_sp = model.hparams.get("smoothing_params", None)
        else:
            user_sp = None

        has_factor_smooth_fs = any(
            str(getattr(tb, "term_type", "")).lower() == "factor_smooth_fs"
            for tb in _term_blocks_seq(model)
        )
        use_design_balance_init = user_sp is None and (
            exact_gaussian
            or (not bool(getattr(model.family, "supports_closed_form_solve", False)))
            or ml_reml_backend == "gaussian_dynamic"
            or has_factor_smooth_fs
        )
        if use_design_balance_init:
            if use_joint_gamma_reml_scale:
                init = _initial_smoothing_params_mgcv_style(model, y)
                if init is None:
                    init = _initial_smoothing_params_from_design_balance(model, y)
            elif exact_gaussian:
                init = _initial_smoothing_params_mgcv_style(model, y)
                if init is None:
                    init = _initial_smoothing_params_from_design_balance(model, y)
            else:
                init = _initial_smoothing_params_from_design_balance(model, y)
            if init is None:
                init_free = np.asarray(
                    model.smoothing_params[free_mask], dtype=np.float64
                )
            else:
                init_free = np.asarray(init[free_mask], dtype=np.float64)
        else:
            init_free = np.asarray(model.smoothing_params[free_mask], dtype=np.float64)
    else:
        init = np.asarray(initial_smoothing_params, dtype=np.float64)
        if init.shape == (_n_smoothing_params(model),):
            init_free = np.asarray(init[free_mask], dtype=np.float64)
        elif init.shape == (n_free,):
            init_free = init.copy()
        else:
            raise ValueError(
                f"Expected initial smoothing params of shape "
                f"({_n_smoothing_params(model)},) or ({n_free},), got {init.shape}."
            )

    if np.any(~np.isfinite(init_free)) or np.any(init_free <= 0):
        raise ValueError("Initial free smoothing parameters must be finite and > 0.")

    min_sp = (
        np.zeros(_n_smoothing_params(model), dtype=np.float64)
        if model.min_sp_ is None
        else np.asarray(model.min_sp_, dtype=np.float64)
    )

    init_free = np.maximum(init_free, min_sp[free_mask])
    x0 = np.log(np.maximum(init_free, LOG_GUARD_MIN))

    bounds = []
    for lower_sp in min_sp[free_mask]:
        if lower_sp > 0:
            lo = max(float(model.sp_log_bounds[0]), float(np.log(lower_sp)))
        else:
            lo = float(model.sp_log_bounds[0])
        bounds.append((lo, float(model.sp_log_bounds[1])))

    if use_joint_gaussian_reml_scale:
        # Mirror `mgcv/R/mgcv.r::get.null.coef` + `scale.as.sp` initialization:
        # `log.scale <- log(null.scale / 10)`, where
        # `null.scale <- sum(dev.resids(y, mum, weights)) / nrow(X)`.
        yv = np.asarray(y, dtype=np.float64).ravel()
        mu_null = np.repeat(float(np.mean(yv)), model.n_samples_)
        null_scale = float(
            model.family.deviance(
                yv,
                mu_null,
                weights=getattr(model, "prior_weights_", None),
            )
        ) / float(model.n_samples_)
        sigma20 = max(null_scale / 10.0, LOG_GUARD_MIN)
        y_scale = (
            float(np.var(yv))
            if yv.size > 1
            else float(np.maximum(np.abs(float(yv[0])), LOG_GUARD_MIN))
        )
        hi_sigma2 = max(sigma20 * 1e8, y_scale * 1e8, 1e-30)
        x0 = np.concatenate([x0, np.array([float(np.log(sigma20))], dtype=np.float64)])
        bounds = list(bounds) + [
            (float(np.log(LOG_GUARD_MIN)), float(np.log(hi_sigma2)))
        ]

    model._gaussian_reml_sigma2_opt_ = None
    model._gaussian_reml_last_scale_est_ = None

    if use_joint_negbin_reml_theta:
        mgcv_result = _optimize_negbin_reml_joint_native(
            model, y, x0, free_mask, method, bounds
        )
        if mgcv_result is not None:
            model.family.theta = float(np.exp(float(mgcv_result.joint_log_theta)))
            model._pirls_disable_theta_efs_ = True
            model.smoothing_params = np.asarray(
                mgcv_result.joint_negbin_selected_full_sp, dtype=np.float64
            ).copy()
            model._optim_method = method
            model._optim_result = mgcv_result
            model._optim_trace = None
            model._optim_used_gradient = False
            model._optim_used_hessian = False
            model.smoothing_score_ = float(mgcv_result.fun)
            return model
        raise NotImplementedError(
            "Negative-binomial REML/LAML with estimate_theta=True has no "
            "native upstream-supported joint optimizer path in this build."
        )

    branch_m = "LAML" if method == "laml" else "REML"
    if use_joint_gaussian_reml_scale and optimizer != "bfgs":
        objective = _GaussianRemlJointObjective(
            model=model,
            y=y,
            branch_method=branch_m,
        )
    elif exact_gaussian and method in {"reml", "laml"}:
        objective = _GaussianRemlProfiledObjective(
            model=model,
            y=y,
            branch_method=branch_m,
        )
    else:
        objective = _CriterionObjective(
            model, y, method=method, use_gradient=use_gradient
        )
    if bool(getattr(model.family, "supports_pirls", False)):
        # Carry P-IRLS coefficient warm-starts between outer criterion evaluations.
        model._pirls_coef_start_ = None
        model._pirls_eta_start_ = None
        model._pirls_mu_start_ = None
    result = None

    if not use_joint_gamma_reml_scale and result is None:
        if optimizer == "efs":
            result = _optimize_outer_efs_mgcv(
                model=model,
                y=y,
                x0=x0,
                bounds=bounds,
                method=method,
            )
        elif optimizer == "bfgs":
            result = _optimize_outer_bfgs_mgcv(
                objective=objective,
                x0=x0,
                bounds=bounds,
                score_type=method,
            )
        elif method in {"ml", "reml", "laml"}:
            result = _optimize_outer_newton_indefinite_hessian(
                objective=objective,
                x0=x0,
                bounds=bounds,
            )
            result.indefinite_hessian_outer_newton = True
        elif optimizer == "lbfgsb":
            result = minimize(
                fun=objective.fun,
                x0=x0,
                method="L-BFGS-B",
                jac=objective.jac if use_gradient else None,
                bounds=bounds,
                options={"maxfun": 25000, "ftol": 1e-13, "gtol": 1e-12},
            )
        else:
            result = _optimize_outer_newton(
                objective=objective,
                x0=x0,
                bounds=bounds,
            )

    if use_joint_gamma_reml_scale:
        mu_null = np.repeat(
            float(np.mean(np.asarray(y, dtype=np.float64).ravel())), model.n_samples_
        )
        null_scale = float(
            model.family.deviance(np.asarray(y, dtype=np.float64).ravel(), mu_null)
        ) / float(model.n_samples_)
        phi0 = max(null_scale / 10.0, 1e-12)
        if phi0 is not None and np.isfinite(float(phi0)) and float(phi0) > 0.0:
            phi0 = float(phi0)
            y_eff = (
                np.asarray(y, dtype=np.float64).ravel()
                if model.offset_train_ is None
                else (np.asarray(y, dtype=np.float64).ravel() - model.offset_train_)
            )
            y_scale = (
                float(np.var(y_eff))
                if y_eff.size > 1
                else float(np.maximum(np.abs(float(y_eff[0])), LOG_GUARD_MIN))
            )
            hi_phi = max(phi0 * 1e8, y_scale * 1e8, 1e-30)
            joint_bounds = list(bounds) + [
                (float(np.log(LOG_GUARD_MIN)), float(np.log(hi_phi)))
            ]
            x_joint0 = np.concatenate(
                [x0.copy(), np.array([np.log(phi0)], dtype=np.float64)]
            )
            j_obj = _JointGammaPirlsRemlObjective(model, y, branch_m)
            result_joint = _optimize_outer_newton_indefinite_hessian(
                objective=j_obj,
                x0=x_joint0,
                bounds=joint_bounds,
                conv_tol=1e-6,
            )
            x_joint = np.asarray(result_joint.x, dtype=np.float64).ravel()
            x_selected = np.asarray(x_joint[:-1], dtype=np.float64).ravel()
            result = OptimizeResult(
                x=np.asarray(x_selected, dtype=np.float64).copy(),
                fun=float(objective.fun(x_selected)),
                jac=np.asarray(objective.jac(x_selected), dtype=np.float64),
                hess=np.asarray(objective.hess(x_selected), dtype=np.float64),
                success=bool(getattr(result_joint, "success", False)),
                status=int(getattr(result_joint, "status", 0)),
                message=str(getattr(result_joint, "message", "")),
                nit=int(getattr(result_joint, "nit", 0)),
                nfev=int(getattr(result_joint, "nfev", j_obj.n_fun)),
                njev=int(getattr(result_joint, "njev", j_obj.n_jac)),
                nhev=int(getattr(result_joint, "nhev", j_obj.n_hess)),
            )
            result.joint_gamma_reml_outer = True
            result.joint_log_phi = float(x_joint[-1])
            result.joint_gamma_message = str(getattr(result_joint, "message", ""))
            _ = criterion_hessian_ml_reml_pirls_exact(model, y, result.x, branch_m)
            gamma_state = getattr(model, "_pirls_reml_gamma_state_", None)
            phi_opt = None
            if isinstance(gamma_state, dict):
                phi_opt = gamma_state.get("phi", None)
            if (
                phi_opt is not None
                and np.isfinite(float(phi_opt))
                and float(phi_opt) > 0.0
            ):
                model._gamma_reml_phi_opt_ = float(phi_opt)

    if not result.success:
        warnings.warn(
            f"Smoothing optimisation did not converge: {result.message}",
            stacklevel=2,
        )

    if use_joint_gaussian_reml_scale and result is not None and result.x is not None:
        x_joint = np.asarray(result.x, dtype=np.float64).ravel()
        if x_joint.size == n_free + 1:
            joint_log_sigma2 = float(x_joint[-1])
            model._gaussian_reml_sigma2_opt_ = float(
                max(np.exp(joint_log_sigma2), LOG_GUARD_MIN)
            )
            result.joint_gaussian_reml_outer = True
            result.joint_log_sigma2 = joint_log_sigma2
            result.joint_x = x_joint.copy()
            result.x = np.asarray(x_joint[:-1], dtype=np.float64).copy()
            if getattr(result, "jac", None) is not None:
                result.jac = np.asarray(result.jac, dtype=np.float64).ravel()[:-1]
            if getattr(result, "hess", None) is not None:
                result.hess = np.asarray(result.hess, dtype=np.float64)[:-1, :-1]

    if (
        exact_gaussian
        and method in {"reml", "laml"}
        and result is not None
        and result.x is not None
        and getattr(model, "_gaussian_reml_sigma2_opt_", None) is None
    ):
        branch_m = "LAML" if method == "laml" else "REML"
        try:
            scale_out = _gaussian_dynamic_reml_derivative_terms(
                model, y, np.asarray(result.x, dtype=np.float64).ravel(), branch_m
            )
            if isinstance(scale_out, dict) and bool(scale_out.get("valid", False)):
                F = float(scale_out.get("F", np.nan))
                coeff = float(scale_out.get("coeff", np.nan))
                gamma_val = float(model.score_gamma)
                prof_df = gamma_val * coeff
                if (
                    np.isfinite(F)
                    and F > 0.0
                    and np.isfinite(prof_df)
                    and prof_df > 0.0
                    and np.isfinite(gamma_val)
                    and gamma_val > 0.0
                ):
                    model._gaussian_reml_sigma2_opt_ = float(
                        max(F / prof_df, LOG_GUARD_MIN)
                    )
        except Exception:
            pass

    model.smoothing_params = np.asarray(model.smoothing_params, dtype=np.float64).copy()
    model.smoothing_params[free_mask] = np.exp(result.x)
    model.smoothing_params = np.maximum(model.smoothing_params, min_sp)

    model._optim_method = method
    model._optim_result = result
    if (
        getattr(result, "optim_trace", None) is not None
        and (
            not bool(getattr(result, "joint_negbin_reml_outer", False))
            or not bool(getattr(result, "joint_negbin_efs_outer", False))
        )
        and not bool(getattr(result, "joint_gamma_reml_outer", False))
    ):
        trace_rows = []
        uses_joint_log_scale = bool(getattr(objective, "uses_joint_log_scale", False))
        for row in list(getattr(result, "optim_trace", []) or []):
            row_dict = dict(row)
            log_sp = np.asarray(row_dict.get("log_sp", []), dtype=np.float64).ravel()
            log_scale = None
            if uses_joint_log_scale and log_sp.size > 0:
                log_scale = float(log_sp[-1])
                log_sp = log_sp[:-1]
            trace_rows.append(
                {
                    "iter": int(row_dict.get("iter", 0)),
                    "log_sp": log_sp.tolist(),
                    "log_scale": log_scale,
                    "log_theta": (
                        None
                        if row_dict.get("log_theta", None) is None
                        else float(row_dict.get("log_theta"))
                    ),
                    "criterion": (
                        None
                        if row_dict.get("criterion", None) is None
                        else float(row_dict.get("criterion"))
                    ),
                    "gradient": (
                        None
                        if row_dict.get("gradient", None) is None
                        else np.asarray(
                            row_dict.get("gradient"), dtype=np.float64
                        ).tolist()
                    ),
                    "hessian": (
                        None
                        if row_dict.get("hessian", None) is None
                        else np.asarray(
                            row_dict.get("hessian"), dtype=np.float64
                        ).tolist()
                    ),
                    "accepted_step_norm": float(
                        row_dict.get("accepted_step_norm", 0.0)
                    ),
                    "rank_info": row_dict.get("rank_info", None),
                }
            )
        model._optim_trace = trace_rows
        result.optim_trace = trace_rows
    if (
        getattr(result, "optim_trace", None) is None
        and bool(getattr(result, "joint_negbin_reml_outer", False))
        and bool(getattr(result, "joint_negbin_efs_outer", False))
    ):
        outer_info = getattr(result, "outer_info", {}) or {}
        score_hist = list(outer_info.get("score_hist", []))
        log_theta_hist = list(outer_info.get("log_theta_hist", []))
        log_sp_hist = list(outer_info.get("log_sp_hist", []))
        trace_rows = []
        prev_x = None
        n_rows = min(len(score_hist), len(log_theta_hist), len(log_sp_hist))
        for i in range(n_rows):
            x_row = np.asarray(log_sp_hist[i], dtype=np.float64)
            step_norm = (
                0.0 if prev_x is None else float(np.linalg.norm(x_row - prev_x, ord=2))
            )
            trace_rows.append(
                {
                    "iter": int(i + 1),
                    "log_sp": x_row.tolist(),
                    "log_theta": float(log_theta_hist[i]),
                    "criterion": float(score_hist[i]),
                    "gradient": None,
                    "hessian": None,
                    "accepted_step_norm": step_norm,
                    "rank_info": {
                        "joint_negbin_reml_outer": True,
                    },
                }
            )
            prev_x = x_row
        if trace_rows:
            model._optim_trace = trace_rows
            result.optim_trace = trace_rows
    elif (
        getattr(result, "optim_trace", None) is None
        and getattr(objective, "trace", None) is not None
        and (
            not bool(getattr(result, "joint_negbin_reml_outer", False))
            or not bool(getattr(result, "joint_negbin_efs_outer", False))
        )
        and not bool(getattr(result, "joint_gamma_reml_outer", False))
    ):
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
                    "log_theta": None,
                    "criterion": None if row["fun"] is None else float(row["fun"]),
                    "gradient": (
                        None
                        if row["grad"] is None
                        else np.asarray(row["grad"], dtype=np.float64).tolist()
                    ),
                    "hessian": (
                        None
                        if row["hess"] is None
                        else np.asarray(row["hess"], dtype=np.float64).tolist()
                    ),
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
    elif (
        getattr(result, "optim_trace", None) is None
        and getattr(objective, "accepted_trace", None) is not None
        and (
            not bool(getattr(result, "joint_negbin_reml_outer", False))
            or not bool(getattr(result, "joint_negbin_efs_outer", False))
        )
        and not bool(getattr(result, "joint_gamma_reml_outer", False))
    ):
        trace_rows = []
        uses_joint_log_scale = bool(getattr(objective, "uses_joint_log_scale", False))
        prev_x = None
        for i, row in enumerate(objective.accepted_trace):
            x_row_full = np.asarray(row["x"], dtype=np.float64)
            x_row = (
                np.asarray(x_row_full[:-1], dtype=np.float64)
                if uses_joint_log_scale and x_row_full.size > 0
                else x_row_full
            )
            step_norm = float(row.get("accepted_step_norm", 0.0))
            if prev_x is not None and not np.isfinite(step_norm):
                step_norm = float(np.linalg.norm(x_row - prev_x, ord=2))
            trace_rows.append(
                {
                    "iter": int(i + 1),
                    "log_sp": x_row.tolist(),
                    "log_theta": None,
                    "criterion": None if row.get("fun") is None else float(row["fun"]),
                    "gradient": None,
                    "hessian": None,
                    "accepted_step_norm": step_norm,
                    "rank_info": {"accepted_outer_step": True},
                }
            )
            prev_x = x_row
        model._optim_trace = trace_rows
        result.optim_trace = trace_rows
    model.smoothing_score_ = float(result.fun)
    model._optim_used_gradient = bool(use_gradient)
    model._optim_used_hessian = bool(use_hessian)
    return model
