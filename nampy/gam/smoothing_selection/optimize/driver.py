"""Entry points: supports_*, expand_*, and optimize_smoothing_params."""

import warnings

import numpy as np
from scipy.optimize import OptimizeResult, minimize, minimize_scalar

from ..._mgcv_constants import LOG_GUARD_MIN
from ..._model_state import _coef_column_offset, _term_blocks_seq
from ..criteria import (
    _pirls_ml_reml_objective_from_solution,
    _stable_penalty_logdet_derivatives,
    _static_penalty_null_dim,
    criterion_gradient_ml_reml_gaussian_dynamic_joint,
    criterion_gradient_ml_reml_pirls_exact,
    criterion_hessian_ml_reml_pirls_exact,
    criterion_ml_reml_gaussian_dynamic_joint,
    resolve_ml_reml_scoring_backend,
)
from .basics import (
    _initial_smoothing_params_from_design_balance,
    _initial_smoothing_params_mgcv_style,
    supports_criterion_gradient,
    supports_criterion_hessian,
)
from .heuristics.rollback import (
    _accept_flat_boundary_result,
    _accept_stationary_abnormal_result,
    _accept_tiny_step_line_search_result,
    _rollback_working_infinite_smoothing_params,
)
from .heuristics.stabilize import (
    _collapse_near_zero_smoothing_params,
    _coordinate_refine_smoothing_params,
    _refine_null_space_smoothing_params,
    _snap_gaussian_random_effect_boundary,
    _stabilize_factor_smooth_shared_ridge,
    _stabilize_flat_smoothing_params,
    _stabilize_joint_negbin_flat_ridge,
)
from .objectives import (
    _approx_derivative,
    _CriterionObjective,
    _JointGammaPirlsRemlObjective,
    _JointGaussianRemlObjective,
    _JointNegbinPirlsRemlObjective,
)
from .outer import _optimize_outer_newton, _optimize_outer_newton_indefinite_hessian


def _joint_negbin_efs_update_terms(model, sol, sp):
    """mgcv::efsudr update terms for log(sp), using current PIRLS endpoint."""
    beta = np.asarray(sol["coef_full"], dtype=np.float64).ravel()
    A_inv = np.asarray(sol["A_inv"], dtype=np.float64)
    n_sp = int(model.n_smoothing_params_ or 0)
    quad = np.zeros(n_sp, dtype=np.float64)
    tr_vs = np.zeros(n_sp, dtype=np.float64)
    offset0 = _coef_column_offset(model)

    P_derivs = [np.zeros_like(A_inv, dtype=np.float64) for _ in range(n_sp)]
    for pb in getattr(model, "penalty_blocks_", None) or ():
        k = int(getattr(pb, "smoothing_index", -1))
        if k < 0 or k >= n_sp:
            continue
        sl = getattr(pb, "coef_slice", None)
        if sl is None:
            continue
        full_sl = slice(offset0 + sl.start, offset0 + sl.stop)
        P_loc = np.asarray(pb.matrix, dtype=np.float64)
        beta_loc = beta[full_sl]
        quad[k] += float(beta_loc @ (P_loc @ beta_loc))
        P_derivs[k][full_sl, full_sl] += float(sp[k]) * P_loc

    _, det_s1, _ = _stable_penalty_logdet_derivatives(model, sp, order=1)
    for k, Pk in enumerate(P_derivs):
        spk = float(sp[k])
        if spk <= 0.0 or not np.any(Pk):
            continue
        tr_vs[k] = float(np.trace(A_inv @ Pk)) / spk

    a = np.maximum(
        0.0,
        np.asarray(det_s1, dtype=np.float64) / np.asarray(sp, dtype=np.float64) - tr_vs,
    )
    return a, quad, tr_vs


def _pirls_state_from_solution(sol):
    return {
        "coef": np.asarray(sol["coef_full"], dtype=np.float64).copy(),
        "eta": np.asarray(sol["eta"], dtype=np.float64).copy(),
        "mu": np.asarray(sol["mu"], dtype=np.float64).copy(),
        "theta": None,
    }


def _set_model_pirls_start_state(model, state):
    if not isinstance(state, dict):
        return
    coef = state.get("coef", None)
    eta = state.get("eta", None)
    mu = state.get("mu", None)
    model._pirls_coef_start_ = (
        None if coef is None else np.asarray(coef, dtype=np.float64).copy()
    )
    model._pirls_eta_start_ = (
        None if eta is None else np.asarray(eta, dtype=np.float64).copy()
    )
    model._pirls_mu_start_ = (
        None if mu is None else np.asarray(mu, dtype=np.float64).copy()
    )
    theta = state.get("theta", None)
    if theta is not None and np.isfinite(float(theta)) and float(theta) > 0.0:
        model.family.theta = float(theta)


def _evaluate_joint_negbin_efs_state(
    model, y, log_sp, log_theta_seed, method, *, start_state=None
):
    # mgcv::efsudr always refits each trial from the current accepted PIRLS
    # state (`start <- fit$coefficients`, plus `mustart` carried through
    # `gam.fit3/4`). Rejected trials must not replace the accepted warm start.
    start_coef = None if start_state is None else start_state.get("coef", None)
    start_eta = None if start_state is None else start_state.get("eta", None)
    start_mu = None if start_state is None else start_state.get("mu", None)
    model._pirls_eval_start_ = (
        None if start_coef is None else np.asarray(start_coef, dtype=np.float64).copy()
    )
    model._pirls_eval_eta_start_ = (
        None if start_eta is None else np.asarray(start_eta, dtype=np.float64).copy()
    )
    model._pirls_eval_mu_start_ = (
        None if start_mu is None else np.asarray(start_mu, dtype=np.float64).copy()
    )
    model._pirls_lock_start_ = True
    try:
        model.family.theta = float(np.exp(float(log_theta_seed)))
        sp = model._expand_smoothing_params_from_log(log_sp)
        sol = model._solve_pirls_given_smoothing(y, sp)
        score = _pirls_ml_reml_objective_from_solution(model, y, sol, sp, method)
        theta_fit = float(
            np.log(max(float(getattr(model.family, "theta", 1.0)), 1e-12))
        )
        sol_state = _pirls_state_from_solution(sol)
        sol_state["theta"] = float(np.exp(theta_fit))
        return float(score), sol, theta_fit, sol_state
    finally:
        model._pirls_eval_start_ = None
        model._pirls_eval_eta_start_ = None
        model._pirls_eval_mu_start_ = None
        model._pirls_lock_start_ = False


def _optimize_joint_negbin_reml_efs(model, y, x0, bounds, free_mask, method):
    x = np.asarray(x0, dtype=np.float64).copy()
    free_idx = np.flatnonzero(np.asarray(free_mask, dtype=bool))
    if free_idx.size != x.size or free_idx.size == 0:
        return None

    branch_m = "LAML" if method == "laml" else "REML"
    log_theta = float(np.log(max(float(getattr(model.family, "theta", 1.0)), 1e-6)))
    log_theta_init = float(log_theta)
    mult = 1.0
    score_hist: list[float] = []
    log_theta_hist: list[float] = []
    x_hist: list[np.ndarray] = []
    n_eval = 0
    best_sol = current_sol = None
    best_state = current_state = {
        "coef": getattr(model, "_pirls_coef_start_", None),
        "eta": getattr(model, "_pirls_eta_start_", None),
        "mu": getattr(model, "_pirls_mu_start_", None),
        "theta": float(np.exp(log_theta)),
    }
    old_dev = None

    # mgcv::efsudr perturbs the working log(sp) upward before the first PIRLS
    # call (`lsp[spind] <- lsp[spind] + 2.5`). Mirror that initialization.
    for j, (lo, hi) in enumerate(bounds):
        x[j] = min(max(float(x[j] + 2.5), float(lo)), float(hi))

    current_score, current_sol, log_theta, current_state = (
        _evaluate_joint_negbin_efs_state(
            model, y, x, log_theta, branch_m, start_state=current_state
        )
    )
    n_eval += 1
    if not np.isfinite(current_score):
        return None
    _set_model_pirls_start_state(model, current_state)
    best_sol = current_sol
    best_state = current_state

    for it in range(1, 201):
        sp = model._expand_smoothing_params_from_log(x)
        if np.any(~np.isfinite(sp)) or np.any(sp <= 0.0):
            break

        a, b_sb, _ = _joint_negbin_efs_update_terms(model, current_sol, sp)
        phi = float(current_sol["scale"])
        with np.errstate(divide="ignore", invalid="ignore"):
            r = a / np.maximum(b_sb, LOG_GUARD_MIN) * phi
        same_zero = (a == 0.0) & (b_sb == 0.0)
        r[same_zero] = 1.0
        r[~np.isfinite(r)] = 1e6
        r = np.maximum(r, LOG_GUARD_MIN)

        delta = np.log(r[free_idx]) * mult
        x1 = np.asarray(x, dtype=np.float64).copy()
        for j, (lo, hi) in enumerate(bounds):
            x1[j] = min(max(float(x[j] + delta[j]), float(lo)), float(hi))
        max_step = float(np.max(np.abs(x1 - x))) if x1.size else 0.0
        old_score = float(current_score)

        cand_score, cand_sol, cand_theta, cand_state = _evaluate_joint_negbin_efs_state(
            model, y, x1, log_theta, branch_m, start_state=current_state
        )
        n_eval += 1
        accepted_x = x
        accepted_score = old_score
        accepted_sol = current_sol
        accepted_theta = log_theta
        accepted_state = current_state

        if np.isfinite(cand_score) and cand_score <= old_score:
            accepted_x = x1
            accepted_score = cand_score
            accepted_sol = cand_sol
            accepted_theta = cand_theta
            accepted_state = cand_state
            if max_step < 0.05:
                x2 = np.asarray(x, dtype=np.float64).copy()
                for j, (lo, hi) in enumerate(bounds):
                    x2[j] = min(
                        max(
                            float(x[j] + np.log(r[free_idx][j]) * mult * 2.0), float(lo)
                        ),
                        float(hi),
                    )
                ext_score, ext_sol, ext_theta, ext_state = (
                    _evaluate_joint_negbin_efs_state(
                        model, y, x2, log_theta, branch_m, start_state=current_state
                    )
                )
                n_eval += 1
                if np.isfinite(ext_score) and ext_score < accepted_score:
                    accepted_x = x2
                    accepted_score = ext_score
                    accepted_sol = ext_sol
                    accepted_theta = ext_theta
                    accepted_state = ext_state
                    mult *= 2.0
        else:
            while (
                not np.isfinite(cand_score) or cand_score > old_score
            ) and mult > 1.0:
                mult /= 2.0
                x1 = np.asarray(x, dtype=np.float64).copy()
                for j, (lo, hi) in enumerate(bounds):
                    x1[j] = min(
                        max(float(x[j] + np.log(r[free_idx][j]) * mult), float(lo)),
                        float(hi),
                    )
                max_step = float(np.max(np.abs(x1 - x))) if x1.size else 0.0
                cand_score, cand_sol, cand_theta, cand_state = (
                    _evaluate_joint_negbin_efs_state(
                        model, y, x1, log_theta, branch_m, start_state=current_state
                    )
                )
                n_eval += 1
            if np.isfinite(cand_score):
                accepted_x = x1
                accepted_score = cand_score
                accepted_sol = cand_sol
                accepted_theta = cand_theta
                accepted_state = cand_state
            if mult < 1.0:
                mult = 1.0

        x = np.asarray(accepted_x, dtype=np.float64)
        current_score = float(accepted_score)
        current_sol = accepted_sol
        log_theta = float(accepted_theta)
        current_state = accepted_state
        _set_model_pirls_start_state(model, current_state)
        best_sol = current_sol
        best_state = current_state
        score_hist.append(current_score)
        log_theta_hist.append(float(log_theta))
        x_hist.append(np.asarray(x, dtype=np.float64).copy())

        dev = float(current_sol["deviance"])
        if (
            it > 3
            and max_step < 0.05
            and max(abs(score_hist[-k] - score_hist[-k - 1]) for k in range(1, 4))
            < 1e-7
        ):
            break
        if old_dev is not None and abs(old_dev - dev) < 100.0 * np.finfo(
            np.float64
        ).eps * abs(dev):
            break
        old_dev = dev

    if best_sol is None:
        return None

    result = OptimizeResult()
    result.x = x.copy()
    result.fun = float(current_score)
    _set_model_pirls_start_state(model, best_state)
    model.family.theta = float(np.exp(log_theta))
    result.jac = np.asarray(
        criterion_gradient_ml_reml_pirls_exact(model, y, result.x, branch_m),
        dtype=np.float64,
    )
    result.hess = np.asarray(
        criterion_hessian_ml_reml_pirls_exact(model, y, result.x, branch_m),
        dtype=np.float64,
    )
    result.success = True
    result.status = 0
    result.message = (
        "iteration limit reached" if len(score_hist) >= 200 else "full convergence"
    )
    result.nit = int(len(score_hist))
    result.nfev = int(n_eval)
    result.njev = 1
    result.nhev = 1
    result.joint_negbin_reml_outer = True
    result.joint_negbin_efs_outer = True
    result.joint_negbin_initial_log_theta = log_theta_init
    result.joint_log_theta = float(log_theta)
    result.joint_negbin_message = str(result.message)
    result.joint_negbin_fun = float(result.fun)
    result.joint_negbin_nfev = int(result.nfev)
    result.joint_negbin_njev = int(result.njev)
    result.joint_negbin_selected_x = np.asarray(result.x, dtype=np.float64).copy()
    result.outer_info = {
        "iter": int(len(score_hist)),
        "score_hist": list(score_hist),
        "log_theta_hist": list(log_theta_hist),
        "log_sp_hist": [np.asarray(v, dtype=np.float64).tolist() for v in x_hist],
    }
    result.joint_negbin_state = best_state
    return result


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

    if model.family.known_scale is not None and getattr(
        model.family, "supports_ubre", False
    ):
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


def optimize_smoothing_params(
    model, y, initial_smoothing_params=None, method="gcv", optimizer="lbfgsb"
):
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
    use_hessian = optimizer == "outer_newton" and supports_criterion_hessian(
        model, method
    )

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
    family_name = str(getattr(model.family, "name", "")).lower()
    use_joint_gamma_reml_scale = (
        family_name == "gamma"
        and method in {"reml", "laml"}
        and ml_reml_backend == "pirls_laplace"
    )
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
            model._criterion(y, np.empty((0,), dtype=np.float64), method=method)
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
            (not bool(getattr(model.family, "supports_closed_form_solve", False)))
            or ml_reml_backend == "gaussian_dynamic"
            or has_factor_smooth_fs
        )
        if use_design_balance_init:
            if use_joint_gamma_reml_scale:
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
        raise ValueError("Initial free smoothing parameters must be finite and > 0.")

    min_sp = (
        np.zeros(model.n_smoothing_params_, dtype=np.float64)
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

    model._gaussian_reml_sigma2_opt_ = None
    # Gaussian REML/LAML uses a joint (log sp, log sigma^2) outer loop in mgcv's
    # reported objective (`gcv.ubre`). Using that same geometry for both exact and
    # dynamic Gaussian backends removes the last optimizer-level discrepancy in
    # machine-precision parity cases such as `tp(..., pc=...)`.
    use_joint_gaussian_reml_scale = (
        exact_gaussian
        and method in {"reml", "laml"}
        and ml_reml_backend in {"gaussian_exact", "gaussian_dynamic"}
    )

    if use_joint_gaussian_reml_scale:
        sp0 = expand_smoothing_params_from_log(model, x0)
        sol0 = model._solve_gaussian_given_smoothing(y, sp0)
        F0 = float(sol0["rss"]) + float(sol0["penalty_quadratic"] or 0.0)
        Mp = float(
            _static_penalty_null_dim(model)
            + _coef_column_offset(model)
        )
        nu0 = float(model.n_samples_ - Mp)
        if not np.isfinite(nu0) or nu0 <= 0.0:
            log_s2_0 = np.log(LOG_GUARD_MIN)
        else:
            log_s2_0 = float(np.log(max(F0 / nu0, LOG_GUARD_MIN)))
        x_joint0 = np.concatenate(
            [
                np.asarray(x0, dtype=np.float64).ravel(),
                np.array([log_s2_0], dtype=np.float64),
            ]
        )
        y_eff = (
            np.asarray(y, dtype=np.float64).ravel()
            if model.offset_train_ is None
            else (np.asarray(y, dtype=np.float64).ravel() - model.offset_train_)
        )
        yv = (
            float(np.var(y_eff))
            if y_eff.size > 1
            else float(np.maximum(np.abs(float(y_eff[0])), LOG_GUARD_MIN))
        )
        hi_scale = max(yv * 1e8, max(F0 / max(nu0, LOG_GUARD_MIN), LOG_GUARD_MIN) * 1e8, 1e-30)
        joint_bounds = list(bounds) + [(float(np.log(LOG_GUARD_MIN)), float(np.log(hi_scale)))]
        branch_m = "LAML" if method == "laml" else "REML"
        j_obj = _JointGaussianRemlObjective(model, y, branch_m, str(ml_reml_backend))
        callback_state = {"last_x": np.asarray(x_joint0, dtype=np.float64).copy()}

        def _joint_callback(xk):
            xk = np.asarray(xk, dtype=np.float64).ravel()
            prev = np.asarray(callback_state["last_x"], dtype=np.float64).ravel()
            step_norm = float(np.linalg.norm(xk - prev))
            j_obj.record_iter(xk, step_norm)
            callback_state["last_x"] = xk.copy()

        # Provide a local finite-difference `jac` even for `gaussian_exact` so
        # SciPy does not invoke its internal `_numdiff` path on ill-scaled joint
        # (log sp, log sigma^2) probes.
        use_jac = True
        joint_options = (
            {"maxfun": 50000, "ftol": 1e-14, "gtol": 1e-14}
            if str(ml_reml_backend) == "gaussian_exact"
            else {"maxfun": 50000, "ftol": 1e-14, "gtol": 1e-13}
        )
        result_joint = minimize(
            fun=j_obj.fun,
            x0=x_joint0,
            method="L-BFGS-B",
            jac=j_obj.jac if use_jac else None,
            bounds=joint_bounds,
            callback=_joint_callback,
            options=joint_options,
        )
        if str(ml_reml_backend) == "gaussian_dynamic" and np.isfinite(
            float(getattr(result_joint, "fun", np.nan))
        ):
            joint_polish = minimize(
                fun=j_obj.fun,
                x0=np.asarray(result_joint.x, dtype=np.float64),
                method="L-BFGS-B",
                jac=j_obj.jac if use_jac else None,
                bounds=joint_bounds,
                callback=_joint_callback,
                options={"maxfun": 50000, "ftol": 1e-15, "gtol": 1e-14},
            )
            if joint_polish.success or (
                np.isfinite(float(getattr(joint_polish, "fun", np.nan)))
                and float(joint_polish.fun) <= float(result_joint.fun)
            ):
                result_joint = joint_polish
        sigma2_bounds = joint_bounds[-1]
        has_random_effect_term = any(
            str(getattr(tb, "term_type", "")).lower() == "random_effect"
            for tb in _term_blocks_seq(model)
        )
        if str(ml_reml_backend) == "gaussian_dynamic" and has_random_effect_term:
            x_joint = np.asarray(result_joint.x, dtype=np.float64).ravel()
            x_sp_cur = np.asarray(x_joint[:-1], dtype=np.float64).ravel()
            if x_sp_cur.size > 0 and np.any(x_sp_cur < -20.0):
                x_sp_snap = x_sp_cur.copy()
                for j, (lo, _hi) in enumerate(bounds):
                    if x_sp_snap[j] < -20.0:
                        x_sp_snap[j] = max(float(lo), -64.0)

                def _sigma2_obj_dynamic(log_sigma2_scalar: float):
                    return float(
                        criterion_ml_reml_gaussian_dynamic_joint(
                            model,
                            y,
                            x_sp_snap,
                            float(log_sigma2_scalar),
                            method=branch_m,
                        )
                    )

                sigma2_res = minimize_scalar(
                    _sigma2_obj_dynamic,
                    bounds=sigma2_bounds,
                    method="bounded",
                    options={"xatol": 1e-10, "maxiter": 200},
                )
                if bool(sigma2_res.success) and np.isfinite(float(sigma2_res.fun)):
                    result_joint.x = np.concatenate(
                        [x_sp_snap, np.array([float(sigma2_res.x)], dtype=np.float64)]
                    )
                    result_joint.fun = float(sigma2_res.fun)
                    result_joint.success = True
                    result_joint.message = "Snapped Gaussian random-effect smoothing parameter to the lower boundary."
        if str(ml_reml_backend) == "gaussian_exact" and n_free == 1:

            def _refine_sigma2_for_log_sp(log_sp_scalar: float):
                def _sigma2_obj(log_sigma2_scalar: float):
                    return float(
                        criterion_ml_reml_gaussian_dynamic_joint(
                            model,
                            y,
                            np.array([float(log_sp_scalar)], dtype=np.float64),
                            float(log_sigma2_scalar),
                            method=branch_m,
                        )
                    )

                sigma2_res = minimize_scalar(
                    _sigma2_obj,
                    bounds=sigma2_bounds,
                    method="bounded",
                    options={"xatol": 1e-10, "maxiter": 200},
                )
                return float(sigma2_res.fun), float(sigma2_res.x)

            def _outer_obj(log_sp_scalar: float):
                return _refine_sigma2_for_log_sp(float(log_sp_scalar))[0]

            scalar_res = minimize_scalar(
                _outer_obj,
                bounds=bounds[0],
                method="bounded",
                options={"xatol": 1e-10, "maxiter": 200},
            )
            if bool(scalar_res.success) and np.isfinite(float(scalar_res.fun)):
                refined_fun, refined_log_s2 = _refine_sigma2_for_log_sp(
                    float(scalar_res.x)
                )
                if refined_fun <= float(result_joint.fun) + 1e-12:
                    result_joint.x = np.array(
                        [float(scalar_res.x), float(refined_log_s2)],
                        dtype=np.float64,
                    )
                    result_joint.fun = float(refined_fun)
                    result_joint.success = True
                    result_joint.message = "Refined exact Gaussian REML joint optimum with nested scalar search."
        joint_dim = int(n_free + 1)
        if (not bool(result_joint.success)) and np.isfinite(
            float(getattr(result_joint, "fun", np.nan))
        ):
            msg_u = str(getattr(result_joint, "message", "")).upper()
            if "ABNORMAL" in msg_u and joint_dim <= 32:
                result_joint.success = True
                result_joint.message = "Accepted L-BFGS-B ABNORMAL termination on joint Gaussian REML outer problem."

        x_sp = np.asarray(result_joint.x[:-1], dtype=np.float64).ravel()
        log_s2_opt = float(result_joint.x[-1])

        factor_smooth_shared_ridge_stabilized = False
        factor_smooth_shared_ridge_shift = None
        if ml_reml_backend == "gaussian_exact":

            def _joint_exact_refine_sigma2(x_sp_vec):
                def _sigma2_obj_exact(log_sigma2_scalar: float):
                    return float(
                        criterion_ml_reml_gaussian_dynamic_joint(
                            model,
                            y,
                            np.asarray(x_sp_vec, dtype=np.float64),
                            float(log_sigma2_scalar),
                            method=branch_m,
                        )
                    )

                sigma2_res = minimize_scalar(
                    _sigma2_obj_exact,
                    bounds=sigma2_bounds,
                    method="bounded",
                    options={"xatol": 1e-10, "maxiter": 200},
                )
                return (
                    float(sigma2_res.fun),
                    float(sigma2_res.x),
                    bool(sigma2_res.success),
                )

            # mgcv's default Gaussian REML path uses outer Newton with an explicit
            # joint (log sp, log sigma^2) parameterization. After the joint L-BFGS-B
            # solve, profile sigma^2 and coordinate-polish log sp to tighten the
            # endpoint on small multi-smoothing problems such as factor-by smooths.
            x_sp_work = x_sp.copy()
            log_s2_work = float(log_s2_opt)
            score_work = float(result_joint.fun)
            score_tol = max(1e-12, (1.0 + abs(score_work)) * 1e-12)
            improved_exact = False

            for _ in range(3):
                improved_pass = False
                for j, (lo, hi) in enumerate(bounds):
                    lo = float(lo)
                    hi = float(hi)
                    if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
                        continue

                    def _profiled_obj(log_sp_scalar: float, j=j, x_sp_work=x_sp_work):
                        trial = x_sp_work.copy()
                        trial[j] = float(log_sp_scalar)
                        trial_fun, _trial_log_s2, trial_ok = _joint_exact_refine_sigma2(
                            trial
                        )
                        if not trial_ok:
                            return np.inf
                        return float(trial_fun)

                    opt = minimize_scalar(
                        _profiled_obj,
                        bounds=(lo, hi),
                        method="bounded",
                        options={"xatol": 1e-10, "maxiter": 200},
                    )
                    if not bool(getattr(opt, "success", False)) or not np.isfinite(
                        getattr(opt, "fun", np.nan)
                    ):
                        continue

                    if float(opt.fun) + score_tol < score_work:
                        trial = x_sp_work.copy()
                        trial[j] = float(opt.x)
                        trial_fun, trial_log_s2, trial_ok = _joint_exact_refine_sigma2(
                            trial
                        )
                        if (
                            trial_ok
                            and np.isfinite(trial_fun)
                            and trial_fun + score_tol < score_work
                        ):
                            x_sp_work = trial
                            log_s2_work = float(trial_log_s2)
                            score_work = float(trial_fun)
                            improved_pass = True
                            improved_exact = True
                if not improved_pass:
                    break

            if improved_exact:
                x_sp = x_sp_work
                log_s2_opt = float(log_s2_work)
                result_joint.x = np.concatenate(
                    [
                        np.asarray(x_sp, dtype=np.float64),
                        np.array([log_s2_opt], dtype=np.float64),
                    ]
                )
                result_joint.fun = float(score_work)
                result_joint.success = True
                result_joint.message = "Refined exact Gaussian REML joint optimum with profiled coordinate search."

            full_to_free = {
                int(full): int(i) for i, full in enumerate(np.flatnonzero(free_mask))
            }
            fs_groups = []
            for tb in _term_blocks_seq(model):
                if str(getattr(tb, "term_type", "")).lower() != "factor_smooth_fs":
                    continue
                group = sorted(
                    {
                        int(pb.smoothing_index)
                        for pb in (getattr(model, "penalty_blocks_", None) or ())
                        if pb.coef_slice == tb.coef_slice
                    }
                )
                group_free = [full_to_free[g] for g in group if g in full_to_free]
                if group_free:
                    fs_groups.append(group_free)

            if fs_groups:
                score_tol = max(2e-5, (1.0 + abs(float(result_joint.fun))) * 1e-7)
                x_sp_work = x_sp.copy()
                log_s2_work = float(log_s2_opt)
                score_work = float(result_joint.fun)
                improved_fs_ridge = False
                fs_shift_by_group = []

                for group in fs_groups:
                    local_best = x_sp_work.copy()
                    local_best_log_s2 = float(log_s2_work)
                    local_best_score = float(score_work)
                    log_step = 0.25
                    max_shift = 4.0

                    for direction in (-1.0, 1.0):
                        local = x_sp_work.copy()
                        total_shift = 0.0

                        while total_shift + log_step <= max_shift + 1e-12:
                            trial = local.copy()
                            stop = False
                            for j in group:
                                bound = (
                                    float(bounds[j][0])
                                    if direction < 0.0
                                    else float(bounds[j][1])
                                )
                                trial[j] = trial[j] + direction * log_step
                                if direction < 0.0:
                                    trial[j] = max(bound, trial[j])
                                    if trial[j] >= local[j] - 1e-12:
                                        stop = True
                                else:
                                    trial[j] = min(bound, trial[j])
                                    if trial[j] <= local[j] + 1e-12:
                                        stop = True
                            if stop:
                                break

                            trial_fun, trial_log_s2, trial_ok = (
                                _joint_exact_refine_sigma2(trial)
                            )
                            if (
                                (not trial_ok)
                                or (not np.isfinite(trial_fun))
                                or trial_fun > float(result_joint.fun) + score_tol
                            ):
                                break

                            local = trial
                            total_shift += log_step
                            if (
                                trial_fun + 1e-12 < local_best_score
                                or (
                                    abs(trial_fun - local_best_score) <= score_tol
                                    and float(np.mean(trial[group]))
                                    < float(np.mean(local_best[group])) - 1e-12
                                )
                            ):
                                local_best = trial.copy()
                                local_best_log_s2 = float(trial_log_s2)
                                local_best_score = float(trial_fun)

                    shift_vec = (local_best - x_sp_work)[group]
                    if np.any(np.abs(shift_vec) > 1e-12):
                        x_sp_work = local_best
                        log_s2_work = local_best_log_s2
                        score_work = local_best_score
                        improved_fs_ridge = True
                        fs_shift_by_group.append(
                            {
                                "free_indices": [int(j) for j in group],
                                "log_sp_shift": [float(v) for v in shift_vec],
                            }
                        )

                if improved_fs_ridge:
                    x_sp = x_sp_work
                    log_s2_opt = float(log_s2_work)
                    result_joint.x = np.concatenate(
                        [
                            np.asarray(x_sp, dtype=np.float64),
                            np.array([log_s2_opt], dtype=np.float64),
                        ]
                    )
                    result_joint.fun = float(score_work)
                    result_joint.factor_smooth_shared_ridge_stabilized = True
                    result_joint.factor_smooth_shared_ridge_shift = fs_shift_by_group
                    factor_smooth_shared_ridge_stabilized = True
                    factor_smooth_shared_ridge_shift = fs_shift_by_group

            if not factor_smooth_shared_ridge_stabilized:
                profiled_objective = _CriterionObjective(
                    model, y, method=method, use_gradient=True
                )
                profiled_newton = _optimize_outer_newton(
                    objective=profiled_objective,
                    x0=np.asarray(x_sp, dtype=np.float64),
                    bounds=bounds,
                )
                profiled_fun = float(getattr(profiled_newton, "fun", np.nan))
                if bool(getattr(profiled_newton, "success", False)) and np.isfinite(
                    profiled_fun
                ):
                    trial_x_sp = np.asarray(profiled_newton.x, dtype=np.float64).ravel()
                    trial_fun, trial_log_s2, trial_ok = _joint_exact_refine_sigma2(
                        trial_x_sp
                    )
                    if trial_ok and np.isfinite(trial_fun):
                        current_grad = np.asarray(
                            profiled_objective.jac(np.asarray(x_sp, dtype=np.float64)),
                            dtype=np.float64,
                        ).ravel()
                        trial_grad = np.asarray(
                            profiled_objective.jac(trial_x_sp), dtype=np.float64
                        ).ravel()
                        current_grad_norm = (
                            float(np.max(np.abs(current_grad)))
                            if current_grad.size
                            else 0.0
                        )
                        trial_grad_norm = (
                            float(np.max(np.abs(trial_grad)))
                            if trial_grad.size
                            else 0.0
                        )
                        if (
                            trial_fun <= float(result_joint.fun) + 1e-10
                            and trial_grad_norm + 1e-10 < current_grad_norm
                        ):
                            x_sp = trial_x_sp
                            log_s2_opt = float(trial_log_s2)
                            result_joint.x = np.concatenate(
                                [
                                    np.asarray(x_sp, dtype=np.float64),
                                    np.array([log_s2_opt], dtype=np.float64),
                                ]
                            )
                            result_joint.fun = float(trial_fun)
                            result_joint.success = True
                            result_joint.message = "Refined exact Gaussian REML joint optimum with profiled outer Newton."

        model.smoothing_params = np.asarray(
            model.smoothing_params, dtype=np.float64
        ).copy()
        model.smoothing_params[free_mask] = np.exp(x_sp)
        model.smoothing_params = np.maximum(model.smoothing_params, min_sp)
        model._gaussian_reml_sigma2_opt_ = float(np.exp(log_s2_opt))

        x_full_opt = np.concatenate(
            [
                np.asarray(x_sp, dtype=np.float64).ravel(),
                np.array([log_s2_opt], dtype=np.float64),
            ]
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
        result.joint_gaussian_reml_outer = True
        result.joint_log_sigma2 = float(log_s2_opt)
        if factor_smooth_shared_ridge_stabilized:
            result.factor_smooth_shared_ridge_stabilized = True
            result.factor_smooth_shared_ridge_shift = factor_smooth_shared_ridge_shift

        model._optim_method = method
        model._optim_result = result
        trace_grad = None
        if g_full is not None:
            trace_grad = np.asarray(g_full, dtype=np.float64).tolist()
        final_joint_x = np.asarray(result_joint.x, dtype=np.float64).ravel()
        if len(j_obj.accepted_trace) == 0 or not np.array_equal(
            np.asarray(j_obj.accepted_trace[-1]["x"], dtype=np.float64).ravel(),
            final_joint_x,
        ):
            step_norm = float(
                np.linalg.norm(
                    final_joint_x
                    - np.asarray(callback_state["last_x"], dtype=np.float64).ravel()
                )
            )
            j_obj.record_iter(final_joint_x, step_norm)
        model._optim_trace = []
        for i, row in enumerate(j_obj.accepted_trace):
            x_row = np.asarray(row["x"], dtype=np.float64).ravel()
            model._optim_trace.append(
                {
                    "iter": int(i + 1),
                    "log_sp": np.asarray(x_row[:-1], dtype=np.float64).tolist(),
                    "criterion": float(row["fun"]),
                    "gradient": (
                        trace_grad if i == len(j_obj.accepted_trace) - 1 else None
                    ),
                    "hessian": None,
                    "accepted_step_norm": float(row.get("accepted_step_norm", 0.0)),
                    "rank_info": {
                        "joint_gaussian_reml_outer": True,
                        "factor_smooth_shared_ridge_stabilized": (
                            factor_smooth_shared_ridge_stabilized
                        ),
                        "factor_smooth_shared_ridge_shift": factor_smooth_shared_ridge_shift,
                    },
                }
            )
        model._optim_used_gradient = True
        model._optim_used_hessian = False
        model.smoothing_score_ = float(result_joint.fun)

        if not result_joint.success:
            warnings.warn(
                f"Smoothing optimisation did not converge: {result_joint.message}",
                stacklevel=2,
            )
        return model

    objective = _CriterionObjective(model, y, method=method, use_gradient=use_gradient)
    if bool(getattr(model.family, "supports_pirls", False)):
        # Carry P-IRLS coefficient warm-starts between outer criterion evaluations.
        model._pirls_coef_start_ = None
        model._pirls_eta_start_ = None
        model._pirls_mu_start_ = None
    indefinite_hessian_newton_for_pirls = (
        method in {"ml", "reml", "laml"}
        and bool(getattr(model.family, "supports_pirls", False))
        and not bool(getattr(model.family, "supports_closed_form_solve", False))
    )

    if optimizer == "lbfgsb":
        if indefinite_hessian_newton_for_pirls and supports_criterion_hessian(
            model, method
        ):
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
                    options={"maxfun": 25000, "ftol": 1e-13, "gtol": 1e-12},
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
                options={"maxfun": 25000, "ftol": 1e-13, "gtol": 1e-12},
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
        if indefinite_hessian_newton_for_pirls and supports_criterion_hessian(
            model, method
        ):
            result = _optimize_outer_newton_indefinite_hessian(
                objective=objective,
                x0=x0,
                bounds=bounds,
            )
            result.indefinite_hessian_outer_newton = True
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
                options={"maxfun": 25000, "ftol": 1e-13, "gtol": 1e-12},
            )
            lbfgsb_result.outer_newton_fallback = True
            lbfgsb_result.outer_newton_message = str(result.message)
            result = lbfgsb_result

    has_null_space_penalty = any(
        bool(getattr(pb, "is_null_space_penalty", False))
        for pb in (getattr(model, "penalty_blocks_", None) or [])
    )
    apply_generic_pirls_rollback = method in {"ml", "reml", "laml"} and (
        (not exact_gaussian) and (not model._has_tensor_terms())
    )
    if apply_generic_pirls_rollback:
        result = _rollback_working_infinite_smoothing_params(
            objective=objective,
            result=result,
            x0=x0,
            bounds=bounds,
            method=method,
        )
    if has_null_space_penalty:
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
    result = _stabilize_factor_smooth_shared_ridge(
        objective=objective,
        result=result,
        bounds=bounds,
        method=method,
    )
    result = _snap_gaussian_random_effect_boundary(
        objective=objective,
        result=result,
        bounds=bounds,
        method=method,
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
    result = _accept_stationary_abnormal_result(
        objective=objective,
        result=result,
    )

    if use_joint_gamma_reml_scale:
        branch_m = "LAML" if method == "laml" else "REML"
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
                conv_tol=1e-7,
            )
            lbfgsb_retry = minimize(
                fun=j_obj.fun,
                x0=np.asarray(result_joint.x, dtype=np.float64),
                method="L-BFGS-B",
                jac=j_obj.jac,
                bounds=joint_bounds,
                options={"maxfun": 25000, "ftol": 1e-11, "gtol": 1e-10},
            )
            if lbfgsb_retry.success or (
                np.isfinite(getattr(lbfgsb_retry, "fun", np.inf))
                and float(lbfgsb_retry.fun) <= float(result_joint.fun)
            ):
                result_joint = lbfgsb_retry

            if np.isfinite(float(getattr(result_joint, "fun", np.inf))):
                x_joint = np.asarray(result_joint.x, dtype=np.float64).ravel()
                x_selected = np.asarray(x_joint[:-1], dtype=np.float64).ravel()
                grad_joint = np.asarray(
                    (
                        result_joint.jac
                        if getattr(result_joint, "jac", None) is not None
                        else j_obj.jac(x_joint)
                    ),
                    dtype=np.float64,
                )
                hess_joint = np.asarray(
                    (
                        result_joint.hess
                        if getattr(result_joint, "hess", None) is not None
                        else j_obj.hess(x_joint)
                    ),
                    dtype=np.float64,
                )
                if (
                    grad_joint.ndim == 1
                    and hess_joint.ndim == 2
                    and grad_joint.size == x_joint.size
                    and hess_joint.shape == (x_joint.size, x_joint.size)
                ):
                    grad2 = np.diag(hess_joint)
                    flat = np.where(
                        np.abs(grad2[:n_free]) < np.abs(grad_joint[:n_free]) * 100.0
                    )[0]
                    if flat.size > 0:
                        target = float(result_joint.fun) + 0.02
                        x_edge = x_joint.copy()
                        for j in flat.tolist():
                            step_j = 1.0 if x0[j] > x_edge[j] else -1.0
                            x_try = x_edge.copy()
                            x_try[j] = min(
                                max(x_try[j] + step_j, joint_bounds[j][0]),
                                joint_bounds[j][1],
                            )
                            if abs(x_try[j] - x_edge[j]) <= 1e-12:
                                continue
                            score_try = float(j_obj.fun(x_try))
                            if np.isfinite(score_try) and score_try < target:
                                x_edge = x_try
                        x_selected = np.asarray(x_edge[:-1], dtype=np.float64).ravel()
                        x_joint = x_edge
                _ = criterion_hessian_ml_reml_pirls_exact(
                    model, y, x_selected, branch_m
                )
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
                result.x = np.asarray(x_selected, dtype=np.float64).copy()
                result.fun = float(objective.fun(result.x))
                result.jac = np.asarray(objective.jac(result.x), dtype=np.float64)
                result.hess = np.asarray(objective.hess(result.x), dtype=np.float64)
                result.joint_gamma_reml_outer = True
                result.joint_log_phi = float(x_joint[-1])
                result.joint_gamma_message = str(getattr(result_joint, "message", ""))

    if use_joint_negbin_reml_theta:
        branch_m = "LAML" if method == "laml" else "REML"
        theta0 = float(max(getattr(model.family, "theta", 1.0), 1e-6))
        log_theta0 = float(np.log(theta0))
        theta_bounds = [(-12.0, 12.0)]
        joint_bounds = list(bounds) + theta_bounds
        x_joint0 = np.concatenate([x0.copy(), np.array([log_theta0], dtype=np.float64)])
        result_joint_nb_init = _optimize_joint_negbin_reml_efs(
            model=model,
            y=y,
            x0=x0,
            bounds=bounds,
            free_mask=free_mask,
            method=method,
        )
        if result_joint_nb_init is not None:
            x_joint0 = np.concatenate(
                [
                    np.asarray(result_joint_nb_init.x, dtype=np.float64).ravel(),
                    np.array(
                        [
                            float(
                                getattr(
                                    result_joint_nb_init, "joint_log_theta", log_theta0
                                )
                            )
                        ],
                        dtype=np.float64,
                    ),
                ]
            )
        j_obj = _JointNegbinPirlsRemlObjective(model, y, branch_m)
        result_joint_nb = minimize(
            fun=j_obj.fun,
            x0=x_joint0,
            method="L-BFGS-B",
            jac=j_obj.jac,
            bounds=joint_bounds,
            options={"maxfun": 25000, "ftol": 1e-11, "gtol": 1e-10},
        )
        if not bool(getattr(result_joint_nb, "success", False)) and not np.isfinite(
            getattr(result_joint_nb, "fun", np.nan)
        ):
            raise RuntimeError(
                "Negative binomial joint REML outer smoothing optimization failed."
            )
        result_joint_nb = _coordinate_refine_smoothing_params(
            objective=j_obj,
            result=result_joint_nb,
            bounds=joint_bounds,
            improve_tol=1e-4,
        )
        x_joint = np.asarray(result_joint_nb.x, dtype=np.float64).ravel()
        x_selected_nb = np.asarray(x_joint[:-1], dtype=np.float64).ravel()
        log_theta_opt = float(x_joint[-1])

        model.family.theta = float(np.exp(log_theta_opt))
        model._pirls_disable_theta_efs_ = True
        result.x = np.asarray(x_selected_nb, dtype=np.float64).copy()
        result.fun = float(getattr(result_joint_nb, "fun", np.nan))
        jac_joint = getattr(result_joint_nb, "jac", None)
        if jac_joint is not None:
            jac_joint = np.asarray(jac_joint, dtype=np.float64).ravel()
            result.jac = jac_joint[:-1].copy()
        else:
            result.jac = None
        result.hess = None
        result.joint_negbin_reml_outer = True
        result.joint_negbin_efs_outer = False
        result.joint_negbin_initial_log_theta = log_theta0
        result.joint_log_theta = log_theta_opt
        result.joint_negbin_message = str(getattr(result_joint_nb, "message", ""))
        result.joint_negbin_fun = float(getattr(result_joint_nb, "fun", np.nan))
        result.joint_negbin_nfev = int(getattr(result_joint_nb, "nfev", 0))
        result.joint_negbin_njev = int(getattr(result_joint_nb, "njev", 0))
        result.joint_negbin_selected_x = np.asarray(result.x, dtype=np.float64).copy()
        result.message = str(getattr(result_joint_nb, "message", ""))
        result.nfev = int(getattr(result_joint_nb, "nfev", 0))
        result.njev = int(getattr(result_joint_nb, "njev", 0))
        result.nit = int(getattr(result_joint_nb, "nit", 0))
        if getattr(j_obj, "trace", None):
            score_hist = []
            log_theta_hist = []
            log_sp_hist = []
            for row in j_obj.trace:
                x_row = np.asarray(row["x"], dtype=np.float64).ravel()
                if x_row.size != n_free + 1:
                    continue
                score = row.get("fun", None)
                if score is None or not np.isfinite(float(score)):
                    continue
                log_sp_hist.append(x_row[:-1].tolist())
                log_theta_hist.append(float(x_row[-1]))
                score_hist.append(float(score))
            result.outer_info = {
                "iter": int(len(score_hist)),
                "score_hist": score_hist,
                "log_theta_hist": log_theta_hist,
                "log_sp_hist": log_sp_hist,
            }

        if apply_generic_pirls_rollback:
            result = _rollback_working_infinite_smoothing_params(
                objective=objective,
                result=result,
                x0=x0,
                bounds=bounds,
                method=method,
            )
        result = _stabilize_joint_negbin_flat_ridge(
            objective=objective,
            result=result,
            bounds=bounds,
            score_tol=1.0e-4,
        )
        if has_null_space_penalty:
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
            result = _refine_null_space_smoothing_params(
                objective=objective,
                result=result,
                bounds=bounds,
            )
        result.joint_negbin_postprocessed = True

    if not result.success:
        warnings.warn(
            f"Smoothing optimisation did not converge: {result.message}",
            stacklevel=2,
        )

    model.smoothing_params = np.asarray(model.smoothing_params, dtype=np.float64).copy()
    model.smoothing_params[free_mask] = np.exp(result.x)
    model.smoothing_params = np.maximum(model.smoothing_params, min_sp)

    model._optim_method = method
    model._optim_result = result
    if bool(getattr(result, "joint_negbin_reml_outer", False)):
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
    if (
        not bool(getattr(result, "joint_negbin_reml_outer", False))
        and getattr(objective, "trace", None) is not None
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
    model.smoothing_score_ = float(result.fun)
    model._optim_used_gradient = bool(use_gradient)
    model._optim_used_hessian = bool(use_hessian)
    return model
