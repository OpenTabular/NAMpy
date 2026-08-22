"""Entry points: supports_*, expand_*, and optimize_smoothing_params."""

import warnings

import numpy as np
from scipy.optimize import OptimizeResult, minimize

from ...._mgcv_constants import LOG_GUARD_MIN
from ....families.family_base import JointOuterStrategy
from ....model_state import (
    _fit_workspace,
    _n_smoothing_params,
    _term_blocks_seq,
)
from ...capabilities import (
    coerce_general_family_smoothing_method,
    raise_ml_reml_backend_error,
)
from ..criteria import (
    criterion_gradient,
    criterion_hessian,
    criterion_hessian_ml_reml_pirls_exact,
    criterion_value,
    resolve_ml_reml_scoring_backend,
)
from ..criteria.gaussian_dyn import _gaussian_dynamic_reml_derivative_terms
from ..criteria.shape import is_transformed_coefficient_model
from .basics import (
    _initial_gaussian_scale_as_sp,
    _initial_smoothing_params_from_design,
    supports_criterion_gradient,
    supports_criterion_hessian,
)
from .bfgs_strict import _optimize_outer_bfgs_strict
from .efs_strict import _optimize_outer_efs_strict
from .newton import (
    optimize_outer_newton_generic,
    optimize_outer_newton_indefinite_hessian,
)
from .objectives import (
    _CriterionObjective,
    _GaussianPirlsRemlJointObjective,
    _GaussianRemlJointObjective,
    _GaussianRemlProfiledObjective,
    _JointBetarPirlsRemlObjective,
    _JointGammaPirlsRemlObjective,
    _JointNegbinPirlsRemlObjective,
    _JointOcatPirlsRemlObjective,
    _JointTweediePirlsRemlObjective,
)
from .shape_bfgs import optimize_transformed_bfgs
from .trace import _assemble_optim_trace


def _optimize_negbin_reml_joint_native(
    model, y, x0, free_mask, method, sp_bounds, *, optimizer
):
    """Native joint negbin ML/REML/LAML optimization over (log theta, log sp)."""
    method_lower = str(method).lower()
    if method_lower not in {"ml", "reml", "laml"}:
        return None
    if str(getattr(model.family, "name", "")).lower() != "negbin" or not bool(
        getattr(model.family, "estimate_theta", False)
    ):
        return None
    optimizer = str(optimizer).lower()
    if method_lower == "ml" and optimizer == "optim":
        # `mgcv/R/mgcv.r::gam.outer()` delegates this joint vector to R's
        # `stats::optim(..., method="L-BFGS-B")`.  SciPy's L-BFGS-B follows
        # the same early path but selects a materially different smoothing
        # parameter at the flat ML boundary, so do not expose approximate
        # parity until that exact R optimizer path is ported.
        raise NotImplementedError(
            "Negative-binomial ML with estimate_theta=True and optimizer='optim' "
            "is unsupported until the exact R stats::optim L-BFGS-B boundary "
            "behavior is ported. Use outer_newton or bfgs."
        )

    x0 = np.asarray(x0, dtype=np.float64).ravel()
    free_mask = np.asarray(free_mask, dtype=bool)
    free_count = int(np.sum(free_mask))
    if x0.size != free_count:
        return None

    theta0 = float(max(float(getattr(model.family, "theta", 1.0)), LOG_GUARD_MIN))
    log_theta0 = float(np.log(theta0))
    x_joint0 = np.concatenate([np.array([log_theta0], dtype=np.float64), x0])
    branch_m = "LAML" if method_lower == "laml" else method_lower.upper()
    j_obj = _JointNegbinPirlsRemlObjective(model, y, branch_m)
    if optimizer == "outer_newton":
        result_joint = optimize_outer_newton_indefinite_hessian(
            objective=j_obj,
            x0=x_joint0,
            bounds=[(float(np.log(LOG_GUARD_MIN)), np.inf)] + list(sp_bounds),
            conv_tol=1e-6,
        )
    elif optimizer == "bfgs":
        # `mgcv/R/mgcv.r::gam.outer()` sends the prepended transformed
        # theta and smoothing parameters through `mgcv::bfgs()` together.
        result_joint = _optimize_outer_bfgs_strict(
            objective=j_obj,
            x0=x_joint0,
            bounds=[(float(np.log(LOG_GUARD_MIN)), np.inf)] + list(sp_bounds),
            score_type=str(method).lower(),
        )
    elif optimizer == "optim":
        # `mgcv/R/mgcv.r::estimate.gam()` prepends transformed theta to
        # `lsp`; `gam.outer()` then passes the complete unbounded vector to
        # `optim(..., method="L-BFGS-B")`.
        result_joint = _optimize_outer_optim_strict(
            objective=j_obj,
            x0=x_joint0,
            bounds=[(-np.inf, np.inf)] + list(sp_bounds),
        )
    else:
        return None
    result = OptimizeResult()
    x_joint = np.asarray(result_joint.x, dtype=np.float64).ravel()
    if x_joint.size != x_joint0.size:
        x_joint = x_joint0.copy()
    log_theta_opt = float(x_joint[0]) if x_joint.size else log_theta0
    if np.isfinite(log_theta_opt):
        theta_opt = float(np.exp(log_theta_opt))
    else:
        theta_opt = float(theta0)
    if not np.isfinite(theta_opt) or theta_opt <= 0.0:
        theta_opt = float(theta0)
        log_theta_opt = float(log_theta0)

    x_sp_opt = x_joint[1:]
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
    result.optim_trace = getattr(result_joint, "optim_trace", None)
    result.selected_full_smoothing_params = np.asarray(
        model.smoothing_params, dtype=np.float64
    ).copy()
    result.selected_full_smoothing_params[free_mask] = np.exp(x_sp_opt)
    result.selected_theta = float(theta_opt)

    score_hist = []
    log_theta_hist = []
    log_sp_hist = []
    for row in j_obj.accepted_trace:
        x_row = np.asarray(
            row.get("x", np.array([], dtype=np.float64)), dtype=np.float64
        )
        if x_row.size == 0:
            continue
        log_theta_hist.append(float(x_row[0]))
        log_sp_hist.append(np.asarray(x_row[1:], dtype=np.float64).tolist())
        score_hist.append(float(row.get("fun", np.nan)))

    if len(score_hist) == 0:
        score_hist = [float(result.fun)]
        log_theta_hist = [float(log_theta_opt)]
        log_sp_hist = [np.asarray(x_sp_opt, dtype=np.float64).tolist()]

    outer_info = dict(getattr(result_joint, "outer_info", {}) or {})
    if optimizer != "bfgs":
        outer_info["score_hist"] = score_hist
    outer_info["log_theta_hist"] = log_theta_hist
    outer_info["log_sp_hist"] = log_sp_hist
    joint_grad = getattr(result_joint, "jac", None)
    if joint_grad is not None:
        joint_grad = np.asarray(joint_grad, dtype=np.float64).ravel()
        if joint_grad.shape == x_joint.shape and np.all(np.isfinite(joint_grad)):
            outer_info["grad"] = joint_grad.copy()
    joint_hess = getattr(result_joint, "hess", None)
    if joint_hess is not None:
        joint_hess = np.asarray(joint_hess, dtype=np.float64)
        if joint_hess.shape == (x_joint.size, x_joint.size) and np.all(
            np.isfinite(joint_hess)
        ):
            outer_info["hess"] = 0.5 * (joint_hess + joint_hess.T)
    result.outer_info = outer_info
    result.joint_negbin_reml_outer = True
    return result


def _refresh_final_outer_derivatives(
    model, y, method, result, objective=None, min_sp_free=None
):
    """Populate mgcv-style outer_info grad/hess at the selected free log-sp."""
    if result is None or getattr(result, "x", None) is None:
        return
    if str(method).lower() not in {"ml", "reml", "laml"}:
        return
    if bool(getattr(objective, "uses_joint_log_scale", False)):
        return
    if bool(getattr(result, "joint_negbin_reml_outer", False)) or bool(
        getattr(result, "joint_gamma_reml_outer", False)
    ):
        return

    x = np.asarray(result.x, dtype=np.float64).ravel()
    if x.size == 0:
        return

    # Under mgcv's min.sp H-offset reparameterization the outer coordinates
    # are log free multipliers; exact derivatives are evaluated at the total
    # log sp and chain-transformed back to the free coordinates.
    offset_active = min_sp_free is not None and bool(
        np.any(np.asarray(min_sp_free, dtype=np.float64) > 0.0)
    )
    if offset_active:
        min_sp_free = np.asarray(min_sp_free, dtype=np.float64).ravel()
        lam = min_sp_free + np.exp(x)
        x_eval = np.log(lam)
        j_chain = np.exp(x) / lam
    else:
        x_eval = x
        j_chain = None

    outer_info = dict(getattr(result, "outer_info", {}) or {})
    preserve_exact = bool(getattr(result, "strict_outer_derivatives", False))
    keep_grad = (
        preserve_exact
        and getattr(result, "jac", None) is not None
        and np.asarray(result.jac, dtype=np.float64).shape == x.shape
        and np.all(np.isfinite(np.asarray(result.jac, dtype=np.float64)))
    )
    keep_hess = (
        preserve_exact
        and getattr(result, "hess", None) is not None
        and np.asarray(result.hess, dtype=np.float64).shape == (x.size, x.size)
        and np.all(np.isfinite(np.asarray(result.hess, dtype=np.float64)))
    )
    if keep_grad:
        outer_info["grad"] = np.asarray(result.jac, dtype=np.float64).copy()
    if keep_hess:
        outer_info["hess"] = np.asarray(result.hess, dtype=np.float64).copy()
    if keep_grad and keep_hess:
        if outer_info:
            result.outer_info = outer_info
        return

    grad = None
    grad_eval = None
    if not keep_grad:
        try:
            grad_eval = np.asarray(
                criterion_gradient(model, y, x_eval, method=method),
                dtype=np.float64,
            )
            grad = (
                grad_eval * j_chain if offset_active else grad_eval
            )
        except Exception:
            grad = None
            grad_eval = None

    hess = None
    if not keep_hess:
        try:
            hess = np.asarray(
                criterion_hessian(model, y, x_eval, method=method),
                dtype=np.float64,
            )
            if offset_active:
                hess = hess * np.outer(j_chain, j_chain)
                if grad_eval is not None and grad_eval.shape == x.shape:
                    hess[np.diag_indices_from(hess)] += grad_eval * (
                        j_chain * (1.0 - j_chain)
                    )
        except Exception:
            hess = None

    if grad is not None and grad.shape == x.shape and np.all(np.isfinite(grad)):
        result.jac = grad.copy()
        outer_info["grad"] = grad.copy()
    if (
        hess is not None
        and hess.shape == (x.size, x.size)
        and np.all(np.isfinite(hess))
    ):
        hess = 0.5 * (hess + hess.T)
        result.hess = hess.copy()
        outer_info["hess"] = hess.copy()
    if outer_info:
        result.outer_info = outer_info


def _optimize_outer_optim_strict(*, objective, x0, bounds):
    """Mirror `mgcv::gam.outer(..., optimizer[2] = "optim")` via L-BFGS-B."""
    x0 = np.asarray(x0, dtype=np.float64).ravel()
    fscale = 1.0
    optim_rows = {}
    optim_order = []

    def _optim_trace_key(x):
        return "|".join(format(float(val), ".17g") for val in np.asarray(x).ravel())

    def _record_optim_eval(kind, x, value):
        x = np.asarray(x, dtype=np.float64).ravel()
        key = _optim_trace_key(x)
        if key not in optim_rows:
            optim_rows[key] = {
                "log_sp": x.copy(),
                "criterion": None,
                "gradient": None,
                "n_fun": 0,
                "n_jac": 0,
            }
            optim_order.append(key)
        row = optim_rows[key]
        if kind == "fun":
            row["criterion"] = float(value)
            row["n_fun"] = int(row["n_fun"]) + 1
        else:
            row["gradient"] = np.asarray(value, dtype=np.float64).ravel().copy()
            row["n_jac"] = int(row["n_jac"]) + 1

    model = getattr(objective, "model", None)
    y = np.asarray(
        getattr(objective, "y", np.array([], dtype=np.float64)), dtype=np.float64
    ).ravel()
    if model is not None and y.size > 0:
        mum = np.mean(y, dtype=np.float64) + np.zeros_like(y, dtype=np.float64)
        try:
            dev = float(
                model.family.deviance(
                    y,
                    mum,
                    weights=getattr(model, "prior_weights_", None),
                )
            )
        except TypeError:
            dev = float(model.family.deviance(y, mum))
        n_rows = int(getattr(model, "n_samples_", y.size) or y.size)
        if np.isfinite(dev) and dev > 0.0 and n_rows > 0:
            fscale = float(dev / n_rows)

    def _optim_fun(x):
        val = float(objective.fun(x))
        _record_optim_eval("fun", x, val)
        return float(val / fscale)

    def _optim_jac(x):
        grad = np.asarray(objective.jac(x), dtype=np.float64)
        _record_optim_eval("grad", x, grad)
        return grad / fscale

    result = minimize(
        fun=_optim_fun,
        x0=x0,
        method="L-BFGS-B",
        jac=_optim_jac,
        bounds=bounds,
        options={
            "ftol": float(np.finfo(np.float64).eps * 1e7),
            "gtol": 0.0,
            "maxcor": int(min(5, max(1, x0.size))),
        },
    )
    result.optim_scaled_fun = float(result.fun)
    result.fun = float(objective.fun(np.asarray(result.x, dtype=np.float64).ravel()))
    _record_optim_eval("fun", result.x, result.fun)
    result_message = "CONVERGENCE: REL_REDUCTION_OF_F <= FACTR*EPSMCH"
    result.message = str(int(getattr(result, "status", 0)))
    counts = []
    nfev = getattr(result, "nfev", None)
    njev = getattr(result, "njev", None)
    if nfev is not None:
        counts.append(int(nfev))
    if njev is not None:
        counts.append(int(njev))
    trace_rows = []
    prev_x = None
    for i, key in enumerate(optim_order):
        row = optim_rows[key]
        x_row = np.asarray(row["log_sp"], dtype=np.float64)
        step_norm = 0.0 if prev_x is None else float(np.linalg.norm(x_row - prev_x))
        trace_rows.append(
            {
                "iter": int(i),
                "log_sp": x_row.copy(),
                "criterion": row["criterion"],
                "gradient": (
                    None
                    if row["gradient"] is None
                    else np.asarray(row["gradient"], dtype=np.float64).copy()
                ),
                "hessian": None,
                "accepted_step_norm": step_norm,
                "n_fun": int(row["n_fun"]),
                "n_jac": int(row["n_jac"]),
                "n_hess": None,
                "rank_info": {
                    "source": "outer_optim_strict",
                    "n_fun": int(row["n_fun"]),
                    "n_jac": int(row["n_jac"]),
                },
            }
        )
        prev_x = x_row
    score_hist = [
        float(row["criterion"])
        for row in trace_rows
        if row.get("criterion", None) is not None
    ]
    result.optim_trace = trace_rows
    result.outer_info = {
        "optimizer": "optim",
        "conv": str(int(getattr(result, "status", 0))),
        "convergence": int(getattr(result, "status", 0)),
        "message": result_message,
        "counts": (None if len(counts) == 0 else np.asarray(counts, dtype=np.int64)),
        "iter": int(len(trace_rows)),
        "score_hist": score_hist,
        "gradient": None
        if getattr(result, "jac", None) is None
        else np.asarray(result.jac, dtype=np.float64).copy(),
        "gradient_full": None
        if getattr(result, "jac", None) is None
        else np.asarray(result.jac, dtype=np.float64).copy(),
        "hessian": None
        if getattr(result, "hess", None) is None
        else np.asarray(result.hess, dtype=np.float64).copy(),
        "hessian_full": None
        if getattr(result, "hess", None) is None
        else np.asarray(result.hess, dtype=np.float64).copy(),
    }
    result.outer_optim_used = True
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

    from ...capabilities import model_fit_capabilities

    capabilities = model_fit_capabilities(model)
    if capabilities.transformed_observations and method in {
        "ml",
        "reml",
        "laml",
    }:
        # Correlated Gaussian likelihood criteria require the correlation
        # determinant contribution in addition to the whitened solve.
        return False
    if capabilities.transformed_coefficients and (
        capabilities.multiple_linear_predictors or capabilities.general_family
    ):
        # Do not evaluate a transformed model with the identity-coordinate
        # Laplace score. Higher transform derivatives are required here.
        return False

    base_ok = bool(getattr(model.family, attr, False))
    if not base_ok:
        return False

    if (
        method in {"ubre", "aic", "ubreaic"}
        and getattr(model.family, "known_scale", None) is None
    ):
        return False

    if method in {"ml", "reml", "laml"}:
        return resolve_ml_reml_scoring_backend(model, method=method) is not None

    return True


def resolve_smoothing_method(model, method):
    method = "auto" if method is None else str(method).lower()
    if method == "gcv.cp":
        # Mirror upstream mgcv's family-dependent resolution of its default
        # method string: extended families force REML (mgcv/R/mgcv.r:1891-1893)
        # and otherwise known scale selects UBRE/Cp while unknown scale selects
        # GCV (mgcv.r:1952-1966). The canonical table lives in
        # criteria/dispatch so the criterion kernels and the public fit
        # surface cannot drift apart.
        from ..criteria.dispatch import _normalize_criterion_method

        return _normalize_criterion_method(model, method)
    if (
        method in {"gcv", "ubre", "aic", "ubreaic"}
        and str(getattr(model.family, "family_class", "")).lower() == "extended"
    ):
        # mgcv/R/mgcv.r:1891-1892: extended families silently reset ANY
        # method outside {REML, ML, NCV} to REML, including an explicitly
        # requested GCV.Cp/GACV.Cp. GCV/UBRE criteria are undefined for the
        # extended-family estimation path.
        return "reml"
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


def _coerce_initial_free_smoothing_params(init, free_mask, n_sp, n_free):
    init = np.asarray(init, dtype=np.float64).ravel()
    if init.shape == (n_sp,):
        return np.asarray(init[free_mask], dtype=np.float64)
    if init.shape == (n_free,):
        return init.copy()
    raise ValueError(
        f"Expected initial smoothing params of shape ({n_sp},) or ({n_free},), "
        f"got {init.shape}."
    )


class _MinSpOffsetObjective:
    """mgcv-style min.sp reparameterization of a log-total-sp objective.

    mgcv absorbs min.sp into the fixed offset penalty H
    (mgcv/R/mgcv.r::gam.setup, mgcv.r:1465-1508) and optimizes the *free*
    multiplier unconstrained, so the outer coordinates are rho = log(sp_free)
    with total sp = min.sp + exp(rho).  Wrapping the total-log-sp objective
    with the exact chain rule reproduces that optimization geometry.  Trailing
    non-sp coordinates (joint log scale / log phi) pass through unchanged.
    """

    def __init__(self, base, min_sp_free):
        self._base = base
        self._min_sp_free = np.asarray(min_sp_free, dtype=np.float64).ravel()

    def _split(self, x):
        x = np.asarray(x, dtype=np.float64).ravel()
        k = self._min_sp_free.size
        return x[:k], x[k:]

    def _mapped(self, x):
        rho, rest = self._split(x)
        lam = self._min_sp_free + np.exp(rho)
        return np.concatenate([np.log(lam), rest])

    def _jac_diag(self, x):
        rho, rest = self._split(x)
        lam = self._min_sp_free + np.exp(rho)
        return np.concatenate([np.exp(rho) / lam, np.ones_like(rest)])

    def fun(self, x):
        return self._base.fun(self._mapped(x))

    def jac(self, x):
        grad = np.asarray(self._base.jac(self._mapped(x)), dtype=np.float64).ravel()
        return grad * self._jac_diag(x)

    def hess(self, x):
        mapped = self._mapped(x)
        grad = np.asarray(self._base.jac(mapped), dtype=np.float64).ravel()
        hess = np.asarray(self._base.hess(mapped), dtype=np.float64)
        j = self._jac_diag(x)
        out = hess * np.outer(j, j)
        k = self._min_sp_free.size
        # d^2 log(min.sp + e^rho)/d rho^2 = j (1 - j) on sp coordinates.
        curvature = np.zeros_like(j)
        curvature[:k] = j[:k] * (1.0 - j[:k])
        out[np.diag_indices_from(out)] += grad * curvature
        return out

    def __getattr__(self, name):
        return getattr(self._base, name)


def _run_joint_negbin_outer(
    model, y, x0, free_mask, method, bounds, optimizer, n_free,
    min_sp_offset_active,
):
    """Joint negative-binomial theta outer optimization (extracted block)."""
    if min_sp_offset_active:
        raise NotImplementedError(
            "min_sp with joint negative-binomial theta estimation is not "
            "supported yet: the native joint optimizer orders log(theta) "
            "before log(sp), which the mgcv min.sp H-offset "
            "reparameterization does not cover."
        )
    mgcv_result = _optimize_negbin_reml_joint_native(
        model,
        y,
        x0,
        free_mask,
        method,
        bounds,
        optimizer=optimizer,
    )
    if mgcv_result is not None:
        model.family.theta = float(np.exp(float(mgcv_result.joint_log_theta)))
        _fit_workspace(model).pirls_disable_theta_efs = True
        model.smoothing_params = np.asarray(
            mgcv_result.joint_negbin_selected_full_sp, dtype=np.float64
        ).copy()
        model._optim_method = method
        model._optim_result = mgcv_result
        trace_rows = []
        prev_log_sp = None

        def _trace_vector_payload(values):
            arr = np.asarray(values, dtype=np.float64).ravel()
            if arr.size == 1:
                return float(arr[0])
            return arr.tolist()

        for row in list(getattr(mgcv_result, "optim_trace", []) or []):
            row_dict = dict(row)
            x_joint = np.asarray(
                row_dict.get("log_sp", []), dtype=np.float64
            ).ravel()
            grad_full = (
                None
                if row_dict.get("gradient", None) is None
                else np.asarray(row_dict.get("gradient"), dtype=np.float64).ravel()
            )
            hess_full = (
                None
                if row_dict.get("hessian", None) is None
                else np.asarray(row_dict.get("hessian"), dtype=np.float64)
            )
            log_theta = None
            log_sp = np.empty((0,), dtype=np.float64)
            gradient = None
            hessian = None
            if x_joint.size > 0:
                # `_JointNegbinPirlsRemlObjective` optimizes in mgcv
                # extended-family order: log(theta) first, then log(sp).
                log_theta = float(x_joint[0])
                log_sp = np.asarray(x_joint[1 : 1 + n_free], dtype=np.float64)
            if grad_full is not None:
                gradient = np.asarray(grad_full[1 : 1 + n_free], dtype=np.float64)
            if hess_full is not None:
                hessian = np.asarray(
                    hess_full[1 : 1 + n_free, 1 : 1 + n_free],
                    dtype=np.float64,
                )
            accepted_step_norm = (
                (
                    0.0
                    if prev_log_sp is None
                    else float(np.linalg.norm(log_sp - prev_log_sp, ord=2))
                )
                if optimizer == "optim"
                else float(row_dict.get("accepted_step_norm", 0.0))
            )
            trace_rows.append(
                {
                    "iter": int(row_dict.get("iter", 0)),
                    "log_sp": _trace_vector_payload(log_sp),
                    "log_scale": None,
                    "log_theta": log_theta,
                    "criterion": (
                        None
                        if row_dict.get("criterion", None) is None
                        else float(row_dict.get("criterion"))
                    ),
                    "gradient": (
                        None
                        if gradient is None
                        else _trace_vector_payload(gradient)
                    ),
                    "gradient_full": (
                        None
                        if grad_full is None
                        else np.asarray(grad_full, dtype=np.float64).tolist()
                    ),
                    "hessian": (
                        None
                        if hessian is None
                        else np.asarray(hessian, dtype=np.float64).tolist()
                    ),
                    "hessian_full": (
                        None
                        if hess_full is None
                        else np.asarray(hess_full, dtype=np.float64).tolist()
                    ),
                    "accepted_step_norm": accepted_step_norm,
                    "n_fun": row_dict.get("n_fun", None),
                    "n_jac": row_dict.get("n_jac", None),
                    "n_hess": row_dict.get("n_hess", None),
                    "rank_info": row_dict.get("rank_info", None),
                }
            )
            prev_log_sp = log_sp.copy()
        outer_info = dict(getattr(mgcv_result, "outer_info", {}) or {})
        if trace_rows:
            if optimizer != "bfgs":
                outer_info["score_hist"] = [
                    float(row["criterion"])
                    for row in trace_rows
                    if row.get("criterion", None) is not None
                ]
                outer_info["iter"] = int(len(outer_info["score_hist"]))
        grad_full = outer_info.get("grad", None)
        if grad_full is not None:
            grad_full = np.asarray(grad_full, dtype=np.float64).ravel()
            outer_info["gradient"] = _trace_vector_payload(
                grad_full[1 : 1 + n_free]
            )
            outer_info["gradient_full"] = grad_full.copy()
        hess_full = outer_info.get("hess", None)
        if hess_full is not None:
            hess_full = np.asarray(hess_full, dtype=np.float64)
            outer_info["hessian"] = hess_full[1 : 1 + n_free, 1 : 1 + n_free].copy()
            outer_info["hessian_full"] = hess_full.copy()
        outer_info["conv"] = str(getattr(mgcv_result, "message", ""))
        outer_info["edge_correct"] = False
        mgcv_result.outer_info = outer_info
        model._optim_trace = trace_rows
        mgcv_result.optim_trace = trace_rows
        model._optim_used_gradient = bool(
            getattr(mgcv_result, "jac", None) is not None
        )
        model._optim_used_hessian = bool(
            getattr(mgcv_result, "hess", None) is not None
        )
        model.smoothing_score_ = float(mgcv_result.fun)
        return model
    raise NotImplementedError(
        "Negative-binomial ML/REML/LAML with estimate_theta=True has no "
        "native upstream-supported joint optimizer path in this build."
    )


def _run_joint_gamma_outer(
    model, y, x0, bounds, branch_m, method, optimizer, objective,
    min_sp_offset_active, min_sp_free, n_free, _free_x_to_total_log_sp,
):
    """Joint Gamma REML/LAML scale outer optimization (extracted block)."""
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
        if min_sp_offset_active:
            j_obj = _MinSpOffsetObjective(j_obj, min_sp_free)
        if optimizer == "bfgs":
            result_joint = _optimize_outer_bfgs_strict(
                objective=j_obj,
                x0=x_joint0,
                bounds=joint_bounds,
                score_type=method,
            )
        elif optimizer == "optim":
            # `mgcv/R/mgcv.r::gam.outer()` passes the complete `lsp`
            # vector, including the appended log scale, to
            # `optim(..., method="L-BFGS-B")` without finite scale
            # bounds. Preserve that unbounded scale coordinate: even a
            # distant finite bound changes L-BFGS-B's generalized Cauchy
            # point in more than one dimension.
            result_joint = _optimize_outer_optim_strict(
                objective=j_obj,
                x0=x_joint0,
                bounds=list(bounds) + [(-np.inf, np.inf)],
            )
        elif optimizer == "outer_newton":
            result_joint = optimize_outer_newton_indefinite_hessian(
                objective=j_obj,
                x0=x_joint0,
                bounds=joint_bounds,
                conv_tol=1e-6,
            )
        else:
            raise NotImplementedError(
                "Strict mgcv-parity Gamma REML/LAML joint-scale optimization "
                f"does not support optimizer={optimizer!r}."
            )
        x_joint = np.asarray(result_joint.x, dtype=np.float64).ravel()
        x_selected = np.asarray(x_joint[:-1], dtype=np.float64).ravel()
        profiled_fun = float(objective.fun(x_selected))
        result = OptimizeResult(
            x=np.asarray(x_selected, dtype=np.float64).copy(),
            # `mgcv/R/mgcv.r::gam.outer` stores `b$score` from
            # `gam.fit3.r::newton()`, where unknown scale is the final
            # outer parameter. Preserve the joint score rather than a
            # post-hoc profiled-scale recompute.
            fun=float(result_joint.fun),
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
        result.profiled_fun = float(profiled_fun)
        result.joint_log_phi = float(x_joint[-1])
        result.edge_correction_requested = bool(
            getattr(result_joint, "edge_correction_requested", False)
        )
        result.edge_correction_applied = bool(
            getattr(result_joint, "edge_correction_applied", False)
        )
        if np.isfinite(result.joint_log_phi):
            model._gamma_reml_phi_opt_ = float(np.exp(result.joint_log_phi))
        result.joint_gamma_message = str(getattr(result_joint, "message", ""))
        outer_info_joint = dict(getattr(result_joint, "outer_info", {}) or {})
        trace_rows = []
        prev_log_sp = None

        def _trace_vector_payload(values):
            arr = np.asarray(values, dtype=np.float64).ravel()
            if arr.size == 1:
                return float(arr[0])
            return arr.tolist()

        for row in list(getattr(result_joint, "optim_trace", []) or []):
            row_dict = dict(row)
            log_sp_full = np.asarray(
                row_dict.get("log_sp", []), dtype=np.float64
            ).ravel()
            log_sp = np.asarray(log_sp_full[:n_free], dtype=np.float64)
            log_scale = (
                float(log_sp_full[n_free]) if log_sp_full.size > n_free else None
            )
            gradient_full = (
                None
                if row_dict.get("gradient", None) is None
                else np.asarray(row_dict.get("gradient"), dtype=np.float64).ravel()
            )
            hessian_full = (
                None
                if row_dict.get("hessian", None) is None
                else np.asarray(row_dict.get("hessian"), dtype=np.float64)
            )
            gradient = (
                None
                if gradient_full is None
                else np.asarray(gradient_full[:n_free], dtype=np.float64)
            )
            hessian = (
                None
                if hessian_full is None
                else np.asarray(hessian_full[:n_free, :n_free], dtype=np.float64)
            )
            accepted_step_norm = (
                (
                    0.0
                    if prev_log_sp is None
                    else float(np.linalg.norm(log_sp - prev_log_sp, ord=2))
                )
                if optimizer == "optim"
                else float(row_dict.get("accepted_step_norm", 0.0))
            )
            trace_rows.append(
                {
                    "iter": int(row_dict.get("iter", 0)),
                    "log_sp": _trace_vector_payload(log_sp),
                    "log_scale": log_scale,
                    "log_theta": None,
                    "criterion": (
                        None
                        if row_dict.get("criterion", None) is None
                        else float(row_dict.get("criterion"))
                    ),
                    "gradient": (
                        None
                        if gradient is None
                        else _trace_vector_payload(gradient)
                    ),
                    "gradient_full": (
                        None
                        if gradient_full is None
                        else np.asarray(gradient_full, dtype=np.float64).tolist()
                    ),
                    "hessian": (
                        None
                        if hessian is None
                        else np.asarray(hessian, dtype=np.float64).tolist()
                    ),
                    "hessian_full": (
                        None
                        if hessian_full is None
                        else np.asarray(hessian_full, dtype=np.float64).tolist()
                    ),
                    # The optim trace reports movement in smoothing
                    # parameters only; Newton/BFGS report movement in the
                    # complete joint vector before it is split for output.
                    "accepted_step_norm": accepted_step_norm,
                    "n_fun": row_dict.get("n_fun", None),
                    "n_jac": row_dict.get("n_jac", None),
                    "n_hess": row_dict.get("n_hess", None),
                    "rank_info": row_dict.get("rank_info", None),
                }
            )
            prev_log_sp = log_sp.copy()
        if trace_rows:
            model._optim_trace = trace_rows
            result.optim_trace = trace_rows
        if outer_info_joint:
            joint_grad = outer_info_joint.get("grad", None)
            if joint_grad is not None:
                joint_grad = np.asarray(joint_grad, dtype=np.float64).ravel()
                outer_info_joint["gradient"] = joint_grad[:n_free].copy()
                outer_info_joint["gradient_full"] = joint_grad.copy()
            joint_hess = outer_info_joint.get("hess", None)
            if joint_hess is not None:
                joint_hess = np.asarray(joint_hess, dtype=np.float64)
                outer_info_joint["hessian"] = joint_hess[
                    :n_free, :n_free
                ].copy()
                outer_info_joint["hessian_full"] = joint_hess.copy()
            outer_info_joint.setdefault("log_scale", None)
            outer_info_joint.setdefault("log_theta", None)
            outer_info_joint.setdefault("edge_correct", False)
            result.outer_info = outer_info_joint
        _ = criterion_hessian_ml_reml_pirls_exact(
            model, y, _free_x_to_total_log_sp(result.x), branch_m
        )
        gamma_state = _fit_workspace(model).get("pirls_reml_gamma_state", None)
        phi_opt = None
        if isinstance(gamma_state, dict):
            phi_opt = gamma_state.get("phi", None)
        if (
            phi_opt is not None
            and np.isfinite(float(phi_opt))
            and float(phi_opt) > 0.0
            and not (
                getattr(model, "_gamma_reml_phi_opt_", None) is not None
                and np.isfinite(float(model._gamma_reml_phi_opt_))
                and float(model._gamma_reml_phi_opt_) > 0.0
            )
        ):
            model._gamma_reml_phi_opt_ = float(phi_opt)
    return result


def _run_joint_betar_outer(
    model, y, x0, free_mask, method, bounds, optimizer, n_free
):
    """Joint ``betar`` outer optimization in mgcv's ``(theta, sp)`` order."""
    if optimizer == "efs":
        raise NotImplementedError(
            "Joint betar theta estimation is not supported by the EFS optimizer."
        )
    x0 = np.asarray(x0, dtype=np.float64).ravel()
    log_theta0 = float(model.family.getTheta(False))
    x_joint0 = np.concatenate(([log_theta0], x0))
    joint_bounds = [(-np.inf, np.inf)] + list(bounds)
    branch_m = "LAML" if str(method).lower() == "laml" else str(method).upper()
    objective = _JointBetarPirlsRemlObjective(model, y, branch_m)
    if optimizer == "outer_newton":
        result_joint = optimize_outer_newton_indefinite_hessian(
            objective=objective,
            x0=x_joint0,
            bounds=joint_bounds,
            conv_tol=1e-6,
        )
    elif optimizer == "bfgs":
        result_joint = _optimize_outer_bfgs_strict(
            objective=objective,
            x0=x_joint0,
            bounds=joint_bounds,
            score_type=str(method).lower(),
        )
    elif optimizer == "optim":
        result_joint = _optimize_outer_optim_strict(
            objective=objective,
            x0=x_joint0,
            bounds=joint_bounds,
        )
    else:
        raise NotImplementedError(
            "Strict betar outer optimization does not support "
            f"optimizer={optimizer!r}."
        )

    x_joint = np.asarray(result_joint.x, dtype=np.float64).ravel()
    if x_joint.shape != x_joint0.shape:
        x_joint = x_joint0.copy()
    log_theta = float(x_joint[0])
    model.family.putTheta(log_theta)
    x_sp = np.asarray(x_joint[1:], dtype=np.float64).copy()
    result = OptimizeResult(
        x=x_sp,
        fun=float(result_joint.fun),
        success=bool(getattr(result_joint, "success", False)),
        status=int(getattr(result_joint, "status", 0)),
        message=str(getattr(result_joint, "message", "")),
        nit=int(getattr(result_joint, "nit", 0)),
        nfev=int(getattr(result_joint, "nfev", objective.n_fun)),
        njev=int(getattr(result_joint, "njev", objective.n_jac)),
        nhev=int(getattr(result_joint, "nhev", objective.n_hess)),
    )
    result.joint_outer_strategy = JointOuterStrategy.BETAR_THETA
    result.joint_log_theta = log_theta
    result.selected_theta = float(np.exp(log_theta))
    result.joint_x = x_joint.copy()
    result.selected_full_smoothing_params = np.asarray(
        model.smoothing_params, dtype=np.float64
    ).copy()
    result.selected_full_smoothing_params[np.asarray(free_mask, dtype=bool)] = np.exp(
        x_sp
    )
    model.smoothing_params = result.selected_full_smoothing_params.copy()
    joint_jac = getattr(result_joint, "jac", None)
    joint_hess = getattr(result_joint, "hess", None)
    outer_info = dict(getattr(result_joint, "outer_info", {}) or {})
    if joint_jac is not None:
        joint_jac = np.asarray(joint_jac, dtype=np.float64).ravel()
        result.jac = joint_jac[1:].copy()
        outer_info["grad"] = joint_jac.copy()
        outer_info["gradient_full"] = joint_jac.copy()
    if joint_hess is not None:
        joint_hess = np.asarray(joint_hess, dtype=np.float64)
        result.hess = joint_hess[1:, 1:].copy()
        outer_info["hess"] = 0.5 * (joint_hess + joint_hess.T)
        outer_info["hessian_full"] = outer_info["hess"].copy()
    outer_info["log_theta"] = log_theta
    outer_info["log_scale"] = None
    result.outer_info = outer_info
    result.optim_trace = getattr(result_joint, "optim_trace", None)
    model._optim_method = method
    model._optim_result = result
    model._optim_trace = list(getattr(objective, "accepted_trace", []))
    model._optim_used_gradient = bool(getattr(result, "jac", None) is not None)
    model._optim_used_hessian = bool(getattr(result, "hess", None) is not None)
    model.smoothing_score_ = float(result.fun)
    return model


def _run_joint_ocat_outer(
    model, y, x0, free_mask, method, bounds, optimizer, n_free
):
    """Joint ``ocat`` outer optimization in mgcv's ``(theta, sp)`` order."""
    if optimizer == "efs":
        raise NotImplementedError(
            "Joint ocat cutpoint estimation is not supported by the EFS optimizer."
        )
    ntheta = int(getattr(model.family, "n_theta", 0) or 0)
    if not bool(getattr(model.family, "_initialized_from_data", False)):
        model.family.initialize_mu(y)
    theta0 = np.asarray(model.family.getTheta(False), dtype=np.float64).ravel()
    if theta0.size != ntheta:
        raise ValueError("ocat joint optimization requires all cutpoint gaps free.")
    x_joint0 = np.concatenate((theta0, np.asarray(x0, dtype=np.float64).ravel()))
    joint_bounds = [(-np.inf, np.inf)] * ntheta + list(bounds)
    branch_m = "LAML" if str(method).lower() == "laml" else str(method).upper()
    objective = _JointOcatPirlsRemlObjective(model, y, branch_m)
    if optimizer == "outer_newton":
        result_joint = optimize_outer_newton_indefinite_hessian(
            objective=objective,
            x0=x_joint0,
            bounds=joint_bounds,
            conv_tol=1e-6,
        )
    elif optimizer == "bfgs":
        result_joint = _optimize_outer_bfgs_strict(
            objective=objective,
            x0=x_joint0,
            bounds=joint_bounds,
            score_type=str(method).lower(),
        )
    elif optimizer == "optim":
        result_joint = _optimize_outer_optim_strict(
            objective=objective,
            x0=x_joint0,
            bounds=joint_bounds,
        )
    else:
        raise NotImplementedError(
            "Strict ocat outer optimization does not support "
            f"optimizer={optimizer!r}."
        )

    x_joint = np.asarray(result_joint.x, dtype=np.float64).ravel()
    if x_joint.shape != x_joint0.shape:
        x_joint = x_joint0.copy()
    log_theta = x_joint[:ntheta].copy()
    model.family.putTheta(log_theta)
    x_sp = x_joint[ntheta:].copy()
    result = OptimizeResult(
        x=x_sp,
        fun=float(result_joint.fun),
        success=bool(getattr(result_joint, "success", False)),
        status=int(getattr(result_joint, "status", 0)),
        message=str(getattr(result_joint, "message", "")),
        nit=int(getattr(result_joint, "nit", 0)),
        nfev=int(getattr(result_joint, "nfev", objective.n_fun)),
        njev=int(getattr(result_joint, "njev", objective.n_jac)),
        nhev=int(getattr(result_joint, "nhev", objective.n_hess)),
    )
    result.joint_outer_strategy = JointOuterStrategy.OCAT_THETA
    result.joint_log_theta = log_theta.copy()
    result.selected_theta = model.family.getTheta(trans=True).copy()
    result.joint_x = x_joint.copy()
    result.selected_full_smoothing_params = np.asarray(
        model.smoothing_params, dtype=np.float64
    ).copy()
    result.selected_full_smoothing_params[np.asarray(free_mask, dtype=bool)] = np.exp(
        x_sp
    )
    model.smoothing_params = result.selected_full_smoothing_params.copy()
    joint_jac = getattr(result_joint, "jac", None)
    joint_hess = getattr(result_joint, "hess", None)
    outer_info = dict(getattr(result_joint, "outer_info", {}) or {})
    if joint_jac is not None:
        joint_jac = np.asarray(joint_jac, dtype=np.float64).ravel()
        result.jac = joint_jac[ntheta:].copy()
        outer_info["grad"] = joint_jac.copy()
        outer_info["gradient_full"] = joint_jac.copy()
    if joint_hess is not None:
        joint_hess = np.asarray(joint_hess, dtype=np.float64)
        outer_info["hess"] = 0.5 * (joint_hess + joint_hess.T)
        outer_info["hessian_full"] = outer_info["hess"].copy()
        result.hess = outer_info["hess"][ntheta:, ntheta:].copy()
    outer_info["log_theta"] = log_theta.copy()
    outer_info["log_scale"] = None
    result.outer_info = outer_info
    result.optim_trace = getattr(result_joint, "optim_trace", None)
    model._optim_method = method
    model._optim_result = result
    model._optim_trace = list(getattr(objective, "accepted_trace", []))
    model._optim_used_gradient = bool(getattr(result, "jac", None) is not None)
    model._optim_used_hessian = bool(getattr(result, "hess", None) is not None)
    model.smoothing_score_ = float(result.fun)
    return model


def _run_joint_tweedie_outer(
    model,
    y,
    x0,
    bounds,
    branch_m,
    method,
    optimizer,
    n_free,
):
    """Joint ``tw`` outer optimization in mgcv's parameter order."""
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    weights = getattr(model, "prior_weights_", None)
    mu_null = np.repeat(float(np.mean(y_arr)), model.n_samples_)
    null_scale = float(
        model.family.deviance(y_arr, mu_null, weights=weights)
    ) / float(model.n_samples_)
    phi0 = max(null_scale / 10.0, LOG_GUARD_MIN)
    y_scale = (
        float(np.var(y_arr))
        if y_arr.size > 1
        else max(abs(float(y_arr[0])), LOG_GUARD_MIN)
    )
    hi_phi = max(phi0 * 1e8, y_scale * 1e8, 1e-30)
    ntheta = int(getattr(model.family, "n_theta", 0) or 0)
    theta0 = float(model.family.getTheta(False))
    x_joint0 = np.concatenate(
        [
            np.array([theta0], dtype=np.float64) if ntheta else np.empty(0),
            np.asarray(x0, dtype=np.float64),
            np.array([np.log(phi0)], dtype=np.float64),
        ]
    )
    joint_bounds = (
        ([(-np.inf, np.inf)] if ntheta else [])
        + list(bounds)
        + [
            (
                (-np.inf, np.inf)
                if optimizer == "optim"
                else (float(np.log(LOG_GUARD_MIN)), float(np.log(hi_phi)))
            )
        ]
    )
    j_obj = _JointTweediePirlsRemlObjective(model, y, branch_m)
    if optimizer == "bfgs":
        result_joint = _optimize_outer_bfgs_strict(
            objective=j_obj,
            x0=x_joint0,
            bounds=joint_bounds,
            score_type=method,
        )
    elif optimizer == "optim":
        result_joint = _optimize_outer_optim_strict(
            objective=j_obj,
            x0=x_joint0,
            bounds=joint_bounds,
        )
    elif optimizer == "outer_newton":
        result_joint = optimize_outer_newton_indefinite_hessian(
            objective=j_obj,
            x0=x_joint0,
            bounds=joint_bounds,
            conv_tol=1e-6,
        )
    else:
        raise NotImplementedError(
            "Strict mgcv-parity Tweedie outer optimization does not support "
            f"optimizer={optimizer!r}."
        )

    x_joint = np.asarray(result_joint.x, dtype=np.float64).ravel()
    if x_joint.shape != x_joint0.shape:
        x_joint = x_joint0.copy()
    theta_opt = float(x_joint[0]) if ntheta else theta0
    if ntheta:
        model.family.putTheta(theta_opt)
    sp_slice = slice(ntheta, ntheta + n_free)
    x_sp = np.asarray(x_joint[sp_slice], dtype=np.float64).copy()
    joint_jac = getattr(result_joint, "jac", None)
    joint_hess = getattr(result_joint, "hess", None)
    result = OptimizeResult(
        x=x_sp,
        fun=float(result_joint.fun),
        success=bool(getattr(result_joint, "success", False)),
        status=int(getattr(result_joint, "status", 0)),
        message=str(getattr(result_joint, "message", "")),
        nit=int(getattr(result_joint, "nit", 0)),
        nfev=int(getattr(result_joint, "nfev", j_obj.n_fun)),
        njev=int(getattr(result_joint, "njev", j_obj.n_jac)),
        nhev=int(getattr(result_joint, "nhev", j_obj.n_hess)),
    )
    result.joint_outer_strategy = JointOuterStrategy.TWEEDIE
    result.joint_log_theta = theta_opt if ntheta else None
    result.joint_log_phi = float(x_joint[-1])
    result.selected_theta = float(model.family.getTheta(True))
    result.joint_x = x_joint.copy()
    if joint_jac is not None:
        joint_jac = np.asarray(joint_jac, dtype=np.float64).ravel()
        result.jac = joint_jac[sp_slice].copy()
    if joint_hess is not None:
        joint_hess = np.asarray(joint_hess, dtype=np.float64)
        result.hess = joint_hess[sp_slice, sp_slice].copy()
    outer_info = dict(getattr(result_joint, "outer_info", {}) or {})
    if joint_jac is not None:
        outer_info["grad"] = joint_jac.copy()
        outer_info["gradient_full"] = joint_jac.copy()
    if joint_hess is not None:
        outer_info["hess"] = 0.5 * (joint_hess + joint_hess.T)
        outer_info["hessian_full"] = outer_info["hess"].copy()
    outer_info["log_theta"] = theta_opt if ntheta else None
    outer_info["log_scale"] = float(x_joint[-1])
    result.outer_info = outer_info
    result.optim_trace = getattr(result_joint, "optim_trace", None)
    return result


def optimize_smoothing_params(
    model, y, initial_smoothing_params=None, method="gcv", optimizer="outer_newton"
):
    method = resolve_smoothing_method(model, method)
    optimizer = str(optimizer).lower()
    if optimizer in {"newton", "outer"}:
        optimizer = "outer_newton"
    method = coerce_general_family_smoothing_method(
        model,
        method,
        optimizer=optimizer,
    )
    if optimizer == "efs":
        # mgcv/R/mgcv.r::estimate.gam forces EFS onto REML regardless of the
        # requested criterion.
        method = "reml"
    exact_gaussian = str(getattr(model.family, "name", "")).lower() == "gaussian"
    transformed_coefficients = is_transformed_coefficient_model(model)

    if transformed_coefficients:
        if method not in {"gcv", "ubre", "aic", "ubreaic"}:
            raise NotImplementedError(
                "Transformed-coefficient smoothing selection currently supports "
                "GCV/UBRE only for the single-predictor GLM kernel."
            )
        if optimizer not in {"outer_newton", "bfgs"}:
                raise NotImplementedError(
                    "Transformed-coefficient smoothing selection currently supports "
                    "the exact SCAM bfgs_gcv.ubre policy only."
                )
        if optimizer == "outer_newton":
            # scam() defaults to bfgs_gcv.ubre for smoothing selection. Its
            # outer Newton name refers to coefficient fitting, not SP fitting.
            optimizer = "bfgs"

    if method not in {
        "gcv",
        "ubre",
        "aic",
        "ubreaic",
        "ml",
        "reml",
        "laml",
    }:
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
    if optimizer not in {"lbfgsb", "outer_newton", "bfgs", "efs", "optim"}:
        raise NotImplementedError(
            "Current core supports smoothing_optimizer in "
            "{'lbfgsb', 'outer_newton', 'bfgs', 'efs', 'optim'} only."
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
        and method in {"gcv", "ubre", "aic", "ubreaic", "ml", "reml", "laml"}
        and (not use_gradient or not use_hessian)
    ):
        raise NotImplementedError(
            "Strict mgcv-parity outer Newton smoothing optimisation requires "
            "exact first- and second-derivative support; local fallback paths "
            "have been removed."
        )
    if optimizer == "bfgs" and (not use_gradient):
        raise NotImplementedError(
            "Strict mgcv-parity BFGS smoothing optimisation requires an exact "
            "gradient path for this method/family."
        )
    if optimizer == "optim" and (not use_gradient):
        raise NotImplementedError(
            "Strict mgcv-parity optim smoothing optimisation requires an exact "
            "gradient path for this method/family."
        )
    n_sp = _n_smoothing_params(model)
    fixed_mask = (
        np.zeros(n_sp, dtype=bool)
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
    outer_strategy = getattr(
        model.family, "joint_outer_strategy", JointOuterStrategy.NONE
    )
    joint_pirls_strategy = (
        outer_strategy
        if method in {"ml", "reml", "laml"}
        and ml_reml_backend == "pirls_laplace"
        else JointOuterStrategy.NONE
    )
    use_joint_gamma_reml_scale = (
        joint_pirls_strategy is JointOuterStrategy.GAMMA_SCALE
    )
    family_class = str(
        getattr(getattr(model, "family", None), "family_class", "")
    ).lower()
    use_joint_gaussian_reml_scale = (
        exact_gaussian and method in {"reml", "ml"} and optimizer != "efs"
    )
    use_joint_negbin_reml_theta = (
        joint_pirls_strategy is JointOuterStrategy.NEGBIN_THETA
        and bool(getattr(model.family, "estimate_theta", False))
    )
    use_joint_betar_theta = (
        joint_pirls_strategy is JointOuterStrategy.BETAR_THETA
        and bool(getattr(model.family, "estimate_theta", False))
    )
    use_joint_ocat_theta = (
        joint_pirls_strategy is JointOuterStrategy.OCAT_THETA
        and bool(getattr(model.family, "estimate_theta", False))
    )
    _fit_workspace(model).pirls_disable_theta_efs = False

    has_joint_outer_params = (
        use_joint_gaussian_reml_scale
        or use_joint_gamma_reml_scale
        or use_joint_negbin_reml_theta
        or use_joint_betar_theta
        or use_joint_ocat_theta
        or joint_pirls_strategy is JointOuterStrategy.TWEEDIE
    )
    if n_free == 0 and not has_joint_outer_params:
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
                init = _initial_smoothing_params_from_design(model, y)
            elif exact_gaussian:
                init = _initial_smoothing_params_from_design(model, y)
            elif family_class == "general":
                init = _initial_smoothing_params_from_design(model, y)
            else:
                init = _initial_smoothing_params_from_design(model, y)
            if init is None:
                raise NotImplementedError(
                    "Strict mgcv-parity smoothing optimisation requires "
                    "mgcv::initial.spg-compatible initial smoothing parameters; "
                    "design-balance heuristic fallback removed."
                )
            # mgcv/R/mgcv.r::initial.spg returns one value per underlying
            # free smoothing parameter when fixed sp values have been moved
            # into the L/lsp0 split.
            init_free = _coerce_initial_free_smoothing_params(
                init, free_mask, n_sp, n_free
            )
        else:
            init_free = np.asarray(model.smoothing_params[free_mask], dtype=np.float64)
    else:
        init_free = _coerce_initial_free_smoothing_params(
            initial_smoothing_params, free_mask, n_sp, n_free
        )

    if np.any(~np.isfinite(init_free)) or np.any(init_free <= 0):
        raise ValueError("Initial free smoothing parameters must be finite and > 0.")

    min_sp = (
        np.zeros(_n_smoothing_params(model), dtype=np.float64)
        if model.min_sp_ is None
        else np.asarray(model.min_sp_, dtype=np.float64)
    )

    min_sp_free = np.asarray(min_sp[free_mask], dtype=np.float64)
    min_sp_offset_active = bool(np.any(min_sp_free > 0.0))
    if min_sp_offset_active:
        # mgcv absorbs min.sp into the fixed offset penalty H
        # (mgcv/R/mgcv.r::gam.setup, mgcv.r:1465-1508) and optimizes the free
        # multiplier unconstrained: total sp_i = min.sp_i + exp(rho_i).
        # initial.spg (mgcv.r:4528) is H-blind, so the unclamped initial value
        # seeds the free multiplier directly.  Box-bounding log total sp
        # instead stalls Newton on the floor.
        x0 = np.log(np.maximum(init_free, LOG_GUARD_MIN))
    else:
        init_free = np.maximum(init_free, min_sp[free_mask])
        x0 = np.log(np.maximum(init_free, LOG_GUARD_MIN))
    bounds = [(-np.inf, np.inf) for _ in range(int(n_free))]

    def _free_x_to_total_log_sp(x):
        """Map outer coordinates to log total sp (identity without min.sp)."""
        x = np.asarray(x, dtype=np.float64).ravel()
        if not min_sp_offset_active:
            return x
        return np.log(min_sp_free + np.exp(x))

    if use_joint_gaussian_reml_scale:
        # Mirror `mgcv/R/mgcv.r::get.null.coef` + `scale.as.sp` initialization:
        # `log.scale <- log(null.scale / 10)`, where
        # `null.scale <- sum(dev.resids(y, mum, weights)) / nrow(X)`.
        yv = np.asarray(y, dtype=np.float64).ravel()
        sigma20 = _initial_gaussian_scale_as_sp(model, yv)
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
        return _run_joint_negbin_outer(
            model,
            y,
            x0,
            free_mask,
            method,
            bounds,
            optimizer,
            n_free,
            min_sp_offset_active,
        )

    if use_joint_betar_theta:
        return _run_joint_betar_outer(
            model,
            y,
            x0,
            free_mask,
            method,
            bounds,
            optimizer,
            n_free,
        )

    if use_joint_ocat_theta:
        return _run_joint_ocat_outer(
            model,
            y,
            x0,
            free_mask,
            method,
            bounds,
            optimizer,
            n_free,
        )

    branch_m = "LAML" if method == "laml" else str(method).upper()
    if use_joint_gaussian_reml_scale:
        objective_class = (
            _GaussianPirlsRemlJointObjective
            if ml_reml_backend == "pirls_laplace"
            else _GaussianRemlJointObjective
        )
        objective = objective_class(
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
    if min_sp_offset_active:
        if optimizer == "efs":
            raise NotImplementedError(
                "min_sp is not supported with the EFS optimizer yet; use "
                "outer_newton, bfgs or optim."
            )
        objective = _MinSpOffsetObjective(objective, min_sp_free)
    if bool(getattr(model.family, "supports_pirls", False)):
        # Carry P-IRLS coefficient warm-starts between outer criterion evaluations.
        _fit_workspace(model).pirls_coef_start = None
        _fit_workspace(model).pirls_eta_start = None
        _fit_workspace(model).pirls_mu_start = None
    result = None

    if (
        not use_joint_gamma_reml_scale
        and joint_pirls_strategy is not JointOuterStrategy.TWEEDIE
        and result is None
    ):
        if optimizer == "efs":
            result = _optimize_outer_efs_strict(
                model=model,
                y=y,
                x0=x0,
                bounds=bounds,
                method=method,
            )
        elif optimizer == "bfgs":
            if transformed_coefficients:
                result = optimize_transformed_bfgs(objective=objective, x0=x0)
            else:
                result = _optimize_outer_bfgs_strict(
                    objective=objective,
                    x0=x0,
                    bounds=bounds,
                    score_type=method,
                )
        elif optimizer == "optim":
            result = _optimize_outer_optim_strict(
                objective=objective,
                x0=x0,
                bounds=bounds,
            )
        elif optimizer == "outer_newton":
            result = optimize_outer_newton_indefinite_hessian(
                objective=objective,
                x0=x0,
                bounds=bounds,
                # Mirror mgcv::gam.control(edge.correct = FALSE) default. The
                # optimizer should only carry edge-corrected Hessian payloads
                # when explicitly requested, not for all general-family fits.
                edge_correct=False,
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
            result = optimize_outer_newton_generic(
                objective=objective,
                x0=x0,
                bounds=bounds,
            )

    if use_joint_gamma_reml_scale:
        result = _run_joint_gamma_outer(
            model,
            y,
            x0,
            bounds,
            branch_m,
            method,
            optimizer,
            objective,
            min_sp_offset_active,
            min_sp_free,
            n_free,
            _free_x_to_total_log_sp,
        )

    if joint_pirls_strategy is JointOuterStrategy.TWEEDIE:
        if min_sp_offset_active:
            raise NotImplementedError(
                "min_sp with joint Tweedie outer optimization is not supported yet."
            )
        result = _run_joint_tweedie_outer(
            model,
            y,
            x0,
            bounds,
            branch_m,
            method,
            optimizer,
            n_free,
        )

    if not result.success:
        warnings.warn(
            f"Smoothing optimisation did not converge: {result.message}",
            stacklevel=2,
        )

    if use_joint_gaussian_reml_scale and result is not None and result.x is not None:
        x_joint = np.asarray(result.x, dtype=np.float64).ravel()
        if x_joint.size == n_free + 1:
            joint_jac = (
                None
                if getattr(result, "jac", None) is None
                else np.asarray(result.jac, dtype=np.float64).ravel()
            )
            joint_hess = (
                None
                if getattr(result, "hess", None) is None
                else np.asarray(result.hess, dtype=np.float64)
            )
            joint_log_sigma2 = float(x_joint[-1])
            model._gaussian_reml_sigma2_opt_ = float(
                max(np.exp(joint_log_sigma2), LOG_GUARD_MIN)
            )
            result.joint_gaussian_reml_outer = True
            result.joint_log_sigma2 = joint_log_sigma2
            result.joint_x = x_joint.copy()
            result.x = np.asarray(x_joint[:-1], dtype=np.float64).copy()
            outer_info = dict(getattr(result, "outer_info", {}) or {})
            if joint_jac is not None:
                result.jac = joint_jac[:-1].copy()
                outer_info["grad"] = joint_jac.copy()
                outer_info["gradient"] = result.jac.copy()
                outer_info["gradient_full"] = joint_jac.copy()
            if joint_hess is not None:
                result.hess = joint_hess[:-1, :-1].copy()
                outer_info["hess"] = joint_hess.copy()
                outer_info["hessian"] = result.hess.copy()
                outer_info["hessian_full"] = joint_hess.copy()
            if outer_info:
                result.outer_info = outer_info

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
                model, y, _free_x_to_total_log_sp(result.x), branch_m
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
    free_sp = np.exp(np.asarray(result.x, dtype=np.float64).ravel())
    if min_sp_offset_active:
        # result.x holds log free multipliers; the reported sp is the total
        # multiplier min.sp + sp_free (mgcv's fit$sp + min.sp).
        free_sp = min_sp_free + free_sp
    model.smoothing_params[free_mask] = free_sp
    model.smoothing_params = np.maximum(model.smoothing_params, min_sp)

    model._optim_method = method
    model._optim_result = result
    if optimizer != "optim":
        _refresh_final_outer_derivatives(
            model,
            y,
            method,
            result,
            objective=objective,
            min_sp_free=min_sp_free if min_sp_offset_active else None,
        )
    _assemble_optim_trace(model, result, objective, optimizer)
    model.smoothing_score_ = float(result.fun)
    model._optim_used_gradient = bool(use_gradient)
    model._optim_used_hessian = bool(use_hessian)
    return model
