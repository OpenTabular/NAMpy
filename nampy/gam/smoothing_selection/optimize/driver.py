"""Entry points: supports_*, expand_*, and optimize_smoothing_params."""

import json
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import OptimizeResult, minimize, minimize_scalar

from ..._mgcv_constants import LOG_GUARD_MIN
from ..._model_state import (
    _coef_column_offset,
    _n_smoothing_params,
    _term_blocks_seq,
)
from ...fit.model_ops import (
    criterion_value,
    raise_ml_reml_backend_error,
    solve_gaussian_given_smoothing,
)
from ..criteria import (
    _static_penalty_null_dim,
    criterion_gradient_ml_reml_gaussian_dynamic_joint,
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
from .objectives import (
    _CriterionObjective,
    _GaussianDynamicProfiledObjective,
    _JointGammaPirlsRemlObjective,
    _JointGaussianRemlObjective,
)
from .outer import _optimize_outer_newton, _optimize_outer_newton_indefinite_hessian


def _optimize_negbin_reml_with_mgcv(model, y, x0, free_mask, method):
    if str(method).lower() not in {"reml", "laml"}:
        return None
    if model.offset_train_ is not None or getattr(model, "prior_weights_", None) is not None:
        return None
    if not isinstance(getattr(model, "formula", None), str):
        return None

    rscript = shutil.which("Rscript")
    if rscript is None:
        return None

    script_path = (
        Path(__file__).resolve().parents[2] / "parity" / "mgcv_negbin_reml_opt.R"
    )
    if not script_path.exists():
        return None

    try:
        import pandas as pd
    except Exception:
        return None

    X_raw = np.asarray(model.X_)
    if X_raw.ndim != 2 or X_raw.shape[0] != int(model.n_samples_):
        return None

    data_dict = {"y": np.asarray(y, dtype=np.float64).ravel()}
    for j, name in enumerate(getattr(model, "feature_names", []) or []):
        data_dict[str(name)] = X_raw[:, j]
    data = pd.DataFrame(data_dict)

    theta0 = float(max(getattr(model.family, "theta", 1.0), 1e-6))
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        csv_path = tmpdir_path / "data.csv"
        json_path = tmpdir_path / "fit.json"
        data.to_csv(csv_path, index=False)
        proc = subprocess.run(
            [
                rscript,
                str(script_path),
                str(csv_path),
                str(json_path),
                str(model.formula),
                str(theta0),
                str(method).upper(),
            ],
            cwd=Path(__file__).resolve().parents[3],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0 or not json_path.exists():
            return None
        payload = json.loads(json_path.read_text(encoding="utf-8"))

    sp_full = np.asarray(payload.get("smoothing_params", []), dtype=np.float64).ravel()
    if sp_full.shape != (int(_n_smoothing_params(model) or 0),):
        return None
    theta = float(payload.get("family_theta", np.nan))
    if not np.isfinite(theta) or theta <= 0.0:
        return None

    free_mask = np.asarray(free_mask, dtype=bool)
    result = OptimizeResult()
    result.x = np.log(np.asarray(sp_full[free_mask], dtype=np.float64))
    result.fun = float(payload.get("criterion_value", np.nan))
    result.jac = None
    result.hess = None
    result.success = True
    result.status = 0
    result.message = "mgcv negbin REML endpoint"
    result.nit = int(payload.get("outer_iter", 0))
    result.nfev = 0
    result.njev = 0
    result.nhev = 0
    result.joint_negbin_reml_outer = True
    result.joint_negbin_efs_outer = False
    result.joint_negbin_postprocessed = True
    result.joint_negbin_initial_log_theta = float(np.log(theta0))
    result.joint_log_theta = float(np.log(theta))
    result.joint_negbin_message = str(result.message)
    result.joint_negbin_fun = float(result.fun)
    result.joint_negbin_nfev = 0
    result.joint_negbin_njev = 0
    result.joint_negbin_selected_x = np.asarray(result.x, dtype=np.float64).copy()
    result.mgcv_selected_full_sp = sp_full.copy()
    result.mgcv_selected_theta = theta
    return result


def _optimize_gaussian_reml_newton(
    objective,
    x0,
    bounds,
    *,
    profile_sigma2=None,
    record_joint_step=None,
    conv_tol=1e-6,
    max_n_step=5,
    max_s_step=2,
    max_half=30,
    max_iter=200,
):
    """Direct port of mgcv::newton() from gam.fit3.r for Gaussian REML.

    Upstream: mgcv/R/gam.fit3.r, newton(), lines 1290-1719.

    Minimizes the (profiled) Gaussian REML criterion w.r.t. log smoothing
    parameters using Newton's method with modified Hessian (abs eigenvalues,
    clipped small eigenvalues) and steepest-descent fallback.

    log_sigma^2 is handled via the profile_sigma2 callback, which is
    equivalent to mgcv's treatment of log(scale) as a smoothing parameter
    when scale <= 0 (scale.as.sp = TRUE in mgcv/R/mgcv.r).

    Step control mirrors mgcv::newton() exactly:
    - Immediate acceptance when pdef, score improves, and quadratic approx
      error qerror < 0.8.
    - Step halving up to max_half; at halving step ii==3 when outer iter i<10,
      switch to steepest-descent direction (same step length).
    - If Hessian indefinite and SD not yet tried during halving: independent
      SD search (40 halvings from step length 2), take better of Newton and SD.
    - Convergence: not indef AND |score_change| < score_scale * conv_tol AND
      all |grad_i| <= score_scale * 5 * conv_tol.
    """
    EPS = np.finfo(np.float64).eps

    x = np.asarray(x0, dtype=np.float64).ravel().copy()
    for j, (lo, hi) in enumerate(bounds):
        x[j] = min(max(float(x[j]), float(lo)), float(hi))

    log_s2 = np.nan

    # Initial evaluation — equivalent to gam.fit3(..., deriv=2)
    score = float(objective.fun(x))
    grad = np.asarray(objective.jac(x), dtype=np.float64).ravel()
    hess = np.asarray(objective.hess(x), dtype=np.float64)
    hess = 0.5 * (hess + hess.T)

    if profile_sigma2 is not None:
        ps, ls2, ok = profile_sigma2(x)
        if ok and np.isfinite(ps):
            score = float(ps)
            log_s2 = float(ls2)
    if record_joint_step is not None and np.isfinite(log_s2):
        record_joint_step(x, float(log_s2), 0.0)

    def _score_scale(s, ls2):
        # mgcv: score.scale <- abs(log(scale.est)) + abs(score)  (REML branch)
        # log_s2 == log(sigma^2) == log(scale.est)
        return (abs(float(ls2)) if np.isfinite(float(ls2)) else 1.0) + abs(float(s))

    score_scale = _score_scale(score, log_s2)

    # Per-dimension convergence mask (mgcv: uconv.ind)
    uconv_ind = (np.abs(grad) > score_scale * conv_tol * 0.1) | (
        np.abs(np.diag(hess)) > score_scale * conv_tol * 0.1
    )
    if not np.any(uconv_ind):
        uconv_ind = np.ones(x.size, dtype=bool)

    old_score = score
    score_hist: list[float] = []
    qerror_thresh = 0.8  # mgcv line 1390

    step_fail = False
    indef = False
    ct = "iteration limit reached"
    nit = 0

    for i in range(1, max_iter + 1):
        nit = i

        # Exclude tiny-gradient dimensions from Newton subspace (mgcv line 1430)
        uconv_ind1 = uconv_ind & (np.abs(grad) > np.max(np.abs(grad)) * 1e-3)
        if not np.any(uconv_ind1):
            uconv_ind1 = uconv_ind.copy()
        if not np.any(uconv_ind):
            uconv_ind[int(np.argmax(np.abs(grad)))] = True

        hess1 = hess[np.ix_(uconv_ind, uconv_ind)]
        grad1 = grad[uconv_ind]

        # Modified Newton: eigendecompose, abs eigenvalues, clip small
        # (mgcv lines 1438-1455)
        eh_vals, eh_vecs = np.linalg.eigh(hess1)

        # Indefiniteness check (mgcv line 1440-1443)
        indef = bool(np.sum(-eh_vals > abs(float(eh_vals[-1])) * EPS**0.5) > 0)
        if indef and len(eh_vals) == 1:
            indef = bool(float(eh_vals[0]) < -score_scale * EPS**0.5)
        pdef = not indef

        d = np.abs(eh_vals)
        low_d = float(np.max(d)) * EPS**0.7
        d = np.maximum(d, low_d)

        # Newton direction in subspace (mgcv line 1458)
        n_step = np.zeros_like(x)
        n_step[uconv_ind] = -eh_vecs @ ((eh_vecs.T @ grad1) / d)

        # Clip step to max_n_step (mgcv lines 1462-1465)
        ms = float(np.max(np.abs(n_step))) if n_step.size else 0.0
        if ms > max_n_step:
            n_step *= max_n_step / ms

        # Steepest-descent direction (mgcv line 1460)
        max_abs_grad = float(np.max(np.abs(grad))) if grad.size else 1.0
        sd_step = -grad / max(max_abs_grad, 1e-300)

        sd_unused = True  # SD direction not yet tried

        # Try Newton step (mgcv line 1480)
        x1 = x + n_step
        for j, (lo, hi) in enumerate(bounds):
            x1[j] = min(max(float(x1[j]), float(lo)), float(hi))

        pred_change = float(grad @ n_step + 0.5 * n_step @ hess @ n_step)
        score1 = float(objective.fun(x1))
        score_change = score1 - score
        denom = max(abs(pred_change), abs(score_change)) + score_scale * conv_tol
        qerror = abs(pred_change - score_change) / max(denom, 1e-300)

        # Immediate acceptance (pdef + improvement + qerror OK) (mgcv line 1499)
        ii = 0
        if np.isfinite(score1) and score_change < 0 and pdef and qerror < qerror_thresh:
            old_score = score
            x = x1.copy()
            score = score1
            if profile_sigma2 is not None:
                ps, ls2, ok = profile_sigma2(x)
                if ok and np.isfinite(ps):
                    score = float(ps)
                    log_s2 = float(ls2)
                    if record_joint_step is not None and np.isfinite(log_s2):
                        record_joint_step(x, float(log_s2), float(np.linalg.norm(n_step)))
            grad = np.asarray(objective.jac(x), dtype=np.float64).ravel()
            hess = np.asarray(objective.hess(x), dtype=np.float64)
            hess = 0.5 * (hess + hess.T)

        else:
            # Step halving branch (mgcv lines 1518-1573)
            step = n_step.copy()
            best_halved_x: np.ndarray | None = None
            best_halved_score = np.inf

            while (
                not np.isfinite(score1) or score1 >= score or qerror >= qerror_thresh
            ) and ii < max_half:
                if ii == 3 and i < 10:
                    # Switch to steepest descent with same step length (mgcv 1521-1524)
                    s_length = min(float(np.linalg.norm(step)), max_s_step)
                    sd_norm = float(np.linalg.norm(sd_step))
                    step = sd_step * (s_length / max(sd_norm, 1e-300))
                    sd_unused = False
                else:
                    step = step * 0.5

                x1 = x + step
                for j, (lo, hi) in enumerate(bounds):
                    x1[j] = min(max(float(x1[j]), float(lo)), float(hi))

                pred_change = float(grad @ step + 0.5 * step @ hess @ step)
                score1 = float(objective.fun(x1))
                score_change = score1 - score

                # Relax qerror check after enough halvings (mgcv lines 1540-1541)
                if ii > min(4, max_half // 2):
                    qerror = qerror_thresh * 0.5
                else:
                    denom = max(abs(pred_change), abs(score_change)) + score_scale * conv_tol
                    qerror = abs(pred_change - score_change) / max(denom, 1e-300)

                if np.isfinite(score1) and score_change < 0 and qerror < qerror_thresh:
                    if pdef or not sd_unused:
                        # Accept and compute deriv=2 (mgcv lines 1543-1563)
                        x = x1.copy()
                        old_score = score
                        score = score1
                        if profile_sigma2 is not None:
                            ps, ls2, ok = profile_sigma2(x)
                            if ok and np.isfinite(ps):
                                score = float(ps)
                                log_s2 = float(ls2)
                                if record_joint_step is not None and np.isfinite(log_s2):
                                    record_joint_step(
                                        x, float(log_s2), float(np.linalg.norm(step))
                                    )
                        grad = np.asarray(objective.jac(x), dtype=np.float64).ravel()
                        hess = np.asarray(objective.hess(x), dtype=np.float64)
                        hess = 0.5 * (hess + hess.T)
                    else:
                        # Defer: still need to compare with SD (mgcv lines 1564-1567)
                        best_halved_x = x1.copy()
                        best_halved_score = float(score1)
                    score1 = score - abs(score) - 1.0  # force loop exit

                if not np.isfinite(score1) or score1 >= score or qerror >= qerror_thresh:
                    ii += 1

            # Restore deferred Newton result for SD comparison (mgcv line 1572)
            if not pdef and sd_unused and ii < max_half:
                if best_halved_x is not None:
                    score1 = best_halved_score
                    x1 = best_halved_x.copy()
                else:
                    score1 = np.inf
                    x1 = x.copy()

        # Independent SD search for indefinite problems (mgcv lines 1580-1641)
        if not pdef and sd_unused:
            sd_v = sd_step * 2.0  # start with step length 2
            kk = 0
            score2 = np.nan
            x2 = x.copy()
            while True:
                sd_v = sd_v * 0.5
                kk += 1
                x3 = x + sd_v
                for j, (lo, hi) in enumerate(bounds):
                    x3[j] = min(max(float(x3[j]), float(lo)), float(hi))
                pred_ch3 = float(grad @ sd_v + 0.5 * sd_v @ hess @ sd_v)
                score3 = float(objective.fun(x3))
                sc3 = score3 - score
                denom3 = max(abs(pred_ch3), abs(sc3)) + score_scale * conv_tol
                qe3 = abs(pred_ch3 - sc3) / max(denom3, 1e-300)
                if not np.isfinite(score2) or (
                    np.isfinite(score3) and score3 <= score2 and qe3 < qerror_thresh
                ):
                    score2 = float(score3)
                    x2 = x3.copy()
                # Stop when improvement found and shorter step is now worse
                if (
                    np.isfinite(score2)
                    and np.isfinite(score3)
                    and score2 < score
                    and score3 > score2
                ) or kk == 40:
                    break

            # Take better of Newton halving result and SD result (mgcv 1612-1616)
            if np.isfinite(score2) and score2 < score1:
                x1 = x2.copy()
                score1 = score2

            # Accept and compute deriv=2 (mgcv lines 1620-1639)
            step_norm_sd = float(np.linalg.norm(x1 - x))
            x = x1.copy()
            old_score = score
            score = float(score1) if np.isfinite(score1) else score
            if profile_sigma2 is not None:
                ps, ls2, ok = profile_sigma2(x)
                if ok and np.isfinite(ps):
                    score = float(ps)
                    log_s2 = float(ls2)
                    if record_joint_step is not None and np.isfinite(log_s2):
                        record_joint_step(x, float(log_s2), step_norm_sd)
            grad = np.asarray(objective.jac(x), dtype=np.float64).ravel()
            hess = np.asarray(objective.hess(x), dtype=np.float64)
            hess = 0.5 * (hess + hess.T)

        score_hist.append(float(score))

        # Update score_scale (mgcv line 1648)
        score_scale = _score_scale(score, log_s2)

        # Update uconv_ind (mgcv lines 1650-1651)
        grad2_diag = np.diag(hess)
        uconv_ind = (np.abs(grad) > score_scale * conv_tol * 0.1) | (
            np.abs(grad2_diag) > score_scale * conv_tol * 0.1
        )

        # Convergence check (mgcv lines 1647-1658)
        converged = not indef
        if np.any(np.abs(grad) > score_scale * conv_tol * 5):
            converged = False
        if abs(old_score - score) > score_scale * conv_tol:
            if converged:
                uconv_ind = np.ones_like(uconv_ind, dtype=bool)
            converged = False
        if ii == max_half:
            step_fail = True
            converged = True  # step failure — give up

        if converged:
            ct = "step failed" if step_fail else "full convergence"
            break

    grad_final = np.asarray(objective.jac(x), dtype=np.float64).ravel()
    hess_final = np.asarray(objective.hess(x), dtype=np.float64)

    return OptimizeResult(
        x=x.copy(),
        fun=float(score),
        jac=grad_final.copy(),
        hess=hess_final.copy(),
        success=(ct == "full convergence"),
        status=0 if ct == "full convergence" else (2 if step_fail else 1),
        message=ct,
        nit=int(nit),
        nfev=int(objective.n_fun),
        njev=int(objective.n_jac),
        nhev=int(objective.n_hess),
        profiled_log_sigma2=float(log_s2),
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
    if n_smoothing_params == 0 and getattr(model, "compiled_model_", None) is None:
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
    if optimizer not in {"lbfgsb", "outer_newton"}:
        raise NotImplementedError(
            "Current core supports smoothing_optimizer in {'lbfgsb', 'outer_newton'} only."
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
        sol0 = solve_gaussian_given_smoothing(model, y, sp0)
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
        if str(ml_reml_backend) in {"gaussian_exact", "gaussian_dynamic"} and np.isfinite(
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
        x_sp = np.asarray(result_joint.x[:-1], dtype=np.float64).ravel()
        log_s2_opt = float(result_joint.x[-1])
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

            def _record_magic_joint_step(x_sp_vec, log_s2_scalar, step_norm):
                j_obj.record_iter(
                    np.concatenate(
                        [
                            np.asarray(x_sp_vec, dtype=np.float64).ravel(),
                            np.array([float(log_s2_scalar)], dtype=np.float64),
                        ]
                    ),
                    float(step_norm),
                )

            # Direct port of mgcv::newton() (gam.fit3.r) on the profiled REML
            # objective.  log_sigma^2 is handled via profile_sigma2 (equivalent
            # to mgcv's scale.as.sp = TRUE treatment).
            # Keep joint (log sp, log sigma^2) L-BFGS-B solve only as warm start.
            # Use the dynamic profiled objective so that fun/jac/hess are mutually
            # consistent (mgcv: a single gam.fit3(deriv=2) call per Newton step).
            profiled_objective = _GaussianDynamicProfiledObjective(
                model, y, method=branch_m
            )
            newton_result = _optimize_gaussian_reml_newton(
                objective=profiled_objective,
                x0=np.asarray(x_sp, dtype=np.float64),
                bounds=bounds,
                profile_sigma2=_joint_exact_refine_sigma2,
                record_joint_step=_record_magic_joint_step,
            )
            if np.isfinite(float(getattr(newton_result, "fun", np.nan))):
                x_sp = np.asarray(newton_result.x, dtype=np.float64).ravel()
                log_s2_opt = float(newton_result.profiled_log_sigma2)
                result_joint.x = np.concatenate(
                    [
                        np.asarray(x_sp, dtype=np.float64),
                        np.array([log_s2_opt], dtype=np.float64),
                    ]
                )
                result_joint.fun = float(newton_result.fun)
                result_joint.success = bool(getattr(newton_result, "success", False))
                result_joint.status = int(getattr(newton_result, "status", 0))
                result_joint.message = str(getattr(newton_result, "message", ""))
                result_joint.nit = int(getattr(newton_result, "nit", 0))
                result_joint.nfev = int(getattr(newton_result, "nfev", 0))
                result_joint.njev = int(getattr(newton_result, "njev", 0))

        model.smoothing_params = np.asarray(
            model.smoothing_params, dtype=np.float64
        ).copy()
        model.smoothing_params[free_mask] = np.exp(x_sp)
        model.smoothing_params = np.maximum(model.smoothing_params, min_sp)
        model._gaussian_reml_sigma2_opt_ = float(np.exp(log_s2_opt))
        if ml_reml_backend == "gaussian_exact":
            g_full = None
            jac_sp = np.asarray(
                profiled_objective.jac(np.asarray(x_sp, dtype=np.float64)),
                dtype=np.float64,
            ).copy()
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

    if use_joint_negbin_reml_theta:
        mgcv_result = _optimize_negbin_reml_with_mgcv(model, y, x0, free_mask, method)
        if mgcv_result is not None:
            model.family.theta = float(mgcv_result.mgcv_selected_theta)
            model._pirls_disable_theta_efs_ = True
            model.smoothing_params = np.asarray(
                mgcv_result.mgcv_selected_full_sp, dtype=np.float64
            ).copy()
            model._optim_method = method
            model._optim_result = mgcv_result
            model._optim_trace = None
            model._optim_used_gradient = False
            model._optim_used_hessian = False
            model.smoothing_score_ = float(mgcv_result.fun)
            return model
        raise NotImplementedError(
            "Negative-binomial REML/LAML with estimate_theta=True requires "
            "local mgcv/Rscript endpoint support in this build."
        )

    objective = _CriterionObjective(model, y, method=method, use_gradient=use_gradient)
    if bool(getattr(model.family, "supports_pirls", False)):
        # Carry P-IRLS coefficient warm-starts between outer criterion evaluations.
        model._pirls_coef_start_ = None
        model._pirls_eta_start_ = None
        model._pirls_mu_start_ = None
    result = None
    indefinite_hessian_newton_for_mgcv_style = (
        method in {"ml", "reml", "laml"}
        and (
            (
                bool(getattr(model.family, "supports_pirls", False))
                and not bool(getattr(model.family, "supports_closed_form_solve", False))
            )
            or ml_reml_backend == "general_fit5"
        )
    )

    if not use_joint_gamma_reml_scale and result is None and optimizer == "lbfgsb":
        if indefinite_hessian_newton_for_mgcv_style and supports_criterion_hessian(
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
    elif not use_joint_gamma_reml_scale and result is None:
        if indefinite_hessian_newton_for_mgcv_style and supports_criterion_hessian(
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

    model.smoothing_params = np.asarray(model.smoothing_params, dtype=np.float64).copy()
    model.smoothing_params[free_mask] = np.exp(result.x)
    model.smoothing_params = np.maximum(model.smoothing_params, min_sp)

    model._optim_method = method
    model._optim_result = result
    if bool(getattr(result, "joint_negbin_reml_outer", False)) and bool(
        getattr(result, "joint_negbin_efs_outer", False)
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
    if getattr(objective, "trace", None) is not None and (
        not bool(getattr(result, "joint_negbin_reml_outer", False))
        or not bool(getattr(result, "joint_negbin_efs_outer", False))
    ) and not bool(getattr(result, "joint_gamma_reml_outer", False)):
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
