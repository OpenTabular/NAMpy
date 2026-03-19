import warnings

import numpy as np
from scipy.optimize import OptimizeResult
from scipy.optimize import minimize

from ..fit.penalized_system import build_full_design
from .criteria import (
    criterion_infinite_sp_signal,
    criterion_gradient,
    criterion_hessian,
    criterion_value,
    resolve_ml_reml_scoring_backend,
)


class _CriterionObjective:
    def __init__(self, model, y, method, use_gradient):
        self.model = model
        self.y = y
        self.method = method
        self.use_gradient = bool(use_gradient)
        self._last_x = None
        self._last_fun = None
        self._last_grad = None
        self._last_hess = None
        self.n_fun = 0
        self.n_jac = 0
        self.n_hess = 0
        self.capture_trace = bool(getattr(model, "exact_mgcv_mode", False))
        self.trace = []
        self._trace_index_by_x = {}

    def _same_x(self, x):
        return self._last_x is not None and np.array_equal(self._last_x, x)

    def fun(self, x):
        x = np.asarray(x, dtype=np.float64).ravel()
        if self._same_x(x) and self._last_fun is not None:
            return float(self._last_fun)
        val = float(criterion_value(self.model, self.y, x, method=self.method))
        self.n_fun += 1
        self._last_x = x.copy()
        self._last_fun = val
        self._last_grad = None
        self._last_hess = None
        if self.capture_trace:
            key = tuple(np.asarray(x, dtype=np.float64).tolist())
            idx = self._trace_index_by_x.get(key, None)
            if idx is None:
                idx = len(self.trace)
                self._trace_index_by_x[key] = idx
                self.trace.append(
                    {
                        "x": np.asarray(x, dtype=np.float64).copy(),
                        "fun": float(val),
                        "grad": None,
                        "hess": None,
                        "n_fun": int(self.n_fun),
                        "n_jac": int(self.n_jac),
                        "n_hess": int(self.n_hess),
                    }
                )
            else:
                self.trace[idx]["fun"] = float(val)
                self.trace[idx]["n_fun"] = int(self.n_fun)
        return val

    def jac(self, x):
        x = np.asarray(x, dtype=np.float64).ravel()
        if self._same_x(x) and self._last_grad is not None:
            return self._last_grad.copy()

        if not self._same_x(x) or self._last_fun is None:
            self.fun(x)

        grad = np.asarray(
            criterion_gradient(self.model, self.y, x, method=self.method),
            dtype=np.float64,
        )
        self.n_jac += 1
        self._last_x = x.copy()
        self._last_grad = grad.copy()
        self._last_hess = None
        if self.capture_trace:
            key = tuple(np.asarray(x, dtype=np.float64).tolist())
            idx = self._trace_index_by_x.get(key, None)
            if idx is None:
                idx = len(self.trace)
                self._trace_index_by_x[key] = idx
                self.trace.append(
                    {
                        "x": np.asarray(x, dtype=np.float64).copy(),
                        "fun": None,
                        "grad": grad.copy(),
                        "hess": None,
                        "n_fun": int(self.n_fun),
                        "n_jac": int(self.n_jac),
                        "n_hess": int(self.n_hess),
                    }
                )
            else:
                self.trace[idx]["grad"] = grad.copy()
                self.trace[idx]["n_jac"] = int(self.n_jac)
        return grad

    def hess(self, x):
        x = np.asarray(x, dtype=np.float64).ravel()
        if self._same_x(x) and self._last_hess is not None:
            return self._last_hess.copy()

        if not self._same_x(x) or self._last_fun is None:
            self.fun(x)

        hess = np.asarray(
            criterion_hessian(self.model, self.y, x, method=self.method),
            dtype=np.float64,
        )
        self.n_hess += 1
        self._last_x = x.copy()
        self._last_hess = hess.copy()
        if self.capture_trace:
            key = tuple(np.asarray(x, dtype=np.float64).tolist())
            idx = self._trace_index_by_x.get(key, None)
            if idx is None:
                idx = len(self.trace)
                self._trace_index_by_x[key] = idx
                self.trace.append(
                    {
                        "x": np.asarray(x, dtype=np.float64).copy(),
                        "fun": None,
                        "grad": None,
                        "hess": hess.copy(),
                        "n_fun": int(self.n_fun),
                        "n_jac": int(self.n_jac),
                        "n_hess": int(self.n_hess),
                    }
                )
            else:
                self.trace[idx]["hess"] = hess.copy()
                self.trace[idx]["n_hess"] = int(self.n_hess)
        return hess


def supports_criterion_gradient(model, method):
    method = str(method).lower()
    return method in {"gcv", "ubre", "aic", "ubreaic", "ml", "reml", "laml"}


def supports_criterion_hessian(model, method):
    method = str(method).lower()
    return method in {"gcv", "ubre", "aic", "ubreaic", "ml", "reml", "laml"}


def _project_to_bounds(x, bounds):
    x = np.asarray(x, dtype=np.float64).copy()
    for i, (lo, hi) in enumerate(bounds):
        x[i] = min(max(x[i], lo), hi)
    return x


def _mgcv_style_initial_smoothing_params(model, y):
    penalty_blocks = getattr(model, "penalty_blocks_", None)
    n_sp = int(getattr(model, "n_smoothing_params_", 0) or 0)
    if not penalty_blocks or n_sp == 0:
        return None

    X = build_full_design(model.Z, fit_intercept=model.fit_intercept)
    y = np.asarray(y, dtype=np.float64).ravel()

    try:
        mu0 = np.asarray(model.family.initialize_mu(y), dtype=np.float64)
        eta0 = np.asarray(model.family.link(mu0), dtype=np.float64)
        mu_eta = np.asarray(model.family.mu_eta(eta0), dtype=np.float64)
        var_mu = np.asarray(model.family.variance(mu0), dtype=np.float64)
    except Exception:
        return None

    weights = np.sqrt(
        np.clip(mu_eta * mu_eta / np.maximum(var_mu, 1e-12), 1e-12, None)
    )
    Xw = weights[:, None] * X
    ldxx = np.sum(Xw * Xw, axis=0)
    ldss = np.zeros_like(ldxx)
    def_sp = np.zeros(n_sp, dtype=np.float64)
    counts = np.zeros(n_sp, dtype=np.int64)
    penalized = np.zeros_like(ldxx, dtype=bool)

    for pb in penalty_blocks:
        S = np.asarray(pb.matrix, dtype=np.float64)
        if S.size == 0:
            continue
        start = int(pb.coef_slice.start)
        stop = int(pb.coef_slice.stop)
        dS = np.diag(np.abs(S))
        if dS.size == 0:
            continue

        maS = float(np.max(np.abs(S)))
        if not np.isfinite(maS) or maS <= 0.0:
            continue
        thresh = np.finfo(np.float64).eps ** 0.8 * maS
        rsS = np.mean(np.abs(S), axis=1)
        csS = np.mean(np.abs(S), axis=0)
        ind = (rsS > thresh) & (csS > thresh) & (dS > thresh)
        if not np.any(ind):
            continue

        xx = ldxx[start:stop][ind]
        ss = dS[ind]
        if xx.size == 0 or ss.size == 0:
            continue

        sizeXX = float(np.mean(xx))
        sizeS = float(np.mean(ss))
        if not np.isfinite(sizeXX) or not np.isfinite(sizeS) or sizeS <= 0.0:
            continue

        lam = sizeXX / sizeS
        j = int(pb.smoothing_index)
        def_sp[j] += lam
        counts[j] += 1
        ldss[start:stop] += lam * np.diag(S)
        penalized[start:stop] |= ind

    ok = counts > 0
    if not np.any(ok):
        return None
    def_sp[ok] /= counts[ok]
    def_sp[~ok] = 1.0

    use = (ldss > 0.0) & penalized & (ldxx > 0.0)
    if np.any(use):
        xx = ldxx[use]
        ss = ldss[use]
        ratio = float(np.mean(xx / (xx + ss)))
        while ratio > 0.4:
            def_sp *= 10.0
            ss *= 10.0
            ratio = float(np.mean(xx / (xx + ss)))
        while ratio < 0.4:
            def_sp /= 10.0
            ss /= 10.0
            ratio = float(np.mean(xx / (xx + ss)))

    def_sp = np.maximum(def_sp, 1e-12)
    return def_sp


def _rollback_working_infinite_smoothing_params(
    objective,
    result,
    x0,
    bounds,
    method,
    *,
    conv_tol=1e-6,
    max_iter=5,
):
    method = str(method).lower()
    if method not in {"ml", "reml", "laml"}:
        return result

    if getattr(result, "_rolled_retry", False):
        return result
    x = np.asarray(result.x, dtype=np.float64).copy()
    if x.size == 0:
        return result

    score = float(objective.fun(x))
    grad_signal, dvkk = criterion_infinite_sp_signal(
        objective.model, objective.y, x, method=method
    )
    grad = np.asarray(grad_signal, dtype=np.float64)
    if grad.ndim != 1 or grad.shape[0] != x.size:
        return result

    score_scale = 1.0 + abs(score)
    informative = np.abs(grad) > score_scale * conv_tol * 0.1
    if dvkk.shape == grad.shape:
        informative |= np.abs(dvkk) > score_scale * conv_tol * 0.1
    if np.all(informative):
        return result

    x_roll = x.copy()
    informative0 = informative.copy()
    rolled = False

    shrink_counts = np.zeros_like(x_roll, dtype=int)
    for _ in range(int(max_iter)):
        stuck = ~informative
        if not np.any(stuck):
            break
        next_counts = shrink_counts.copy()
        next_counts[stuck] += 1
        if np.any(next_counts > 1):
            break
        trial = x_roll.copy()
        trial[stuck] = 0.8 * trial[stuck] + 0.2 * x0[stuck]
        trial = _project_to_bounds(trial, bounds)
        if np.array_equal(trial, x_roll):
            break

        trial_score = float(objective.fun(trial))
        trial_grad, trial_dvkk = criterion_infinite_sp_signal(
            objective.model, objective.y, trial, method=method
        )
        trial_grad = np.asarray(trial_grad, dtype=np.float64)
        if trial_grad.ndim != 1 or trial_grad.shape[0] != trial.size:
            break

        x_roll = trial
        score = trial_score
        grad = trial_grad
        dvkk = (
            np.asarray(trial_dvkk, dtype=np.float64)
            if np.asarray(trial_dvkk).shape == trial_grad.shape
            else np.full_like(trial_grad, np.nan)
        )

        informative = np.abs(grad) > score_scale * conv_tol * 20.0
        informative |= np.abs(dvkk) > score_scale * conv_tol * 20.0
        informative |= informative0
        rolled = True
        shrink_counts = next_counts

    if not rolled:
        return result

    rolled_result = OptimizeResult()
    rolled_result.x = x_roll.copy()
    rolled_result.fun = float(score)
    rolled_result.jac = np.asarray(objective.jac(x_roll), dtype=np.float64)
    rolled_result.hess = np.asarray(objective.hess(x_roll), dtype=np.float64)
    rolled_result.success = bool(getattr(result, "success", True))
    rolled_result.status = int(getattr(result, "status", 0))
    rolled_result.message = str(getattr(result, "message", "rolled-back endpoint"))
    rolled_result.nit = int(getattr(result, "nit", 0))
    rolled_result.nfev = int(getattr(result, "nfev", getattr(objective, "n_fun", 0)))
    rolled_result.njev = int(getattr(result, "njev", getattr(objective, "n_jac", 0)))
    rolled_result.nhev = int(getattr(result, "nhev", getattr(objective, "n_hess", 0)))
    rolled_result.rolled_back_infinite_sp = True
    rolled_result.rollback_start_x = x.copy()
    rolled_result.rollback_final_x = x_roll.copy()

    stuck0 = ~informative0

    # Retry once from rolled point, but guard against drifting back to
    # effectively infinite smoothing on weakly identified dimensions.
    retry = minimize(
        fun=objective.fun,
        x0=x_roll,
        method="L-BFGS-B",
        jac=objective.jac if objective.use_gradient else None,
        bounds=bounds,
    )
    retry._rolled_retry = True
    retry.rolled_back_infinite_sp = True
    retry.rollback_start_x = x.copy()
    retry.rollback_final_x = x_roll.copy()
    if not hasattr(retry, "hess"):
        retry.hess = np.asarray(objective.hess(retry.x), dtype=np.float64)

    retry_x = np.asarray(retry.x, dtype=np.float64)
    bad_retry = False
    if retry_x.shape != x_roll.shape:
        bad_retry = True
    else:
        # If previously stuck dimensions inflate again, keep the rolled point.
        if np.any(retry_x[stuck0] > (x_roll[stuck0] + 0.5)):
            bad_retry = True
        # Also reject retry if objective meaningfully worsens.
        if float(retry.fun) > float(rolled_result.fun) + 1e-8 * (1.0 + abs(float(rolled_result.fun))):
            bad_retry = True

    if bad_retry:
        rolled_result.retry_rejected = True
        return rolled_result

    retry.retry_rejected = False
    return retry


def _stabilize_flat_smoothing_params(objective, result, x0, bounds, method, *, conv_tol=1e-6):
    method = str(method).lower()
    if method not in {"ml", "reml", "laml"}:
        return result

    x = np.asarray(result.x, dtype=np.float64).copy()
    if x.size == 0:
        return result

    score = float(objective.fun(x))
    grad_signal, dvkk = criterion_infinite_sp_signal(
        objective.model, objective.y, x, method=method
    )
    grad = np.asarray(grad_signal, dtype=np.float64)
    if grad.ndim != 1 or grad.shape[0] != x.size:
        return result

    score_scale = 1.0 + abs(score)
    flat = np.abs(grad) <= score_scale * conv_tol * 0.5

    if not np.any(flat):
        return result

    improved = False
    x_work = x.copy()
    score_work = score
    score_tol = max(1e-5, score_scale * 1e-7)

    for j in np.flatnonzero(flat):
        local_x = x_work.copy()
        local_best = local_x[j]
        local_best_score = score_work
        for _ in range(10):
            trial = local_x.copy()
            trial[j] = trial[j] - 0.5
            trial = _project_to_bounds(trial, bounds)
            if trial[j] >= local_x[j] - 1e-12:
                break
            trial_score = float(objective.fun(trial))
            if np.isfinite(trial_score) and trial_score <= local_best_score + score_tol:
                local_x = trial
                local_best = trial[j]
                local_best_score = trial_score
            else:
                break
        if local_best < x_work[j] - 1e-12:
            x_work[j] = local_best
            score_work = local_best_score
            improved = True

    if not improved:
        return result

    result.x = x_work
    result.fun = float(score_work)
    result.jac = np.asarray(objective.jac(x_work), dtype=np.float64)
    result.hess = np.asarray(objective.hess(x_work), dtype=np.float64)
    result.flat_sp_stabilized = True
    return result


def _optimize_outer_newton(objective, x0, bounds, max_iter=50, grad_tol=1e-6, step_tol=1e-8):
    x = _project_to_bounds(x0, bounds)
    f = float(objective.fun(x))
    success = False
    message = "maximum iterations reached"
    nit = 0

    for nit in range(1, max_iter + 1):
        g = np.asarray(objective.jac(x), dtype=np.float64)
        if not np.all(np.isfinite(g)):
            message = "non-finite gradient"
            break
        if np.linalg.norm(g, ord=np.inf) <= grad_tol:
            success = True
            message = "gradient tolerance satisfied"
            break

        H = np.asarray(objective.hess(x), dtype=np.float64)
        if not np.all(np.isfinite(H)):
            message = "non-finite Hessian"
            break

        direction = None
        ridge = 0.0
        eye = np.eye(len(x), dtype=np.float64)
        for _ in range(8):
            try:
                direction_try = -np.linalg.solve(H + ridge * eye, g)
            except np.linalg.LinAlgError:
                ridge = max(1e-6, 10.0 * ridge if ridge > 0 else 1e-6)
                continue
            if float(g @ direction_try) < 0.0 and np.all(np.isfinite(direction_try)):
                direction = direction_try
                break
            ridge = max(1e-6, 10.0 * ridge if ridge > 0 else 1e-6)

        if direction is None:
            direction = -g

        step_norm = float(np.linalg.norm(direction))
        if step_norm <= step_tol:
            success = True
            message = "step tolerance satisfied"
            break

        accepted = False
        candidate_directions = [direction]
        if float(g @ direction) >= 0.0:
            candidate_directions = []
        candidate_directions.append(-g)

        for cand_direction in candidate_directions:
            alpha = 1.0
            gtd = float(g @ cand_direction)
            for _ in range(25):
                x_trial = _project_to_bounds(x + alpha * cand_direction, bounds)
                if np.array_equal(x_trial, x):
                    alpha *= 0.5
                    continue
                f_trial = float(objective.fun(x_trial))
                if np.isfinite(f_trial) and f_trial <= f + 1e-4 * alpha * gtd:
                    x = x_trial
                    f = f_trial
                    accepted = True
                    break
                alpha *= 0.5
            if accepted:
                break

        if not accepted:
            message = "line search failed"
            break

    g_final = np.asarray(objective.jac(x), dtype=np.float64)
    h_final = np.asarray(objective.hess(x), dtype=np.float64)
    return OptimizeResult(
        x=x,
        fun=float(f),
        jac=g_final,
        hess=h_final,
        success=bool(success),
        status=0 if success else 1,
        message=message,
        nit=nit,
        nfev=objective.n_fun,
        njev=objective.n_jac,
        nhev=objective.n_hess,
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
    exact_mode = bool(getattr(model, "exact_mgcv_mode", False))
    exact_gaussian = exact_mode and str(getattr(model.family, "name", "")).lower() == "gaussian"

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

        if user_sp is None and not bool(getattr(model.family, "supports_closed_form_solve", False)):
            init = _mgcv_style_initial_smoothing_params(model, y)
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

    objective = _CriterionObjective(model, y, method=method, use_gradient=use_gradient)

    if optimizer == "lbfgsb":
        result = minimize(
            fun=objective.fun,
            x0=x0,
            method="L-BFGS-B",
            jac=objective.jac if use_gradient else None,
            bounds=bounds,
        )
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

    if not result.success:
        warnings.warn(f"Smoothing optimisation did not converge: {result.message}")

    if not exact_gaussian:
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
