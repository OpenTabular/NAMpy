"""
Compatibility wrapper over stable fitted outputs and transient engine state.

`FitCoreSolution` remains the common solver return type during migration, but it
now wraps:

- `FitResult`: stable fitted outputs for consumers
- `FitState`: transient numerical workspace
- `PenalizedSystem`: engine-facing assembled system metadata
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular

from .._model_state import (
    _coef_column_offset,
    _compiled_model,
    _fit_intercept,
    _fit_state,
    _n_smoothing_params,
    _term_blocks_seq,
)
from ..engine.state import FitState, PenalizedSystem
from ..results import FitResult
from .covariance import build_bayes_and_freq_covariances


def _prediction_parameterization_map(model) -> np.ndarray | None:
    compiled_model = _compiled_model(model)
    if compiled_model is None:
        return None
    metadata = dict(getattr(compiled_model, "metadata", {}) or {})
    P = metadata.get("fit_to_prediction_parameterization_map", None)
    if P is None:
        return None
    return np.asarray(P, dtype=np.float64)


def _transform_covariance_to_prediction_space(
    cov: np.ndarray | None, P: np.ndarray
) -> np.ndarray | None:
    if cov is None:
        return None
    cov = np.asarray(cov, dtype=np.float64)
    return 0.5 * (P @ cov @ P.T + (P @ cov @ P.T).T)


def _apply_prediction_parameterization_to_fit_result(model, fit_result, fit_state):
    del fit_state
    P = _prediction_parameterization_map(model)
    if P is None:
        return fit_result

    coef_full = np.asarray(
        P @ np.asarray(fit_result.coef_full, dtype=np.float64),
        dtype=np.float64,
    )
    cov_bayes = _transform_covariance_to_prediction_space(fit_result.cov_bayes, P)
    cov_freq = _transform_covariance_to_prediction_space(fit_result.cov_freq, P)
    cov_unconditional = _transform_covariance_to_prediction_space(
        fit_result.cov_unconditional, P
    )

    beta = np.asarray(coef_full[_coef_column_offset(model) :], dtype=np.float64)
    return replace(
        fit_result,
        coef_full=coef_full,
        intercept=float(coef_full[0]) if _fit_intercept(model) else 0.0,
        beta=beta,
        cov_bayes=cov_bayes,
        cov_freq=cov_freq,
        cov_unconditional=cov_unconditional,
        penalty_quadratic=fit_result.penalty_quadratic,
    )


def _mgcv_dchol(dA: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Mirror mgcv/src/mat.c::dchol() for upper-Cholesky factors."""
    dA = np.asarray(dA, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    p = int(R.shape[0])
    dR = np.zeros_like(R, dtype=np.float64)
    for i in range(p):
        for j in range(i, p):
            x = 0.0
            for k in range(i):
                x += R[k, i] * dR[k, j] + R[k, j] * dR[k, i]
            if j > i:
                dR[i, j] = (dA[i, j] - x - R[i, j] * dR[i, i]) / R[i, i]
            else:
                dR[i, i] = 0.5 * (dA[i, i] - x) / R[i, i]
    return dR


def _mgcv_vcorr(
    dR_list: list[np.ndarray], Vr: np.ndarray, *, trans: bool
) -> np.ndarray:
    """Mirror mgcv/R/misc.r::vcorr() for dense NumPy arrays."""
    if len(dR_list) == 0:
        return np.zeros((0, 0), dtype=np.float64)
    out = np.zeros_like(np.asarray(dR_list[0], dtype=np.float64), dtype=np.float64)
    Vr = np.asarray(Vr, dtype=np.float64)
    for i, dRi in enumerate(dR_list):
        dRi = np.asarray(dRi, dtype=np.float64)
        for j, dRj in enumerate(dR_list):
            w = float(Vr[i, j])
            if w == 0.0:
                continue
            out += w * (dRi.T @ dRj if trans else dRi @ dRj.T)
    return 0.5 * (out + out.T)


def _gaussian_exact_unconditional_postfit(
    model,
    fit_result: FitResult,
    fit_state: FitState,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Mirror mgcv::gam.fit3.post.proc() EDF2 / unconditional covariance assembly.

    This applies only to Gaussian ML/REML/LAML fits after the final solve, where
    mgcv recomputes `edf2` from the fitted-model outer Hessian plus `Vb.corr()`.
    """
    if str(getattr(getattr(model, "family", None), "name", "")).lower() != "gaussian":
        return None, None

    method = str(getattr(model, "_optim_method", "")).lower()
    if method not in {"ml", "reml", "laml"}:
        return None, None

    if (
        fit_state.A is None
        or fit_state.A_inv is None
        or fit_state.XtWX is None
        or fit_result.cov_bayes is None
    ):
        return None, None

    n_sp = int(_n_smoothing_params(model) or 0)
    if n_sp == 0:
        return None, None

    fixed_mask = (
        np.zeros(n_sp, dtype=bool)
        if model.smoothing_fixed_mask_ is None
        else np.asarray(model.smoothing_fixed_mask_, dtype=bool)
    )
    free_mask = ~fixed_mask
    free_idx = np.flatnonzero(free_mask)
    if free_idx.size == 0:
        return None, None

    sp = np.asarray(model.smoothing_params, dtype=np.float64).ravel()
    log_sp = np.log(np.maximum(sp[free_mask], np.finfo(np.float64).tiny))

    from ..smoothing_selection.criteria.laplace import _penalty_derivative_matrices
    from .model_ops import criterion_hessian

    Hsp = np.asarray(
        criterion_hessian(model, model.y_, log_sp, method=method), dtype=np.float64
    )
    if Hsp.shape != (free_idx.size, free_idx.size) or not np.all(np.isfinite(Hsp)):
        return None, None
    Hsp = 0.5 * (Hsp + Hsp.T)

    optim_result = getattr(model, "_optim_result", None)
    if optim_result is not None:
        optim_result.hess = Hsp.copy()

    evals, evecs = np.linalg.eigh(Hsp)
    inv_vals = np.zeros_like(evals)
    pos = evals > 0.0
    inv_vals[pos] = 1.0 / evals[pos]
    Vsp = np.asarray(evecs @ (inv_vals[:, None] * evecs.T), dtype=np.float64)

    reg_vals = np.where(evals <= 0.0, 0.0, evals)
    d_reg = 1.0 / np.sqrt(reg_vals + 0.1)
    Vr = np.asarray(evecs @ ((d_reg * d_reg)[:, None] * evecs.T), dtype=np.float64)

    A = np.asarray(fit_state.A, dtype=np.float64)
    A_inv = np.asarray(fit_state.A_inv, dtype=np.float64)
    XtWX = np.asarray(fit_state.XtWX, dtype=np.float64)
    beta = np.asarray(fit_result.coef_full, dtype=np.float64).ravel()
    Vp = np.asarray(fit_result.cov_bayes, dtype=np.float64)
    H_coef = np.asarray(fit_result.H_coef, dtype=np.float64)
    scale = float(fit_result.scale)
    if not (np.isfinite(scale) and scale > 0.0):
        return None, None

    P_derivs_full = _penalty_derivative_matrices(model, sp)
    P_derivs = [
        np.asarray(P_derivs_full[i], dtype=np.float64).copy() for i in free_idx.tolist()
    ]
    db_drho = np.column_stack([-(A_inv @ (Pj @ beta)) for Pj in P_derivs])
    Vc1 = np.asarray(db_drho @ Vsp @ db_drho.T, dtype=np.float64)

    Vc2 = np.zeros_like(Vp)
    try:
        R = np.linalg.cholesky(A).T
        R_inv = solve_triangular(
            R,
            np.eye(R.shape[0], dtype=np.float64),
            lower=False,
            check_finite=False,
        )
        dR_inv = []
        for Pj in P_derivs:
            dRj = _mgcv_dchol(Pj, R)
            dRj_inv = -(
                solve_triangular(R, dRj, lower=False, check_finite=False) @ R_inv
            )
            dR_inv.append(np.asarray(dRj_inv, dtype=np.float64))
        Vc2 = scale * _mgcv_vcorr(dR_inv, Vr, trans=False)
    except np.linalg.LinAlgError:
        Vc2 = np.zeros_like(Vp)

    Vc = np.asarray(Vp + Vc1 + Vc2, dtype=np.float64)
    Vc = 0.5 * (Vc + Vc.T)

    edf1 = 2.0 * np.diag(H_coef) - np.sum(H_coef * H_coef.T, axis=1)
    edf2 = np.sum(Vc * XtWX, axis=1) / scale
    if float(np.sum(edf2)) > float(np.sum(edf1)):
        edf2 = np.asarray(edf1, dtype=np.float64).copy()

    return Vc, np.asarray(edf2, dtype=np.float64)


@dataclass(frozen=True)
class FitCoreSolution:
    fit_result: FitResult
    fit_state: FitState
    penalized_system: PenalizedSystem

    def __getitem__(self, key):
        return self.get(key)

    def get(self, key, default=None):
        for obj in (self.fit_result, self.fit_state, self.penalized_system):
            if hasattr(obj, key):
                value = getattr(obj, key)
                return default if value is None and not hasattr(self, key) else value
        return default

    def __getattr__(self, key):
        for obj in (self.fit_result, self.fit_state, self.penalized_system):
            if hasattr(obj, key):
                return getattr(obj, key)
        raise AttributeError(key)

    def with_fit_result(self, **changes) -> "FitCoreSolution":
        return replace(self, fit_result=replace(self.fit_result, **changes))

    def with_fit_state(self, **changes) -> "FitCoreSolution":
        new_state = replace(self.fit_state, **changes)
        return replace(
            self, fit_state=new_state, penalized_system=new_state.to_penalized_system()
        )

    @classmethod
    def from_dict(cls, data: dict) -> "FitCoreSolution":
        fit_result = FitResult(
            coef_full=np.asarray(data["coef_full"], dtype=np.float64),
            intercept=float(data["intercept"]),
            beta=np.asarray(data["beta"], dtype=np.float64),
            eta=np.asarray(data["eta"], dtype=np.float64),
            mu=np.asarray(data["mu"], dtype=np.float64),
            rss=(None if data.get("rss", None) is None else float(data["rss"])),
            deviance=float(data["deviance"]),
            edf=float(data["edf"]),
            trace_H=float(data["trace_H"]),
            scale=float(data["scale"]),
            cov_bayes=(
                None
                if data.get("cov_bayes", None) is None
                else np.asarray(data["cov_bayes"], dtype=np.float64)
            ),
            cov_freq=(
                None
                if data.get("cov_freq", None) is None
                else np.asarray(data["cov_freq"], dtype=np.float64)
            ),
            cov_unconditional=(
                None
                if data.get("cov_unconditional", None) is None
                else np.asarray(data["cov_unconditional"], dtype=np.float64)
            ),
            H_coef=np.asarray(data["H_coef"], dtype=np.float64),
            edf2=(
                None
                if data.get("edf2", None) is None
                else np.asarray(data["edf2"], dtype=np.float64)
            ),
            penalty_quadratic=(
                None
                if data.get("penalty_quadratic", None) is None
                else float(data["penalty_quadratic"])
            ),
            loglik=(
                None if data.get("loglik", None) is None else float(data["loglik"])
            ),
            converged=data.get("converged", None),
            iter=data.get("iter", None),
            failed_step=data.get("failed_step", None),
            failure_reason=data.get("failure_reason", None),
            inner_trace=data.get("inner_trace", None),
        )
        fit_state = FitState(
            X=(
                None
                if data.get("X", None) is None
                else np.asarray(data["X"], dtype=np.float64)
            ),
            A=(
                None
                if data.get("A", None) is None
                else np.asarray(data["A"], dtype=np.float64)
            ),
            A_inv=(
                None
                if data.get("A_inv", None) is None
                else np.asarray(data["A_inv"], dtype=np.float64)
            ),
            XtWX=(
                None
                if data.get("XtWX", None) is None
                else np.asarray(data["XtWX"], dtype=np.float64)
            ),
            P=(
                None
                if data.get("P", None) is None
                else np.asarray(data["P"], dtype=np.float64)
            ),
            penalty_matrix=(
                None
                if data.get("penalty_matrix", None) is None
                else np.asarray(data["penalty_matrix"], dtype=np.float64)
            ),
            working_weights=(
                None
                if data.get("working_weights", None) is None
                else np.asarray(data["working_weights"], dtype=np.float64)
            ),
            fisher_weights=(
                None
                if data.get("fisher_weights", None) is None
                else np.asarray(data["fisher_weights"], dtype=np.float64)
            ),
            working_response=(
                None
                if data.get("working_response", None) is None
                else np.asarray(data["working_response"], dtype=np.float64)
            ),
            offset=(
                None
                if data.get("offset", None) is None
                else np.asarray(data["offset"], dtype=np.float64)
            ),
            log_det_XtWX_plus_penalty=(
                None
                if data.get("log_det_XtWX_plus_penalty", None) is None
                else float(data["log_det_XtWX_plus_penalty"])
            ),
            penalized_system_rank=(
                None
                if data.get("penalized_system_rank", None) is None
                else int(data["penalized_system_rank"])
            ),
            dropped_column_indices=(
                None
                if data.get("dropped_column_indices", None) is None
                else np.asarray(data["dropped_column_indices"], dtype=np.int64)
            ),
            scale=float(data["scale"]),
        )
        return cls(
            fit_result=fit_result,
            fit_state=fit_state,
            penalized_system=fit_state.to_penalized_system(),
        )


def compute_edf_by_term(model, H_coef):
    offset0 = _coef_column_offset(model)
    fit_state = _fit_state(model)
    X_full = None if fit_state is None else getattr(fit_state, "X", None)
    A_inv = None if fit_state is None else getattr(fit_state, "A_inv", None)
    w = None if fit_state is None else getattr(fit_state, "working_weights", None)

    edf = []
    for tb in _term_blocks_seq(model):
        sl = slice(
            offset0 + int(tb.coef_slice.start),
            offset0 + int(tb.coef_slice.stop),
        )
        val = float(np.trace(H_coef[sl, sl]))

        if (
            str(getattr(tb, "term_type", "")) == "random_effect"
            and _fit_intercept(model)
            and X_full is not None
            and w is not None
        ):
            try:
                X = np.asarray(X_full, dtype=np.float64)
                w_arr = np.asarray(w, dtype=np.float64).ravel()
                Xp = X[:, :offset0]
                Xt = X[:, sl]
                if Xp.shape[1] > 0 and Xt.shape[1] > 0 and w_arr.shape[0] == X.shape[0]:
                    sqrt_w = np.sqrt(np.clip(w_arr, 0.0, None))
                    Xp_w = sqrt_w[:, None] * Xp
                    Xt_w = sqrt_w[:, None] * Xt
                    coef = np.linalg.lstsq(Xp_w, Xt_w, rcond=None)[0]
                    Xt_eff_w = Xt_w - Xp_w @ coef
                    sp = np.asarray(
                        getattr(model, "smoothing_params", np.empty((0,), dtype=float)),
                        dtype=np.float64,
                    ).ravel()
                    tb_sp = (
                        sp[np.asarray(tb.smoothing_indices, dtype=int)]
                        if len(getattr(tb, "smoothing_indices", [])) > 0 and sp.size > 0
                        else np.empty((0,), dtype=np.float64)
                    )
                    if tb_sp.size > 0 and np.max(tb_sp) <= 1e-20:
                        val = float(np.linalg.matrix_rank(Xt_eff_w))
                    elif A_inv is not None:
                        A_inv_arr = np.asarray(A_inv, dtype=np.float64)
                        val = float(
                            np.trace(
                                (Xt_eff_w.T @ (sqrt_w[:, None] * X) @ A_inv_arr)[:, sl]
                            )
                        )
            except Exception:
                pass

        edf.append(val)
    edf = np.asarray(edf, dtype=np.float64)
    for i, tb in enumerate(_term_blocks_seq(model)):
        by_info = getattr(tb, "by_variable_info", None)
        runtime_by_name = getattr(by_info, "name", None)
        runtime_by_is_constant = getattr(by_info, "is_constant", None)
        deleted = getattr(tb, "deleted_columns", None)
        n_deleted = int(0 if deleted is None else np.asarray(deleted, dtype=int).size)
        if (
            n_deleted > 0
            and runtime_by_name is not None
            and not bool(runtime_by_is_constant)
        ):
            edf[i] += float(n_deleted)
            if i > 0:
                edf[i - 1] -= float(n_deleted)
    return edf


def assign_fit_solution(model, sol: FitCoreSolution):
    fit_result = sol.fit_result
    fit_state = sol.fit_state

    H_post = np.asarray(fit_result.H_coef, dtype=np.float64)
    trace_H_post = float(fit_result.trace_H)
    scale_post = float(fit_result.scale)
    Vp_post = (
        None
        if fit_result.cov_bayes is None
        else np.asarray(fit_result.cov_bayes, dtype=np.float64)
    )
    Vf_post = (
        None
        if fit_result.cov_freq is None
        else np.asarray(fit_result.cov_freq, dtype=np.float64)
    )
    if (
        not bool(getattr(model.family, "canonical_link", False))
        and fit_state.X is not None
        and fit_state.P is not None
        and fit_state.fisher_weights is not None
    ):
        try:
            X = np.asarray(fit_state.X, dtype=np.float64)
            P = np.asarray(fit_state.P, dtype=np.float64)
            fisher_w = np.asarray(fit_state.fisher_weights, dtype=np.float64).ravel()
            XtFX = X.T @ (fisher_w[:, None] * X)
            cA_post, lower_post = cho_factor(
                XtFX + P, overwrite_a=False, check_finite=False
            )
            A_inv_post = cho_solve(
                (cA_post, lower_post),
                np.eye(X.shape[1], dtype=np.float64),
                check_finite=False,
            )
            if str(getattr(model.family, "name", "")).lower() == "gamma":
                scale_post = float(
                    model.family.estimate_dispersion(
                        model.y_,
                        fit_result.mu,
                        edf=trace_H_post,
                        weights=model.prior_weights_,
                    )
                )
            Vp_post, Vf_post, H_post = build_bayes_and_freq_covariances(
                scale_post, A_inv_post, XtFX
            )
            trace_H_post = float(np.trace(H_post))
        except Exception:
            pass
    if (
        str(getattr(model.family, "name", "")).lower() == "gamma"
        and fit_state.X is not None
        and fit_state.P is not None
        and fit_state.fisher_weights is not None
    ):
        X = np.asarray(fit_state.X, dtype=np.float64)
        P = np.asarray(fit_state.P, dtype=np.float64)
        fisher_w = np.asarray(fit_state.fisher_weights, dtype=np.float64).ravel()
        XtFX = X.T @ (fisher_w[:, None] * X)
        cA_post, lower_post = cho_factor(
            XtFX + P, overwrite_a=False, check_finite=False
        )
        A_inv_post = cho_solve(
            (cA_post, lower_post),
            np.eye(X.shape[1], dtype=np.float64),
            check_finite=False,
        )
        H_post = A_inv_post @ XtFX
        trace_H_post = float(np.trace(H_post))
        scale_post = float(
            model.family.estimate_dispersion(
                model.y_,
                fit_result.mu,
                edf=trace_H_post,
                weights=model.prior_weights_,
            )
        )
        Vp_post, Vf_post, H_post = build_bayes_and_freq_covariances(
            scale_post, A_inv_post, XtFX
        )
        trace_H_post = float(np.trace(H_post))
    fit_state = replace(fit_state, scale=float(scale_post))
    penalized_system = fit_state.to_penalized_system()
    fit_result = replace(
        fit_result,
        trace_H=float(trace_H_post),
        edf=float(trace_H_post),
        scale=float(scale_post),
        cov_bayes=(
            None if Vp_post is None else np.asarray(Vp_post, dtype=np.float64).copy()
        ),
        cov_freq=(
            None if Vf_post is None else np.asarray(Vf_post, dtype=np.float64).copy()
        ),
        H_coef=np.asarray(H_post, dtype=np.float64).copy(),
    )
    cov_unconditional_post = (
        None
        if fit_result.cov_unconditional is None
        else np.asarray(fit_result.cov_unconditional, dtype=np.float64).copy()
    )
    edf2_post = (
        None
        if fit_result.edf2 is None
        else np.asarray(fit_result.edf2, dtype=np.float64).copy()
    )
    Vc_gauss, edf2_gauss = _gaussian_exact_unconditional_postfit(
        model, fit_result, fit_state
    )
    if Vc_gauss is not None:
        cov_unconditional_post = np.asarray(Vc_gauss, dtype=np.float64).copy()
    if edf2_gauss is not None:
        edf2_post = np.asarray(edf2_gauss, dtype=np.float64).copy()
    fit_result = replace(
        fit_result,
        cov_unconditional=cov_unconditional_post,
        edf2=edf2_post,
    )
    fit_result = _apply_prediction_parameterization_to_fit_result(
        model, fit_result, fit_state
    )
    sol = replace(
        sol,
        fit_result=fit_result,
        fit_state=fit_state,
        penalized_system=penalized_system,
    )

    model.fit_core_solution_ = sol
    model.gam_result_ = None
    if (
        getattr(model.family, "family_class", "") == "general"
        and getattr(model, "_coef_reduced_to_full_idx", None) is not None
    ):
        full_idx = np.asarray(model._coef_reduced_to_full_idx, dtype=int)
        edf = []
        for tb in _term_blocks_seq(model):
            idx = full_idx[tb.coef_slice]
            edf.append(float(np.trace(H_post[np.ix_(idx, idx)])))
        model._edf_by_term_fit_ = np.asarray(edf, dtype=np.float64)
    else:
        model._edf_by_term_fit_ = compute_edf_by_term(model, H_post)

    if bool(getattr(model, "_fitted", False)):
        from .model_ops import sync_gam_result

        sync_gam_result(model)


__all__ = [
    "FitCoreSolution",
    "FitState",
    "PenalizedSystem",
    "FitResult",
    "assign_fit_solution",
    "compute_edf_by_term",
]
