"""
Stable fitted outputs and transient fit state.

`FitCoreSolution` remains the common solver return type during migration, but it
now wraps:

- `FitResult`: stable fitted outputs for consumers
- `FitState`: transient numerical workspace
- `PenalizedSystem`: assembled penalized-system metadata
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .._model_state import (
    _coef_column_offset,
    _fit_state,
    _term_blocks_seq,
)
from ..linalg import numerical_rank
from ..results import FitResult
from .parameterization import (
    export_fit_result_to_prediction_space,
    prediction_parameterization_map,
)
from .postprocess.unconditional_covariance import (
    _restore_pirls_dbeta_to_original_parameterization as _restore_pirls_dbeta_impl,
)
from .postprocess.unconditional_covariance import (
    _restore_pirls_rank_root_to_original_parameterization as _restore_pirls_rank_root_impl,
)
from .postprocess.unconditional_covariance import (
    apply_unconditional_postfit,
)


@dataclass(frozen=True)
class PenalizedSystem:
    X: np.ndarray | None = None
    A: np.ndarray | None = None
    XtWX: np.ndarray | None = None
    P: np.ndarray | None = None
    penalty_matrix: np.ndarray | None = None
    offset: np.ndarray | None = None
    log_det_XtWX_plus_penalty: float | None = None
    penalized_system_rank: int | None = None
    dropped_column_indices: np.ndarray | None = None


@dataclass(frozen=True)
class FitState:
    X: np.ndarray | None = None
    A: np.ndarray | None = None
    A_inv: np.ndarray | None = None
    XtWX: np.ndarray | None = None
    P: np.ndarray | None = None
    penalty_matrix: np.ndarray | None = None
    working_weights: np.ndarray | None = None
    fisher_weights: np.ndarray | None = None
    working_response: np.ndarray | None = None
    offset: np.ndarray | None = None
    log_det_XtWX_plus_penalty: float | None = None
    penalized_system_rank: int | None = None
    dropped_column_indices: np.ndarray | None = None
    scale: float | None = None

    def to_penalized_system(self) -> PenalizedSystem:
        return PenalizedSystem(
            X=self.X,
            A=self.A,
            XtWX=self.XtWX,
            P=self.P,
            penalty_matrix=self.penalty_matrix,
            offset=self.offset,
            log_det_XtWX_plus_penalty=self.log_det_XtWX_plus_penalty,
            penalized_system_rank=self.penalized_system_rank,
            dropped_column_indices=self.dropped_column_indices,
        )


def _prediction_parameterization_map(model) -> np.ndarray | None:
    return prediction_parameterization_map(model)


def _apply_prediction_parameterization_to_fit_result(model, fit_result, fit_state):
    del fit_state
    return export_fit_result_to_prediction_space(model, fit_result)


def _restore_pirls_dbeta_to_original_parameterization(current, dbeta_rank):
    return _restore_pirls_dbeta_impl(current, dbeta_rank)


def _restore_pirls_rank_root_to_original_parameterization(current, rank_root):
    return _restore_pirls_rank_root_impl(current, rank_root)


def _apply_unconditional_postfit(model, sol, fit_result, fit_state):
    return apply_unconditional_postfit(model, sol, fit_result, fit_state)


@dataclass(frozen=True)
class FitCoreSolution:
    fit_result: FitResult
    fit_state: FitState
    penalized_system: PenalizedSystem

    def _delegate_objects(self):
        state = object.__getattribute__(self, "__dict__")
        return tuple(
            state[name]
            for name in ("fit_result", "fit_state", "penalized_system")
            if name in state
        )

    def __getitem__(self, key):
        return self.get(key)

    def get(self, key, default=None):
        state = object.__getattribute__(self, "__dict__")
        for obj in self._delegate_objects():
            if hasattr(obj, key):
                value = getattr(obj, key)
                return default if value is None and key not in state else value
        return default

    def __getattr__(self, key):
        for obj in self._delegate_objects():
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
            coef_space=str(data.get("coef_space", "fit")),
            cov_bayes_space=str(data.get("cov_bayes_space", "fit")),
            cov_freq_space=str(data.get("cov_freq_space", "fit")),
            cov_unconditional_space=str(data.get("cov_unconditional_space", "fit")),
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
    edf = []
    for tb in _term_blocks_seq(model):
        sl = slice(
            offset0 + int(tb.coef_slice.start),
            offset0 + int(tb.coef_slice.stop),
        )
        edf.append(float(np.trace(H_coef[sl, sl])))
    return np.asarray(edf, dtype=np.float64)


def _fallback_single_smooth_edf(model, trace_H: float):
    blocks = list(_term_blocks_seq(model))
    smooth_blocks = [
        tb
        for tb in blocks
        if str(getattr(tb, "term_type", "")) != "parametric"
    ]
    if len(smooth_blocks) != 1:
        return None
    smooth = smooth_blocks[0]
    if str(getattr(smooth, "basis_name", "")).lower() == "re":
        fit_state = _fit_state(model)
        X = None if fit_state is None else getattr(fit_state, "X", None)
        if X is not None:
            X = np.asarray(X, dtype=np.float64)
            offset0 = _coef_column_offset(model)
            sl = slice(
                offset0 + int(smooth.coef_slice.start),
                offset0 + int(smooth.coef_slice.stop),
            )
            Xt = np.asarray(X[:, sl], dtype=np.float64)
            if offset0 > 0:
                Xp = np.asarray(X[:, :offset0], dtype=np.float64)
                coef = np.linalg.lstsq(Xp, Xt, rcond=None)[0]
                Xt = Xt - Xp @ coef
            return np.asarray([float(numerical_rank(Xt))], dtype=np.float64)
    nsdf = float(_coef_column_offset(model))
    for tb in blocks:
        if str(getattr(tb, "term_type", "")) == "parametric":
            nsdf += float(int(tb.coef_slice.stop) - int(tb.coef_slice.start))
    return np.asarray([max(float(trace_H) - nsdf, 0.0)], dtype=np.float64)


def assign_fit_solution(model, sol: FitCoreSolution):
    fit_result = sol.fit_result
    fit_state = sol.fit_state

    H_post = np.asarray(fit_result.H_coef, dtype=np.float64)
    trace_H_post = float(fit_result.trace_H)
    scale_post = float(fit_result.scale)
    A_post = None
    A_inv_post = None
    XtWX_post = None
    working_weights_post = None
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
    fit_state = replace(
        fit_state,
        scale=float(scale_post),
        A=(
            fit_state.A
            if A_post is None
            else np.asarray(A_post, dtype=np.float64).copy()
        ),
        A_inv=(
            fit_state.A_inv
            if A_inv_post is None
            else np.asarray(A_inv_post, dtype=np.float64).copy()
        ),
        XtWX=(
            fit_state.XtWX
            if XtWX_post is None
            else np.asarray(XtWX_post, dtype=np.float64).copy()
        ),
        working_weights=(
            fit_state.working_weights
            if working_weights_post is None
            else np.asarray(working_weights_post, dtype=np.float64).copy()
        ),
    )
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
    fit_result = _apply_unconditional_postfit(model, sol, fit_result, fit_state)
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
        edf = compute_edf_by_term(model, H_post)
        if (
            not np.all(np.isfinite(edf))
            or np.any(edf < -1e-7)
            or float(np.sum(edf)) > max(float(trace_H_post) + 1.0, 10.0 * float(model.n_samples_))
        ):
            fallback = _fallback_single_smooth_edf(model, trace_H_post)
            if fallback is not None:
                edf = fallback
        model._edf_by_term_fit_ = np.asarray(edf, dtype=np.float64)

    if bool(getattr(model, "_fitted", False)):
        from .result_builders import sync_gam_result

        sync_gam_result(model)


__all__ = [
    "FitCoreSolution",
    "FitState",
    "PenalizedSystem",
    "FitResult",
    "assign_fit_solution",
    "compute_edf_by_term",
]
