"""
Stable fitted outputs and transient fit state.

`FitCoreSolution` is the common solver return type. It wraps:

- `FitResult`: stable fitted outputs for consumers
- `FitState`: transient numerical workspace (design, weights, penalized system)
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .._model_state import (
    _coef_column_offset,
    _term_blocks_seq,
)
from ..results import FitResult
from .parameterization import export_fit_result_to_prediction_space
from .postprocess.unconditional_covariance import apply_unconditional_postfit


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
    gdi1_eta: np.ndarray | None = None
    gdi1_mu: np.ndarray | None = None
    offset: np.ndarray | None = None
    log_det_XtWX_plus_penalty: float | None = None
    penalized_system_rank: int | None = None
    dropped_column_indices: np.ndarray | None = None
    scale: float | None = None


@dataclass(frozen=True)
class FitCoreSolution:
    fit_result: FitResult
    fit_state: FitState

    def _delegate_objects(self):
        state = object.__getattribute__(self, "__dict__")
        return tuple(
            state[name] for name in ("fit_result", "fit_state") if name in state
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
        return replace(self, fit_state=replace(self.fit_state, **changes))

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
            gdi1_eta=(
                None
                if data.get("gdi1_eta", None) is None
                else np.asarray(data["gdi1_eta"], dtype=np.float64)
            ),
            gdi1_mu=(
                None
                if data.get("gdi1_mu", None) is None
                else np.asarray(data["gdi1_mu"], dtype=np.float64)
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
        return cls(fit_result=fit_result, fit_state=fit_state)


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
    fit_result = apply_unconditional_postfit(model, sol, fit_result, fit_state)
    fit_result = export_fit_result_to_prediction_space(model, fit_result)
    sol = replace(sol, fit_result=fit_result, fit_state=fit_state)

    result = model.gam_result_
    if result is None:
        raise RuntimeError("Cannot assign a fit solution before design compilation.")
    if (
        getattr(model.family, "family_class", "") == "general"
        and result.compiled_model.coef_reduced_to_full_idx is not None
    ):
        full_idx = np.asarray(
            result.compiled_model.coef_reduced_to_full_idx, dtype=int
        )
        edf = []
        for tb in _term_blocks_seq(model):
            idx = full_idx[tb.coef_slice]
            edf.append(float(np.trace(H_post[np.ix_(idx, idx)])))
        edf_by_term = np.asarray(edf, dtype=np.float64)
    else:
        # Keep the EDF implied by the fitted coefficient-space hat matrix.
        # mgcv has no single-smooth numerical-rank / trace(H)-nsdf substitute;
        # parity-sensitive callers must see the actual post-fit result.
        edf_by_term = np.asarray(
            compute_edf_by_term(model, H_post), dtype=np.float64
        )

    sol = replace(sol, fit_result=replace(sol.fit_result, edf_by_term=edf_by_term))

    model.gam_result_ = result.with_fit_solution(
        sol,
    )

    if bool(getattr(model, "_fitted", False)):
        from .result_builders import sync_gam_result

        sync_gam_result(model)


__all__ = [
    "FitCoreSolution",
    "FitState",
    "FitResult",
    "assign_fit_solution",
    "compute_edf_by_term",
]
