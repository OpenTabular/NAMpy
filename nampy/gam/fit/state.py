"""
Fit solution container shared by all fitting backends.

:class:`FitCoreSolution` is the common return type of :func:`solve_gaussian_fit`
and :func:`solve_pirls_fit`.  Downstream code (orchestrator, criterion functions,
result builder) always receives this type, regardless of which backend produced it.

:func:`compute_edf_by_term` and :func:`assign_fit_solution` are helpers called by
the model after fitting to populate the model's public attributes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve, qr


@dataclass
class FitCoreSolution:
    coef_full: np.ndarray
    intercept: float
    beta: np.ndarray
    eta: np.ndarray
    mu: np.ndarray

    rss: float | None
    deviance: float
    edf: float
    trace_H: float
    scale: float

    cov_bayes: np.ndarray | None
    cov_freq: np.ndarray | None
    H_coef: np.ndarray

    X: np.ndarray | None = None
    A: np.ndarray | None = None
    A_inv: np.ndarray | None = None
    XtWX: np.ndarray | None = None
    P: np.ndarray | None = None
    penalty_matrix: np.ndarray | None = None
    working_weights: np.ndarray | None = None
    fisher_weights: np.ndarray | None = None
    working_response: np.ndarray | None = None
    penalty_quadratic: float | None = None
    loglik: float | None = None
    offset: np.ndarray | None = None
    log_det_XtWX_plus_penalty: float | None = None

    converged: bool | None = None
    iter: int | None = None
    failed_step: bool | None = None
    failure_reason: str | None = None
    inner_trace: list | None = None

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    @classmethod
    def from_dict(cls, data: dict) -> "FitCoreSolution":
        return cls(
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
            H_coef=np.asarray(data["H_coef"], dtype=np.float64),
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
            penalty_quadratic=(
                None
                if data.get("penalty_quadratic", None) is None
                else float(data["penalty_quadratic"])
            ),
            loglik=(
                None if data.get("loglik", None) is None else float(data["loglik"])
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
            converged=data.get("converged", None),
            iter=data.get("iter", None),
            failed_step=data.get("failed_step", None),
            failure_reason=data.get("failure_reason", None),
            inner_trace=data.get("inner_trace", None),
        )


def compute_edf_by_term(model, H_coef):
    offset0 = 1 if model.fit_intercept else 0
    fit_state = getattr(model, "fit_state_", None)
    X_full = None if fit_state is None else getattr(fit_state, "X", None)
    A_inv = None if fit_state is None else getattr(fit_state, "A_inv", None)
    w = None if fit_state is None else getattr(fit_state, "working_weights", None)

    edf = []
    for tb in model.term_blocks_:
        sl = slice(
            offset0 + int(tb.coef_slice.start),
            offset0 + int(tb.coef_slice.stop),
        )
        val = float(np.trace(H_coef[sl, sl]))

        # mgcv summary EDF for `bs="re"` terms is effectively attributed after
        # removing the intercept direction from the random-effect block. Without
        # that orthogonalization, near-singular Gaussian REML fits can over-credit
        # the RE term by the same tiny amount that belongs to the intercept.
        if (
            str(getattr(tb, "term_type", "")) == "random_effect"
            and bool(getattr(model, "fit_intercept", False))
            and X_full is not None
            and A_inv is not None
            and w is not None
        ):
            try:
                X = np.asarray(X_full, dtype=np.float64)
                A_inv_arr = np.asarray(A_inv, dtype=np.float64)
                w_arr = np.asarray(w, dtype=np.float64).ravel()
                Xp = X[:, :offset0]
                Xt = X[:, sl]
                if Xp.shape[1] > 0 and Xt.shape[1] > 0 and w_arr.shape[0] == X.shape[0]:
                    sqrt_w = np.sqrt(np.clip(w_arr, 0.0, None))
                    coef = np.linalg.lstsq(
                        sqrt_w[:, None] * Xp,
                        sqrt_w[:, None] * Xt,
                        rcond=None,
                    )[0]
                    Xt_eff = Xt - Xp @ coef
                    val = float(
                        np.trace((Xt_eff.T @ (w_arr[:, None] * X) @ A_inv_arr)[:, sl])
                    )
            except Exception:
                pass

        edf.append(val)
    edf = np.asarray(edf, dtype=np.float64)
    # Under predictor-wide side conditions, a later non-constant numeric-by
    # smooth can lose one or more redundant columns against the accumulated
    # design. The reduced fitted block trace then credits that shared null-space
    # EDF to earlier terms, while mgcv reports it with the term that originally
    # owned the deleted columns. Reassign those deleted directions for parity.
    for i, tb in enumerate(model.term_blocks_):
        meta = dict(getattr(tb, "metadata", {}) or {})
        ctor = dict(meta.get("constructor_metadata", {}) or {})
        runtime_by_name = ctor.get("runtime_by_name", None)
        runtime_by_is_constant = ctor.get("runtime_by_is_constant", None)
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
    model.intercept_ = float(sol.intercept)
    model.coef_full_ = np.asarray(sol.coef_full, dtype=np.float64)
    model.coef_ = np.asarray(sol.beta, dtype=np.float64)
    model.beta = [model.coef_[tb.coef_slice] for tb in model.term_blocks_]

    H_post = np.asarray(sol.H_coef, dtype=np.float64)
    trace_H_post = float(sol.trace_H)
    if (
        not bool(getattr(model.family, "canonical_link", False))
        and sol.X is not None
        and sol.P is not None
        and sol.fisher_weights is not None
    ):
        try:
            X = np.asarray(sol.X, dtype=np.float64)
            P = np.asarray(sol.P, dtype=np.float64)
            fisher_w = np.asarray(sol.fisher_weights, dtype=np.float64).ravel()
            XtFX = X.T @ (fisher_w[:, None] * X)
            cA_post, lower_post = cho_factor(
                XtFX + P, overwrite_a=False, check_finite=False
            )
            H_post = cho_solve((cA_post, lower_post), XtFX, check_finite=False)
            trace_H_post = float(np.trace(H_post))
        except Exception:
            H_post = np.asarray(sol.H_coef, dtype=np.float64)
            trace_H_post = float(sol.trace_H)

    model.edf_ = trace_H_post
    model.trace_H_ = trace_H_post
    model.scale_ = float(sol.scale)
    model.rss_ = None if sol.rss is None else float(sol.rss)
    model.deviance_ = float(sol.deviance)

    model.Vp_ = (
        None if sol.cov_bayes is None else np.asarray(sol.cov_bayes, dtype=np.float64)
    )
    model.Vf_ = (
        None if sol.cov_freq is None else np.asarray(sol.cov_freq, dtype=np.float64)
    )
    model._H_coef = H_post

    model._fitted_eta = np.asarray(sol.eta, dtype=np.float64)
    model._fitted_mu = np.asarray(sol.mu, dtype=np.float64)
    model.fit_state_ = sol
    model._summary_R_ = None
    if sol.X is not None and sol.working_weights is not None:
        X = np.asarray(sol.X, dtype=np.float64)
        w = np.asarray(sol.working_weights, dtype=np.float64).ravel()
        if X.ndim == 2 and w.shape[0] == X.shape[0]:
            # Use unpivoted QR to match mgcv's fit$R convention.
            _, R_nat = qr(
                np.sqrt(np.clip(w, 0.0, None))[:, None] * X,
                mode="economic",
            )
            model._summary_R_ = R_nat

    model.edf_by_term_ = compute_edf_by_term(model, model._H_coef)
