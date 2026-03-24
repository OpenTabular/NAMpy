from __future__ import annotations

import copy

import numpy as np

from .base import _resolve_feature
from .constructed import ConstructedSmooth
from .materialize import materialize_term


def _penalty_rank_and_null_dim(P, tol=1e-10):
    P = np.asarray(P, dtype=np.float64)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("Penalty matrices must be square.")
    if P.shape[0] == 0:
        return 0, 0

    evals = np.linalg.eigvalsh(0.5 * (P + P.T))
    tol_eff = tol * max(1.0, np.max(np.abs(evals)))
    rank = int(np.sum(evals > tol_eff))
    null_dim = int(P.shape[0] - rank)
    return rank, null_dim


def _copy_penalty_defs(penalty_defs):
    return [copy.copy(p) for p in penalty_defs]


def _set_penalty_matrix_and_meta(pdef, P):
    P = np.asarray(P, dtype=np.float64)
    rank, null_dim = _penalty_rank_and_null_dim(P)

    pdef.matrix = P
    if hasattr(pdef, "rank"):
        pdef.rank = rank
    if hasattr(pdef, "null_space_dim"):
        pdef.null_space_dim = null_dim
    return pdef


def _nullspace_transform_from_constraint_matrix(C, d, tol=1e-12):
    """
    Build T with shape (d, d-rank(C)) such that beta_raw = T beta_free
    and C beta_raw = 0.
    """
    C = np.asarray(C, dtype=np.float64)
    if C.ndim == 1:
        C = C.reshape(1, -1)

    if C.ndim != 2:
        raise ValueError("Constraint matrix must be 2D.")
    if C.shape[1] != d:
        raise ValueError(
            f"Constraint matrix has width {C.shape[1]}, expected {d}."
        )

    if C.shape[0] == 0:
        return np.eye(d, dtype=np.float64), 0

    q, r = np.linalg.qr(C.T, mode="complete")
    diag_r = np.abs(np.diag(r[: min(C.shape), :]))
    if diag_r.size == 0:
        return np.eye(d, dtype=np.float64), 0

    rank_tol = max(C.shape) * np.finfo(float).eps * diag_r[0]
    tol_eff = max(float(tol), float(rank_tol))
    rank = int(np.sum(diag_r > tol_eff))

    T = q[:, rank:]
    return T, rank


def _apply_constraint_absorption(B, penalty_defs, C):
    """
    Absorb linear constraints C beta = 0 into the basis / penalties.
    """
    B = np.asarray(B, dtype=np.float64)
    d = B.shape[1]
    T, n_cons = _nullspace_transform_from_constraint_matrix(C, d=d)

    B_new = B @ T
    pdefs_new = _copy_penalty_defs(penalty_defs)
    for pdef in pdefs_new:
        P = np.asarray(pdef.matrix, dtype=np.float64)
        P_new = 0.5 * (T.T @ P @ T + (T.T @ P @ T).T)
        _set_penalty_matrix_and_meta(pdef, P_new)

    return B_new, pdefs_new, T, n_cons


def _extract_runtime_state(runtime):
    B = np.asarray(runtime.basis_train, dtype=np.float64)
    if B.ndim != 2:
        raise ValueError(
            f"Constructed basis for {getattr(runtime, 'label', runtime)!r} "
            f"must be 2D, got shape={B.shape}."
        )

    if hasattr(runtime, "get_penalty_definitions"):
        penalty_defs = list(runtime.get_penalty_definitions())
    else:
        penalty_defs = []

    penalty_defs = _copy_penalty_defs(penalty_defs)
    for pdef in penalty_defs:
        P = np.asarray(pdef.matrix, dtype=np.float64)
        _set_penalty_matrix_and_meta(pdef, P)

    return B, penalty_defs


def smooth_con(
    term_like,
    X,
    feature_names,
    *,
    absorb_cons=True,
    apply_by=True,
    null_space_penalty=False,
):
    """
    smoothCon analogue for the current framework.

    Current scope
    -------------
    - wraps one runtime term object into one ConstructedSmooth
    - records whether by/constraints were already handled by the runtime term
    - supports generic linear-constraint absorption when a future term exposes
      fit_constraint_matrix / predict_constraint_matrix without already absorbing them
    """
    runtime = materialize_term(term_like)
    runtime.fit(X, feature_names)

    B, penalty_defs = _extract_runtime_state(runtime)

    by_done = bool(getattr(runtime, "by_done", True))
    constraints_absorbed = bool(getattr(runtime, "constraints_absorbed", True))

    fit_constraint = getattr(runtime, "fit_constraint_matrix", None)
    predict_constraint = getattr(runtime, "predict_constraint_matrix", None)
    prediction_offset = getattr(runtime, "prediction_offset", None)

    constructor_metadata = {
        "by_done": by_done,
        "constraints_absorbed_by_runtime": constraints_absorbed,
        "runtime_constraint_kind": getattr(runtime, "_constraint_kind", None),
        "runtime_by_name": getattr(runtime, "_by_name", None),
        "runtime_by_is_constant": getattr(runtime, "_by_is_constant", None),
    }

    X0 = None
    predict_fn = None

    by_variable = getattr(runtime, "by", None)
    if by_variable is not None and apply_by and not by_done:
        idx, by_name = _resolve_feature(by_variable, feature_names)
        z_train = np.asarray(X[:, idx], dtype=np.float64).ravel()

        if hasattr(runtime, "basis_train_base") and runtime.basis_train_base is not None:
            X0 = np.asarray(runtime.basis_train_base, dtype=np.float64)
        else:
            raise NotImplementedError(
                f"Runtime term {runtime.__class__.__name__} requested wrapper-level by handling, "
                "but does not expose basis_train_base."
            )

        B = X0 * z_train[:, None]

        if hasattr(runtime, "transform_new_base") and callable(runtime.transform_new_base):
            base_predict_fn = runtime.transform_new_base
        else:
            raise NotImplementedError(
                f"Runtime term {runtime.__class__.__name__} requested wrapper-level by handling, "
                "but does not expose transform_new_base()."
            )

        def predict_fn(X_new, *, _base=base_predict_fn, _idx=idx):
            X_new = np.asarray(X_new, dtype=np.float64)
            z_new = np.asarray(X_new[:, _idx], dtype=np.float64).ravel()
            B0_new = np.asarray(_base(X_new), dtype=np.float64)
            return B0_new * z_new[:, None]

        constructor_metadata["by_handling"] = "wrapper"
    else:
        constructor_metadata["by_handling"] = "runtime" if by_variable is not None else "none"

    if not absorb_cons and constraints_absorbed:
        raise NotImplementedError(
            "absorb_cons=False is incompatible with the current runtime term, because "
            "it already absorbs constraints internally."
        )

    if absorb_cons and (not constraints_absorbed) and fit_constraint is not None:
        B, penalty_defs, T_fit, n_cons = _apply_constraint_absorption(
            B, penalty_defs, fit_constraint
        )

        if predict_fn is None:
            base_predict_fn = runtime.transform_new
        else:
            base_predict_fn = predict_fn

        def predict_fn(X_new, *, _base=base_predict_fn, _T=T_fit):
            return np.asarray(_base(X_new), dtype=np.float64) @ _T

        constraints_absorbed = True
        constructor_metadata["constraint_absorption"] = "wrapper"
        constructor_metadata["n_constraints_absorbed"] = int(n_cons)
    else:
        constructor_metadata["constraint_absorption"] = (
            "runtime" if constraints_absorbed else "none"
        )
        constructor_metadata["n_constraints_absorbed"] = None

    if null_space_penalty:
        raise NotImplementedError(
            "Generic smoothCon-level null-space penalty insertion is not enabled yet in this wrapper. "
            "Use term-level select/null-space penalties for now."
        )

    smooth = ConstructedSmooth(
        label=str(getattr(runtime, "label", str(runtime))),
        runtime=runtime,
        X=np.asarray(B, dtype=np.float64),
        penalty_definitions=penalty_defs,
        basis_name=str(getattr(runtime, "basis_name", "unknown")),
        term_type=str(getattr(runtime, "term_type", "smooth")),
        by_variable=by_variable,
        smoothing_id=(
            None
            if getattr(runtime, "smoothing_id", None) is None
            else str(getattr(runtime, "smoothing_id"))
        ),
        metadata=dict(getattr(runtime, "metadata", {}) or {}),
        fit_constraint=(
            None
            if fit_constraint is None
            else np.asarray(fit_constraint, dtype=np.float64)
        ),
        predict_constraint=(
            None
            if predict_constraint is None
            else np.asarray(predict_constraint, dtype=np.float64)
        ),
        constraints_absorbed=bool(constraints_absorbed),
        prediction_offset=(
            None
            if prediction_offset is None
            else np.asarray(prediction_offset, dtype=np.float64)
        ),
        X0=None if X0 is None else np.asarray(X0, dtype=np.float64),
        constructor_metadata=constructor_metadata,
        _predict_fn=predict_fn,
    )

    return [smooth]