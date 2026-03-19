# gsm/design/side_conditions.py
import warnings

import numpy as np
from scipy.linalg import qr as scipy_qr

from .objects import PenaltyBlock, PredictorDesign, TermBlock


def _orthogonal_residual(B, A):
    """
    Residualize columns of B with respect to the span of A.
    """
    B = np.asarray(B, dtype=np.float64)
    if A is None or A.size == 0 or A.shape[1] == 0:
        return B

    A = np.asarray(A, dtype=np.float64)
    coef, *_ = np.linalg.lstsq(A, B, rcond=None)
    return B - A @ coef


def _independent_column_indices(B, A=None, tol=1e-10):
    """
    Find a subset of columns of B that are linearly independent modulo span(A).

    The result is a list of column indices to keep.
    """
    B = np.asarray(B, dtype=np.float64)
    if B.ndim != 2:
        raise ValueError("B must be a 2D matrix.")
    if B.shape[1] == 0:
        return np.array([], dtype=int)

    Rb = _orthogonal_residual(B, A)

    if np.all(np.abs(Rb) <= tol):
        return np.array([], dtype=int)

    _Q, R, piv = scipy_qr(Rb, mode="economic", pivoting=True)

    diag_R = np.abs(np.diag(R))
    if diag_R.size == 0:
        return np.array([], dtype=int)

    rank_tol = max(B.shape) * np.finfo(float).eps * diag_R[0]
    tol_eff = max(float(tol), float(rank_tol))
    rank = int(np.sum(diag_R > tol_eff))

    if rank <= 0:
        return np.array([], dtype=int)

    keep = np.sort(np.asarray(piv[:rank], dtype=int))
    return keep


def _constraint_null_space(C, tol=1e-10):
    C = np.asarray(C, dtype=np.float64)
    if C.ndim != 2:
        raise ValueError("C must be a 2D matrix.")
    if C.size == 0:
        return np.eye(C.shape[1], dtype=np.float64)

    U, s, Vt = np.linalg.svd(C, full_matrices=True)
    if s.size == 0:
        return np.eye(C.shape[1], dtype=np.float64)

    tol_eff = max(float(tol), float(np.max(s)) * max(C.shape) * np.finfo(float).eps)
    rank = int(np.sum(s > tol_eff))
    return Vt[rank:, :].T.copy()


def apply_global_side_conditions(
    design: PredictorDesign,
    *,
    fit_intercept: bool = True,
    tol: float = 1e-10,
    warn: bool = True,
):
    """
    Apply global identifiability side conditions to one compiled predictor design.

    Strategy
    --------
    Process term blocks in model order. For each term block, keep only those
    columns that are linearly independent of the columns already accumulated from:
    - the intercept (if present)
    - all previously processed term blocks
    """
    if not isinstance(design, PredictorDesign):
        raise TypeError("design must be a PredictorDesign instance.")

    acc = (
        np.ones((design.matrix_train.shape[0], 1), dtype=np.float64)
        if fit_intercept
        else np.empty((design.matrix_train.shape[0], 0), dtype=np.float64)
    )

    new_term_blocks = []
    new_penalty_blocks = []
    design_blocks = []

    deleted_total = 0
    term_reports = []

    start = 0
    for tb in design.term_blocks:
        B = np.asarray(tb.basis_train, dtype=np.float64)
        d = B.shape[1]

        # mgcv-compatible behavior for bs="re":
        # full-rank penalized random effects do not get side conditions applied.
        if tb.term_type in {"random_effect", "factor_smooth_fs", "factor_smooth_sz"}:
            sl_new = slice(start, start + d)

            new_tb = TermBlock(
                label=tb.label,
                coef_slice=sl_new,
                smooth=tb.smooth,
                basis_train=B,
                basis_transform=tb.basis_transform,
                original_n_coef=tb.original_n_coef,
                kept_columns=(
                    np.asarray(tb.kept_columns, dtype=int).copy()
                    if tb.kept_columns is not None
                    else np.arange(d, dtype=int)
                ),
                deleted_columns=(
                    np.asarray(tb.deleted_columns, dtype=int).copy()
                    if tb.deleted_columns is not None
                    else np.array([], dtype=int)
                ),
                smoothing_indices=list(tb.smoothing_indices),
                smoothing_ids=list(tb.smoothing_ids),
                n_penalties=tb.n_penalties,
                term_type=tb.term_type,
                basis_name=tb.basis_name,
                by_variable=tb.by_variable,
                term_id=tb.term_id,
                metadata=dict(tb.metadata),
            )
            new_term_blocks.append(new_tb)

            matches = [pb for pb in design.penalty_blocks if pb.coef_slice == tb.coef_slice]
            for pb in matches:
                new_penalty_blocks.append(
                    PenaltyBlock(
                        label=pb.label,
                        coef_slice=sl_new,
                        matrix=np.asarray(pb.matrix, dtype=np.float64).copy(),
                        smoothing_index=pb.smoothing_index,
                        smoothing_id=pb.smoothing_id,
                        kind=pb.kind,
                        rank=pb.rank,
                        null_space_dim=pb.null_space_dim,
                        is_null_space_penalty=pb.is_null_space_penalty,
                        sp_mode=pb.sp_mode,
                        sp_value=pb.sp_value,
                        metadata=dict(pb.metadata),
                    )
                )

            if d > 0:
                design_blocks.append(B)

            term_reports.append(
                {
                    "label": tb.label,
                    "deleted_columns": [],
                    "kept_columns": (
                        list(np.asarray(tb.kept_columns, dtype=int))
                        if tb.kept_columns is not None
                        else list(range(d))
                    ),
                    "n_deleted": 0,
                }
            )

            start += d
            continue

        if d == 0:
            new_tb = TermBlock(
                label=tb.label,
                coef_slice=slice(start, start),
                smooth=tb.smooth,
                basis_train=B,
                basis_transform=tb.basis_transform,
                original_n_coef=tb.original_n_coef,
                kept_columns=tb.kept_columns,
                deleted_columns=tb.deleted_columns,
                smoothing_indices=list(tb.smoothing_indices),
                smoothing_ids=list(tb.smoothing_ids),
                n_penalties=tb.n_penalties,
                term_type=tb.term_type,
                basis_name=tb.basis_name,
                by_variable=tb.by_variable,
                term_id=tb.term_id,
                metadata=dict(tb.metadata),
            )
            new_term_blocks.append(new_tb)
            term_reports.append(
                {
                    "label": tb.label,
                    "deleted_columns": [],
                    "kept_columns": [],
                    "n_deleted": 0,
                }
            )
            continue

        C_old = (
            np.asarray(tb.basis_transform, dtype=np.float64)
            if tb.basis_transform is not None
            else np.eye(d, dtype=np.float64)
        )

        absorbed_constraint = False
        if fit_intercept and tb.term_type not in {
            "random_effect",
            "factor_smooth_fs",
            "factor_smooth_sz",
        }:
            centering = np.sum(B, axis=0, keepdims=True)
            if np.linalg.norm(centering) > tol:
                T_con = _constraint_null_space(centering, tol=tol)
                if 0 < T_con.shape[1] < B.shape[1]:
                    B = B @ T_con
                    C_old = C_old @ T_con
                    d = B.shape[1]
                    absorbed_constraint = True

        keep = _independent_column_indices(B, A=acc, tol=tol)
        keep = np.asarray(keep, dtype=int)

        if absorbed_constraint:
            old_kept = None
        else:
            old_kept = (
                np.asarray(tb.kept_columns, dtype=int)
                if tb.kept_columns is not None
                else np.arange(d, dtype=int)
            )

        deleted_local = np.setdiff1d(np.arange(d, dtype=int), keep)
        if old_kept is None:
            deleted_orig = None
            kept_orig = None
        else:
            deleted_orig = (
                old_kept[deleted_local] if deleted_local.size > 0 else np.array([], dtype=int)
            )
            kept_orig = old_kept[keep] if keep.size > 0 else np.array([], dtype=int)
        C_new = (
            C_old[:, keep]
            if keep.size > 0
            else np.empty((C_old.shape[0], 0), dtype=np.float64)
        )

        B_new = (
            B[:, keep]
            if keep.size > 0
            else np.empty((B.shape[0], 0), dtype=np.float64)
        )
        d_new = B_new.shape[1]
        sl_new = slice(start, start + d_new)

        tb_meta = dict(tb.metadata)
        if deleted_orig is not None:
            tb_meta["del_index"] = deleted_orig.tolist()
        if absorbed_constraint:
            tb_meta["absorbed_sum_to_zero_constraint"] = True

        new_tb = TermBlock(
            label=tb.label,
            coef_slice=sl_new,
            smooth=tb.smooth,
            basis_train=B_new,
            basis_transform=C_new,
            original_n_coef=tb.original_n_coef if tb.original_n_coef is not None else d,
            kept_columns=kept_orig,
            deleted_columns=deleted_orig,
            smoothing_indices=list(tb.smoothing_indices),
            smoothing_ids=list(tb.smoothing_ids),
            n_penalties=tb.n_penalties,
            term_type=tb.term_type,
            basis_name=tb.basis_name,
            by_variable=tb.by_variable,
            term_id=tb.term_id,
            metadata=tb_meta,
        )
        new_term_blocks.append(new_tb)

        matches = [pb for pb in design.penalty_blocks if pb.coef_slice == tb.coef_slice]
        for pb in matches:
            P = np.asarray(pb.matrix, dtype=np.float64)
            P_new = (
                P[np.ix_(keep, keep)]
                if keep.size > 0
                else np.empty((0, 0), dtype=np.float64)
            )
            new_penalty_blocks.append(
                PenaltyBlock(
                    label=pb.label,
                    coef_slice=sl_new,
                    matrix=P_new,
                    smoothing_index=pb.smoothing_index,
                    smoothing_id=pb.smoothing_id,
                    kind=pb.kind,
                    rank=pb.rank,
                    null_space_dim=pb.null_space_dim,
                    is_null_space_penalty=pb.is_null_space_penalty,
                    sp_mode=pb.sp_mode,
                    sp_value=pb.sp_value,
                    metadata=dict(pb.metadata),
                )
            )

        if d_new > 0:
            design_blocks.append(B_new)
            acc = np.column_stack([acc, B_new])

        n_deleted = int(d - d_new)
        deleted_total += n_deleted
        term_reports.append(
            {
                "label": tb.label,
                "deleted_columns": [] if deleted_orig is None else deleted_orig.tolist(),
                "kept_columns": [] if kept_orig is None else kept_orig.tolist(),
                "n_deleted": n_deleted,
            }
        )

        if warn and n_deleted > 0 and deleted_orig is not None:
            warnings.warn(
                f"Applied side conditions to term {tb.label!r}: "
                f"deleted {n_deleted} redundant column(s) "
                f"(indices {deleted_orig.tolist()})."
            )

        start += d_new

    matrix_train = (
        np.column_stack(design_blocks)
        if design_blocks
        else np.empty((design.matrix_train.shape[0], 0), dtype=np.float64)
    )

    new_design = PredictorDesign(
        name=design.name,
        term_blocks=new_term_blocks,
        penalty_blocks=new_penalty_blocks,
        matrix_train=matrix_train,
        n_coef=start,
        n_smoothing_params=design.n_smoothing_params,
        smoothing_id_map=dict(design.smoothing_id_map),
        smoothing_override_modes=list(design.smoothing_override_modes),
        smoothing_override_values=(
            None
            if design.smoothing_override_values is None
            else np.asarray(design.smoothing_override_values, dtype=np.float64).copy()
        ),
        metadata=dict(design.metadata),
    )

    report = {
        "predictor": design.name,
        "n_deleted_total": deleted_total,
        "term_reports": term_reports,
    }
    return new_design, report
