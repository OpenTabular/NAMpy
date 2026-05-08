"""Predictor-wide identifiability side conditions for compiled GAM designs."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.linalg import eigh

from ..compiler.contracts import CoefficientMap
from ..compiler.structures import (
    CompiledPenalty,
    CompiledPredictor,
    CompiledTerm,
    PenaltySpec,
)
from ..penalties import normalize_penalty_spec
from .transforms import (
    independent_column_indices,
    null_space_basis_from_constraint_matrix,
)


def _penalty_root(S: np.ndarray, tol: float) -> np.ndarray:
    S = 0.5 * (np.asarray(S, dtype=np.float64) + np.asarray(S, dtype=np.float64).T)
    if S.size == 0:
        return np.empty((0, 0), dtype=np.float64)
    evals, evecs = eigh(S)
    if evals.size == 0:
        return np.empty_like(S)
    scale = np.max(np.abs(evals))
    tol_eff = max(float(tol), float(scale) * max(S.shape) * np.finfo(float).eps)
    keep = evals > tol_eff
    if not np.any(keep):
        return np.zeros_like(S)
    root = evecs[:, keep] * np.sqrt(evals[keep])[None, :]
    return root @ evecs[:, keep].T


def _augment_term_matrix(
    B: np.ndarray,
    penalties: list[np.ndarray],
    *,
    coef_slice: slice,
    total_coef: int,
    n_constraint_rows: int = 0,
    tol: float = 1e-10,
) -> np.ndarray:
    """
    Create a penalty-aware dependence-testing matrix similar to mgcv::augment.smX.

    The top block is the observational design matrix. Penalty square-root rows
    occupy the term's global coefficient slice so that only null-space overlap
    remains visible when testing dependence against earlier terms.
    """
    B = np.asarray(B, dtype=np.float64)
    n_obs, d = B.shape
    Xa = np.zeros((n_obs + total_coef + n_constraint_rows, d), dtype=np.float64)
    Xa[:n_obs, :] = B
    if d == 0 or not penalties:
        return Xa

    nz_first = np.any(np.abs(np.asarray(penalties[0], dtype=np.float64)) > tol, axis=0)
    if np.any(nz_first):
        sqrma_x = float(np.mean(np.abs(B[:, nz_first])) ** 2)
    else:
        sqrma_x = float(np.mean(np.abs(B)) ** 2) if B.size else 1.0
    if not np.isfinite(sqrma_x) or sqrma_x <= 0.0:
        sqrma_x = 1.0

    St = np.zeros((d, d), dtype=np.float64)
    for S in penalties:
        S = np.asarray(S, dtype=np.float64)
        active = np.any(np.abs(S) > tol, axis=0)
        if not np.any(active):
            continue
        denom = float(np.mean(np.abs(S[np.ix_(active, active)])))
        if not np.isfinite(denom) or denom <= 0.0:
            continue
        St = St + (sqrma_x / denom) * S

    if np.any(np.abs(St) > tol):
        rS = _penalty_root(St, tol=tol)
        row_slice = slice(
            n_obs + int(coef_slice.start), n_obs + int(coef_slice.start) + d
        )
        Xa[row_slice, :] = rS.T
    return Xa


def apply_global_side_conditions(
    design: CompiledPredictor,
    *,
    fit_intercept: bool = True,
    tol: float = 1e-10,
    warn: bool = True,
):
    """
    Apply predictor-wide identifiability side conditions to a CompiledPredictor.

    It walks each compiled term in order and performs three operations:

    1. **Exempt terms** (random effects, factor smooths):
       Preserve their basis and penalties unchanged.  Still add their columns to
       the span accumulator so that later terms correctly detect linear dependence
       on their subspace (architecture invariant 6.6: exempt terms still span
       predictor space).

    2. **Non-exempt terms** — two sub-steps:

       a. *Centering*: if the model has an intercept and the runtime term did not
          already absorb a sum-to-zero constraint, absorb one now via the
          null-space of the column-sum vector.  The centering transform T is
          applied simultaneously to the basis, the accumulated coefficient
          transform, and every associated penalty matrix (invariant 6.3):
              B  <- B @ T
              C  <- C @ T
              S  <- T.T @ S @ T   for each penalty S

       b. *Column selection*: drop any columns of the (possibly re-parameterised)
          basis that are linearly dependent on the current span accumulator.

       After both steps, ``CompiledTerm.basis_transform`` holds the full mapping
       from the constructed-term coefficient space to the final fitted
       coefficient space. Prediction uses
       ``smooth.predict_matrix(X_new) @ basis_transform`` and nothing else
       (invariant 6.2: ``basis_transform`` is canonical).

    3. **Drop zero-width terms**: terms whose every column was removed are
       excluded from the final compiled predictor.  Keeping zero-width terms
       complicates prediction, summaries, and parity (invariant 6.7).

    Parameters
    ----------
    design : CompiledPredictor
        Output of ``compile_predictors``.
    fit_intercept : bool
        Whether the model has an intercept.  Controls whether the constant
        column is seeded into the span accumulator and whether sum-to-zero
        centering is applied to eligible terms.
    tol : float
        Numerical rank tolerance passed to ``independent_column_indices`` and
        ``null_space_basis_from_constraint_matrix``.
    warn : bool
        If True, emit ``warnings.warn`` when columns are deleted or terms are
        dropped.

    Returns
    -------
    new_design : CompiledPredictor
        Updated predictor.  Every ``CompiledTerm.basis_transform`` is the
        complete, canonical constructed-space-to-fitted coefficient map for
        that term.
    report : dict
        Diagnostic report with per-term deletion counts, centering flags,
        and the total number of dropped zero-width terms.
    """
    if not isinstance(design, CompiledPredictor):
        raise TypeError("design must be a CompiledPredictor instance.")

    n_obs = design.design_matrix.shape[0]

    # Span accumulator.  Seeded with a constant column when fit_intercept=True
    # so that sum-to-zero centering removes the intercept-collinear component.
    acc = (
        np.ones((n_obs, 1), dtype=np.float64)
        if fit_intercept
        else np.empty((n_obs, 0), dtype=np.float64)
    )
    acc_aug = np.zeros((n_obs + design.n_coef, acc.shape[1]), dtype=np.float64)
    if acc.shape[1] > 0:
        acc_aug[:n_obs, :] = acc

    new_term_blocks: list[CompiledTerm] = []
    new_penalty_blocks: list[CompiledPenalty] = []
    design_blocks: list[np.ndarray] = []
    deleted_total = 0
    term_reports = []
    start = 0
    predictor_Q = np.zeros((design.n_coef, design.n_coef), dtype=np.float64)

    for term_idx, tb in enumerate(design.compiled_terms):
        B = np.asarray(tb.basis_train, dtype=np.float64)
        d = B.shape[1]
        d_in = d

        # Collect this term's penalties up-front so the centering transform can
        # be applied to their matrices before column selection.
        term_penalty_objs = [
            pb for pb in design.compiled_penalties if pb.term_index == term_idx
        ]
        pen_matrices = [
            np.asarray(pb.matrix, dtype=np.float64) for pb in term_penalty_objs
        ]

        # ── Exempt terms ──────────────────────────────────────────────────────
        policy = getattr(tb, "side_condition_policy", None)
        exempt = bool(
            policy is not None and policy.exempt_from_predictor_side_conditions
        )
        if exempt:
            sl_new = slice(start, start + d)
            new_idx = len(new_term_blocks)
            if d > 0:
                predictor_Q[tb.coef_slice, sl_new] = np.eye(d, dtype=np.float64)
            new_term_blocks.append(
                CompiledTerm(
                    label=tb.label,
                    coef_slice=sl_new,
                    basis_train=B,
                    predict_fn=tb.predict_fn,
                    predict_coefficient_map=(
                        None
                        if tb.predict_coefficient_map is None
                        else np.asarray(tb.predict_coefficient_map, dtype=np.float64)
                    ),
                    basis_transform=tb.basis_transform,
                    coefficient_maps=tuple(tb.coefficient_maps),
                    feature_info=tb.feature_info,
                    by_variable_info=tb.by_variable_info,
                    side_condition_policy=tb.side_condition_policy,
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
                    term_id=tb.term_id,
                    smoothing_group_id=tb.smoothing_group_id,
                    penalty_specs=tuple(tb.penalty_specs),
                    constructor_metadata=dict(tb.constructor_metadata),
                    metadata=dict(tb.metadata),
                )
            )
            for pb, P in zip(term_penalty_objs, pen_matrices):
                new_penalty_blocks.append(
                    CompiledPenalty(
                        label=pb.label,
                        coef_slice=sl_new,
                        matrix=P.copy(),
                        smoothing_index=pb.smoothing_index,
                        term_index=new_idx,
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
                    "exempt": True,
                    "deleted_columns": [],
                    "kept_columns": (
                        list(np.asarray(tb.kept_columns, dtype=int))
                        if tb.kept_columns is not None
                        else list(range(d))
                    ),
                    "n_deleted": 0,
                    "absorbed_centering": False,
                }
            )
            start += d
            continue

        # ── Non-exempt: centering + column selection ──────────────────────────
        #
        # C accumulates the coefficient transform from the constructed-term
        # coefficient space to the current (pre-side-condition) space.
        # Initialise from whatever the compiler recorded; fall back to identity.
        C = (
            np.asarray(tb.basis_transform, dtype=np.float64)
            if tb.basis_transform is not None
            else np.eye(d, dtype=np.float64)
        )
        Q_term = np.eye(d_in, dtype=np.float64)

        runtime_skip_centering = bool(policy is not None and policy.skip_centering)
        runtime_by_name = tb.by_variable_info.name
        runtime_by_is_constant = tb.by_variable_info.is_constant
        absorbed_centering = False

        # Step (a): optionally absorb a sum-to-zero centering constraint.
        if (
            fit_intercept
            and not runtime_skip_centering
            and (runtime_by_name is None or bool(runtime_by_is_constant))
            and not bool(tb.term_type in {"tensor_interaction"})
        ):
            centering = np.sum(B, axis=0, keepdims=True)
            if np.linalg.norm(centering) > tol:
                T_con, _ = null_space_basis_from_constraint_matrix(centering, tol=tol)
                if 0 < T_con.shape[1] < d:
                    # Invariant 6.3: the same transform must be applied to the
                    # basis, the coefficient map, and every penalty matrix.
                    pen_matrices = [T_con.T @ S @ T_con for S in pen_matrices]
                    B = B @ T_con
                    C = C @ T_con
                    Q_term = Q_term @ T_con
                    d = B.shape[1]
                    absorbed_centering = True

        # Step (b): drop columns linearly dependent on the accumulator.
        if policy is not None and policy.exempt_from_dependency_pruning:
            keep = np.arange(d, dtype=int)
        elif policy is not None and policy.allow_first_numeric_by_unpruned:
            # Ordinary non-constant numeric by-variable smooths should keep their
            # raw term basis for the first occurrence, but later terms still need
            # ordinary cross-term redundancy removal against the accumulated
            # design to match mgcv's side-condition allocation.
            if acc.shape[1] <= int(bool(fit_intercept)):
                keep = np.arange(d, dtype=int)
            else:
                keep = np.asarray(
                    independent_column_indices(B, A=acc, tol=tol), dtype=int
                )
        elif (
            bool(policy is None or policy.requires_penalty_aware_pruning)
            and pen_matrices
        ):
            B_dep = _augment_term_matrix(
                B,
                pen_matrices,
                coef_slice=tb.coef_slice,
                total_coef=design.n_coef,
                tol=tol,
            )
            keep = np.asarray(
                independent_column_indices(B_dep, A=acc_aug, tol=tol), dtype=int
            )
        else:
            keep = np.asarray(independent_column_indices(B, A=acc, tol=tol), dtype=int)
        deleted_local = np.setdiff1d(np.arange(d, dtype=int), keep)

        # C_final is the canonical constructed-space-to-fitted basis transform
        # (invariant 6.2). If basis matrices are right-multiplied by C_final,
        # coefficient vectors pull back in the opposite direction.
        C_final = (
            C[:, keep] if keep.size > 0 else np.empty((C.shape[0], 0), dtype=np.float64)
        )
        Q_term_final = (
            Q_term[:, keep]
            if keep.size > 0
            else np.empty((Q_term.shape[0], 0), dtype=np.float64)
        )
        B_final = (
            B[:, keep] if keep.size > 0 else np.empty((B.shape[0], 0), dtype=np.float64)
        )
        d_final = B_final.shape[1]

        # Subset penalty matrices to the surviving columns and recompute their
        # canonical metadata in the new coefficient space.
        pen_pairs_final = []
        for pb, S in zip(term_penalty_objs, pen_matrices):
            P_new = (
                S[np.ix_(keep, keep)]
                if keep.size > 0
                else np.empty((0, 0), dtype=np.float64)
            )
            p_new = normalize_penalty_spec(
                PenaltySpec(
                    matrix=P_new,
                    smoothing_id=pb.smoothing_id,
                    kind=pb.kind,
                    rank=None,
                    null_space_dim=None,
                    is_null_space_penalty=pb.is_null_space_penalty,
                    sp_mode=pb.sp_mode,
                    sp_value=pb.sp_value,
                    metadata=dict(pb.metadata),
                )
            )
            if int(p_new.rank or 0) > 0:
                pen_pairs_final.append((pb, p_new))
        pen_specs_final = [p_new for _pb, p_new in pen_pairs_final]

        # Track surviving original coefficient indices for diagnostics / parity.
        # When centering was absorbed the mapping through T_con is non-trivial;
        # set to None to signal that exact original indices are unavailable.
        if absorbed_centering:
            kept_orig = None
            deleted_orig = None
        else:
            old_kept = (
                np.asarray(tb.kept_columns, dtype=int)
                if tb.kept_columns is not None
                else np.arange(d, dtype=int)
            )
            kept_orig = old_kept[keep] if keep.size > 0 else np.array([], dtype=int)
            deleted_orig = (
                old_kept[deleted_local]
                if deleted_local.size > 0
                else np.array([], dtype=int)
            )

        tb_meta = dict(tb.metadata)
        if deleted_orig is not None and deleted_orig.size > 0:
            tb_meta["del_index"] = deleted_orig.tolist()
        if absorbed_centering:
            tb_meta["absorbed_sum_to_zero_constraint"] = True

        predictor_maps = list(tb.coefficient_maps)
        if C_final.shape[1] != 0:
            predictor_maps.append(
                CoefficientMap(
                    source_space="local_fit_space",
                    target_space="predictor_fit_space",
                    matrix=np.asarray(C_final, dtype=np.float64),
                    reason="predictor_side_conditions",
                    metadata={
                        "absorbed_centering": bool(absorbed_centering),
                        "n_deleted": int(deleted_local.size),
                    },
                )
            )

        sl_new = slice(start, start + d_final)
        new_idx = len(new_term_blocks)
        if d_final > 0:
            predictor_Q[tb.coef_slice, sl_new] = Q_term_final

        new_term_blocks.append(
            CompiledTerm(
                label=tb.label,
                coef_slice=sl_new,
                basis_train=B_final,
                predict_fn=tb.predict_fn,
                predict_coefficient_map=(
                    None
                    if tb.predict_coefficient_map is None
                    else np.asarray(tb.predict_coefficient_map, dtype=np.float64)
                ),
                basis_transform=C_final,
                coefficient_maps=tuple(predictor_maps),
                feature_info=tb.feature_info,
                by_variable_info=tb.by_variable_info,
                side_condition_policy=tb.side_condition_policy,
                kept_columns=kept_orig,
                deleted_columns=deleted_orig,
                smoothing_indices=[
                    int(pb.smoothing_index) for pb, _p in pen_pairs_final
                ],
                smoothing_ids=[pb.smoothing_id for pb, _p in pen_pairs_final],
                n_penalties=len(pen_pairs_final),
                term_type=tb.term_type,
                basis_name=tb.basis_name,
                term_id=tb.term_id,
                smoothing_group_id=tb.smoothing_group_id,
                penalty_specs=tuple(pen_specs_final),
                constructor_metadata=dict(tb.constructor_metadata),
                metadata=tb_meta,
            )
        )

        for pb, pdef_new in pen_pairs_final:
            new_penalty_blocks.append(
                CompiledPenalty(
                    label=pb.label,
                    coef_slice=sl_new,
                    matrix=np.asarray(pdef_new.matrix, dtype=np.float64),
                    smoothing_index=pb.smoothing_index,
                    term_index=new_idx,
                    smoothing_id=pb.smoothing_id,
                    kind=str(pdef_new.kind),
                    rank=pdef_new.rank,
                    null_space_dim=pdef_new.null_space_dim,
                    is_null_space_penalty=bool(pdef_new.is_null_space_penalty),
                    sp_mode=pdef_new.sp_mode,
                    sp_value=pdef_new.sp_value,
                    metadata=dict(pb.metadata),
                )
            )

        n_deleted = int(d - d_final)
        deleted_total += n_deleted

        if d_final > 0:
            design_blocks.append(B_final)
            acc = np.column_stack([acc, B_final])
            if pen_matrices:
                B_aug_final = _augment_term_matrix(
                    B_final,
                    [np.asarray(p.matrix, dtype=np.float64) for p in pen_specs_final],
                    coef_slice=tb.coef_slice,
                    total_coef=design.n_coef,
                    tol=tol,
                )
            else:
                B_aug_final = np.zeros(
                    (n_obs + design.n_coef, d_final), dtype=np.float64
                )
                B_aug_final[:n_obs, :] = B_final
            acc_aug = np.column_stack([acc_aug, B_aug_final])

        term_reports.append(
            {
                "label": tb.label,
                "exempt": False,
                "deleted_columns": (
                    [] if deleted_orig is None else deleted_orig.tolist()
                ),
                "kept_columns": [] if kept_orig is None else kept_orig.tolist(),
                "n_deleted": n_deleted,
                "absorbed_centering": absorbed_centering,
            }
        )

        # Suppress the warning for non-constant numeric by-variable terms: the
        # cross-term column deletion there is expected orthogonalization (matching
        # mgcv's side-condition allocation), not a surprising constraint application.
        numeric_by_redundancy = runtime_by_name is not None and not bool(
            runtime_by_is_constant
        )
        if warn and n_deleted > 0 and not numeric_by_redundancy:
            col_info = (
                f" (original indices {deleted_orig.tolist()})"
                if deleted_orig is not None and deleted_orig.size > 0
                else ""
            )
            warnings.warn(
                f"Applied side conditions to term {tb.label!r}: "
                f"deleted {n_deleted} redundant column(s){col_info}.",
                stacklevel=2,
            )

        start += d_final

    # mgcv::gam.side keeps smooth objects even if their design is reduced to zero
    # columns; downstream term positions remain stable.
    old_to_final: dict[int, int] = {}
    final_term_blocks: list[CompiledTerm] = []
    for old_i, tb in enumerate(new_term_blocks):
        old_to_final[old_i] = len(final_term_blocks)
        final_term_blocks.append(tb)

    # Keep only penalties whose owning term survived; re-stamp term_index.
    final_penalty_blocks: list[CompiledPenalty] = []
    for pb in new_penalty_blocks:
        new_ti = old_to_final.get(pb.term_index, -1)
        if new_ti >= 0:
            pb.term_index = new_ti
            final_penalty_blocks.append(pb)

    old_to_new_sp: dict[int, int] = {}
    for pb in final_penalty_blocks:
        old_idx = int(pb.smoothing_index)
        if old_idx not in old_to_new_sp:
            old_to_new_sp[old_idx] = len(old_to_new_sp)
        pb.smoothing_index = old_to_new_sp[old_idx]

    smoothing_parameter_map = {
        sid: old_to_new_sp[int(old_idx)]
        for sid, old_idx in dict(design.smoothing_parameter_map).items()
        if int(old_idx) in old_to_new_sp
    }
    for pb in final_penalty_blocks:
        if pb.smoothing_id is not None:
            smoothing_parameter_map.setdefault(
                str(pb.smoothing_id), int(pb.smoothing_index)
            )

    old_modes = list(design.smoothing_override_modes or [])
    old_values = (
        None
        if design.smoothing_override_values is None
        else np.asarray(design.smoothing_override_values, dtype=np.float64)
    )
    n_sp_final = len(old_to_new_sp)
    override_modes = [None] * n_sp_final
    override_values = np.full(n_sp_final, np.nan, dtype=np.float64)
    for old_idx, new_idx in old_to_new_sp.items():
        if old_idx < len(old_modes):
            override_modes[new_idx] = old_modes[old_idx]
        if old_values is not None and old_idx < old_values.size:
            override_values[new_idx] = old_values[old_idx]

    # ── Assemble final CompiledPredictor ──────────────────────────────────────
    matrix_train = (
        np.column_stack(design_blocks)
        if design_blocks
        else np.empty((n_obs, 0), dtype=np.float64)
    )
    term_index_map = {tb.term_id: i for i, tb in enumerate(final_term_blocks)}

    new_design = CompiledPredictor(
        name=design.name,
        design_matrix=matrix_train,
        compiled_terms=tuple(final_term_blocks),
        compiled_penalties=tuple(final_penalty_blocks),
        smoothing_parameter_map=smoothing_parameter_map,
        has_intercept=bool(fit_intercept),
        term_index_map=term_index_map,
        side_condition_Q=predictor_Q[:, :start].copy(),
        n_coef=start,
        n_smoothing_params=n_sp_final,
        smoothing_override_modes=override_modes,
        smoothing_override_values=override_values,
        metadata=dict(design.metadata),
    )
    report = {
        "predictor": design.name,
        "n_deleted_total": deleted_total,
        "n_terms_dropped": len(new_term_blocks) - len(final_term_blocks),
        "term_reports": term_reports,
    }
    return new_design, report


__all__ = [
    "apply_global_side_conditions",
]
