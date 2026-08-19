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
    """Return a public-API PSD root for mgcv's ``augment.smX`` invariant."""
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
    # mgcv::mroot()/augment.smX only identify root @ root.T. The particular root
    # orientation is not unique and is deliberately not part of the contract.
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


def _parametric_span_contains_constant(
    design: CompiledPredictor,
    *,
    fit_intercept: bool,
) -> bool:
    """Mirror the intercept-equivalence check at the start of ``mgcv::gam.side``."""
    if fit_intercept:
        return True

    parametric_blocks = [
        np.asarray(tb.basis_train, dtype=np.float64)
        for tb in design.compiled_terms
        if str(tb.term_type) == "parametric"
    ]
    if not parametric_blocks:
        return False

    Xp = np.column_stack(parametric_blocks)
    constant_tol = np.finfo(np.float64).eps**0.75
    if Xp.shape[0] > 1 and np.any(np.std(Xp, axis=0, ddof=1) < constant_tol):
        return True

    ones = np.ones(Xp.shape[0], dtype=np.float64)
    coef, *_ = np.linalg.lstsq(Xp, ones, rcond=None)
    return bool(np.max(np.abs(Xp @ coef - ones)) < constant_tol)


def _term_is_side_condition_passthrough(tb: CompiledTerm) -> bool:
    policy = getattr(tb, "side_condition_policy", None)
    return bool(
        str(tb.term_type) == "parametric"
        or (
            policy is not None
            and policy.exempt_from_predictor_side_conditions
        )
    )


def _smooth_variable_tokens(tb: CompiledTerm) -> tuple[str, ...]:
    """Mirror the variable-name gate at the start of ``mgcv::gam.side``."""
    metadata = dict(tb.metadata or {})
    factor_by = dict(metadata.get("factor_by", {}) or {})
    by_name = factor_by.get("source_by", None)
    by_level = factor_by.get("level", None)
    if by_name is None:
        runtime_by = getattr(tb.by_variable_info, "name", None)
        if runtime_by is not None and not str(runtime_by).startswith("__gam_by__"):
            by_name = runtime_by

    tokens = []
    for feature in tb.feature_info.feature_names:
        token = str(feature)
        if by_name is not None:
            token += str(by_name)
        if by_level is not None:
            token += str(by_level)
        tokens.append(token)
    return tuple(tokens)


def _unchanged_side_condition_report(design: CompiledPredictor) -> dict:
    term_reports = []
    for tb in design.compiled_terms:
        width = int(np.asarray(tb.basis_train).shape[1])
        kept = (
            np.asarray(tb.kept_columns, dtype=int).tolist()
            if tb.kept_columns is not None
            else list(range(width))
        )
        term_reports.append(
            {
                "label": tb.label,
                "exempt": _term_is_side_condition_passthrough(tb),
                "deleted_columns": [],
                "kept_columns": kept,
                "n_deleted": 0,
                "absorbed_centering": False,
            }
        )
    return {
        "predictor": design.name,
        "n_deleted_total": 0,
        "n_terms_dropped": 0,
        "term_reports": term_reports,
    }


def apply_global_side_conditions(
    design: CompiledPredictor,
    *,
    fit_intercept: bool = True,
    tol: float = float(np.finfo(np.float64).eps**0.5),
    warn: bool = True,
):
    """
    Apply predictor-wide identifiability side conditions to a CompiledPredictor.

    It walks each compiled term in order and performs three operations:

    1. **Passthrough terms** (parametric terms, random effects, factor smooths):
       Preserve their basis and penalties unchanged.  Still add their columns to
       the compiled design, but not to the smooth-nesting accumulator.  In
       particular, ``mgcv::gam.side`` receives the parametric matrix separately
       and never deletes parametric columns.

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
        Whether the model has a literal intercept.  As in ``mgcv::gam.side``, a
        constant contained in the parametric span also seeds the smooth-nesting
        accumulator and triggers eligible sum-to-zero constraints.
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

    # mgcv/R/mgcv.r::gam.side() only uses Xp to determine whether it contains a
    # constant (or an equivalent span). It never includes other parametric
    # columns in smooth-dependence testing.
    intercept_equivalent = _parametric_span_contains_constant(
        design,
        fit_intercept=fit_intercept,
    )

    # mgcv::gam.side returns immediately when there is no repeated smooth
    # variable (`lv == length(unique(v.names))`). The one-smooth case is
    # unambiguous here. Preserve the compiler's matrices byte-for-byte when
    # the runtime has already absorbed its ordinary centering constraint.
    smooth_terms = [
        tb for tb in design.compiled_terms if str(tb.term_type) != "parametric"
    ]
    requires_centering = False
    for smooth in smooth_terms:
        policy = getattr(smooth, "side_condition_policy", None)
        requires_centering = requires_centering or bool(
            intercept_equivalent
            and not _term_is_side_condition_passthrough(smooth)
            and not bool(policy is not None and policy.skip_centering)
            and (
                smooth.by_variable_info.name is None
                or bool(smooth.by_variable_info.is_constant)
            )
            and str(smooth.term_type) != "tensor_interaction"
        )
    variable_tokens = [
        token for smooth in smooth_terms for token in _smooth_variable_tokens(smooth)
    ]
    if len(variable_tokens) == len(set(variable_tokens)) and not requires_centering:
        return design, _unchanged_side_condition_report(design)
    if len(smooth_terms) <= 1:
        if not requires_centering:
            return design, _unchanged_side_condition_report(design)

    # ``mgcv::gam.side`` builds X1 afresh for each smooth from earlier eligible
    # smooths attached to the *same variable token*. Retain both the final
    # observational basis and the pre-deletion augmented basis for that lookup.
    # A predictor-wide accumulator would incorrectly treat two distinct, merely
    # collinear covariates as a nested smooth pair.
    nesting_records: list[dict[str, object]] = []

    processed_terms: dict[int, CompiledTerm] = {}
    new_penalty_blocks: list[CompiledPenalty] = []
    design_blocks: dict[int, np.ndarray] = {}
    predictor_transforms: dict[int, np.ndarray] = {}
    deleted_total = 0
    term_reports_by_index: dict[int, dict] = {}

    # ``gam.side`` traverses smooths by increasing dimension, retaining formula
    # order only as the tie-breaker. The reconstructed predictor is restored to
    # formula order after the side-condition transforms have been computed.
    processing_order = sorted(
        range(len(design.compiled_terms)),
        key=lambda idx: (
            0
            if _term_is_side_condition_passthrough(design.compiled_terms[idx])
            else len(design.compiled_terms[idx].feature_info.feature_names),
            idx,
        ),
    )

    for term_idx in processing_order:
        tb = design.compiled_terms[term_idx]
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

        # ── Passthrough terms ─────────────────────────────────────────────────
        # mgcv::gam.side(sm, Xp, ...) mutates only `sm`. Parametric aliasing is
        # deliberately left for the fitting backend's rank-deficiency handling.
        policy = getattr(tb, "side_condition_policy", None)
        exempt = _term_is_side_condition_passthrough(tb)
        if exempt:
            sl_new = slice(0, d)
            processed_terms[term_idx] = CompiledTerm(
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
            predictor_transforms[term_idx] = np.eye(d, dtype=np.float64)
            for pb, P in zip(term_penalty_objs, pen_matrices, strict=True):
                new_penalty_blocks.append(
                    CompiledPenalty(
                        label=pb.label,
                        coef_slice=sl_new,
                        matrix=P.copy(),
                        smoothing_index=pb.smoothing_index,
                        term_index=term_idx,
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
                design_blocks[term_idx] = B
            term_reports_by_index[term_idx] = {
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

        current_tokens = frozenset(_smooth_variable_tokens(tb))
        current_dim = len(tb.feature_info.feature_names)
        prior_nested = [
            record
            for record in nesting_records
            if int(record["dim"]) <= current_dim
            and bool(current_tokens.intersection(record["tokens"]))
        ]
        acc_blocks = [
            np.asarray(record["basis"], dtype=np.float64)
            for record in prior_nested
        ]
        acc_aug_blocks = [
            np.asarray(record["augmented_basis"], dtype=np.float64)
            for record in prior_nested
        ]
        if intercept_equivalent:
            acc_blocks.insert(0, np.ones((n_obs, 1), dtype=np.float64))
            intercept_aug = np.zeros((n_obs + design.n_coef, 1), dtype=np.float64)
            intercept_aug[:n_obs, 0] = 1.0
            acc_aug_blocks.insert(0, intercept_aug)
        acc = (
            np.column_stack(acc_blocks)
            if acc_blocks
            else np.empty((n_obs, 0), dtype=np.float64)
        )
        acc_aug = (
            np.column_stack(acc_aug_blocks)
            if acc_aug_blocks
            else np.empty((n_obs + design.n_coef, 0), dtype=np.float64)
        )

        # Step (a): optionally absorb a sum-to-zero centering constraint.
        if (
            intercept_equivalent
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
            if acc.shape[1] <= int(intercept_equivalent):
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
        for pb, S in zip(term_penalty_objs, pen_matrices, strict=True):
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

        sl_new = slice(0, d_final)
        predictor_transforms[term_idx] = Q_term_final

        processed_terms[term_idx] = CompiledTerm(
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

        for pb, pdef_new in pen_pairs_final:
            new_penalty_blocks.append(
                CompiledPenalty(
                    label=pb.label,
                    coef_slice=sl_new,
                    matrix=np.asarray(pdef_new.matrix, dtype=np.float64),
                    smoothing_index=pb.smoothing_index,
                    term_index=term_idx,
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
            design_blocks[term_idx] = B_final

        if pen_matrices:
            # `mgcv::gam.side` caches `sm[[i]]$Xa` before deleting columns from
            # the current smooth and reuses that original augmented matrix when
            # testing higher-dimensional nested terms.
            B_aug_for_later = _augment_term_matrix(
                B,
                pen_matrices,
                coef_slice=tb.coef_slice,
                total_coef=design.n_coef,
                tol=tol,
            )
        else:
            B_aug_for_later = np.zeros(
                (n_obs + design.n_coef, B.shape[1]), dtype=np.float64
            )
            B_aug_for_later[:n_obs, :] = B
        nesting_records.append(
            {
                "tokens": current_tokens,
                "dim": current_dim,
                "basis": B_final,
                "augmented_basis": B_aug_for_later,
            }
        )

        term_reports_by_index[term_idx] = {
            "label": tb.label,
            "exempt": False,
            "deleted_columns": [] if deleted_orig is None else deleted_orig.tolist(),
            "kept_columns": [] if kept_orig is None else kept_orig.tolist(),
            "n_deleted": n_deleted,
            "absorbed_centering": absorbed_centering,
        }

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

    # Restore formula order after the dimension-ordered traversal. mgcv keeps
    # the original smooth list order while mutating each smooth in place.
    predictor_Q = np.zeros((design.n_coef, design.n_coef), dtype=np.float64)
    final_term_blocks: list[CompiledTerm] = []
    final_penalty_blocks: list[CompiledPenalty] = []
    final_design_blocks: list[np.ndarray] = []
    term_reports: list[dict] = []
    start = 0
    for term_idx in range(len(design.compiled_terms)):
        tb = processed_terms[term_idx]
        width = int(tb.basis_train.shape[1])
        sl_new = slice(start, start + width)
        tb.coef_slice = sl_new
        final_idx = len(final_term_blocks)
        final_term_blocks.append(tb)
        term_reports.append(term_reports_by_index[term_idx])
        if width > 0:
            final_design_blocks.append(design_blocks[term_idx])
            predictor_Q[design.compiled_terms[term_idx].coef_slice, sl_new] = (
                predictor_transforms[term_idx]
            )
        for pb in new_penalty_blocks:
            if int(pb.term_index) != term_idx:
                continue
            pb.term_index = final_idx
            pb.coef_slice = sl_new
            final_penalty_blocks.append(pb)
        start += width

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
    override_modes: list[str | None] = [None] * n_sp_final
    override_values = np.full(n_sp_final, np.nan, dtype=np.float64)
    for old_idx, new_idx in old_to_new_sp.items():
        if old_idx < len(old_modes):
            override_modes[new_idx] = old_modes[old_idx]
        if old_values is not None and old_idx < old_values.size:
            override_values[new_idx] = old_values[old_idx]

    # ── Assemble final CompiledPredictor ──────────────────────────────────────
    matrix_train = (
        np.column_stack(final_design_blocks)
        if final_design_blocks
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
        "n_terms_dropped": 0,
        "term_reports": term_reports,
    }
    return new_design, report


__all__ = [
    "apply_global_side_conditions",
]
