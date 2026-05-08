"""Central invariant-only policy for non-unique mgcv representation surfaces.

Behavioral parity stays strict. These helpers are for mathematically non-unique
representation layers only.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from nampy.gam.linalg import matrix_self_gram, matrix_summary, row_space_projector
from nampy.gam.linalg import symmetric_spectrum as penalty_spectrum

_GAM_SETUP_INVARIANT_CASE_IDS = frozenset(
    {
        "gaussian_tp_two_dim",
        "gaussian_ts_two_dim",
        "gaussian_fs",
        "gaussian_fs_numeric_by",
    }
)
_GAM_SETUP_DOMINANT_PENALTY_SPECTRUM_CASE_IDS = frozenset({"gaussian_cs_uni"})
_GAM_SIDE_INVARIANT_CLASS_NAMES = frozenset(
    {
        "fs.interaction",
        "tprs.smooth",
        "ts.smooth",
    }
)
_PREOPT_BLOCKS_ALIGN_BASIS_CASE_IDS = frozenset(
    {
        "gaussian_tp_two_dim",
        "gaussian_ts_two_dim",
    }
)
_PREOPT_BLOCKS_SKIP_RANGE_ROOT_REPR_CASE_IDS = (
    _PREOPT_BLOCKS_ALIGN_BASIS_CASE_IDS | frozenset({"gaussian_fs"})
)


def gam_setup_uses_invariant_transform(case_id: str) -> bool:
    """Return whether ``gam.setup`` assembly should compare basis spans only."""
    return case_id in _GAM_SETUP_INVARIANT_CASE_IDS


def gam_setup_compares_dominant_penalty_spectrum(case_id: str) -> bool:
    """Return whether the shrinkage-floor orientation is non-unique."""
    return case_id in _GAM_SETUP_DOMINANT_PENALTY_SPECTRUM_CASE_IDS


def gam_side_uses_invariant_transform(class_name: str) -> bool:
    """Return whether ``gam.side`` should compare basis spans only."""
    return class_name in _GAM_SIDE_INVARIANT_CLASS_NAMES


def final_fit_uses_exact_orientation_parity(
    formula, *, skip_coef_comparison: bool
) -> bool:
    """Return whether full final-fit matrices are stable enough for exact checks."""
    if skip_coef_comparison:
        return False
    text = str(formula)
    return "tp" not in text


def preoptimization_blocks_align_basis_columns(case_id: str) -> bool:
    """Return whether preoptimization block bases should be aligned up to span."""
    return case_id in _PREOPT_BLOCKS_ALIGN_BASIS_CASE_IDS


def preoptimization_blocks_compare_range_root_representation(case_id: str) -> bool:
    """Return whether preoptimization range-root blocks should stay exact."""
    return case_id not in _PREOPT_BLOCKS_SKIP_RANGE_ROOT_REPR_CASE_IDS


def stable_column_space_projector(matrix):
    """Return a symmetric column-space projector with stable SVD rank pruning."""
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError("matrix must be 2D.")
    if mat.shape[1] == 0:
        return np.zeros((mat.shape[0], mat.shape[0]), dtype=np.float64)
    if mat.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float64)

    U, s, _ = np.linalg.svd(mat, full_matrices=False)
    if s.size == 0:
        return np.zeros((mat.shape[0], mat.shape[0]), dtype=np.float64)
    tol = (
        np.finfo(np.float64).eps
        * max(float(mat.shape[0]), float(mat.shape[1]), 1.0)
        * float(s[0])
    )
    rank = int(np.sum(s > tol))
    if rank == 0:
        return np.zeros((mat.shape[0], mat.shape[0]), dtype=np.float64)
    Uq = np.asarray(U[:, :rank], dtype=np.float64)
    proj = Uq @ Uq.T
    return np.asarray(0.5 * (proj + proj.T), dtype=np.float64)


def _copy_raw_value(value):
    if isinstance(value, np.ndarray):
        return np.asarray(value).copy()
    if isinstance(value, dict):
        return {key: _copy_raw_value(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_copy_raw_value(val) for val in value]
    return value


def _normalized_penalties(value):
    if isinstance(value, dict):
        values = list(value.values())
    else:
        values = list(value)
    return [np.asarray(v, dtype=np.float64) for v in values]


def _symmetric_rank(matrix):
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.size == 0:
        return 0
    ev = np.linalg.eigvalsh(0.5 * (mat + mat.T))
    scale = max(float(np.max(np.abs(ev))) if ev.size else 0.0, 1.0)
    tol = np.finfo(np.float64).eps ** 0.8 * scale
    return int(np.sum(ev > tol))


def _sorted_rows(matrix):
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError("matrix must be 2D.")
    if mat.shape[0] == 0:
        return mat.copy()
    order = np.lexsort(mat.T[::-1])
    return np.asarray(mat[order, :], dtype=np.float64)


def _canonicalize_tprs_raw_state(state):
    extra = state["extra"]
    used_supplied_knots = bool(extra.pop("used_supplied_knots", False))
    used_subsampling = bool(extra.pop("used_subsampling", False))
    pure_knot = bool(extra.pop("pure_knot", False))
    Xu = np.asarray(extra["Xu"], dtype=np.float64)

    multivariate_knot_basis = (
        (
            used_supplied_knots
            or (Xu.ndim == 2 and Xu.shape[1] > 1 and state["X"].shape[0] != Xu.shape[0])
        )
        and not pure_knot
        and Xu.ndim == 2
        and Xu.shape[1] > 1
    )
    if used_subsampling or multivariate_knot_basis:
        # For multivariate user-knot and max.knots setups the exact knot/sample
        # choice and Lanczos parameterization can differ while the same
        # penalized subspace and fixed-sp behavior hold.
        state["S"] = [stable_column_space_projector(S) for S in state["S"]]
        state["X"] = matrix_summary(state["X"])
        extra["UZ"] = matrix_summary(extra["UZ"])
        extra["Xu"] = matrix_summary(Xu)
        return state

    state["S"] = [penalty_spectrum(S) for S in state["S"]]
    state["X"] = matrix_self_gram(state["X"])
    extra["UZ"] = stable_column_space_projector(extra["UZ"])
    return state


def _canonicalize_cs_raw_state(state):
    state["S"] = [penalty_spectrum(S) for S in state["S"]]
    return state


def _canonicalize_fs_raw_state(state):
    extra = state["extra"]
    state["S"] = [penalty_spectrum(S) for S in state["S"]]
    state["X"] = matrix_self_gram(state["X"])
    extra["Xb"] = matrix_self_gram(extra["Xb"])
    extra["P"] = stable_column_space_projector(extra["P"])
    return state


def _canonicalize_sz_raw_state(state):
    state["S"] = [penalty_spectrum(S) for S in state["S"]]
    state["X"] = matrix_self_gram(state["X"])
    extra = state["extra"]
    extra["Xb"] = matrix_self_gram(extra["Xb"])
    return state


def _canonicalize_tensor_raw_state(state):
    extra = state["extra"]
    mc = extra.get("mc", [])
    if len(state["S"]) > 2:
        state["S"] = [stable_column_space_projector(S) for S in state["S"]]
    else:
        state["S"] = [penalty_spectrum(S) for S in state["S"]]
    if len(state["S"]) <= 2 and any(bool(v) for v in mc):
        # Centered `ti()` margins with `cs` shrinkage preserve the same tensor
        # penalty subspaces and dominant spectrum, but the tiny shrinkage-floor
        # eigenvalues can drift under absorb.cons-style reparameterization.
        for i, spec in enumerate(state["S"]):
            spec = np.asarray(spec, dtype=np.float64).copy()
            if spec.size == 0:
                continue
            tol = 5e-2 * max(float(np.max(np.abs(spec))), 1.0)
            spec[np.abs(spec) < tol] = 0.0
            state["S"][i] = np.round(spec, decimals=4)

    # `np=TRUE` tensor reparameterization fixes the represented function space,
    # not a unique basis scaling for ill-conditioned marginal inverses.
    state["X"] = stable_column_space_projector(state["X"])

    XP = extra.get("XP", None)
    if isinstance(XP, list):
        extra["XP"] = [None if xp is None else row_space_projector(xp) for xp in XP]

    C = extra.get("C", None)
    if isinstance(C, np.ndarray) and C.ndim == 2:
        extra["C"] = matrix_summary(C)

    return state


def canonicalize_raw_representation_state(state: dict[str, Any]) -> dict[str, Any]:
    """Canonicalize non-unique raw smooth state onto invariant test surfaces."""
    state = _copy_raw_value(state)
    state["S"] = _normalized_penalties(state["S"])
    class_name = str(state["class_name"])

    if class_name == "cs.smooth":
        return _canonicalize_cs_raw_state(state)
    if class_name in {"tprs.smooth", "ts.smooth"}:
        return _canonicalize_tprs_raw_state(state)
    if class_name == "fs.interaction":
        return _canonicalize_fs_raw_state(state)
    if class_name == "sz.interaction":
        return _canonicalize_sz_raw_state(state)
    if class_name == "tensor.smooth":
        return _canonicalize_tensor_raw_state(state)

    state["X"] = np.asarray(state["X"], dtype=np.float64)
    return state


__all__ = [
    "canonicalize_raw_representation_state",
    "final_fit_uses_exact_orientation_parity",
    "gam_setup_uses_invariant_transform",
    "gam_side_uses_invariant_transform",
    "preoptimization_blocks_align_basis_columns",
    "preoptimization_blocks_compare_range_root_representation",
    "stable_column_space_projector",
]
