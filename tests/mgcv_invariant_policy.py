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
        "gaulss_fs",
        "gaussian_sz",
    }
)
_GAM_SETUP_DOMINANT_PENALTY_SPECTRUM_CASE_IDS = frozenset({"gaussian_cs_uni"})
_GAM_SIDE_INVARIANT_CLASS_NAMES = frozenset(
    {
        "fs.interaction",
        "sz.interaction",
        "tprs.smooth",
        "ts.smooth",
    }
)
_PREOPT_BLOCKS_ALIGN_BASIS_CASE_IDS = frozenset(
    {
        "gaussian_tp_two_dim",
        "gaussian_ts_two_dim",
        "gaussian_sz",
    }
)
_PREOPT_BLOCKS_SKIP_RANGE_ROOT_REPR_CASE_IDS = (
    _PREOPT_BLOCKS_ALIGN_BASIS_CASE_IDS | frozenset({"gaussian_fs"})
)
_LPMATRIX_INVARIANT_CASE_IDS = frozenset(
    {
        "gaussian_tp_k20",
        "binomial_logit",
        "poisson",
        "gamma_log",
        "binomial_separation",
        "gaussian_fs_by_factor",
        "gaussian_fs_select_reml",
        "gaussian_ti_cs_ps_reml",
        "factor_smooth_sz",
        "transformed_tp",
        "transformed_ts",
    }
)


def gam_setup_uses_invariant_transform(case_id: str) -> bool:
    """Return whether ``gam.setup`` should compare basis-orientation invariants."""
    return case_id in _GAM_SETUP_INVARIANT_CASE_IDS


def gam_setup_compares_dominant_penalty_spectrum(case_id: str) -> bool:
    """Return whether the shrinkage-floor orientation is non-unique."""
    return case_id in _GAM_SETUP_DOMINANT_PENALTY_SPECTRUM_CASE_IDS


def gam_side_uses_invariant_transform(class_name: str) -> bool:
    """Return whether ``gam.side`` should compare basis spans only."""
    return class_name in _GAM_SIDE_INVARIANT_CLASS_NAMES


def final_fit_uses_exact_orientation_parity(
    model, *, skip_coef_comparison: bool
) -> bool:
    """Return whether compiled terms have uniquely identified coefficient bases."""
    if skip_coef_comparison:
        return False
    result = getattr(model, "gam_result_", None)
    compiled = (
        getattr(result, "compiled_model", None)
        if result is not None
        else getattr(getattr(model, "gam_result_", None), "compiled_model", None)
    )
    if compiled is None:
        raise ValueError("A compiled GAM model is required for representation policy.")
    basis_names = [
        str(getattr(term, "basis_name", "unknown")).lower()
        for term in (getattr(compiled, "compiled_terms", ()) or ())
    ]
    for basis_name in basis_names:
        components = {basis_name}
        if "(" in basis_name and basis_name.endswith(")"):
            components.update(
                part.strip() for part in basis_name.split("(", 1)[1][:-1].split(",")
            )
        if components & {"tp", "ts", "fs", "sz"}:
            return False
    return True


def lpmatrix_uses_invariant_comparison(case_id: str) -> bool:
    """Return whether an lpmatrix is identified only up to basis orientation."""
    return case_id in _LPMATRIX_INVARIANT_CASE_IDS


def preoptimization_blocks_align_basis_columns(case_id: str) -> bool:
    """Return whether preoptimization blocks have non-unique basis orientation."""
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


def penalized_response_operator(design, penalties, *, offsets=None) -> np.ndarray:
    """Return the unit-weight penalized fit operator in observation space."""
    X = np.asarray(design, dtype=np.float64)
    penalty_list = [np.asarray(S, dtype=np.float64) for S in penalties]
    precision = X.T @ X
    if offsets is None:
        for penalty in penalty_list:
            if penalty.shape != precision.shape:
                raise ValueError("Full penalties must match the design width.")
            precision = precision + penalty
    else:
        offset_values = np.asarray(offsets, dtype=np.int64).reshape(-1)
        for penalty, offset in zip(penalty_list, offset_values, strict=True):
            start = int(offset) - 1
            stop = start + penalty.shape[0]
            precision[start:stop, start:stop] += penalty
    precision = 0.5 * (precision + precision.T)
    return np.asarray(X @ np.linalg.pinv(precision, hermitian=True) @ X.T)


def normalize_penalty_scale(matrix) -> np.ndarray:
    """Remove an arbitrary positive scalar from a symmetric penalty.

    This is needed for default ``fs`` smooths only. ``mgcv::nat.param`` leaves
    the repeated null eigenspace orientation to LAPACK, while ``smoothCon``
    subsequently uses an orientation-dependent row-sum norm to scale every
    factor-smooth penalty. The represented penalty subspace is identified; its
    common scalar is not portable across LAPACK builds.
    """
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.size == 0:
        return mat.copy()
    symmetric = 0.5 * (mat + mat.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    scale = float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0
    if scale == 0.0:
        return symmetric
    return np.asarray(symmetric / scale, dtype=np.float64)


def center_term_contributions(matrix) -> np.ndarray:
    """Remove the non-identifiable intercept gauge from term contributions."""
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("term contributions must be a 2D matrix.")
    return np.asarray(values - np.mean(values, axis=0, keepdims=True))


def centered_prediction_covariance(
    lpmatrix, covariance, *, coefficient_indices
) -> np.ndarray:
    """Return covariance of prediction contrasts after removing a constant gauge.

    Default ``fs`` smooths are uncentred. Their TP null-space basis may rotate
    between LAPACK implementations, moving a constant component between the
    intercept and smooth coefficient blocks. Covariance of centred smooth
    contributions is identified even though raw ``type="terms"`` standard
    errors are not.
    """
    design = np.asarray(lpmatrix, dtype=np.float64)
    cov = np.asarray(covariance, dtype=np.float64)
    indices = np.asarray(coefficient_indices, dtype=np.int64).reshape(-1)
    if design.ndim != 2 or cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("lpmatrix and covariance must be two-dimensional.")
    if design.shape[1] != cov.shape[0]:
        raise ValueError("lpmatrix width must match covariance dimension.")
    if np.any(indices < 0) or np.any(indices >= design.shape[1]):
        raise ValueError("coefficient_indices are outside the coefficient space.")
    term_design = design[:, indices]
    term_cov = cov[np.ix_(indices, indices)]
    kernel = np.asarray(term_design @ term_cov @ term_design.T, dtype=np.float64)
    row_mean = np.mean(kernel, axis=1, keepdims=True)
    col_mean = np.mean(kernel, axis=0, keepdims=True)
    return np.asarray(kernel - row_mean - col_mean + np.mean(kernel), dtype=np.float64)


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
        keys = list(value)
        if "S" in value:
            keys = ["S"] + sorted(
                (key for key in keys if key != "S"),
                key=lambda key: (
                    (0, int(key)) if str(key).isdigit() else (1, str(key))
                ),
            )
        values = [value[key] for key in keys]
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


def _canonicalize_duchon_raw_state(state):
    extra = state["extra"]
    extra.pop("used_supplied_knots", False)
    extra.pop("used_subsampling", False)
    extra.pop("pure_knot", False)
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
    if len(state["S"]) > 2:
        state["S"] = [stable_column_space_projector(S) for S in state["S"]]
    else:
        state["S"] = [penalty_spectrum(S) for S in state["S"]]

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
    if class_name == "duchon.spline":
        return _canonicalize_duchon_raw_state(state)
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
    "centered_prediction_covariance",
    "final_fit_uses_exact_orientation_parity",
    "gam_setup_uses_invariant_transform",
    "gam_side_uses_invariant_transform",
    "lpmatrix_uses_invariant_comparison",
    "normalize_penalty_scale",
    "penalized_response_operator",
    "preoptimization_blocks_align_basis_columns",
    "preoptimization_blocks_compare_range_root_representation",
    "stable_column_space_projector",
]
