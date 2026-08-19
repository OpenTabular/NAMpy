"""
Penalty reparameterisation utilities.

Eigendecomposition-based reparameterisation splits a smooth term's basis into
a null-space component (penalty eigenvalue ≈ 0) and a penalised component
(positive eigenvalue).  This is used to convert the penalised regression
problem into a mixed model form ``y = X_fix beta + Z_rand u + epsilon``,
which simplifies REML score computation.

:func:`reparameterize_smooth`
    Split a single smooth basis ``B`` with penalty ``P`` into fixed and random
    design sub-matrices.

:func:`_matrix_sqrt_psd`
    Symmetric square root of a PSD matrix (used for penalty scaling).
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, cast

import numpy as np
from scipy.linalg import qr as scipy_qr
from scipy.linalg import solve_triangular

from .._model_state import (
    _coef_column_offset,
    _compiled_model,
    _design_matrix,
    _fit_intercept,
    _n_coef,
    _n_smoothing_params,
    _penalty_blocks_seq,
    _term_blocks_seq,
)
from ..fit.capabilities import uses_closed_form_solver
from ..linalg import (
    matrix_sqrt_psd,
    mgcv_mroot_chol,
    numerical_rank,
    positive_semidefinite_root,
    symmetric_eigen_partition,
    symmetric_eigh,
    symmetric_eigvalsh,
)


@dataclass
class ReparamState:
    X_fix: Optional[np.ndarray]
    Z_rand: Optional[np.ndarray]
    ZtZ_rand: Optional[np.ndarray]
    sl_blocks: Optional[List["SlBlock"]]
    penalty_range_basis: Optional[np.ndarray] = None
    penalty_null_basis: Optional[np.ndarray] = None
    penalty_range_roots: Optional[List[np.ndarray]] = None
    grouped_penalties: Optional[List["_GroupedPenalty"]] = None


@dataclass
class DynamicReparamDesign:
    X_fix: np.ndarray
    Z_rand: np.ndarray
    ZtZ_rand: np.ndarray
    penalty_logdet: float
    null_dim: int


@dataclass
class PenaltyReparameterizationState:
    Y: np.ndarray
    Z: np.ndarray
    U1: np.ndarray
    UrS: list[np.ndarray]
    rp: dict[str, Any]
    T: np.ndarray
    St: np.ndarray
    Sr: np.ndarray
    Eb: np.ndarray
    Mp: int
    X_range: np.ndarray
    X_fix: np.ndarray
    Z_rand: np.ndarray


@dataclass
class PreOptimizationSetupState:
    """
    Exact mgcv-style pre-optimization setup state.

    Mirrors the objects assembled in ``mgcv/R/mgcv.r::estimate.gam`` from the
    compiled Python design/penalty state:

    - ``G$S``, ``G$off``, ``G$rank``
    - ``G$L``, ``G$lsp0``, ``G$sp``
    - ``G$rS`` from ``mgcv:::mini.roots``
    - ``Ssp <- mgcv:::totalPenaltySpace(...)`` yielding ``Y/Z/E``
    - ``G$Eb``, ``G$U1``, ``G$Mp``, ``G$UrS``
    """

    X: np.ndarray
    offset: np.ndarray | None
    S: list[np.ndarray]
    off: np.ndarray
    rank: np.ndarray
    L: np.ndarray | None
    lsp0: np.ndarray
    sp: np.ndarray
    log_sp_full: np.ndarray
    rS: list[np.ndarray]
    Y: np.ndarray
    Z: np.ndarray
    E: np.ndarray
    Eb: np.ndarray
    U1: np.ndarray
    UrS: list[np.ndarray]
    Mp: int


@dataclass
class SlBlock:
    term_index: int
    repara: bool
    smoothing_index: Optional[int]
    start: int
    stop: int
    ncol: int
    blockSize: int
    lambda_scaling: float = 1.0
    kind: str = "smooth"
    is_null_space_penalty: bool = False


@dataclass
class _GroupedPenalty:
    smoothing_index: int
    matrix_full: np.ndarray
    kind: str
    is_null_space_penalty: bool
    term_indices: tuple[int, ...]


def _assign_reparam_state(
    model, state: Optional[ReparamState]
) -> Optional[ReparamState]:
    model.reparam_state_ = state
    model.sl_blocks_ = None if state is None else list(state.sl_blocks or [])
    return state


def assign_exact_reparam_state(
    model, state: Optional[PenaltyReparameterizationState]
) -> Optional[PenaltyReparameterizationState]:
    model.reparam_state_ = state
    model.sl_blocks_ = None
    return state


def ensure_penalty_reparameterization_state(model) -> ReparamState:
    state = getattr(model, "reparam_state_", None)
    if state is None:
        state = model._build_penalty_reparameterized_system()
    if state is None:
        raise RuntimeError("Penalty reparameterization state is unavailable.")
    return cast(ReparamState, state)


def iter_sl_random_blocks(state: ReparamState):
    for sl_block in list(state.sl_blocks or []):
        if sl_block.repara:
            yield sl_block


def sl_group_indices(state: ReparamState) -> Dict[int, np.ndarray]:
    groups: Dict[int, list[int]] = {}
    for sl_block in iter_sl_random_blocks(state):
        if sl_block.smoothing_index is None:
            continue
        sp_idx = int(sl_block.smoothing_index)
        groups.setdefault(sp_idx, []).extend(
            range(int(sl_block.start), int(sl_block.stop))
        )
    return {
        int(sp_idx): np.asarray(sorted(idxs), dtype=np.int64)
        for sp_idx, idxs in groups.items()
    }


def sl_lambda_vector(state: ReparamState, sp: np.ndarray) -> np.ndarray:
    if not state.sl_blocks:
        return np.empty((0,), dtype=np.float64)
    lam_parts = []
    for sl_block in iter_sl_random_blocks(state):
        if sl_block.smoothing_index is None or int(sl_block.blockSize) == 0:
            continue
        sp_val = float(sp[int(sl_block.smoothing_index)])
        lam_parts.append(
            np.full(
                int(sl_block.blockSize),
                sp_val * float(sl_block.lambda_scaling),
                dtype=np.float64,
            )
        )
    return np.concatenate(lam_parts) if lam_parts else np.empty((0,), dtype=np.float64)


def reparameterize_smooth(B, P, tol=1e-10):
    dec = symmetric_eigen_partition(P, tol=tol)
    U0 = np.asarray(dec["U0"], dtype=np.float64)
    U1 = np.asarray(dec["U1"], dtype=np.float64)
    d_pos = np.asarray(dec["d_pos"], dtype=np.float64)
    B0 = B @ U0
    B1 = B @ U1
    Zr = B1 / np.sqrt(d_pos)[np.newaxis, :] if d_pos.size else B1
    return (
        B0,
        Zr,
        {
            "U0": U0,
            "U1": U1,
            "d_pos": d_pos,
            "n_null": int(dec["null_space_dim"]),
            "n_pen": int(dec["rank"]),
        },
    )


def _matrix_sqrt_psd(M, tol=1e-12):
    del tol
    return matrix_sqrt_psd(M)


def _positive_semidefinite_root(P, *, rank=None, tol=1e-10):
    return positive_semidefinite_root(P, rank=rank, tol=tol)


def _mroot_chol(P, *, rank=None):
    """
    Port of `mgcv::mroot(..., method="chol")` returning `B` with `B B' = P`.
    """
    return mgcv_mroot_chol(P, rank=rank)


def _full_design_matrix(model) -> np.ndarray:
    compiled = _compiled_model(model)
    if compiled is None:
        raise RuntimeError("Compiled model is unavailable.")

    predictors = tuple(getattr(compiled, "predictors", ()) or ())
    if predictors:
        blocks = []
        for predictor in predictors:
            X_pred = np.asarray(predictor.design_matrix, dtype=np.float64)
            if bool(getattr(predictor, "has_intercept", False)):
                ones = np.ones((X_pred.shape[0], 1), dtype=np.float64)
                blocks.append(np.column_stack([ones, X_pred]))
            else:
                blocks.append(X_pred)
        return (
            np.column_stack(blocks)
            if blocks
            else np.empty((int(getattr(model, "n_samples_", 0)), 0), dtype=np.float64)
        )

    X_red = np.asarray(_design_matrix(model), dtype=np.float64)
    if bool(_fit_intercept(model)):
        return np.column_stack([np.ones((X_red.shape[0], 1), dtype=np.float64), X_red])
    return X_red


def _full_coef_indices(model, coef_slice: slice) -> np.ndarray:
    compiled = _compiled_model(model)
    if compiled is None:
        raise RuntimeError("Compiled model is unavailable.")

    start = int(coef_slice.start)
    stop = int(coef_slice.stop)
    width = max(stop - start, 0)

    mapping = getattr(compiled, "coef_reduced_to_full_idx", None)
    if mapping is None:
        off = 1 if bool(_fit_intercept(model)) else 0
        return np.arange(start + off, stop + off, dtype=np.int64)

    idx = np.asarray(mapping[start:stop], dtype=np.int64)
    if idx.shape != (width,):
        raise RuntimeError(
            "Compiled-model reduced->full coefficient map is inconsistent."
        )
    if idx.size == 0:
        return idx
    expected = np.arange(int(idx[0]), int(idx[0]) + idx.size, dtype=np.int64)
    if not np.array_equal(idx, expected):
        raise RuntimeError(
            "Penalty block is not contiguous in the mgcv full-parameter order."
        )
    return idx


def _lift_reduced_matrix_to_full(model, M: np.ndarray, p_full: int) -> np.ndarray:
    M = np.asarray(M, dtype=np.float64)
    if M.shape == (p_full, p_full):
        return M.copy()

    compiled = _compiled_model(model)
    if compiled is None:
        raise RuntimeError("Compiled model is unavailable.")
    mapping = np.asarray(getattr(compiled, "coef_reduced_to_full_idx", None))
    if mapping.shape != (M.shape[0],):
        raise ValueError("Reduced matrix can not be lifted to full coordinates.")

    out = np.zeros((p_full, p_full), dtype=np.float64)
    out[np.ix_(mapping, mapping)] = M
    return out


def _total_penalty_space_from_blocks(
    penalties: list[np.ndarray],
    offsets_1based: np.ndarray,
    p: int,
    *,
    H: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Port of ``mgcv:::totalPenaltySpace(S, H, off, p)`` on packed penalties.
    """

    if H is not None:
        H = np.asarray(H, dtype=np.float64)
        if H.shape != (p, p):
            raise ValueError("H has wrong dimension.")
        Hscale = float(np.sqrt(np.sum(H * H)))
        if Hscale == 0.0:
            H = None
    if H is None:
        St = np.zeros((p, p), dtype=np.float64)
    else:
        St = H / float(np.sqrt(np.sum(H * H)))

    for S_i, off_i in zip(penalties, offsets_1based, strict=True):
        S_i = np.asarray(S_i, dtype=np.float64)
        frob = float(np.sqrt(np.sum(S_i * S_i)))
        if frob <= 0.0:
            continue
        start = int(off_i) - 1
        stop = start + int(S_i.shape[0])
        St[start:stop, start:stop] += S_i / frob

    evals, evecs = symmetric_eigh(St, descending=True)
    max_eval = float(np.max(evals)) if evals.size else 0.0
    pos_mask = evals > max_eval * (np.finfo(np.float64).eps ** 0.66)
    Y = evecs[:, pos_mask]
    Z = evecs[:, ~pos_mask]
    if Y.shape[1] == 0:
        E = np.empty((0, p), dtype=np.float64)
    else:
        E = np.sqrt(np.asarray(evals[pos_mask], dtype=np.float64))[:, np.newaxis] * Y.T
    return Y, Z, E


def _build_log_smoothing_parameter_map(
    model,
    X_full: np.ndarray,
    penalties: list[np.ndarray],
    offsets_1based: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray, np.ndarray]:
    penalty_blocks = list(_penalty_blocks_seq(model))
    n_pen = len(penalty_blocks)
    n_sp = int(_n_smoothing_params(model) or 0)

    if n_pen == 0:
        return (
            None,
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
        )

    full_L = np.zeros((n_pen, n_sp), dtype=np.float64)
    for i, pb in enumerate(penalty_blocks):
        full_L[i, int(pb.smoothing_index)] = 1.0

    sp_raw = getattr(model, "smoothing_params", None)
    if sp_raw is None:
        sp_all = np.ones(n_sp, dtype=np.float64)
    else:
        sp_all = np.asarray(sp_raw, dtype=np.float64).ravel()
    if sp_all.shape != (n_sp,):
        raise ValueError(
            f"Expected smoothing_params with shape ({n_sp},), got {sp_all.shape}."
        )
    fixed_mask = (
        np.zeros(n_sp, dtype=bool)
        if getattr(model, "smoothing_fixed_mask_", None) is None
        else np.asarray(model.smoothing_fixed_mask_, dtype=bool)
    )
    if fixed_mask.shape != (n_sp,):
        raise ValueError(f"Expected smoothing_fixed_mask_ with shape ({n_sp},).")

    # Mirror mgcv/R/mgcv.r: free smoothing parameters remain encoded as -1 in
    # G$sp at setup time; only fixed smoothing parameters retain their numeric
    # values before being folded into lsp0.
    sp_setup = np.full(n_sp, -1.0, dtype=np.float64)
    sp_setup[fixed_mask] = sp_all[fixed_mask]

    if np.any(fixed_mask):
        fixed_vals = np.asarray(sp_setup[fixed_mask], dtype=np.float64).copy()
        zero_mask = fixed_vals == 0.0
        ef0 = np.flatnonzero(zero_mask).astype(np.float64) + 1.0
        if np.any(zero_mask):
            # Mirror the exact mgcv loop in mgcv/R/mgcv.r when constructing
            # effective-zero fixed smoothing parameters for lsp0.
            for i in range(int(np.sum(zero_mask))):
                start = int(offsets_1based[i]) - 1
                stop = start + int(penalties[i].shape[0])
                x_norm = float(np.linalg.norm(X_full[:, start:stop], ord="fro"))
                s_norm = float(np.linalg.norm(penalties[i], ord="fro"))
                ef0[i] = (
                    (x_norm * x_norm / s_norm) * np.finfo(np.float64).eps * 0.1
                    if s_norm > 0.0
                    else 0.0
                )
        fixed_vals[~zero_mask] = np.log(fixed_vals[~zero_mask])
        fixed_vals[zero_mask] = np.log(ef0)
        lsp0 = np.asarray(full_L[:, fixed_mask] @ fixed_vals, dtype=np.float64)
        L_free = np.asarray(full_L[:, ~fixed_mask], dtype=np.float64)
        sp_free = np.asarray(sp_setup[~fixed_mask], dtype=np.float64)
    else:
        lsp0 = np.zeros(n_pen, dtype=np.float64)
        L_free = np.asarray(full_L, dtype=np.float64)
        sp_free = np.asarray(sp_setup, dtype=np.float64)

    if L_free.shape[0] == L_free.shape[1] and np.array_equal(
        L_free, np.eye(L_free.shape[0], dtype=np.float64)
    ):
        L_out = None
        if sp_free.size > 0:
            with np.errstate(invalid="ignore"):
                log_sp_full = np.log(np.asarray(sp_free, dtype=np.float64)) + lsp0
        else:
            log_sp_full = lsp0.copy()
    else:
        L_out = L_free
        if sp_free.size > 0:
            with np.errstate(invalid="ignore"):
                log_sp_full = (
                    np.asarray(L_free @ np.log(sp_free), dtype=np.float64) + lsp0
                )
        else:
            log_sp_full = lsp0.copy()

    return L_out, lsp0, sp_free, log_sp_full


def build_estimate_gam_setup_state(
    model, *, tol: float = 1e-10
) -> PreOptimizationSetupState:
    """
    Reconstruct mgcv's exact pre-optimization block state from the compiled model.

    This mirrors the setup in ``mgcv/R/mgcv.r::estimate.gam`` using the Python
    compiled penalties in their original (ungrouped) order.
    """

    compiled = _compiled_model(model)
    if compiled is None:
        raise RuntimeError("Compiled model is unavailable.")

    X_full = _full_design_matrix(model)
    p_full = int(X_full.shape[1])
    penalty_blocks = list(_penalty_blocks_seq(model))

    penalties = [np.asarray(pb.matrix, dtype=np.float64) for pb in penalty_blocks]
    offsets = np.asarray(
        [
            int(_full_coef_indices(model, pb.coef_slice)[0]) + 1
            for pb in penalty_blocks
        ],
        dtype=np.int64,
    )
    ranks = np.asarray(
        [
            (
                int(pb.rank)
                if getattr(pb, "rank", None) is not None
                else numerical_rank(
                    np.asarray(pb.matrix, dtype=np.float64), hermitian=True
                )
            )
            for pb in penalty_blocks
        ],
        dtype=np.int64,
    )

    L, lsp0, sp_free, log_sp_full = _build_log_smoothing_parameter_map(
        model,
        X_full,
        penalties,
        offsets,
    )

    H = getattr(model, "H", None)
    H_full = None
    if H is not None:
        H_arr = np.asarray(H, dtype=np.float64)
        H_full = (
            H_arr.copy()
            if H_arr.shape == (p_full, p_full)
            else _lift_reduced_matrix_to_full(model, H_arr, p_full)
        )

    roots = []
    for S_i, off_i, rank_i in zip(penalties, offsets, ranks, strict=True):
        root_local = _mroot_chol(S_i, rank=int(rank_i))
        root_full = np.zeros((p_full, root_local.shape[1]), dtype=np.float64)
        start = int(off_i) - 1
        stop = start + int(root_local.shape[0])
        root_full[start:stop, :] = root_local
        roots.append(root_full)

    Y, Z, E = _total_penalty_space_from_blocks(
        penalties,
        offsets,
        p_full,
        H=H_full,
    )
    U1 = np.column_stack([Y, Z]) if p_full > 0 else np.empty((0, 0), dtype=np.float64)
    UrS = [np.asarray(Y.T @ root, dtype=np.float64) for root in roots]
    if H_full is not None:
        UrS.append(np.asarray(Y.T @ _mroot_chol(H_full), dtype=np.float64))

    return PreOptimizationSetupState(
        X=np.asarray(X_full, dtype=np.float64),
        offset=(
            np.zeros(X_full.shape[0], dtype=np.float64)
            if getattr(model, "offset_train_", None) is None
            else np.asarray(model.offset_train_, dtype=np.float64).copy()
        ),
        S=penalties,
        off=offsets,
        rank=ranks,
        L=None if L is None else np.asarray(L, dtype=np.float64),
        lsp0=np.asarray(lsp0, dtype=np.float64),
        sp=np.asarray(sp_free, dtype=np.float64),
        log_sp_full=np.asarray(log_sp_full, dtype=np.float64),
        rS=roots,
        Y=np.asarray(Y, dtype=np.float64),
        Z=np.asarray(Z, dtype=np.float64),
        E=np.asarray(E, dtype=np.float64),
        Eb=np.asarray(E, dtype=np.float64),
        U1=np.asarray(U1, dtype=np.float64),
        UrS=UrS,
        Mp=int(Z.shape[1]),
    )


def _group_exact_setup_roots_by_smoothing_parameter(
    model, setup: PreOptimizationSetupState
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    penalty_blocks = list(_penalty_blocks_seq(model))
    n_sp = int(
        _n_smoothing_params(model)
        or (
            max(
                (int(getattr(pb, "smoothing_index", -1)) for pb in penalty_blocks),
                default=-1,
            )
            + 1
        )
    )
    p_full = int(np.asarray(setup.X, dtype=np.float64).shape[1])
    q = int(np.asarray(setup.Y, dtype=np.float64).shape[1])

    root_parts: list[list[np.ndarray]] = [[] for _ in range(n_sp)]
    range_parts: list[list[np.ndarray]] = [[] for _ in range(n_sp)]
    for pb, root_full, root_range in zip(
        penalty_blocks,
        list(setup.rS),
        list(setup.UrS[: len(penalty_blocks)]),
        strict=True,
    ):
        sp_idx = int(pb.smoothing_index)
        root_parts[sp_idx].append(np.asarray(root_full, dtype=np.float64))
        range_parts[sp_idx].append(np.asarray(root_range, dtype=np.float64))

    roots = [np.empty((p_full, 0), dtype=np.float64) for _ in range(n_sp)]
    range_roots = [np.empty((q, 0), dtype=np.float64) for _ in range(n_sp)]
    S_groups = [np.zeros((q, q), dtype=np.float64) for _ in range(n_sp)]

    for sp_idx in range(n_sp):
        if root_parts[sp_idx]:
            roots[sp_idx] = np.concatenate(root_parts[sp_idx], axis=1)
        if range_parts[sp_idx]:
            range_roots[sp_idx] = np.concatenate(range_parts[sp_idx], axis=1)
            S_groups[sp_idx] = range_roots[sp_idx] @ range_roots[sp_idx].T

    range_roots_with_fixed = list(range_roots)
    for root_range in list(setup.UrS[len(penalty_blocks) :]):
        range_roots_with_fixed.append(np.asarray(root_range, dtype=np.float64))

    return roots, range_roots, range_roots_with_fixed, S_groups


def _grouped_penalties(model) -> list[_GroupedPenalty]:
    term_blocks = list(_term_blocks_seq(model))
    penalty_blocks = list(_penalty_blocks_seq(model))
    p = int(_n_coef(model) or sum(int(tb.basis_train.shape[1]) for tb in term_blocks))
    n_sp = int(
        _n_smoothing_params(model)
        or (max((int(pb.smoothing_index) for pb in penalty_blocks), default=-1) + 1)
    )
    if p == 0 or n_sp == 0:
        return []

    grouped: dict[int, dict[str, Any]] = {}
    for pb in penalty_blocks:
        k = int(pb.smoothing_index)
        entry = grouped.get(k)
        if entry is None:
            entry = {
                "matrix_full": np.zeros((p, p), dtype=np.float64),
                "kind": str(getattr(pb, "kind", "smooth")),
                "is_null_space_penalty": bool(
                    getattr(pb, "is_null_space_penalty", False)
                ),
                "term_indices": set(),
            }
            grouped[k] = entry
        sl = pb.coef_slice
        entry["matrix_full"][sl, sl] += np.asarray(pb.matrix, dtype=np.float64)
        term_index = int(getattr(pb, "term_index", -1))
        if term_index < 0:
            for i, tb in enumerate(term_blocks):
                if tb.coef_slice == sl:
                    term_index = i
                    break
        entry["term_indices"].add(term_index)

    out = []
    for k in sorted(grouped):
        entry = grouped[k]
        out.append(
            _GroupedPenalty(
                smoothing_index=int(k),
                matrix_full=0.5 * (entry["matrix_full"] + entry["matrix_full"].T),
                kind=str(entry["kind"]),
                is_null_space_penalty=bool(entry["is_null_space_penalty"]),
                term_indices=tuple(sorted(entry["term_indices"])),
            )
        )
    return out


def _total_penalty_space(grouped_penalties, p, *, H=None):
    if H is not None:
        H = np.asarray(H, dtype=np.float64)
        Hscale = float(np.sqrt(np.sum(H * H)))
        if Hscale <= 0.0:
            H = None
    if H is None:
        St = np.zeros((p, p), dtype=np.float64)
    else:
        if H.shape != (p, p):
            raise ValueError("H has wrong dimension.")
        St = H / float(np.sqrt(np.sum(H * H)))

    for grp in grouped_penalties:
        Sg = np.asarray(grp.matrix_full, dtype=np.float64)
        frob = float(np.sqrt(np.sum(Sg * Sg)))
        if frob > 0.0:
            St += Sg / frob

    evals, evecs = symmetric_eigh(St, descending=True)
    scale = float(np.max(evals)) if evals.size else 0.0
    pos_mask = evals > scale * (np.finfo(np.float64).eps ** 0.66)
    Y = evecs[:, pos_mask]
    Z = evecs[:, ~pos_mask]
    if Y.shape[1] == 0:
        E = np.empty((0, p), dtype=np.float64)
    else:
        E = np.sqrt(np.asarray(evals[pos_mask], dtype=np.float64))[:, np.newaxis] * Y.T
    return Y, Z, E


def gam_reparam(range_roots, lsp, deriv=2):
    """
    Python port of ``mgcv/R/gam.fit3.r::gam.reparam`` using canonical range roots.

    ``range_roots[i]`` corresponds to ``UrS[[i]]`` in the upstream fit setup.
    The smoothing input is on the log scale, matching upstream ``lsp``.
    """
    lsp = np.asarray(lsp, dtype=np.float64).ravel()
    M = int(lsp.size)
    with np.errstate(over="raise", invalid="raise"):
        sp = np.exp(lsp)
    roots = [np.asarray(r, dtype=np.float64).copy() for r in range_roots]
    q = 0 if not roots else int(roots[0].shape[0])
    Mf = len(roots)
    fixed_penalty = Mf > M
    if q == 0 or Mf == 0:
        return {
            "S": np.empty((q, q), dtype=np.float64),
            "Qs": np.eye(q, dtype=np.float64),
            "rS": roots,
            "E": np.empty((q, q), dtype=np.float64),
            "det": 0.0,
            "det1": np.zeros(M, dtype=np.float64),
            "det2": np.zeros((M, M), dtype=np.float64),
            "fixed_penalty": fixed_penalty,
        }

    for root in roots:
        if int(root.shape[0]) != q:
            raise ValueError("All range penalty roots must have the same row count.")
    spf = np.ones(Mf, dtype=np.float64)
    spf[:M] = sp
    Si = [r @ r.T for r in roots]
    d_tol = float(np.finfo(np.float64).eps ** 0.3)
    r_tol = float(np.finfo(np.float64).eps ** 0.75)

    S_out = np.zeros((q, q), dtype=np.float64)
    Qf = np.eye(q, dtype=np.float64)
    gamma = np.ones(Mf, dtype=bool)
    K = 0
    Q = q
    iteration = 0
    Si_active = [A.copy() for A in Si]
    rS_work = [r.copy() for r in roots]

    while True:
        iteration += 1
        frob = np.array(
            [
                float(np.linalg.norm(Si_active[i], ord="fro")) if gamma[i] else 0.0
                for i in range(Mf)
            ],
            dtype=np.float64,
        )
        max_frob = max([float(frob[i] * spf[i]) for i in range(Mf) if gamma[i]] + [0.0])
        if not np.isfinite(max_frob) or max_frob <= 0.0:
            break

        alpha = np.zeros(Mf, dtype=bool)
        gamma1 = np.zeros(Mf, dtype=bool)
        for i in range(Mf):
            if not gamma[i]:
                continue
            if float(frob[i] * spf[i]) > max_frob * d_tol:
                alpha[i] = True
            else:
                gamma1[i] = True

        if np.any(gamma1):
            Sb = np.zeros((Q, Q), dtype=np.float64)
            for i, A in enumerate(Si_active):
                if alpha[i]:
                    Sb += A / float(frob[i])
            Sb = 0.5 * (Sb + Sb.T)
            ev = symmetric_eigvalsh(Sb)
            if ev.size == 0 or ev[-1] <= 0.0:
                r = 0
            else:
                r = 1
                while r < Q and ev[Q - r - 1] > ev[Q - 1] * r_tol:
                    r += 1
        else:
            r = Q

        if Q == r:
            if iteration == 1:
                S_out.fill(0.0)
                for i, A in enumerate(Si_active):
                    S_out += float(spf[i]) * A
                Qf = np.eye(q, dtype=np.float64)
            break

        Sb = np.zeros((Q, Q), dtype=np.float64)
        Sg = np.zeros((Q, Q), dtype=np.float64)
        for i, A in enumerate(Si_active):
            if alpha[i]:
                Sb += float(spf[i]) * A
            elif gamma1[i]:
                Sg += float(spf[i]) * A

        Sb = 0.5 * (Sb + Sb.T)
        evals, U = symmetric_eigh(Sb, descending=True)

        if iteration == 1:
            Qf[:, :Q] = U
        else:
            Qf[:, K : K + Q] = Qf[:, K : K + Q] @ U

        if K > 0:
            B = S_out[:K, K : K + Q] @ U
            S_out[:K, K : K + Q] = B
            S_out[K : K + Q, :K] = B.T

        C = U.T @ Sg @ U
        if r > 0:
            C[np.arange(r), np.arange(r)] += evals[:r]
        S_out[K : K + Q, K : K + Q] = C

        for k in range(Mf):
            root = rS_work[k]
            cols = int(root.shape[1])
            if cols == 0:
                continue
            work = np.asarray(root[K : K + Q, :], dtype=np.float64)
            if alpha[k]:
                root[K : K + r, :] = U[:, :r].T @ work
                if Q > r:
                    root[K + r : K + Q, :] = 0.0
            elif gamma1[k]:
                root[K : K + Q, :] = U.T @ work

        Un = np.asarray(U[:, r:], dtype=np.float64)
        Si_active = [Un.T @ A @ Un if gamma1[i] else A for i, A in enumerate(Si_active)]
        K += r
        Q -= r
        gamma = gamma1

    S_det = np.array(S_out, dtype=np.float64, copy=True)
    try:
        Q_qr, R_qr, piv = scipy_qr(
            S_det, pivoting=True, mode="full", check_finite=False
        )
        diag_R = np.diag(R_qr)
        if np.any(diag_R == 0.0) or np.any(~np.isfinite(diag_R)):
            logdet = np.inf
            S_inv = np.full_like(S_det, np.nan)
        else:
            logdet = float(np.sum(np.log(np.abs(diag_R))))
            S_inv_piv = solve_triangular(R_qr, Q_qr.T, lower=False, check_finite=False)
            S_inv = np.empty_like(S_inv_piv)
            S_inv[piv, :] = S_inv_piv
    except np.linalg.LinAlgError:
        logdet = np.inf
        S_inv = np.full_like(S_det, np.nan)

    S_out = 0.5 * (S_out + S_out.T)

    p = np.sqrt(np.abs(np.diag(S_out)))
    p[p == 0.0] = 1.0
    St = (S_out / p[:, np.newaxis]) / p[np.newaxis, :]
    St = 0.5 * (St + St.T)
    E_root = _mroot_chol(St, rank=q)
    E = (
        E_root.T * p[np.newaxis, :]
        if E_root.size
        else np.empty((0, q), dtype=np.float64)
    )

    det1 = np.zeros(M, dtype=np.float64)
    det2 = np.zeros((M, M), dtype=np.float64)
    if deriv > 0 and np.all(np.isfinite(S_inv)):
        for i, rS_i in enumerate(rS_work[:M]):
            det1[i] = float(sp[i] * np.trace(S_inv @ (rS_i @ rS_i.T)))
    if deriv > 1 and np.all(np.isfinite(S_inv)):
        Si_trans = [S_inv @ (rS_i @ rS_i.T) for rS_i in rS_work[:M]]
        for i in range(M):
            for j in range(i, M):
                val = -float(sp[i] * sp[j] * np.trace(Si_trans[i] @ Si_trans[j]))
                if i == j:
                    val += det1[i]
                det2[i, j] = det2[j, i] = val

    return {
        "S": S_out,
        "Qs": Qf,
        "rS": rS_work,
        "E": E,
        "det": float(logdet),
        "det1": det1,
        "det2": det2,
        "fixed_penalty": fixed_penalty,
    }


def _canonical_penalty_space(model, *, tol=1e-10) -> dict[str, Any]:
    cache = getattr(model, "_penalty_subspace_cache_", None)
    if cache is not None:
        return cast(dict[str, Any], cache)

    setup = build_estimate_gam_setup_state(model, tol=tol)
    penalty_blocks = list(_penalty_blocks_seq(model))
    grouped = _grouped_penalties(model)
    n_pen = len(penalty_blocks)
    roots = [np.asarray(root, dtype=np.float64) for root in list(setup.rS)]
    range_roots = [
        np.asarray(root, dtype=np.float64) for root in list(setup.UrS[:n_pen])
    ]
    range_roots_with_fixed = [
        np.asarray(root, dtype=np.float64) for root in list(setup.UrS)
    ]
    S_groups = [root @ root.T for root in range_roots]

    cache = {
        "estimate_setup": setup,
        "Y": np.asarray(setup.Y, dtype=np.float64),
        "Z": np.asarray(setup.Z, dtype=np.float64),
        "E": np.asarray(setup.Eb, dtype=np.float64),
        "Mp": int(setup.Mp),
        "grouped_penalties": grouped,
        "roots": roots,
        "range_roots": range_roots,
        "range_roots_with_fixed": range_roots_with_fixed,
        "S_groups": S_groups,
    }
    model._penalty_subspace_cache_ = cache
    return cache


def _penalty_log_smoothing_map(
    model,
    setup: PreOptimizationSetupState,
    sp: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """
    Return mgcv-style per-penalty log smoothing parameters.

    ``mgcv/R/mgcv.r::estimate.gam`` passes ``L %*% log(sp) + lsp0`` to
    ``gam.reparam()``, where rows are in the same order as ``G$S``/``G$UrS``.
    The public Python model stores the full smoothing vector, so this helper
    also returns the full one-hot penalty-to-smoothing map used to lift
    per-penalty derivatives back to the public smoothing-parameter indexing.
    """

    sp = np.asarray(sp, dtype=np.float64).ravel()
    penalty_blocks = list(_penalty_blocks_seq(model))
    n_pen = len(penalty_blocks)
    n_sp = int(_n_smoothing_params(model) or sp.size)
    if sp.shape != (n_sp,):
        raise ValueError(f"Expected smoothing parameter vector of shape ({n_sp},).")
    if len(setup.UrS) < n_pen:
        raise RuntimeError("estimate.gam setup has fewer UrS roots than penalties.")

    L_full = np.zeros((n_pen, n_sp), dtype=np.float64)
    for i, pb in enumerate(penalty_blocks):
        sp_idx = int(pb.smoothing_index)
        if sp_idx < 0 or sp_idx >= n_sp:
            raise ValueError(f"Invalid smoothing parameter index {sp_idx}.")
        L_full[i, sp_idx] = 1.0

    fixed_mask = (
        np.zeros(n_sp, dtype=bool)
        if getattr(model, "smoothing_fixed_mask_", None) is None
        else np.asarray(model.smoothing_fixed_mask_, dtype=bool)
    )
    if fixed_mask.shape != (n_sp,):
        raise ValueError(f"Expected smoothing_fixed_mask_ with shape ({n_sp},).")

    with np.errstate(divide="raise", invalid="raise"):
        log_sp = np.log(sp)
    free_log_sp = np.asarray(log_sp[~fixed_mask], dtype=np.float64)
    lsp0 = np.asarray(setup.lsp0, dtype=np.float64).ravel()
    if lsp0.shape != (n_pen,):
        raise RuntimeError("estimate.gam lsp0 length does not match penalty count.")
    if setup.L is None:
        if free_log_sp.shape != (n_pen,):
            raise RuntimeError(
                "Identity smoothing-parameter map is inconsistent with "
                "the number of penalty blocks."
            )
        lsp = free_log_sp + lsp0
    else:
        L = np.asarray(setup.L, dtype=np.float64)
        if L.shape != (n_pen, free_log_sp.size):
            raise RuntimeError("estimate.gam L shape is inconsistent with sp.")
        lsp = np.asarray(L @ free_log_sp + lsp0, dtype=np.float64)

    roots = [
        np.asarray(root, dtype=np.float64)
        for root in list(setup.UrS[:n_pen]) + list(setup.UrS[n_pen:])
    ]
    return roots, np.asarray(lsp, dtype=np.float64), L_full


def _static_penalty_space(model, *, tol=1e-10):
    return _canonical_penalty_space(model, tol=tol)


def _static_penalty_null_dim(model, *, tol=1e-10):
    cache = _canonical_penalty_space(model, tol=tol)
    if "estimate_setup" in cache:
        return max(int(cache["Mp"]) - _coef_column_offset(model), 0)
    return max(
        int(np.asarray(cache["Z"], dtype=np.float64).shape[1])
        - _coef_column_offset(model),
        0,
    )


def build_penalty_reparameterization_state(
    model, X_full, sp, *, deriv=0, tol=1e-10
) -> PenaltyReparameterizationState:
    """Mirror mgcv's `gam.fit3/gam.fit4` reparameterization on `estimate.gam` setup."""
    X_full = np.asarray(X_full, dtype=np.float64)
    sp = np.asarray(sp, dtype=np.float64).ravel()

    cache = _canonical_penalty_space(model, tol=tol)
    setup = cache["estimate_setup"]
    Y = np.asarray(setup.Y, dtype=np.float64)
    Z = np.asarray(setup.Z, dtype=np.float64)
    UrS, lsp, _L_full = _penalty_log_smoothing_map(model, setup, sp)
    rp = gam_reparam(UrS, lsp, deriv=deriv)
    _roots_grouped, UrS_grouped, UrS_grouped_with_fixed, _S_groups = (
        _group_exact_setup_roots_by_smoothing_parameter(model, setup)
    )
    UrS_public = (
        UrS_grouped_with_fixed
        if len(UrS_grouped_with_fixed) != len(UrS_grouped)
        else UrS_grouped
    )

    q_range = int(Y.shape[1])
    q_full = int(X_full.shape[1])
    if q_full != int(np.asarray(setup.U1, dtype=np.float64).shape[0]):
        raise ValueError(
            "Full design width does not match the compiled estimate.gam setup."
        )
    Mp = int(setup.Mp)
    U1 = np.asarray(setup.U1, dtype=np.float64)

    T_small = np.eye(q_full, dtype=np.float64)
    if q_range > 0:
        T_small[:q_range, :q_range] = np.asarray(rp["Qs"], dtype=np.float64)
    T = U1 @ T_small

    St = np.zeros((q_full, q_full), dtype=np.float64)
    if q_range > 0:
        St[:q_range, :q_range] = np.asarray(rp["S"], dtype=np.float64)

    Sr = np.zeros((q_range, q_full), dtype=np.float64)
    if q_range > 0:
        Sr[:, :q_range] = np.asarray(rp["E"], dtype=np.float64)

    Eb = np.asarray(setup.Eb, dtype=np.float64) @ T

    X_trans = X_full @ T
    X_range = np.asarray(X_trans[:, :q_range], dtype=np.float64)
    X_fix = np.asarray(X_trans[:, q_range:], dtype=np.float64)
    if q_range == 0:
        Z_rand = np.empty((X_full.shape[0], 0), dtype=np.float64)
    else:
        E = np.asarray(rp["E"], dtype=np.float64)
        try:
            Z_rand = np.linalg.solve(E, X_range.T).T
        except np.linalg.LinAlgError:
            Z_rand = np.full_like(X_range, np.nan)

    return PenaltyReparameterizationState(
        Y=Y,
        Z=Z,
        U1=U1,
        UrS=UrS_public,
        rp=rp,
        T=T,
        St=St,
        Sr=Sr,
        Eb=Eb,
        Mp=Mp,
        X_range=X_range,
        X_fix=X_fix,
        Z_rand=Z_rand,
    )


def _static_fixed_and_random_designs(model, X_full, sp, *, tol=1e-10):
    X_full = np.asarray(X_full, dtype=np.float64)
    sp = np.asarray(sp, dtype=np.float64)

    state = build_penalty_reparameterization_state(model, X_full, sp, deriv=0, tol=tol)
    Xf = np.asarray(state.X_fix, dtype=np.float64)
    Zr = np.asarray(state.Z_rand, dtype=np.float64)
    logdet_plus = float(state.rp["det"])

    return (
        Xf,
        Zr,
        {
            "rank": int(np.asarray(state.Y, dtype=np.float64).shape[1]),
            "null_dim": max(int(state.Mp) - _coef_column_offset(model), 0),
            "logdet_plus": logdet_plus,
        },
    )


def dynamic_reparam_design(model, X_full, sp, *, tol=1e-10) -> DynamicReparamDesign:
    Xf, Zr, split = _static_fixed_and_random_designs(model, X_full, sp, tol=tol)
    return DynamicReparamDesign(
        X_fix=np.asarray(Xf, dtype=np.float64),
        Z_rand=np.asarray(Zr, dtype=np.float64),
        ZtZ_rand=np.asarray(Zr, dtype=np.float64).T @ np.asarray(Zr, dtype=np.float64),
        penalty_logdet=float(split["logdet_plus"]),
        null_dim=int(split["null_dim"]),
    )


def _stable_penalty_logdet(model, sp, *, tol=1e-10):
    logdet, _, _ = _stable_penalty_logdet_derivatives(model, sp, tol=tol, order=0)
    return float(logdet)


def _stable_penalty_logdet_derivatives(model, sp, *, tol=1e-10, order=2):
    sp = np.asarray(sp, dtype=np.float64).ravel()
    n_sp = int(_n_smoothing_params(model) or sp.size)
    grad = np.zeros(n_sp, dtype=np.float64)
    hess = np.zeros((n_sp, n_sp), dtype=np.float64)

    cache = _canonical_penalty_space(model, tol=tol)
    Y = np.asarray(cache["Y"], dtype=np.float64)
    if Y.shape[1] == 0:
        return 0.0, grad, hess

    setup = cache["estimate_setup"]
    UrS, lsp, L_full = _penalty_log_smoothing_map(model, setup, sp)
    rp = gam_reparam(UrS, lsp, deriv=min(int(order), 2))
    logdet = float(rp["det"])
    if not np.isfinite(logdet):
        return np.inf, np.full(n_sp, np.nan), np.full((n_sp, n_sp), np.nan)
    if order <= 0:
        return logdet, grad, hess

    det1 = np.asarray(rp["det1"], dtype=np.float64)
    if det1.shape != (L_full.shape[0],):
        raise RuntimeError("gam.reparam determinant gradient length mismatch.")
    grad = np.asarray(L_full.T @ det1, dtype=np.float64)
    if order <= 1:
        return logdet, grad, hess

    det2 = np.asarray(rp["det2"], dtype=np.float64)
    if det2.shape != (L_full.shape[0], L_full.shape[0]):
        raise RuntimeError("gam.reparam determinant Hessian shape mismatch.")
    hess = np.asarray(L_full.T @ det2 @ L_full, dtype=np.float64)
    return logdet, grad, hess


def can_use_simple_ml_reml_structure(model):
    """
    Conservative structural gate for the current ML/REML paths.

    It is enabled when every penalized smooth term can be decomposed into
    connected penalty-support components without cross-component null-space
    couplings. This mirrors the local structural requirement used when
    assembling the ``UrS``-like blocks for the Laplace ML/REML path.
    """
    if _compiled_model(model) is None:
        return False

    for pb in _penalty_blocks_seq(model):
        width = int(pb.coef_slice.stop - pb.coef_slice.start)
        P = np.asarray(pb.matrix, dtype=np.float64)
        if P.shape != (width, width):
            return False
        if str(getattr(pb, "kind", "smooth")) not in {
            "smooth",
            "random_effect",
            "null_space",
        }:
            return False

    return True


def can_use_exact_gaussian_ml_reml(model):
    return bool(uses_closed_form_solver(model)) and can_use_simple_ml_reml_structure(
        model
    )


def build_penalty_reparameterized_system(model):
    if _compiled_model(model) is None:
        return _assign_reparam_state(model, None)

    fix_blocks = []
    cache = _canonical_penalty_space(model)
    setup = cache["estimate_setup"]
    grouped = list(cache["grouped_penalties"])
    penalty_blocks = list(_penalty_blocks_seq(model))
    term_blocks = list(_term_blocks_seq(model))
    X_full = np.asarray(setup.X, dtype=np.float64)
    X_pen = np.asarray(_design_matrix(model), dtype=np.float64)
    p = int(X_pen.shape[1])
    Y = np.asarray(setup.Y, dtype=np.float64)
    Z = np.asarray(setup.Z, dtype=np.float64)

    if Z.shape[1] > 0:
        X_null = X_full @ Z
        if X_null.shape[1] > 0:
            fix_blocks.append(X_null)

    rand_blocks = []
    sl_blocks: list[SlBlock] = []
    rand_start = 0

    if Y.shape[1] > 0 and penalty_blocks:
        B_range = X_full @ Y
        range_roots = list(cache["range_roots"])
        range_penalties = [
            (
                (Ur @ Ur.T)
                if Ur.size
                else np.zeros((Y.shape[1], Y.shape[1]), dtype=np.float64)
            )
            for Ur in range_roots
        ]

        P_sum = np.zeros((Y.shape[1], Y.shape[1]), dtype=np.float64)
        for Pk in range_penalties:
            P_sum += np.asarray(Pk, dtype=np.float64)

        _B0, Zr_main, meta = reparameterize_smooth(B_range, P_sum)
        U_range = np.asarray(meta["U1"], dtype=np.float64)

        for pb, Pk in zip(penalty_blocks, range_penalties, strict=True):
            Pk = np.asarray(Pk, dtype=np.float64)
            if U_range.shape[1] == 0 or not np.any(Pk):
                continue
            P_proj = U_range.T @ (np.asarray(Pk, dtype=np.float64) @ U_range)
            P_proj = 0.5 * (P_proj + P_proj.T)
            norm_val = float(np.linalg.norm(P_proj, ord=2))
            if norm_val <= 0.0:
                continue
            R = _matrix_sqrt_psd(P_proj / norm_val)
            if R.size == 0:
                continue
            keep = np.linalg.norm(R, axis=0) > 1e-14
            if not np.any(keep):
                continue
            R = R[:, keep]
            Z_block = np.asarray(Zr_main @ R, dtype=np.float64)
            if Z_block.shape[1] == 0:
                continue
            block_slice = slice(rand_start, rand_start + Z_block.shape[1])
            rand_blocks.append(Z_block)
            rand_start += Z_block.shape[1]
            term_index = int(getattr(pb, "term_index", -1))
            if term_index < 0:
                coef_slice = getattr(pb, "coef_slice", None)
                for i, tb in enumerate(term_blocks):
                    if getattr(tb, "coef_slice", None) == coef_slice:
                        term_index = i
                        break
            sl_blocks.append(
                SlBlock(
                    term_index=term_index,
                    repara=True,
                    smoothing_index=int(pb.smoothing_index),
                    start=int(block_slice.start),
                    stop=int(block_slice.stop),
                    ncol=int(p),
                    blockSize=int(Z_block.shape[1]),
                    lambda_scaling=float(norm_val),
                    kind=str(getattr(pb, "kind", "smooth")),
                    is_null_space_penalty=bool(
                        getattr(pb, "is_null_space_penalty", False)
                    ),
                )
            )
    if fix_blocks:
        X_fix_raw = np.column_stack(fix_blocks)
        _Q, R, piv = scipy_qr(X_fix_raw, pivoting=True)

        if R.size == 0:
            rank = 0
            keep_cols = np.array([], dtype=int)
        else:
            diag_R = np.abs(np.diag(R[: min(X_fix_raw.shape), :]))
            if diag_R.size == 0:
                rank = 0
                keep_cols = np.array([], dtype=int)
            else:
                rank_tol = (
                    max(X_fix_raw.shape) * np.finfo(float).eps * diag_R[0]
                    if diag_R[0] > 0
                    else 1e-12
                )
                rank = int(np.sum(diag_R > rank_tol))
                keep_cols = np.sort(piv[:rank])

        X_fix = (
            X_fix_raw[:, keep_cols]
            if rank > 0
            else np.empty((model.n_samples_, 0), dtype=np.float64)
        )
    else:
        X_fix = np.empty((model.n_samples_, 0), dtype=np.float64)

    if rand_blocks:
        Z_rand = np.column_stack(rand_blocks)
    else:
        Z_rand = np.empty((model.n_samples_, 0), dtype=np.float64)

    ZtZ_rand = Z_rand.T @ Z_rand
    state = ReparamState(
        X_fix=X_fix,
        Z_rand=Z_rand,
        ZtZ_rand=ZtZ_rand,
        sl_blocks=sl_blocks,
        penalty_range_basis=Y,
        penalty_null_basis=Z,
        penalty_range_roots=list(cache["range_roots"]),
        grouped_penalties=grouped,
    )
    return _assign_reparam_state(model, state)


def build_gaussian_reparameterized_system(model):
    return build_penalty_reparameterized_system(model)
