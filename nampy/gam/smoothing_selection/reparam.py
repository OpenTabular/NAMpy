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
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.linalg import qr as scipy_qr

from .._model_state import _fit_intercept


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
class CanonicalGamReparamState:
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


def _assign_reparam_state(model, state: Optional[ReparamState]) -> Optional[ReparamState]:
    model.reparam_state_ = state
    model.sl_blocks_ = None if state is None else list(state.sl_blocks or [])
    return state


def ensure_penalty_reparameterization_state(model) -> ReparamState:
    state = getattr(model, "reparam_state_", None)
    if state is None:
        state = model._build_penalty_reparameterized_system()
    if state is None:
        raise RuntimeError("Penalty reparameterization state is unavailable.")
    return state


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


def sl_penalty_rank_scaling_derivatives(
    state: ReparamState, n_smoothing_params: int
) -> tuple[np.ndarray, np.ndarray]:
    detS1 = np.zeros(int(n_smoothing_params), dtype=np.float64)
    detS2 = np.zeros((int(n_smoothing_params), int(n_smoothing_params)), dtype=np.float64)
    if not state.sl_blocks:
        return detS1, detS2
    for sl_block in iter_sl_random_blocks(state):
        j = -1 if sl_block.smoothing_index is None else int(sl_block.smoothing_index)
        n_pen = int(sl_block.blockSize)
        if 0 <= j < int(n_smoothing_params) and n_pen > 0:
            detS1[j] += float(n_pen)
    return detS1, detS2


def reparameterize_smooth(B, P, tol=1e-10):
    P_sym = 0.5 * (P + P.T)
    evals, U = np.linalg.eigh(P_sym)
    idx = np.argsort(evals)
    evals = evals[idx]
    U = U[:, idx]
    tol_eff = tol * max(1.0, np.max(np.abs(evals)))
    null_mask = evals <= tol_eff
    pos_mask = ~null_mask
    U0 = U[:, null_mask]
    U1 = U[:, pos_mask]
    d_pos = evals[pos_mask]
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
            "n_null": int(null_mask.sum()),
            "n_pen": int(pos_mask.sum()),
        },
    )


def _matrix_sqrt_psd(M, tol=1e-12):
    M = np.asarray(M, dtype=np.float64)
    if M.size == 0:
        return np.empty((0, 0), dtype=np.float64)
    M = 0.5 * (M + M.T)
    evals, vecs = np.linalg.eigh(M)
    evals = np.clip(evals, 0.0, None)
    sqrt_evals = np.sqrt(evals)
    return vecs @ np.diag(sqrt_evals)


def _positive_semidefinite_root(P, *, rank=None, tol=1e-10):
    P = np.asarray(P, dtype=np.float64)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("Penalty root requires a square matrix.")
    if P.shape[0] == 0:
        return np.empty((0, 0), dtype=np.float64)

    P_sym = 0.5 * (P + P.T)
    evals, U = np.linalg.eigh(P_sym)
    idx = np.argsort(evals)[::-1]
    evals = np.asarray(evals[idx], dtype=np.float64)
    U = np.asarray(U[:, idx], dtype=np.float64)
    tol_eff = float(tol) * max(1.0, float(np.max(np.abs(evals))))
    keep = np.flatnonzero(evals > tol_eff)
    if rank is not None and int(rank) >= 0:
        keep = keep[: min(int(rank), keep.size)]
    if keep.size == 0:
        return np.empty((P.shape[0], 0), dtype=np.float64)
    return U[:, keep] * np.sqrt(evals[keep])[np.newaxis, :]


def _grouped_penalties(model) -> list[_GroupedPenalty]:
    p = int(
        getattr(model, "n_coef_", 0)
        or sum(int(tb.basis_train.shape[1]) for tb in getattr(model, "term_blocks_", []) or [])
    )
    n_sp = int(
        getattr(model, "n_smoothing_params_", 0)
        or (
            max((int(pb.smoothing_index) for pb in getattr(model, "penalty_blocks_", []) or []), default=-1)
            + 1
        )
    )
    if p == 0 or n_sp == 0:
        return []

    grouped = {}
    term_blocks = list(getattr(model, "term_blocks_", []) or [])
    for pb in model.penalty_blocks_:
        k = int(pb.smoothing_index)
        entry = grouped.get(k)
        if entry is None:
            entry = {
                "matrix_full": np.zeros((p, p), dtype=np.float64),
                "kind": str(getattr(pb, "kind", "smooth")),
                "is_null_space_penalty": bool(getattr(pb, "is_null_space_penalty", False)),
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

    evals, evecs = np.linalg.eigh(0.5 * (St + St.T))
    idx = np.argsort(evals)
    evals = np.asarray(evals[idx], dtype=np.float64)
    evecs = np.asarray(evecs[:, idx], dtype=np.float64)
    scale = max(float(np.max(evals)) if evals.size else 0.0, 1.0)
    pos_mask = evals > scale * (np.finfo(np.float64).eps ** 0.66)
    Y = evecs[:, pos_mask]
    Z = evecs[:, ~pos_mask]
    if Y.shape[1] == 0:
        E = np.empty((0, p), dtype=np.float64)
    else:
        E = np.sqrt(np.asarray(evals[pos_mask], dtype=np.float64))[:, np.newaxis] * Y.T
    return Y, Z, E


def mini_roots(grouped_penalties, p, *, tol=1e-10):
    roots = []
    for grp in grouped_penalties:
        rank = int(np.linalg.matrix_rank(grp.matrix_full)) if grp.matrix_full.size else 0
        roots.append(_positive_semidefinite_root(grp.matrix_full, rank=rank, tol=tol))
    return roots


def gam_reparam(range_roots, sp, deriv=2):
    """
    Python port of `mgcv` `gam.reparam()` interface, using canonical range roots.

    `range_roots[i]` corresponds to `UrS[[i]]` in `mgcv/R/mgcv.r`.
    """
    sp = np.asarray(sp, dtype=np.float64).ravel()
    M = int(sp.size)
    roots = [np.asarray(r, dtype=np.float64).copy() for r in range_roots]
    q = 0 if not roots else int(roots[0].shape[0])
    fixed_penalty = len(roots) > M
    if q == 0 or M == 0:
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

    Mf = len(roots)
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
        max_frob = max(
            [float(frob[i] * spf[i]) for i in range(Mf) if gamma[i]] + [0.0]
        )
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
            ev = np.linalg.eigvalsh(Sb)
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
        evals, U = np.linalg.eigh(Sb)
        idx = np.argsort(evals)[::-1]
        evals = np.asarray(evals[idx], dtype=np.float64)
        U = np.asarray(U[:, idx], dtype=np.float64)

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

        for k in range(M):
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

    S_out = 0.5 * (S_out + S_out.T)
    sign, logdet = np.linalg.slogdet(S_out)
    if sign <= 0 or not np.isfinite(logdet):
        logdet = np.inf
    try:
        S_inv = np.linalg.inv(S_out)
    except np.linalg.LinAlgError:
        S_inv = np.full_like(S_out, np.nan)

    p = np.sqrt(np.abs(np.diag(S_out)))
    p[p == 0.0] = 1.0
    St = (S_out / p[:, np.newaxis]) / p[np.newaxis, :]
    St = 0.5 * (St + St.T)
    E_root = _positive_semidefinite_root(St, rank=q)
    E = E_root.T * p[np.newaxis, :] if E_root.size else np.empty((0, q), dtype=np.float64)

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
        return cache

    p_pen = int(
        getattr(model, "n_coef_", 0)
        or sum(int(tb.basis_train.shape[1]) for tb in getattr(model, "term_blocks_", []) or [])
    )
    n_sp = int(
        getattr(model, "n_smoothing_params_", 0)
        or (
            max(
                (
                    int(getattr(pb, "smoothing_index", -1))
                    for pb in getattr(model, "penalty_blocks_", []) or []
                ),
                default=-1,
            )
            + 1
        )
    )
    grouped = _grouped_penalties(model)
    H = getattr(model, "H", None)

    if p_pen == 0 or (not grouped and H is None):
        cache = {
            "Y": np.empty((p_pen, 0), dtype=np.float64),
            "Z": np.eye(p_pen, dtype=np.float64),
            "E": np.empty((0, p_pen), dtype=np.float64),
            "grouped_penalties": grouped,
            "roots": [np.empty((p_pen, 0), dtype=np.float64) for _ in range(n_sp)],
            "range_roots": [np.empty((0, 0), dtype=np.float64) for _ in range(n_sp)],
            "range_roots_with_fixed": [np.empty((0, 0), dtype=np.float64) for _ in range(n_sp)],
            "S_groups": [np.empty((0, 0), dtype=np.float64) for _ in range(n_sp)],
        }
        model._penalty_subspace_cache_ = cache
        return cache

    roots = [np.empty((p_pen, 0), dtype=np.float64) for _ in range(n_sp)]
    grouped_roots = mini_roots(grouped, p_pen, tol=tol)
    for grp, root in zip(grouped, grouped_roots):
        roots[int(grp.smoothing_index)] = np.asarray(root, dtype=np.float64)

    Y, Z, E = _total_penalty_space(grouped, p_pen, H=H)
    q = int(Y.shape[1])
    range_roots = [np.empty((q, 0), dtype=np.float64) for _ in range(n_sp)]
    S_groups = [np.zeros((q, q), dtype=np.float64) for _ in range(n_sp)]
    if q > 0:
        YT = Y.T
        for sp_idx, root in enumerate(roots):
            if root.shape[1] == 0:
                continue
            Ur = YT @ root
            range_roots[sp_idx] = Ur
            S_groups[sp_idx] = Ur @ Ur.T

    range_roots_with_fixed = list(range_roots)
    if H is not None:
        H_root = _positive_semidefinite_root(np.asarray(H, dtype=np.float64), tol=tol)
        range_roots_with_fixed = list(range_roots_with_fixed) + [Y.T @ H_root]

    cache = {
        "Y": np.asarray(Y, dtype=np.float64),
        "Z": np.asarray(Z, dtype=np.float64),
        "E": np.asarray(E, dtype=np.float64),
        "grouped_penalties": grouped,
        "roots": roots,
        "range_roots": range_roots,
        "range_roots_with_fixed": range_roots_with_fixed,
        "S_groups": S_groups,
    }
    model._penalty_subspace_cache_ = cache
    return cache


def _static_penalty_space(model, *, tol=1e-10):
    return _canonical_penalty_space(model, tol=tol)


def _static_penalty_null_dim(model, *, tol=1e-10):
    cache = _canonical_penalty_space(model, tol=tol)
    return int(np.asarray(cache["Z"], dtype=np.float64).shape[1])


def build_canonical_gam_reparam_state(
    model, X_full, sp, *, deriv=0, tol=1e-10
) -> CanonicalGamReparamState:
    """Mirror mgcv's `U1/UrS/gam.reparam/T/St/Sr/Eb/Mp` transform objects."""
    X_full = np.asarray(X_full, dtype=np.float64)
    sp = np.asarray(sp, dtype=np.float64).ravel()

    off = 1 if _fit_intercept(model) else 0
    if off > 0:
        X_pen = X_full[:, off:]
    else:
        X_pen = X_full

    cache = _canonical_penalty_space(model, tol=tol)
    Y = np.asarray(cache["Y"], dtype=np.float64)
    Z = np.asarray(cache["Z"], dtype=np.float64)
    UrS = [np.asarray(root, dtype=np.float64) for root in cache["range_roots_with_fixed"]]
    rp = gam_reparam(UrS, sp, deriv=deriv)

    q_range = int(Y.shape[1])
    q_null_pen = int(Z.shape[1])
    Mp = int(off + q_null_pen)
    q_full = int(off + X_pen.shape[1])

    U1 = np.zeros((q_full, q_full), dtype=np.float64)
    if q_range > 0:
        U1[off:, :q_range] = Y
    if off > 0:
        U1[:off, q_range : q_range + off] = np.eye(off, dtype=np.float64)
    if q_null_pen > 0:
        U1[off:, q_range + off :] = Z

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

    Eb0 = np.zeros((q_range, q_full), dtype=np.float64)
    if q_range > 0:
        Eb0[:, off:] = np.asarray(cache["E"], dtype=np.float64)
    Eb = Eb0 @ T

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

    return CanonicalGamReparamState(
        Y=Y,
        Z=Z,
        U1=U1,
        UrS=UrS,
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

    state = build_canonical_gam_reparam_state(model, X_full, sp, deriv=0, tol=tol)
    Xf = np.asarray(state.X_fix, dtype=np.float64)
    Zr = np.asarray(state.Z_rand, dtype=np.float64)
    logdet_plus = float(state.rp["det"])

    return (
        Xf,
        Zr,
        {
            "rank": int(np.asarray(state.Y, dtype=np.float64).shape[1]),
            "null_dim": int(np.asarray(state.Z, dtype=np.float64).shape[1]),
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
    n_sp = int(model.n_smoothing_params_ or sp.size)
    grad = np.zeros(n_sp, dtype=np.float64)
    hess = np.zeros((n_sp, n_sp), dtype=np.float64)

    cache = _canonical_penalty_space(model, tol=tol)
    Y = np.asarray(cache["Y"], dtype=np.float64)
    if Y.shape[1] == 0:
        return 0.0, grad, hess

    rp = gam_reparam(cache["range_roots_with_fixed"], sp, deriv=min(int(order), 2))
    logdet = float(rp["det"])
    if not np.isfinite(logdet):
        return np.inf, np.full(n_sp, np.nan), np.full((n_sp, n_sp), np.nan)
    if order <= 0:
        return logdet, grad, hess

    grad[: min(n_sp, rp["det1"].shape[0])] = np.asarray(rp["det1"], dtype=np.float64)[
        : min(n_sp, rp["det1"].shape[0])
    ]
    if order <= 1:
        return logdet, grad, hess

    m = min(n_sp, rp["det2"].shape[0])
    hess[:m, :m] = np.asarray(rp["det2"], dtype=np.float64)[:m, :m]
    return logdet, grad, hess


def can_use_simple_ml_reml_structure(model):
    """
    Conservative structural gate for the current ML/REML paths.

    It is enabled when every penalized smooth term can be decomposed into
    connected penalty-support components without cross-component null-space
    couplings. This mirrors the local structural requirement used when
    assembling the ``UrS``-like blocks for the Laplace ML/REML path.
    """
    if model.design_ is None:
        return False

    for pb in model.penalty_blocks_:
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
    return bool(model._uses_closed_form_solver()) and can_use_simple_ml_reml_structure(model)


def build_penalty_reparameterized_system(model):
    if model.design_ is None:
        return _assign_reparam_state(model, None)

    fix_blocks = []
    if _fit_intercept(model):
        fix_blocks.append(np.ones((model.n_samples_, 1), dtype=np.float64))

    cache = _canonical_penalty_space(model)
    grouped = list(cache["grouped_penalties"])
    X_pen = getattr(model, "Z", None)
    if X_pen is None:
        blocks = [
            np.asarray(tb.basis_train, dtype=np.float64)
            for tb in getattr(model, "term_blocks_", []) or []
            if int(np.asarray(tb.basis_train).shape[1]) > 0
        ]
        X_pen = (
            np.column_stack(blocks)
            if blocks
            else np.empty((model.n_samples_, 0), dtype=np.float64)
        )
    else:
        X_pen = np.asarray(X_pen, dtype=np.float64)
    p = int(X_pen.shape[1])
    Y = np.asarray(cache["Y"], dtype=np.float64)
    Z = np.asarray(cache["Z"], dtype=np.float64)

    if Z.shape[1] > 0:
        X_null = X_pen @ Z
        if X_null.shape[1] > 0:
            fix_blocks.append(X_null)

    rand_blocks = []
    sl_blocks: list[SlBlock] = []
    rand_start = 0

    if Y.shape[1] > 0 and grouped:
        B_range = X_pen @ Y
        range_roots = list(cache["range_roots"])
        range_penalties = [
            (Ur @ Ur.T) if Ur.size else np.zeros((Y.shape[1], Y.shape[1]), dtype=np.float64)
            for Ur in range_roots
        ]

        P_sum = np.zeros((Y.shape[1], Y.shape[1]), dtype=np.float64)
        for Pk in range_penalties:
            P_sum += np.asarray(Pk, dtype=np.float64)

        _B0, Zr_main, meta = reparameterize_smooth(B_range, P_sum)
        U_range = np.asarray(meta["U1"], dtype=np.float64)

        for grp, Pk in zip(grouped, range_penalties):
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
            sl_blocks.append(
                SlBlock(
                    term_index=int(grp.term_indices[0]) if grp.term_indices else -1,
                    repara=True,
                    smoothing_index=int(grp.smoothing_index),
                    start=int(block_slice.start),
                    stop=int(block_slice.stop),
                    ncol=int(p),
                    blockSize=int(Z_block.shape[1]),
                    lambda_scaling=float(norm_val),
                    kind=str(grp.kind),
                    is_null_space_penalty=bool(grp.is_null_space_penalty),
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
