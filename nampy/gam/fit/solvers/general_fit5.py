"""
General-family fixed-smoothing backend using mgcv-style ``gam.fit5``.

Mirrors mgcv ``gam.fit5`` / ``gam.fit5.post.proc`` from ``mgcv/R/gam.fit4.r``
for multi-linear-predictor GAMLSS-style families.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..._model_state import _penalty_blocks_seq, _predictor_designs, _term_blocks_seq
from ..model_ops import expand_smoothing_params_from_log
from ..state import FitCoreSolution
from .gam_fit5 import GamFit5Control, _sl_ldetS, gam_fit5, gam_fit5_post_proc


@dataclass
class _GeneralPredictorLayout:
    X_full: np.ndarray
    jj: list[np.ndarray]
    reduced_to_full_idx: np.ndarray
    predictor_full_slices: list[slice]


@dataclass
class GamFit5SlBlock:
    """
    Python materialization of one upstream ``Sl[[b]]`` block.

    ``start``/``stop`` mirror mgcv's 1-based inclusive indexing.
    """

    start: int
    stop: int
    S: list[np.ndarray]
    lambda_: np.ndarray
    repara: bool
    linear: bool
    rank: int | None = None
    ldet: float = 0.0
    ind: np.ndarray | None = None
    D: np.ndarray | None = None
    Di: np.ndarray | None = None
    rS: list[np.ndarray] = field(default_factory=list)
    St: np.ndarray | None = None

    @property
    def start0(self) -> int:
        return int(self.start) - 1

    @property
    def stop0(self) -> int:
        return int(self.stop)

    @property
    def width(self) -> int:
        return int(self.stop) - int(self.start) + 1


@dataclass
class GamFit5SlSetup:
    """
    Python materialization of the upstream ``Sl`` object from ``mgcv::Sl.setup``.
    """

    blocks: list[GamFit5SlBlock]
    E: np.ndarray
    S: np.ndarray
    lambda_: np.ndarray
    cholesky: bool

    def __len__(self) -> int:
        return len(self.blocks)

    def __iter__(self):
        return iter(self.blocks)

    def __getitem__(self, idx):
        return self.blocks[idx]


@dataclass
class GamFit5SetupState:
    """
    Exact setup owner for the general-family ``gam.fit5`` route.

    This centralizes the solver inputs that correspond to the upstream
    ``Sl.setup`` / ``ldetS`` preparation feeding ``mgcv::gam.fit5``:

    - stacked design and predictor index blocks (``x`` / ``attr(x, "lpi")``)
    - exact ``Sl`` block list plus ``attr(Sl, "E/S/lambda/cholesky")``
    - full penalty matrix and unscaled penalty blocks
    - penalty log-determinant terms and total penalty null dimension
    """

    layout: _GeneralPredictorLayout
    sl: GamFit5SlSetup
    X_full: np.ndarray
    X_initial: np.ndarray
    jj: list[np.ndarray]
    reduced_to_full_idx: np.ndarray
    predictor_full_slices: list[slice]
    offset_list: list[np.ndarray | None] | None
    smoothing_params: np.ndarray
    log_sp: np.ndarray
    St: np.ndarray
    S_blocks: list[np.ndarray]
    ldetS: float
    ldetS1: np.ndarray
    ldetS2: np.ndarray
    Mp: int
    score_type: str

    @property
    def Sl(self) -> GamFit5SlSetup:
        return self.sl


def _build_general_predictor_layout(model) -> _GeneralPredictorLayout:
    blocks = []
    jj: list[np.ndarray] = []
    predictor_full_slices: list[slice] = []
    reduced_to_full: list[int] = []
    full_start = 0

    for pred in _predictor_designs(model):
        Z = np.asarray(pred.design_matrix, dtype=np.float64)
        if bool(pred.has_intercept):
            Xp = np.column_stack([np.ones(Z.shape[0], dtype=np.float64), Z])
            local_idx = np.arange(full_start, full_start + Z.shape[1] + 1, dtype=int)
            reduced_to_full.extend(
                list(
                    np.arange(
                        full_start + 1,
                        full_start + 1 + Z.shape[1],
                        dtype=int,
                    )
                )
            )
        else:
            Xp = Z
            local_idx = np.arange(full_start, full_start + Z.shape[1], dtype=int)
            reduced_to_full.extend(
                list(np.arange(full_start, full_start + Z.shape[1], dtype=int))
            )
        blocks.append(Xp)
        jj.append(local_idx)
        predictor_full_slices.append(slice(full_start, full_start + Xp.shape[1]))
        full_start += Xp.shape[1]

    X_full = np.column_stack(blocks) if blocks else np.empty((model.n_samples_, 0))
    return _GeneralPredictorLayout(
        X_full=np.asarray(X_full, dtype=np.float64),
        jj=jj,
        reduced_to_full_idx=np.asarray(reduced_to_full, dtype=int),
        predictor_full_slices=predictor_full_slices,
    )


def _symmetrize_dense(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M, dtype=np.float64)
    return 0.5 * (M + M.T)


def _r_matrix_norm(M: np.ndarray) -> float:
    """Mirror R ``norm(M)`` default used in ``mgcv/R/fast-REML.r::Sl.setup``."""
    return float(np.linalg.norm(np.asarray(M, dtype=np.float64), ord=1))


def _mroot_chol_local(P: np.ndarray, *, rank: int | None = None) -> np.ndarray:
    from ...smoothing_selection.reparam import _mroot_chol

    return _mroot_chol(P, rank=rank)


def _sl_rank_from_eigenvalues(values: np.ndarray) -> int:
    values = np.asarray(values, dtype=np.float64).ravel()
    if values.size == 0:
        return 0
    vmax = float(np.max(values))
    if not np.isfinite(vmax) or vmax <= 0.0:
        return 0
    return int(np.sum(values > (np.finfo(np.float64).eps ** 0.8) * vmax))


def _sl_single_penalty_block(
    S_local: np.ndarray,
    *,
    start: int,
    stop: int,
    rank: int | None,
    repara: bool,
) -> GamFit5SlBlock:
    S_local = _symmetrize_dense(S_local)
    ut = np.triu_indices_from(S_local, k=1)
    diag_only = bool(np.sum(np.abs(S_local[ut])) == 0.0)

    if diag_only:
        D = np.asarray(np.diag(S_local), dtype=np.float64).copy()
        ind = np.asarray(D > 0.0, dtype=bool)
        rank_use = int(np.sum(ind))
        D[ind] = 1.0 / np.sqrt(D[ind])
        D[~ind] = 1.0
        return GamFit5SlBlock(
            start=int(start),
            stop=int(stop),
            rank=rank_use,
            S=[S_local],
            lambda_=np.array([1.0], dtype=np.float64),
            repara=bool(repara),
            linear=True,
            ldet=0.0,
            ind=ind,
            D=np.asarray(D, dtype=np.float64),
            Di=None,
        )

    values, vectors = np.linalg.eigh(S_local)
    order = np.argsort(values)[::-1]
    values = np.asarray(values[order], dtype=np.float64)
    vectors = np.asarray(vectors[:, order], dtype=np.float64)
    rank_use = _sl_rank_from_eigenvalues(values) if rank is None else int(rank)
    ind = np.zeros(values.shape[0], dtype=bool)
    ind[: min(rank_use, ind.size)] = True
    D_vals = values.copy()
    D_vals[ind] = 1.0 / np.sqrt(D_vals[ind])
    D_vals[~ind] = 1.0
    return GamFit5SlBlock(
        start=int(start),
        stop=int(stop),
        rank=int(rank_use),
        S=[S_local],
        lambda_=np.array([1.0], dtype=np.float64),
        repara=bool(repara),
        linear=True,
        ldet=0.0,
        ind=ind,
        D=np.asarray(vectors * D_vals[np.newaxis, :], dtype=np.float64),
        Di=np.asarray(vectors.T / D_vals[:, np.newaxis], dtype=np.float64),
    )


def _sl_multi_penalty_block(
    S_local: list[np.ndarray],
    *,
    start: int,
    stop: int,
    rank: int | None,
    repara: bool,
) -> GamFit5SlBlock:
    if not repara:
        raise NotImplementedError(
            "Non-reparameterized multi-penalty general-family Sl blocks are unsupported."
        )

    S_work = [_symmetrize_dense(Si) for Si in S_local]
    St_sum = np.zeros_like(S_work[0], dtype=np.float64)
    for Si in S_work:
        St_sum += Si

    values, vectors = np.linalg.eigh(St_sum)
    order = np.argsort(values)[::-1]
    values = np.asarray(values[order], dtype=np.float64)
    vectors = np.asarray(vectors[:, order], dtype=np.float64)
    rank_use = _sl_rank_from_eigenvalues(values) if rank is None else int(rank)
    ind = np.zeros(vectors.shape[1], dtype=bool)
    ind[: min(rank_use, ind.size)] = True
    Ur = np.asarray(vectors[:, ind], dtype=np.float64)

    transformed = []
    roots = []
    for Si in S_work:
        bob = _symmetrize_dense(Ur.T @ (Si @ Ur))
        transformed.append(np.asarray(bob, dtype=np.float64))
        roots.append(_mroot_chol_local(bob, rank=int(rank_use)))

    St = np.zeros_like(transformed[0], dtype=np.float64)
    for Si in transformed:
        S_norm = _r_matrix_norm(Si)
        if S_norm <= 0.0:
            raise RuntimeError(
                "Encountered zero-norm penalty in multi-penalty Sl block."
            )
        St += Si / S_norm
    St = _symmetrize_dense(St)

    return GamFit5SlBlock(
        start=int(start),
        stop=int(stop),
        rank=int(rank_use),
        S=transformed,
        lambda_=np.ones(len(transformed), dtype=np.float64),
        repara=True,
        linear=True,
        ldet=0.0,
        ind=ind,
        D=np.asarray(vectors, dtype=np.float64),
        Di=None,
        rS=[np.asarray(root, dtype=np.float64) for root in roots],
        St=None,
    )


def _materialize_sl_attrs(
    blocks: list[GamFit5SlBlock],
    *,
    n_param: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    E = np.zeros((n_param, n_param), dtype=np.float64)
    S = np.zeros((n_param, n_param), dtype=np.float64)
    lambda_values = []

    for block in blocks:
        if len(block.S) <= 1:
            lambda_values.extend(np.asarray(block.lambda_, dtype=np.float64).tolist())
        else:
            lambda_values.extend(
                [
                    1.0 / _r_matrix_norm(np.asarray(Si, dtype=np.float64))
                    for Si in block.S
                ]
            )
        if not block.repara:
            continue
        if len(block.S) == 1:
            ind = (
                np.zeros(block.width, dtype=bool)
                if block.ind is None
                else np.asarray(block.ind, dtype=bool)
            )
            if np.any(ind):
                idx = (block.start - 1) + np.flatnonzero(ind)
                E[idx, idx] = 1.0
                S[idx, idx] = 1.0
        else:
            St = np.zeros_like(np.asarray(block.S[0], dtype=np.float64), dtype=np.float64)
            for Si in block.S:
                Si_arr = np.asarray(Si, dtype=np.float64)
                St += Si_arr / _r_matrix_norm(Si_arr)
            St = _symmetrize_dense(St)
            if St.size == 0:
                continue
            start0 = block.start - 1
            rows = np.arange(start0, start0 + St.shape[0], dtype=int)
            E[np.ix_(rows, rows)] = np.asarray(
                _mroot_chol_local(St, rank=int(block.rank)).T,
                dtype=np.float64,
            )
            S[np.ix_(rows, rows)] = St

    return (
        np.asarray(E, dtype=np.float64),
        np.asarray(S, dtype=np.float64),
        np.asarray(lambda_values, dtype=np.float64),
    )


def build_gam_fit5_sl_setup(model, layout: _GeneralPredictorLayout) -> GamFit5SlSetup:
    """
    Materialize the upstream ``Sl`` list for the implemented linear general-family path.
    """

    penalty_blocks = list(_penalty_blocks_seq(model))
    term_blocks = list(_term_blocks_seq(model))
    full_idx = np.asarray(layout.reduced_to_full_idx, dtype=int)
    used_penalties: set[int] = set()
    blocks: list[GamFit5SlBlock] = []

    def _term_block_start_stop(term_penalties: list[int]) -> tuple[int, int]:
        idx = np.asarray(
            [full_idx[penalty_blocks[i].coef_slice] for i in term_penalties],
            dtype=object,
        )
        idx0 = np.asarray(idx[0], dtype=int)
        if idx0.size == 0:
            raise RuntimeError("Zero-width penalty block in general-family Sl setup.")
        expected = np.arange(int(idx0[0]), int(idx0[0]) + idx0.size, dtype=int)
        if not np.array_equal(idx0, expected):
            raise NotImplementedError(
                "General-family Sl setup requires contiguous term penalty blocks."
            )
        return int(idx0[0]) + 1, int(idx0[-1]) + 1

    def _append_block(block: GamFit5SlBlock) -> None:
        blocks.append(block)

    for term_index, _term in enumerate(term_blocks):
        term_penalty_idx = [
            i for i, pb in enumerate(penalty_blocks) if int(pb.term_index) == term_index
        ]
        if not term_penalty_idx:
            continue
        used_penalties.update(term_penalty_idx)
        start, stop = _term_block_start_stop(term_penalty_idx)
        local_penalties = [
            _symmetrize_dense(np.asarray(penalty_blocks[i].matrix, dtype=np.float64))
            for i in term_penalty_idx
        ]
        local_ranks = [
            (
                int(penalty_blocks[i].rank)
                if getattr(penalty_blocks[i], "rank", None) is not None
                else int(np.linalg.matrix_rank(local_penalties[j]))
            )
            for j, i in enumerate(term_penalty_idx)
        ]

        if len(local_penalties) == 1:
            _append_block(
                _sl_single_penalty_block(
                    local_penalties[0],
                    start=start,
                    stop=stop,
                    rank=local_ranks[0],
                    repara=True,
                )
            )
            continue

        nb = int(local_penalties[0].shape[0])
        sbdiag = []
        sb_start = []
        sb_stop = []
        for Si in local_penalties:
            ut = np.triu_indices(nb, k=1)
            sbdiag.append(bool(np.sum(np.abs(Si[ut])) == 0.0))
            active = np.flatnonzero(np.sum(np.abs(Si), axis=1) > 0.0)
            if active.size == 0:
                raise RuntimeError("Zero-support penalty encountered in Sl.setup.")
            sb_start.append(int(active[0]))
            sb_stop.append(int(active[-1]))

        split_ok = True
        for j in range(len(local_penalties)):
            itot = np.zeros(nb, dtype=bool)
            if all(sbdiag):
                active_j = np.diag(local_penalties[j]) != 0.0
                for k in range(len(local_penalties)):
                    if j == k:
                        continue
                    itot |= np.diag(local_penalties[k]) != 0.0
                if int(np.sum(itot[active_j])) > 0:
                    split_ok = False
                    break
            else:
                for k in range(len(local_penalties)):
                    if j == k:
                        continue
                    itot[sb_start[k] : sb_stop[k] + 1] = True
                if int(np.sum(itot[sb_start[j] : sb_stop[j] + 1])) > 0:
                    split_ok = False
                    break

        if split_ok:
            for j, Si in enumerate(local_penalties):
                ind = slice(sb_start[j], sb_stop[j] + 1)
                _append_block(
                    _sl_single_penalty_block(
                        Si[ind, ind],
                        start=start + sb_start[j],
                        stop=start + sb_stop[j],
                        rank=local_ranks[j],
                        repara=True,
                    )
                )
        else:
            _append_block(
                _sl_multi_penalty_block(
                    local_penalties,
                    start=start,
                    stop=stop,
                    rank=None,
                    repara=True,
                )
            )

    for i, pb in enumerate(penalty_blocks):
        if i in used_penalties:
            continue
        idx = np.asarray(full_idx[pb.coef_slice], dtype=int)
        if idx.size == 0:
            continue
        expected = np.arange(int(idx[0]), int(idx[0]) + idx.size, dtype=int)
        if not np.array_equal(idx, expected):
            raise NotImplementedError(
                "General-family Sl setup requires contiguous fallback penalty blocks."
            )
        _append_block(
            _sl_single_penalty_block(
                np.asarray(pb.matrix, dtype=np.float64),
                start=int(idx[0]) + 1,
                stop=int(idx[-1]) + 1,
                rank=(
                    int(pb.rank)
                    if getattr(pb, "rank", None) is not None
                    else int(
                        np.linalg.matrix_rank(np.asarray(pb.matrix, dtype=np.float64))
                    )
                ),
                repara=True,
            )
        )

    E, S, lambda_values = _materialize_sl_attrs(
        blocks, n_param=int(layout.X_full.shape[1])
    )
    return GamFit5SlSetup(
        blocks=blocks,
        E=E,
        S=S,
        lambda_=lambda_values,
        cholesky=False,
    )


def sl_initial_repara(
    Sl: GamFit5SlSetup,
    X: np.ndarray,
    *,
    inverse: bool = False,
    both_sides: bool = True,
    cov: bool = True,
) -> np.ndarray:
    """
    Mirror ``mgcv::Sl.initial.repara`` for the implemented linear ``Sl`` blocks.
    """

    X_arr = np.asarray(X, dtype=np.float64).copy()
    if len(Sl) == 0:
        return X_arr

    is_matrix = X_arr.ndim == 2
    for block in Sl:
        if not block.repara:
            continue
        ind = np.arange(block.start0, block.stop0, dtype=int)
        D = np.asarray(block.D, dtype=np.float64)

        if inverse:
            if is_matrix:
                if cov:
                    if D.ndim == 2:
                        if both_sides:
                            X_arr[ind, :] = D @ X_arr[ind, :]
                        X_arr[:, ind] = X_arr[:, ind] @ D.T
                    else:
                        X_arr[:, ind] = X_arr[:, ind] * D[np.newaxis, :]
                        if both_sides:
                            X_arr[ind, :] = D[:, np.newaxis] * X_arr[ind, :]
                else:
                    if D.ndim == 2:
                        Di = (
                            D.T
                            if block.Di is None
                            else np.asarray(block.Di, dtype=np.float64)
                        )
                        if both_sides:
                            X_arr[ind, :] = Di.T @ X_arr[ind, :]
                        X_arr[:, ind] = X_arr[:, ind] @ Di
                    else:
                        Di = 1.0 / D
                        X_arr[:, ind] = X_arr[:, ind] * Di[np.newaxis, :]
                        if both_sides:
                            X_arr[ind, :] = Di[:, np.newaxis] * X_arr[ind, :]
            else:
                if D.ndim == 2:
                    X_arr[ind] = D @ X_arr[ind]
                else:
                    X_arr[ind] = D * X_arr[ind]
        else:
            if is_matrix:
                if D.ndim == 2:
                    if both_sides:
                        X_arr[ind, :] = D.T @ X_arr[ind, :]
                    X_arr[:, ind] = X_arr[:, ind] @ D
                else:
                    if both_sides:
                        X_arr[ind, :] = D[:, np.newaxis] * X_arr[ind, :]
                    X_arr[:, ind] = X_arr[:, ind] * D[np.newaxis, :]
            else:
                if both_sides:
                    if D.ndim == 2:
                        X_arr[ind] = D.T @ X_arr[ind]
                    else:
                        X_arr[ind] = D * X_arr[ind]
                else:
                    if D.ndim == 2:
                        Di = (
                            D.T
                            if block.Di is None
                            else np.asarray(block.Di, dtype=np.float64)
                        )
                        X_arr[ind] = Di @ X_arr[ind]
                    else:
                        X_arr[ind] = X_arr[ind] / D

    return np.asarray(X_arr, dtype=np.float64)


def _build_general_penalty_matrix(
    model, smoothing_params, layout
) -> tuple[np.ndarray, list[np.ndarray]]:
    smoothing_params = np.asarray(smoothing_params, dtype=np.float64).ravel()
    p_full = layout.X_full.shape[1]
    St = np.zeros((p_full, p_full), dtype=np.float64)
    S_blocks: list[np.ndarray] = []

    full_idx = layout.reduced_to_full_idx
    for pb in _penalty_blocks_seq(model):
        S_full = np.zeros((p_full, p_full), dtype=np.float64)
        idx = full_idx[pb.coef_slice]
        S_full[np.ix_(idx, idx)] = np.asarray(pb.matrix, dtype=np.float64)
        S_blocks.append(S_full)
        St += float(smoothing_params[pb.smoothing_index]) * S_full

    return St, S_blocks


def _offset_list(model, n_pred: int):
    offset = getattr(model, "offset_train_", None)
    if offset is None:
        return [None]
    if isinstance(offset, (list, tuple)):
        return list(offset)
    return [np.asarray(offset, dtype=np.float64)]


def _general_fit_score_type_name(method: str) -> str:
    method_l = str(method).lower()
    if method_l in {"reml", "laml"}:
        return "REML"
    if method_l == "ml":
        return "ML"
    return method_l.upper()


def _general_penalty_null_dim(St: np.ndarray) -> int:
    St = np.asarray(St, dtype=np.float64)
    evals = np.linalg.eigvalsh(0.5 * (St + St.T))
    if evals.size == 0:
        return 0
    tol = max(float(np.max(evals)), 0.0) * np.finfo(np.float64).eps ** 0.75
    return int(St.shape[0] - np.count_nonzero(evals > tol))


def build_gam_fit5_setup_state(
    model,
    smoothing_params,
    *,
    score_type=None,
) -> GamFit5SetupState:
    layout = _build_general_predictor_layout(model)
    sl = build_gam_fit5_sl_setup(model, layout)
    X_initial = sl_initial_repara(sl, layout.X_full, both_sides=False)
    smoothing_params = np.asarray(smoothing_params, dtype=np.float64).ravel()
    St, S_blocks = _build_general_penalty_matrix(model, smoothing_params, layout)
    log_sp = np.log(np.clip(smoothing_params, 1e-300, None))
    ldet_state = _sl_ldetS(
        sl,
        rho=log_sp,
        fixed=np.zeros_like(smoothing_params, dtype=bool),
        np_=layout.X_full.shape[1],
        root=False,
        Stot=False,
        deriv=2,
    )
    score_name = _general_fit_score_type_name(
        getattr(model, "_optim_method", "REML") if score_type is None else score_type
    )

    return GamFit5SetupState(
        layout=layout,
        sl=sl,
        X_full=np.asarray(layout.X_full, dtype=np.float64),
        X_initial=np.asarray(X_initial, dtype=np.float64),
        jj=[np.asarray(j, dtype=int) for j in layout.jj],
        reduced_to_full_idx=np.asarray(layout.reduced_to_full_idx, dtype=int),
        predictor_full_slices=list(layout.predictor_full_slices),
        offset_list=_offset_list(model, len(layout.jj)),
        smoothing_params=smoothing_params.copy(),
        log_sp=np.asarray(log_sp, dtype=np.float64),
        St=np.asarray(St, dtype=np.float64),
        S_blocks=[np.asarray(S, dtype=np.float64) for S in S_blocks],
        ldetS=float(ldet_state["ldetS"]),
        ldetS1=np.asarray(ldet_state["ldet1"], dtype=np.float64),
        ldetS2=np.asarray(ldet_state["ldet2"], dtype=np.float64),
        Mp=_general_penalty_null_dim(St),
        score_type=score_name,
    )


def _record_outer_derivative_mode(model, *, gradient_source=None, hessian_source=None):
    info = dict(getattr(model, "_general_fit5_outer_derivative_info", {}) or {})
    family = getattr(model, "family", None)
    family_name = str(getattr(family, "name", "")).lower()
    supports_analytic = bool(
        getattr(family, "supports_analytic_outer_derivatives", False)
    )
    supports_analytic_hessian = _supports_analytic_outer_hessian(family)
    if gradient_source is not None:
        info["gradient_source"] = str(gradient_source)
    if hessian_source is not None:
        info["hessian_source"] = str(hessian_source)
    info["supports_analytic_outer_derivatives"] = supports_analytic
    info["penalty_logdet_source"] = "analytic"
    info["uses_exact_penalty_logdet"] = True
    if (
        not supports_analytic
        and not supports_analytic_hessian
        and family_name in {"gevlss", "shashlss"}
    ):
        info["fallback_reason"] = (
            "fully analytic outer Hessian is not exposed by this family; "
            "use analytic gradient and finite-difference Hessian fallback"
        )
    else:
        info.pop("fallback_reason", None)
    model._general_fit5_outer_derivative_info = info


def _supports_analytic_outer_gradient(family) -> bool:
    return bool(
        getattr(family, "supports_analytic_outer_derivatives", False)
        or getattr(family, "supports_analytic_outer_gradient", False)
    )


def _supports_analytic_outer_hessian(family) -> bool:
    return bool(
        getattr(family, "supports_analytic_outer_derivatives", False)
        or getattr(family, "supports_analytic_outer_hessian", False)
    )


def _run_general_fit5(
    model,
    y,
    smoothing_params,
    *,
    weights=None,
    deriv=2,
    score_type=None,
):
    setup = build_gam_fit5_setup_state(
        model,
        smoothing_params,
        score_type=score_type,
    )
    from ...smoothing_selection.reparam import _stable_penalty_logdet_derivatives

    ldetS, ldetS1, ldetS2 = _stable_penalty_logdet_derivatives(
        model,
        np.asarray(smoothing_params, dtype=np.float64),
        order=2,
    )
    ctl = GamFit5Control(
        maxit=int(getattr(model, "max_irls_iter", 200)),
        epsilon=float(getattr(model, "irls_tol", 1e-7)),
        trace=bool(getattr(model, "hparams", {}).get("trace", False)),
    )
    fit = gam_fit5(
        setup.X_initial,
        np.asarray(y, dtype=np.float64),
        setup.jj,
        setup.log_sp,
        setup.St,
        setup.S_blocks,
        ldetS=float(ldetS),
        ldetS1=np.asarray(ldetS1, dtype=np.float64),
        ldetS2=np.asarray(ldetS2, dtype=np.float64),
        family=model.family,
        weights=weights,
        offset=setup.offset_list,
        deriv=deriv,
        score_type=setup.score_type,
        control=ctl,
        Mp=setup.Mp,
        Sl=setup.Sl,
    )
    return {
        "layout": setup.layout,
        "setup": setup,
        "fit": fit,
        "offset_list": setup.offset_list,
        "smoothing_params": setup.smoothing_params,
        "log_sp": setup.log_sp,
    }


def criterion_ml_reml_general_fit5(model, y, log_sp, method):
    sp = expand_smoothing_params_from_log(model, log_sp)
    run = _run_general_fit5(
        model, y, sp, weights=model.prior_weights_, deriv=0, score_type=method
    )
    return float(run["fit"]["score"])


def criterion_gradient_ml_reml_general_fit5(model, y, log_sp, method):
    if not _supports_analytic_outer_gradient(model.family):
        raise NotImplementedError(
            "General-family ML/REML outer optimization requires analytic outer "
            "gradients for strict mgcv parity; finite-difference fallback removed."
        )
    _record_outer_derivative_mode(model, gradient_source="analytic")
    sp = expand_smoothing_params_from_log(model, log_sp)
    run = _run_general_fit5(
        model, y, sp, weights=model.prior_weights_, deriv=1, score_type=method
    )
    grad = run["fit"].get("score1", None)
    if grad is None:
        return np.empty((0,), dtype=np.float64)
    return np.asarray(grad, dtype=np.float64)


def criterion_hessian_ml_reml_general_fit5(model, y, log_sp, method):
    if _supports_analytic_outer_hessian(model.family):
        _record_outer_derivative_mode(model, hessian_source="analytic")
        sp = expand_smoothing_params_from_log(model, log_sp)
        run = _run_general_fit5(
            model, y, sp, weights=model.prior_weights_, deriv=2, score_type=method
        )
        hess = run["fit"].get("score2", None)
        if hess is None:
            return np.empty((0, 0), dtype=np.float64)
        return np.asarray(hess, dtype=np.float64)
    raise NotImplementedError(
        "General-family ML/REML outer optimization requires analytic outer "
        "Hessians for strict mgcv parity; finite-difference fallback removed."
    )


def solve_general_fit(model, y, smoothing_params, weights=None):
    need_postproc_derivs = len(tuple(_penalty_blocks_seq(model))) > 0 and (
        _supports_analytic_outer_gradient(model.family)
        or _supports_analytic_outer_hessian(model.family)
    )
    deriv_order = (
        2
        if need_postproc_derivs and _supports_analytic_outer_hessian(model.family)
        else 1 if need_postproc_derivs else 0
    )
    run = _run_general_fit5(
        model,
        y,
        smoothing_params,
        weights=weights,
        deriv=deriv_order,
        score_type=getattr(model, "_optim_method", "REML"),
    )
    setup = run["setup"]
    fit = run["fit"]
    outer_hess = None
    optim_result = getattr(model, "_optim_result", None)
    if optim_result is not None and getattr(optim_result, "hess", None) is not None:
        outer_hess = np.asarray(optim_result.hess, dtype=np.float64)

    post = gam_fit5_post_proc(
        fit,
        Sl=setup.Sl,
        L_map=None,
        lsp0=None,
        S_blocks=setup.S_blocks,
        off=[1] * len(setup.S_blocks),
        outer_hess=outer_hess,
        smoothing_params=setup.smoothing_params,
    )

    coef_full = np.asarray(fit["coef"], dtype=np.float64)
    if len(setup.Sl) > 0:
        coef_full = np.asarray(
            sl_initial_repara(
                setup.Sl,
                coef_full,
                inverse=True,
                both_sides=False,
                cov=False,
            ),
            dtype=np.float64,
        )

    eta_cols = []
    for k, sl in enumerate(setup.predictor_full_slices):
        eta_k = setup.X_full[:, sl] @ np.asarray(coef_full[sl], dtype=np.float64)
        if setup.offset_list is not None and k < len(setup.offset_list):
            off_k = setup.offset_list[k]
            if off_k is not None:
                eta_k = eta_k + np.asarray(off_k, dtype=np.float64)
        eta_cols.append(np.asarray(eta_k, dtype=np.float64))
    eta = (
        np.column_stack(eta_cols)
        if eta_cols
        else np.empty((len(y), 0), dtype=np.float64)
    )

    mu = np.asarray(model.family.predict(eta=eta), dtype=np.float64)

    RTR = np.asarray(post["R"].T @ post["R"], dtype=np.float64)
    H_coef = np.asarray(post["Vp"] @ RTR, dtype=np.float64)

    Vc = np.asarray(post.get("Vc", post["Vp"]), dtype=np.float64)

    beta = np.asarray(coef_full[setup.reduced_to_full_idx], dtype=np.float64)
    intercept = float(coef_full[0]) if setup.predictor_full_slices else 0.0
    deviance = float(-2.0 * float(fit["l"]))
    family_residuals = getattr(model.family, "residuals", None)
    if callable(family_residuals):
        try:
            rsd = np.asarray(
                family_residuals(
                    np.asarray(y, dtype=np.float64),
                    mu,
                    rtype="deviance",
                    eta=eta,
                ),
                dtype=np.float64,
            ).ravel()
        except TypeError:
            try:
                rsd = np.asarray(
                    family_residuals(
                        np.asarray(y, dtype=np.float64),
                        mu,
                        rtype="deviance",
                    ),
                    dtype=np.float64,
                ).ravel()
            except Exception:
                rsd = np.empty((0,), dtype=np.float64)
        except Exception:
            rsd = np.empty((0,), dtype=np.float64)
        if (
            rsd.size == len(np.asarray(y, dtype=np.float64).ravel())
            and np.isfinite(rsd).all()
        ):
            deviance = float(np.sum(rsd**2))

    return FitCoreSolution.from_dict(
        {
            "coef_full": coef_full,
            "intercept": intercept,
            "beta": beta,
            "eta": eta,
            "mu": mu,
            "rss": None,
            "deviance": deviance,
            "edf": float(np.sum(np.asarray(post["edf"], dtype=np.float64))),
            "trace_H": float(np.trace(H_coef)),
            "scale": 1.0,
            "cov_bayes": np.asarray(post["Vp"], dtype=np.float64),
            "cov_freq": np.asarray(post["Ve"], dtype=np.float64),
            "cov_unconditional": Vc,
            "H_coef": H_coef,
            "edf2": np.asarray(post["edf2"], dtype=np.float64),
            "X": setup.X_full,
            "A": np.asarray(-fit["lbb"], dtype=np.float64)
            + np.asarray(fit["St_full"], dtype=np.float64),
            "A_inv": np.asarray(post["Vp"], dtype=np.float64),
            "XtWX": None,
            "P": np.asarray(setup.St, dtype=np.float64),
            "penalty_matrix": np.asarray(setup.St, dtype=np.float64),
            "working_weights": None,
            "fisher_weights": None,
            "working_response": None,
            "penalty_quadratic": 0.5
            * float(coef_full @ (np.asarray(setup.St, dtype=np.float64) @ coef_full)),
            "loglik": float(fit["l"]),
            "offset": None,
            "log_det_XtWX_plus_penalty": float(fit["ldetHp"]),
            "converged": (len(fit.get("warn", [])) == 0),
            "iter": int(fit["iter"]),
            "failed_step": bool(len(fit.get("warn", [])) > 0),
            "failure_reason": (
                None if len(fit.get("warn", [])) == 0 else "; ".join(fit["warn"])
            ),
            "inner_trace": None,
        }
    )
