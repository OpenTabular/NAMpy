"""
General-family fixed-smoothing backend using mgcv-style ``gam.fit5``.

Mirrors mgcv ``gam.fit5`` / ``gam.fit5.post.proc`` from ``mgcv/R/gam.fit4.r``
for multi-linear-predictor GAMLSS-style families.
"""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.linalg import eigh as scipy_eigh

from ....linalg.matrix import symmetrize_matrix
from ....linalg.norms import r_matrix_norm_one
from ....model_state import (
    _fit_workspace,
    _penalty_blocks_seq,
    _predictor_designs,
    _term_blocks_seq,
)
from ...parameterization import (
    PREDICTION_PARAMETER_SPACE,
    prediction_parameterization_map,
)
from ...smoothing_params import expand_smoothing_params_from_log
from ...state import FitCoreSolution
from . import newton as general_newton
from .sl_transforms import sl_inirep as sl_inirep
from .sl_transforms import sl_initial_repara


@dataclass
class _GeneralPredictorLayout:
    X_full: np.ndarray
    jj: list[np.ndarray]
    reduced_to_full_idx: np.ndarray
    predictor_full_slices: list[slice]


@dataclass
class GeneralPenaltyBlock:
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
    St: Any | None = None
    penalty_indices: tuple[int, ...] = field(default_factory=tuple)
    smoothing_indices: tuple[int, ...] = field(default_factory=tuple)
    n_sp: int | None = None
    inisp: Any | None = None
    updateS: Any | None = None
    AS: Any | None = None
    AdS: Any | None = None
    ldS: Any | None = None
    nlinfo: Any | None = None

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
class GeneralPenaltySetup:
    """
    Python materialization of the upstream ``Sl`` object from ``mgcv::Sl.setup``.
    """

    blocks: list[Any]
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
class GeneralFamilySetupState:
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
    sl: GeneralPenaltySetup
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
    penalty_derivatives: list[np.ndarray]
    ldetS: float
    ldetS1: np.ndarray
    ldetS2: np.ndarray
    Mp: int
    score_type: str

    @property
    def Sl(self) -> GeneralPenaltySetup:
        return self.sl


def _build_general_predictor_layout(model) -> _GeneralPredictorLayout:
    blocks = []
    from ....model_state import _compiled_model, _predictor_full_indices

    predictor_full_slices: list[np.ndarray] = []
    full_start = 0

    for pred in _predictor_designs(model):
        Z = np.asarray(pred.design_matrix, dtype=np.float64)
        if bool(pred.prediction_has_intercept):
            Xp = np.column_stack([np.ones(Z.shape[0], dtype=np.float64), Z])
        else:
            Xp = Z
        blocks.append(Xp)
        full_start += Xp.shape[1]

    X_full = np.column_stack(blocks) if blocks else np.empty((model.n_samples_, 0))
    compiled = _compiled_model(model)
    if compiled is None:
        raise RuntimeError("General-family layout requires a compiled model.")
    jj = [np.asarray(indices, dtype=int) for indices in _predictor_full_indices(model)]
    predictor_full_slices.extend(jj)
    return _GeneralPredictorLayout(
        X_full=np.asarray(X_full, dtype=np.float64),
        jj=jj,
        reduced_to_full_idx=np.asarray(
            compiled.coef_reduced_to_full_idx, dtype=int
        ),
        predictor_full_slices=predictor_full_slices,
    )


def _mroot_chol_local(P: np.ndarray, *, rank: int | None = None) -> np.ndarray:
    from ...selection.reparam import _mroot_chol

    root: np.ndarray = _mroot_chol(P, rank=rank)
    return root


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
    penalty_indices: tuple[int, ...] = (),
) -> GeneralPenaltyBlock:
    S_local = symmetrize_matrix(S_local)
    ut = np.triu_indices_from(S_local, k=1)
    diag_only = bool(np.sum(np.abs(S_local[ut])) == 0.0)

    if diag_only:
        D = np.asarray(np.diag(S_local), dtype=np.float64).copy()
        ind = np.asarray(D > 0.0, dtype=bool)
        rank_use = int(np.sum(ind))
        D[ind] = 1.0 / np.sqrt(D[ind])
        D[~ind] = 1.0
        return GeneralPenaltyBlock(
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
            penalty_indices=tuple(int(v) for v in penalty_indices),
        )

    # mgcv::Sl.setup() uses R's `eigen(..., symmetric=TRUE)` result directly.
    # SciPy's `lower=True` path matches the LAPACK triangle convention R uses
    # here; changing it alters `Sl.initial.repara` and general-family starts.
    values, vectors = scipy_eigh(
        S_local,
        lower=True,
        check_finite=False,
    )
    order = np.argsort(values)[::-1]
    values = np.asarray(values[order], dtype=np.float64)
    vectors = np.asarray(vectors[:, order], dtype=np.float64)
    rank_use = _sl_rank_from_eigenvalues(values) if rank is None else int(rank)
    ind = np.zeros(values.shape[0], dtype=bool)
    ind[: min(rank_use, ind.size)] = True
    D_vals = values.copy()
    D_vals[ind] = 1.0 / np.sqrt(D_vals[ind])
    D_vals[~ind] = 1.0
    return GeneralPenaltyBlock(
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
        penalty_indices=tuple(int(v) for v in penalty_indices),
    )


def _sl_multi_penalty_block(
    S_local: list[np.ndarray],
    *,
    start: int,
    stop: int,
    rank: int | None,
    repara: bool,
    penalty_indices: tuple[int, ...] = (),
) -> GeneralPenaltyBlock:
    if not repara:
        raise NotImplementedError(
            "Non-reparameterized multi-penalty general-family Sl blocks are unsupported."
        )

    S_work = [symmetrize_matrix(Si) for Si in S_local]
    St_sum = np.zeros_like(S_work[0], dtype=np.float64)
    for Si in S_work:
        St_sum += Si

    # `mgcv/R/fast-REML.r::Sl.setup` calls `eigen(St, symmetric=TRUE)` using
    # the lower-triangle convention. Keep the same convention as the singleton
    # Sl path above.
    values, vectors = scipy_eigh(
        St_sum,
        lower=True,
        check_finite=False,
    )
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
        bob = symmetrize_matrix(Ur.T @ (Si @ Ur))
        transformed.append(np.asarray(bob, dtype=np.float64))
        roots.append(_mroot_chol_local(bob, rank=int(rank_use)))

    St = np.zeros_like(transformed[0], dtype=np.float64)
    for Si in transformed:
        S_norm = r_matrix_norm_one(Si)
        if S_norm <= 0.0:
            raise RuntimeError(
                "Encountered zero-norm penalty in multi-penalty Sl block."
            )
        St += Si / S_norm
    St = symmetrize_matrix(St)

    return GeneralPenaltyBlock(
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
        penalty_indices=tuple(int(v) for v in penalty_indices),
    )


_GENERAL_FAMILY_NONLINEAR_SL_KEY = "general_family_nonlinear_sl"


def _general_family_term_start_stop(
    term,
    *,
    full_idx: np.ndarray,
) -> tuple[int, int]:
    idx = np.asarray(full_idx[term.coef_slice], dtype=int)
    if idx.size == 0:
        raise RuntimeError("Zero-width term block in general-family Sl setup.")
    expected = np.arange(int(idx[0]), int(idx[0]) + idx.size, dtype=int)
    if not np.array_equal(idx, expected):
        raise NotImplementedError(
            "General-family nonlinear Sl setup requires contiguous term coefficient blocks."
        )
    return int(idx[0]) + 1, int(idx[-1]) + 1


def _coerce_general_family_nonlinear_sl_block(
    spec: Any,
    *,
    start: int,
    stop: int,
    penalty_indices: tuple[int, ...],
    smoothing_indices: tuple[int, ...],
    n_sp: int,
) -> Any:
    if isinstance(spec, dict):
        block = GeneralPenaltyBlock(
            start=int(start),
            stop=int(stop),
            S=[np.asarray(Si, dtype=np.float64) for Si in spec.get("S", ())],
            lambda_=np.asarray(
                spec.get("lambda_", np.zeros(int(spec.get("n_sp", n_sp)))),
                dtype=np.float64,
            ).copy(),
            repara=bool(spec.get("repara", False)),
            linear=False,
            rank=spec.get("rank", None),
            ldet=float(spec.get("ldet", 0.0)),
            ind=(
                None
                if spec.get("ind", None) is None
                else np.asarray(spec.get("ind"), dtype=bool).copy()
            ),
            D=(
                None
                if spec.get("D", None) is None
                else np.asarray(spec.get("D"), dtype=np.float64).copy()
            ),
            Di=(
                None
                if spec.get("Di", None) is None
                else np.asarray(spec.get("Di"), dtype=np.float64).copy()
            ),
            rS=[np.asarray(root, dtype=np.float64) for root in spec.get("rS", ())],
            St=spec.get("St", None),
            penalty_indices=tuple(int(v) for v in penalty_indices),
            smoothing_indices=tuple(int(v) for v in smoothing_indices),
            n_sp=int(spec.get("n_sp", n_sp)),
            inisp=spec.get("inisp", None),
            updateS=spec.get("updateS", None),
            AS=spec.get("AS", None),
            AdS=spec.get("AdS", None),
            ldS=spec.get("ldS", None),
            nlinfo=spec.get("nlinfo", None),
        )
    else:
        block = copy(spec)
        block.start = int(start)
        block.stop = int(stop)
        block.linear = False
        block.repara = bool(getattr(block, "repara", False))
        block.penalty_indices = tuple(int(v) for v in penalty_indices)
        block.smoothing_indices = tuple(int(v) for v in smoothing_indices)
        if getattr(block, "n_sp", None) is None:
            block.n_sp = int(n_sp)
        if getattr(block, "lambda_", None) is None:
            block.lambda_ = np.zeros(int(block.n_sp), dtype=np.float64)
        else:
            block.lambda_ = np.asarray(block.lambda_, dtype=np.float64).copy()
        if getattr(block, "S", None) is None:
            block.S = []
        else:
            block.S = [np.asarray(Si, dtype=np.float64) for Si in block.S]

    if bool(getattr(block, "repara", False)):
        raise NotImplementedError(
            "Reparameterized nonlinear general-family Sl blocks are unsupported."
        )

    n_sp_use = int(getattr(block, "n_sp", n_sp))
    if n_sp_use <= 0:
        raise ValueError("Nonlinear general-family Sl blocks require `n_sp > 0`.")
    if penalty_indices and n_sp_use != len(penalty_indices):
        raise ValueError(
            "Nonlinear general-family Sl block smoothing-parameter count does not "
            "match the compiled term penalties."
        )

    missing = [
        name
        for name in ("updateS", "AS", "AdS", "ldS", "St")
        if getattr(block, name, None) is None
    ]
    if missing:
        raise TypeError(
            "Nonlinear general-family Sl blocks require "
            + ", ".join(f"`{name}`" for name in missing)
            + "."
        )

    return block


def _term_nonlinear_sl_block(
    term,
    *,
    term_penalty_idx: list[int],
    penalty_blocks: list[Any],
    full_idx: np.ndarray,
) -> Any | None:
    metadata = dict(getattr(term, "metadata", {}) or {})
    spec = metadata.get(_GENERAL_FAMILY_NONLINEAR_SL_KEY, None)
    if spec is None:
        return None

    if not term_penalty_idx:
        raise ValueError(
            "Compiled general-family nonlinear Sl terms require corresponding "
            "compiled penalties to define smoothing-parameter ownership."
        )

    start, stop = _general_family_term_start_stop(term, full_idx=full_idx)
    penalty_indices = tuple(int(v) for v in term_penalty_idx)
    smoothing_indices = tuple(
        int(penalty_blocks[i].smoothing_index) for i in term_penalty_idx
    )
    if callable(spec):
        spec = general_newton._sl_call(
            spec,
            [
                (term, penalty_indices, start, stop),
                (term, penalty_indices),
                (term,),
                (),
            ],
        )
    if spec is None:
        return None
    return _coerce_general_family_nonlinear_sl_block(
        spec,
        start=start,
        stop=stop,
        penalty_indices=penalty_indices,
        smoothing_indices=smoothing_indices,
        n_sp=len(penalty_indices),
    )


def _materialize_sl_attrs(
    blocks: list[Any],
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
                    1.0 / r_matrix_norm_one(np.asarray(Si, dtype=np.float64))
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
            St = np.zeros_like(
                np.asarray(block.S[0], dtype=np.float64), dtype=np.float64
            )
            for Si in block.S:
                Si_arr = np.asarray(Si, dtype=np.float64)
                St += Si_arr / r_matrix_norm_one(Si_arr)
            St = symmetrize_matrix(St)
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


def build_general_penalty_setup(
    model, layout: _GeneralPredictorLayout
) -> GeneralPenaltySetup:
    """
    Materialize the upstream ``Sl`` list for general-family fits.

    Mirrors `mgcv::Sl.setup()` for linear blocks and term-owned nonlinear
    blocks carried through compiled-term metadata.
    """

    penalty_blocks = list(_penalty_blocks_seq(model))
    term_blocks = list(_term_blocks_seq(model))
    full_idx = np.asarray(layout.reduced_to_full_idx, dtype=int)
    used_penalties: set[int] = set()
    blocks: list[Any] = []

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

    def _append_block(block: Any) -> None:
        blocks.append(block)

    for term_index, term in enumerate(term_blocks):
        term_penalty_idx = [
            i for i, pb in enumerate(penalty_blocks) if int(pb.term_index) == term_index
        ]
        if not term_penalty_idx:
            continue
        nonlinear_block = _term_nonlinear_sl_block(
            term,
            term_penalty_idx=term_penalty_idx,
            penalty_blocks=penalty_blocks,
            full_idx=full_idx,
        )
        if nonlinear_block is not None:
            used_penalties.update(term_penalty_idx)
            _append_block(nonlinear_block)
            continue
        used_penalties.update(term_penalty_idx)
        start, stop = _term_block_start_stop(term_penalty_idx)
        local_penalties = [
            symmetrize_matrix(np.asarray(penalty_blocks[i].matrix, dtype=np.float64))
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
                    penalty_indices=(int(term_penalty_idx[0]),),
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
                        penalty_indices=(int(term_penalty_idx[j]),),
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
                    penalty_indices=tuple(int(v) for v in term_penalty_idx),
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
                penalty_indices=(int(i),),
            )
        )

    E, S, lambda_values = _materialize_sl_attrs(
        blocks, n_param=int(layout.X_full.shape[1])
    )
    return GeneralPenaltySetup(
        blocks=blocks,
        E=E,
        S=S,
        lambda_=lambda_values,
        cholesky=False,
    )


def _general_penalty_setup_has_nonlinear_blocks(Sl: GeneralPenaltySetup) -> bool:
    return any(not bool(getattr(block, "linear", True)) for block in Sl)


def _nonlinear_penalty_indices(Sl: GeneralPenaltySetup) -> set[int]:
    out: set[int] = set()
    for block in Sl:
        if not bool(getattr(block, "linear", True)):
            out.update(int(v) for v in getattr(block, "penalty_indices", ()))
    return out


def _current_nonlinear_penalty_matrix(
    sl_blocks: list[Any],
    *,
    n_param: int,
) -> np.ndarray:
    St = np.zeros((n_param, n_param), dtype=np.float64)
    for block in sl_blocks:
        if bool(getattr(block, "linear", True)):
            continue
        base_ind = np.arange(int(block.start) - 1, int(block.stop), dtype=int)
        current = general_newton._sl_block_st(block, 0)
        S_block = symmetrize_matrix(np.asarray(current["S"], dtype=np.float64))
        if S_block.shape != (base_ind.size, base_ind.size):
            raise ValueError(
                "Nonlinear general-family Sl block returned a penalty matrix with "
                "shape inconsistent with its coefficient span."
            )
        St[np.ix_(base_ind, base_ind)] += S_block
    return np.asarray(St, dtype=np.float64)


def _current_general_penalty_derivatives(
    model,
    smoothing_params: np.ndarray,
    layout: _GeneralPredictorLayout,
    *,
    sl_blocks: list[Any],
) -> list[np.ndarray]:
    smoothing_params = np.asarray(smoothing_params, dtype=np.float64).ravel()
    n_sp = int(smoothing_params.size)
    p_full = int(layout.X_full.shape[1])
    derivs = [np.zeros((p_full, p_full), dtype=np.float64) for _ in range(n_sp)]
    full_idx = np.asarray(layout.reduced_to_full_idx, dtype=int)
    compiled_penalties = list(_penalty_blocks_seq(model))
    used_nonlinear_penalties: set[int] = set()
    eye_p = np.eye(p_full, dtype=np.float64)

    for block in sl_blocks:
        if bool(getattr(block, "linear", True)):
            continue
        penalty_indices = tuple(int(v) for v in getattr(block, "penalty_indices", ()))
        smoothing_indices = getattr(block, "smoothing_indices", None)
        if smoothing_indices is None:
            smoothing_indices = tuple(
                int(compiled_penalties[i].smoothing_index) for i in penalty_indices
            )
        smoothing_indices = tuple(int(v) for v in smoothing_indices)
        if len(smoothing_indices) != int(
            getattr(block, "n_sp", len(smoothing_indices))
        ):
            raise ValueError(
                "Nonlinear general-family Sl block smoothing-index metadata is inconsistent "
                "with its number of smoothing parameters."
            )
        for j, sp_idx in enumerate(smoothing_indices):
            derivs[sp_idx] += np.asarray(
                general_newton._sl_mult([block], eye_p, j + 1, full=True),
                dtype=np.float64,
            )
        used_nonlinear_penalties.update(penalty_indices)

    for i, pb in enumerate(compiled_penalties):
        if i in used_nonlinear_penalties:
            continue
        sp_idx = int(pb.smoothing_index)
        idx = np.asarray(full_idx[pb.coef_slice], dtype=int)
        if idx.size == 0:
            continue
        derivs[sp_idx][np.ix_(idx, idx)] += float(
            smoothing_params[sp_idx]
        ) * np.asarray(pb.matrix, dtype=np.float64)

    return [np.asarray(block, dtype=np.float64) for block in derivs]


def _build_general_penalty_matrix(
    model, smoothing_params, layout, *, exclude_penalties: set[int] | None = None
) -> tuple[np.ndarray, list[np.ndarray]]:
    smoothing_params = np.asarray(smoothing_params, dtype=np.float64).ravel()
    p_full = layout.X_full.shape[1]
    St = np.zeros((p_full, p_full), dtype=np.float64)
    S_blocks: list[np.ndarray] = []
    excluded = (
        set() if exclude_penalties is None else {int(v) for v in exclude_penalties}
    )

    full_idx = layout.reduced_to_full_idx
    for i, pb in enumerate(_penalty_blocks_seq(model)):
        if i in excluded:
            continue
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


def build_general_family_setup_state(
    model,
    smoothing_params,
    *,
    score_type=None,
) -> GeneralFamilySetupState:
    layout = _build_general_predictor_layout(model)
    sl = build_general_penalty_setup(model, layout)
    X_initial = sl_initial_repara(sl, layout.X_full, both_sides=False)
    smoothing_params = np.asarray(smoothing_params, dtype=np.float64).ravel()
    log_sp = np.log(np.clip(smoothing_params, 1e-300, None))
    nonlinear_penalties = _nonlinear_penalty_indices(sl)
    St_linear, S_blocks = _build_general_penalty_matrix(
        model,
        smoothing_params,
        layout,
        exclude_penalties=nonlinear_penalties,
    )
    ldet_state = general_newton._sl_ldetS(
        sl,
        rho=log_sp,
        fixed=np.zeros_like(smoothing_params, dtype=bool),
        np_=layout.X_full.shape[1],
        root=False,
        Stot=False,
        deriv=2,
    )
    St = np.asarray(St_linear, dtype=np.float64)
    if _general_penalty_setup_has_nonlinear_blocks(sl):
        St = np.asarray(
            St
            + _current_nonlinear_penalty_matrix(
                list(ldet_state["Sl"]),
                n_param=int(layout.X_full.shape[1]),
            ),
            dtype=np.float64,
        )
    penalty_derivatives = _current_general_penalty_derivatives(
        model,
        smoothing_params,
        layout,
        sl_blocks=list(ldet_state["Sl"]),
    )
    score_name = _general_fit_score_type_name(
        getattr(model, "_optim_method", "REML") if score_type is None else score_type
    )

    return GeneralFamilySetupState(
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
        penalty_derivatives=[
            np.asarray(block, dtype=np.float64) for block in penalty_derivatives
        ],
        ldetS=float(ldet_state["ldetS"]),
        ldetS1=np.asarray(ldet_state["ldet1"], dtype=np.float64),
        ldetS2=np.asarray(ldet_state["ldet2"], dtype=np.float64),
        Mp=_general_penalty_null_dim(St),
        score_type=score_name,
    )


def _record_outer_derivative_mode(model, *, gradient_source=None, hessian_source=None):
    info = dict(getattr(model, "_general_family_outer_derivative_info", {}) or {})
    family = getattr(model, "family", None)
    supports_analytic = bool(
        getattr(family, "supports_analytic_outer_derivatives", False)
    )
    if gradient_source is not None:
        info["gradient_source"] = str(gradient_source)
    if hessian_source is not None:
        info["hessian_source"] = str(hessian_source)
    info["supports_analytic_outer_derivatives"] = supports_analytic
    info["penalty_logdet_source"] = "analytic"
    info["uses_exact_penalty_logdet"] = True
    info.pop("fallback_reason", None)
    model._general_family_outer_derivative_info = info


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


def _copy_optional_float_array(x):
    if x is None:
        return None
    return np.asarray(x, dtype=np.float64).copy()


def _same_optional_float_array(lhs, rhs) -> bool:
    if lhs is None or rhs is None:
        return lhs is None and rhs is None
    return np.array_equal(
        np.asarray(lhs, dtype=np.float64),
        np.asarray(rhs, dtype=np.float64),
    )


def _run_general_family_fixed_smoothing_cached(
    model,
    y,
    log_sp,
    *,
    deriv: int,
    method,
):
    log_sp = np.asarray(log_sp, dtype=np.float64).ravel()
    method_name = str(method).upper()
    weights = _copy_optional_float_array(getattr(model, "prior_weights_", None))
    cache = _fit_workspace(model).get("general_family_outer_eval_cache", None)
    if (
        isinstance(cache, dict)
        and str(cache.get("method", "")).upper() == method_name
        and int(cache.get("deriv", -1)) >= int(deriv)
        and np.array_equal(
            np.asarray(cache.get("log_sp", np.array([], dtype=np.float64))),
            log_sp,
        )
        and _same_optional_float_array(cache.get("weights", None), weights)
    ):
        return cache["run"]

    sp = expand_smoothing_params_from_log(model, log_sp)
    run = run_general_family_fixed_smoothing(
        model,
        y,
        sp,
        weights=getattr(model, "prior_weights_", None),
        deriv=int(deriv),
        score_type=method_name,
    )
    _fit_workspace(model).general_family_outer_eval_cache = {
        "method": method_name,
        "deriv": int(deriv),
        "log_sp": log_sp.copy(),
        "weights": weights,
        "run": run,
    }
    return run


def run_general_family_fixed_smoothing(
    model,
    y,
    smoothing_params,
    *,
    weights=None,
    deriv=2,
    score_type=None,
):
    setup = build_general_family_setup_state(
        model,
        smoothing_params,
        score_type=score_type,
    )
    if _general_penalty_setup_has_nonlinear_blocks(setup.Sl):
        ldetS = float(setup.ldetS)
        ldetS1 = np.asarray(setup.ldetS1, dtype=np.float64)
        ldetS2 = np.asarray(setup.ldetS2, dtype=np.float64)
    else:
        from ...selection.reparam import _stable_penalty_logdet_derivatives

        ldetS, ldetS1, ldetS2 = _stable_penalty_logdet_derivatives(
            model,
            np.asarray(smoothing_params, dtype=np.float64),
            order=2,
        )
    ctl = general_newton.GeneralNewtonControl(
        maxit=int(getattr(model, "max_irls_iter", 200)),
        epsilon=float(getattr(model, "irls_tol", 1e-7)),
        trace=bool(getattr(model, "hparams", {}).get("trace", False)),
    )
    start_coef = _fit_workspace(model).get("pirls_eval_start", None)
    if start_coef is None:
        start_coef = _fit_workspace(model).get("pirls_coef_start", None)
    if start_coef is not None:
        start_coef = np.asarray(start_coef, dtype=np.float64).copy()
    fit = general_newton.solve_general_newton_fit(
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
        start=start_coef,
    )
    coef_fit = fit.get("coef", None)
    if coef_fit is not None:
        coef_fit = np.asarray(coef_fit, dtype=np.float64).copy()
        _fit_workspace(model).pirls_last_coef = coef_fit
    offset_list = setup.offset_list
    if hasattr(model.family, "_offset_list"):
        offset_list = model.family._offset_list(offset_list)
    if coef_fit is not None and hasattr(model.family, "_stacked_eta"):
        _fit_workspace(model).pirls_last_eta = np.asarray(
            model.family._stacked_eta(
                setup.X_initial,
                setup.jj,
                coef_fit,
                offset=offset_list,
            ),
            dtype=np.float64,
        )
    if coef_fit is not None and hasattr(model.family, "predict_fitted"):
        _fit_workspace(model).pirls_last_mu = np.asarray(
            model.family.predict_fitted(
                setup.X_initial,
                setup.jj,
                coef_fit,
                offset=offset_list,
            ),
            dtype=np.float64,
        )
    return {
        "layout": setup.layout,
        "setup": setup,
        "fit": fit,
        "offset_list": setup.offset_list,
        "smoothing_params": setup.smoothing_params,
        "log_sp": setup.log_sp,
    }


def criterion_ml_reml_general_family(model, y, log_sp, method):
    run = _run_general_family_fixed_smoothing_cached(
        model,
        y,
        log_sp,
        deriv=0,
        method=method,
    )
    return float(run["fit"]["score"])


def criterion_gradient_ml_reml_general_family(model, y, log_sp, method):
    if not _supports_analytic_outer_gradient(model.family):
        raise NotImplementedError(
            "General-family ML/REML outer optimization requires analytic outer "
            "gradients for strict mgcv parity; finite-difference fallback removed."
        )
    _record_outer_derivative_mode(model, gradient_source="analytic")
    run = _run_general_family_fixed_smoothing_cached(
        model,
        y,
        log_sp,
        deriv=1,
        method=method,
    )
    grad = run["fit"].get("score1", None)
    if grad is None:
        return np.empty((0,), dtype=np.float64)
    return np.asarray(grad, dtype=np.float64)


def criterion_hessian_ml_reml_general_family(model, y, log_sp, method):
    if _supports_analytic_outer_hessian(model.family):
        _record_outer_derivative_mode(
            model,
            hessian_source="analytic",
        )
        run = _run_general_family_fixed_smoothing_cached(
            model,
            y,
            log_sp,
            deriv=2,
            method=method,
        )
        info = dict(getattr(model, "_general_family_outer_derivative_info", {}) or {})
        db_drho = run["fit"].get("db_drho", None)
        if db_drho is None:
            info.pop("db_drho", None)
        else:
            info["db_drho"] = np.asarray(db_drho, dtype=np.float64).copy()
        model._general_family_outer_derivative_info = info
        score2 = run["fit"].get("score2", None)
        if score2 is None:
            return np.empty((0, 0), dtype=np.float64)
        return np.asarray(score2, dtype=np.float64)
    raise NotImplementedError(
        "General-family ML/REML outer optimization requires analytic outer "
        "Hessians for strict mgcv parity; finite-difference fallback removed."
    )


def solve_general_family_fit(model, y, smoothing_params, weights=None):
    need_postproc_derivs = len(tuple(_penalty_blocks_seq(model))) > 0 and (
        _supports_analytic_outer_gradient(model.family)
        or _supports_analytic_outer_hessian(model.family)
    )
    deriv_order = (
        2
        if need_postproc_derivs and _supports_analytic_outer_hessian(model.family)
        else 1 if need_postproc_derivs else 0
    )
    run = run_general_family_fixed_smoothing(
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
    outer_info = None
    has_nonlinear_sl = any(
        not bool(getattr(block, "linear", True)) for block in setup.Sl
    )
    if optim_result is not None and getattr(optim_result, "hess", None) is not None:
        outer_hess = np.asarray(optim_result.hess, dtype=np.float64)
    if optim_result is not None:
        outer_info = dict(getattr(optim_result, "outer_info", {}) or {})
        outer_hess_raw = outer_info.get("hess", None)
        if outer_hess_raw is not None:
            outer_hess = np.asarray(outer_hess_raw, dtype=np.float64)

    from ...selection.reparam import build_estimate_gam_setup_state

    exact_setup = build_estimate_gam_setup_state(model)

    post = general_newton.postprocess_general_newton_fit(
        fit,
        L_map=None if exact_setup.L is None else np.asarray(exact_setup.L, dtype=np.float64),
        Sl=setup.Sl,
        lsp0=np.asarray(exact_setup.lsp0, dtype=np.float64),
        S_blocks=setup.S_blocks,
        off=None,
        outer_hess=outer_hess,
        outer_info=outer_info,
        smoothing_params=setup.smoothing_params,
        penalty_matrix=setup.St if has_nonlinear_sl else None,
        penalty_derivatives=setup.penalty_derivatives if has_nonlinear_sl else None,
        # Upstream efsud/optim run the final gam.fit5 at deriv=0, so
        # gam.fit5.post.proc's correction gate (gam.fit4.r:1648) never fires
        # for those optimizers; the local inner fit still carries REML2 /
        # db_drho, so suppress explicitly instead of relying on absent state.
        suppress_smoothing_uncertainty=(
            str(getattr(model, "smoothing_optimizer", "")).lower()
            in {"efs", "optim"}
        ),
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
    # `post["Vc"]` is kept in fit space to mirror the upstream post-proc object.
    cov_unconditional_space = PREDICTION_PARAMETER_SPACE
    # Mirror mgcv/R/mgcv.r::estimate.gam(): when G$P is present, coefficients,
    # Vp, and Ve are mapped to prediction parameterization and the raw fit-space
    # Vc is exported through the same fit-result bridge.
    #
    # `summary.gam()` / `anova.gam()` also consume the post-proc `R` from
    # `gam.fit5.post.proc()`. Keep the exact mgcv-shaped factor available on the
    # fitted model whenever the public coefficients remain in fit space.
    model._summary_R_ = None
    if prediction_parameterization_map(model) is None:
        model._summary_R_ = np.asarray(post["R"], dtype=np.float64).copy()

    beta = np.asarray(coef_full[setup.reduced_to_full_idx], dtype=np.float64)
    intercept = (
        float(coef_full[int(setup.predictor_full_slices[0][0])])
        if setup.predictor_full_slices
        and np.asarray(setup.predictor_full_slices[0]).size
        else 0.0
    )
    deviance = float(-2.0 * float(fit["l"]))

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
            "cov_unconditional_space": cov_unconditional_space,
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
