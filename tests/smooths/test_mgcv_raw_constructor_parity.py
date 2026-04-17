from __future__ import annotations

from dataclasses import dataclass
from numbers import Number

import numpy as np
import pandas as pd
import pytest

from nampy.gam.compiler.factory import instantiate_term
from nampy.gam.formula import extract_formula_terms, parse_gam_formula
from nampy.gam.penalties import tensor_product_penalties
from nampy.gam.smooths.algebra import rowwise_kronecker, t2_marginal_reparameterization
from nampy.gam.smooths.categorical.categorical_utils import (
    as_object_1d,
    factor_indicator_matrix,
)
from nampy.gam.smooths.categorical.factor_smooth import (
    FSmoothInteractionTerm,
    SZSmoothInteractionTerm,
    _block_penalty_for_group,
)
from nampy.gam.smooths.categorical.mrf import MarkovRandomFieldTerm
from nampy.gam.smooths.categorical.random_effect import RandomEffectTerm
from nampy.gam.smooths.tensor.marginals import build_tensor_product_components
from nampy.gam.smooths.tensor.t2 import TensorANOVASplineTerm
from nampy.gam.smooths.tensor.t2_basis import build_t2_basis_and_penalties
from nampy.gam.smooths.tensor.te import TensorProductSplineTerm
from nampy.gam.smooths.tensor.ti import InteractionTensorProductSplineTerm
from nampy.gam.smooths.univariate.cubic_regression import SplineTerm1D
from nampy.gam.smooths.univariate.gp import GPSmoothTerm
from nampy.gam.smooths.univariate.pspline import PSplineTerm1D
from nampy.gam.smooths.univariate.thin_plate import ThinPlateSplineTerm
from nampy.gam.specs.build import build_formula_model
from nampy.splines.mrf import nat_param_type0, nat_param_type1
from nampy.splines.univariate_bases import (
    add_full_rank_shrinkage,
    cyclic_cubic_bd,
    cyclic_cubic_predict_matrix,
)
from tests.mgcv_parity_utils import (
    _make_fs_data,
    _make_gaussian_data,
    _make_mrf_data,
    _make_mrf_low_rank_data,
    _make_random_effect_data,
    _make_sz_data,
    _normalize_python_formula_text,
    _run_mgcv_raw_constructor,
)


@dataclass(frozen=True)
class RawConstructorCase:
    case_id: str
    data_factory: object
    formula: str
    atol: float = 1e-10


def _make_univariate_data(seed=31, n=140):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.0, 2.0, size=n)
    y = np.sin(1.1 * x) + 0.2 * x**2 + rng.normal(scale=0.1, size=n)
    return pd.DataFrame({"y": y, "x": x})


def _make_cyclic_data(seed=77, n=160):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 2 * np.pi, size=n)
    y = np.sin(x) + 0.25 * np.cos(2.0 * x) + rng.normal(scale=0.1, size=n)
    return pd.DataFrame({"y": y, "x": x})


def _make_gp_data(seed=91, n=150):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3.0, 3.0, size=n)
    y = np.exp(-0.5 * x**2) + 0.35 * np.sin(x) + rng.normal(scale=0.1, size=n)
    return pd.DataFrame({"y": y, "x": x})


CASES = [
    RawConstructorCase(
        case_id="cr_basic",
        data_factory=lambda: _make_univariate_data(seed=31),
        formula='y ~ s(x, bs="cr", k=7)',
    ),
    RawConstructorCase(
        case_id="cs_basic",
        data_factory=lambda: _make_univariate_data(seed=32),
        formula='y ~ s(x, bs="cs", k=7)',
    ),
    RawConstructorCase(
        case_id="cc_basic",
        data_factory=lambda: _make_cyclic_data(seed=77),
        formula='y ~ s(x, bs="cc", k=8)',
    ),
    RawConstructorCase(
        case_id="ps_m_ordered",
        data_factory=lambda: _make_univariate_data(seed=33),
        formula='y ~ s(x, bs="ps", k=10, m=[2, 3])',
    ),
    RawConstructorCase(
        case_id="tp_drop_null",
        data_factory=lambda: _make_gaussian_data(seed=34, n=120),
        formula='y ~ s(x0, x1, bs="tp", k=10, m=[2, 0])',
    ),
    RawConstructorCase(
        case_id="ts_basic",
        data_factory=lambda: _make_gaussian_data(seed=35, n=120),
        formula='y ~ s(x0, x1, bs="ts", k=10)',
    ),
    RawConstructorCase(
        case_id="gp_stationary_powerexp",
        data_factory=lambda: _make_gp_data(seed=91),
        formula='y ~ s(x, bs="gp", k=9, m=[-2, 0.6, 1.7])',
    ),
    RawConstructorCase(
        case_id="mrf_full_rank",
        data_factory=_make_mrf_data,
        formula=(
            'y ~ s(region, bs="mrf", '
            'xt=list(nb=list(A=c("B"), B=c("A","C"), C=c("B"))))'
        ),
    ),
    RawConstructorCase(
        case_id="mrf_low_rank",
        data_factory=_make_mrf_low_rank_data,
        formula=(
            'y ~ s(region, bs="mrf", k=3, '
            'xt=list(nb=list(A=c("B"), B=c("A","C"), C=c("B","D"), D=c("C"))))'
        ),
    ),
    RawConstructorCase(
        case_id="re_factor",
        data_factory=_make_random_effect_data,
        formula='y ~ s(f, bs="re")',
    ),
    RawConstructorCase(
        case_id="re_numeric_factor",
        data_factory=lambda: pd.DataFrame(
            {
                "y": [0.0, 1.0, 2.0, 3.0],
                "x": [1.0, 2.0, 3.0, 4.0],
                "f": ["b", "a", "c", "a"],
            }
        ),
        formula='y ~ s(x, f, bs="re")',
    ),
    RawConstructorCase(
        case_id="fs_default_tp",
        data_factory=_make_fs_data,
        formula='y ~ s(f, x, bs="fs")',
    ),
    RawConstructorCase(
        case_id="fs_ps_xt",
        data_factory=_make_fs_data,
        formula='y ~ s(f, x, bs="fs", xt=list(bs="ps", m=2, k=7))',
    ),
    RawConstructorCase(
        case_id="sz_default",
        data_factory=_make_sz_data,
        formula='y ~ s(f1, f2, x, bs="sz", k=6)',
    ),
    RawConstructorCase(
        case_id="sz_shared_id",
        data_factory=_make_sz_data,
        formula='y ~ s(f1, f2, x, bs="sz", k=6, id="shared")',
    ),
    RawConstructorCase(
        case_id="te_cr",
        data_factory=lambda: _make_gaussian_data(seed=36, n=90),
        formula='y ~ te(x0, x1, bs=["cr", "cr"], k=[5, 6])',
    ),
    RawConstructorCase(
        case_id="te_ps_m",
        data_factory=lambda: _make_gaussian_data(seed=37, n=90),
        formula='y ~ te(x0, x1, bs=["ps", "ps"], k=[6, 7], m=[1, 3])',
    ),
    RawConstructorCase(
        case_id="ti_cr",
        data_factory=lambda: _make_gaussian_data(seed=38, n=90),
        formula='y ~ ti(x0, x1, bs=["cr", "cr"], k=[5, 6])',
    ),
    RawConstructorCase(
        case_id="ti_custom_mc",
        data_factory=lambda: _make_gaussian_data(seed=39, n=90),
        formula='y ~ ti(x0, x1, bs=["cr", "ps"], k=[5, 6], m=[2, 2], mc=[True, False])',
    ),
    RawConstructorCase(
        case_id="t2_default",
        data_factory=lambda: _make_gaussian_data(seed=40, n=90),
        formula='y ~ t2(x0, x1, bs=["cr", "cr"], k=[5, 6])',
    ),
    RawConstructorCase(
        case_id="t2_full",
        data_factory=lambda: _make_gaussian_data(seed=41, n=90),
        formula='y ~ t2(x0, x1, bs=["cr", "cr"], k=[5, 6], full=True)',
    ),
    RawConstructorCase(
        case_id="t2_ord",
        data_factory=lambda: _make_gaussian_data(seed=42, n=90),
        formula='y ~ t2(x0, x1, bs=["cr", "cr"], k=[5, 6], ord=[1])',
    ),
]

KNOWN_GAP_REASONS = {
    "tp_drop_null": "Thin-plate raw constructor still differs from mgcv in Lanczos/LAPACK-dependent column orientation.",
    "ts_basic": "Shrinkage thin-plate raw constructor still differs from mgcv in Lanczos/LAPACK-dependent column orientation.",
    "gp_stationary_powerexp": "GP raw constructor still differs from mgcv in Lanczos/LAPACK-dependent column orientation.",
    "fs_ps_xt": "fs raw constructor still differs from mgcv in nat.param(type=1) column orientation for ps marginals.",
    "t2_default": "t2 raw constructor still differs from mgcv in raw nat.param(type=3) marginal/block orientation.",
    "t2_full": "t2(full=TRUE) raw constructor still differs from mgcv in raw nat.param(type=3) marginal/block orientation.",
    "t2_ord": "t2(ord=...) raw constructor still differs from mgcv in raw nat.param(type=3) marginal/block orientation.",
}

CASE_PARAMS = [
    pytest.param(
        case,
        id=case.case_id,
        marks=(
            [
                pytest.mark.status_known_gap,
                pytest.mark.xfail(
                    strict=True,
                    reason=KNOWN_GAP_REASONS[case.case_id],
                ),
            ]
            if case.case_id in KNOWN_GAP_REASONS
            else []
        ),
    )
    for case in CASES
]


def _scalar_or_list(values):
    vals = list(values)
    if len(vals) == 1:
        return vals[0]
    return vals


def _sym_rank(S: np.ndarray) -> int:
    S = np.asarray(S, dtype=np.float64)
    if S.size == 0:
        return 0
    ev = np.linalg.eigvalsh(0.5 * (S + S.T))
    tol = np.finfo(np.float64).eps ** 0.8 * max(float(np.max(ev)), 1.0)
    return int(np.sum(ev > tol))


def _common_raw_state(class_name, X, penalties, *, rank, null_space_dim, extra):
    return {
        "class_name": class_name,
        "X": np.asarray(X, dtype=np.float64),
        "S": [np.asarray(S, dtype=np.float64) for S in penalties],
        "rank": rank,
        "null_space_dim": int(null_space_dim),
        "extra": extra,
    }


def _build_runtime_term(data: pd.DataFrame, formula: str):
    parsed = parse_gam_formula(formula)
    extracted = extract_formula_terms(parsed)
    built = build_formula_model(extracted, data=data, y=np.zeros(len(data)))
    predictor = built.predictor_specs[0]
    assert len(predictor.terms) == 1
    term = instantiate_term(predictor.terms[0])
    term.fit(built.X, built.feature_names)
    return term, built.X, built.feature_names


def _serialize_base_summary(base_term, X):
    raw = _serialize_term_raw(base_term, X)
    names = list(base_term.resolved_feature_names_list())
    out = {
        "class_name": raw["class_name"],
        "bs_dim": int(raw["X"].shape[1]),
        "rank": raw["rank"],
        "null_space_dim": raw["null_space_dim"],
        "term": _scalar_or_list(names),
    }
    if len(names) > 1:
        out["dim"] = len(names)
    return out


def _serialize_cubic_raw(term, X):
    basis_name = str(term.basis_name).lower()
    if basis_name in {"cr", "cs"}:
        B = np.asarray(term._spline.raw_basis, dtype=np.float64)
        S = np.asarray(term._spline.raw_penalty_unscaled, dtype=np.float64)
        if basis_name == "cs":
            S = add_full_rank_shrinkage(S, shrink=0.1)
            rank = int(B.shape[1])
            null_dim = 0
            class_name = "cs.smooth"
        else:
            rank = int(B.shape[1] - 2)
            null_dim = 2
            class_name = "cr.smooth"
        return _common_raw_state(
            class_name,
            B,
            [S],
            rank=rank,
            null_space_dim=null_dim,
            extra={
                "xp": np.asarray(term._spline.knots, dtype=np.float64),
                "F": np.asarray(term._spline.F, dtype=np.float64).reshape(-1),
                "noterp": True,
            },
        )

    x = np.asarray(X[:, term._feature_index], dtype=np.float64).ravel()
    B = cyclic_cubic_predict_matrix(x, term._cc_knots, term._cc_bd)
    BD, _, D = cyclic_cubic_bd(term._cc_knots)
    S = 0.5 * (D.T @ BD + (D.T @ BD).T)
    return _common_raw_state(
        "cyclic.smooth",
        B,
        [S],
        rank=int(B.shape[1] - 1),
        null_space_dim=1,
        extra={
            "xp": np.asarray(term._cc_knots, dtype=np.float64),
            "BD": np.asarray(term._cc_bd, dtype=np.float64),
            "noterp": True,
        },
    )


def _serialize_ps_raw(term):
    setup = term._setup
    B = np.asarray(setup.basis_train, dtype=np.float64)
    return _common_raw_state(
        "pspline.smooth",
        B,
        [np.asarray(setup.penalty, dtype=np.float64)],
        rank=int(setup.rank),
        null_space_dim=int(B.shape[1] - setup.rank),
        extra={
            "knots": np.asarray(setup.knots, dtype=np.float64),
            "m": _scalar_or_list([int(setup.basis_order), int(setup.penalty_order)]),
        },
    )


def _serialize_tprs_raw(term):
    setup = term._setup
    B = np.asarray(setup.basis_train, dtype=np.float64)
    class_name = "ts.smooth" if str(term.basis_name).lower() == "ts" else "tprs.smooth"
    return _common_raw_state(
        class_name,
        B,
        [np.asarray(setup.penalty, dtype=np.float64)],
        rank=int(setup.rank),
        null_space_dim=int(B.shape[1] - setup.rank),
        extra={
            "Xu": np.asarray(setup.Xu, dtype=np.float64),
            "UZ": np.asarray(setup.UZ, dtype=np.float64),
            "shift": np.asarray(setup.shift, dtype=np.float64),
            "drop_null": bool(setup.drop_null_requested),
        },
    )


def _serialize_gp_raw(term):
    setup = term._setup
    return _common_raw_state(
        "gp.smooth",
        np.asarray(setup.basis_train, dtype=np.float64),
        [np.asarray(setup.penalty, dtype=np.float64)],
        rank=int(setup.rank),
        null_space_dim=int(setup.null_space_dim),
        extra={
            "shift": np.asarray(setup.shift, dtype=np.float64),
            "gp_defn": {
                "type": int(setup.gp_defn["type"]),
                "stationary": bool(setup.gp_defn["stationary"]),
                "rho": float(setup.gp_defn["rho"]),
                "power": float(setup.gp_defn["power"]),
            },
            "UZ": np.asarray(setup.UZ, dtype=np.float64),
            "knt": np.asarray(setup.knt, dtype=np.float64),
        },
    )


def _serialize_mrf_raw(term, X):
    x = as_object_1d(X[:, term._feature_index])
    area_names = list(term._area_names)
    X_full = factor_indicator_matrix(x, area_names)
    full_penalty = np.asarray(term._full_penalty, dtype=np.float64)
    n_areas = len(area_names)
    bs_dim = n_areas if term.k < 0 else int(term.k)

    if bs_dim < n_areas:
        miss = np.where(np.sum(X_full, axis=0) == 0.0)[0]
        X_aug = X_full
        if miss.size > 0:
            X_aug = np.vstack([np.zeros((miss.size, n_areas), dtype=np.float64), X_aug])
            for i, j in enumerate(miss):
                X_aug[i, j] = 1.0
        rp = nat_param_type0(X_aug, full_penalty, rank=None, tol=None, unit_fnorm=True)
        ind = np.arange(n_areas - bs_dim, n_areas, dtype=int)
        B = np.asarray(rp["X"][miss.size :, :][:, ind], dtype=np.float64)
        P = np.asarray(rp["P"][:, ind], dtype=np.float64)
        D_red = np.zeros(bs_dim, dtype=np.float64)
        rank_full = int(rp["rank"])
        penalized = ind[ind < rank_full]
        if penalized.size > 0:
            D_red[np.where(ind < rank_full)[0]] = rp["D"][penalized]
        S = np.diag(D_red)
    else:
        B = np.asarray(X_full, dtype=np.float64)
        S = np.asarray(full_penalty, dtype=np.float64)
        P = None

    rank = _sym_rank(S)
    return _common_raw_state(
        "mrf.smooth",
        B,
        [S],
        rank=rank,
        null_space_dim=int(B.shape[1] - rank),
        extra={
            "knots": _scalar_or_list(area_names),
            "P": P,
            "plot_me": bool(term._plot_polys is not None),
            "te_ok": 2,
            "noterp": True,
        },
    )


def _serialize_re_raw(term):
    B = np.asarray(term._basis_train, dtype=np.float64)
    q = int(B.shape[1])
    if term.xt is None or term.xt.get("S", None) is None:
        penalties = [np.eye(q, dtype=np.float64)]
        rank = int(q)
    else:
        S_in = term.xt["S"]
        S_list = S_in if isinstance(S_in, list) else [S_in]
        penalties = [
            0.5 * (np.asarray(S, dtype=np.float64) + np.asarray(S, dtype=np.float64).T)
            for S in S_list
        ]
        ranks = [
            int(v) for v in np.asarray(term.xt["rank"], dtype=int).ravel().tolist()
        ]
        rank = _scalar_or_list(ranks)
    return _common_raw_state(
        "random.effect",
        B,
        penalties,
        rank=rank,
        null_space_dim=0,
        extra={
            "C": np.zeros((0, q), dtype=np.float64),
            "random": True,
            "noterp": True,
        },
    )


def _serialize_fs_raw(term, X):
    base_term = term._base_term
    base_raw = _serialize_term_raw(base_term, X)
    B0 = np.asarray(base_raw["X"], dtype=np.float64)
    S0 = np.asarray(base_raw["S"][0], dtype=np.float64)
    base_rank = int(base_raw["rank"])
    base_null = int(base_raw["null_space_dim"])

    fac_idx = term._factor_feature_indices[0]
    fac = as_object_1d(X[:, fac_idx])
    levels = list(term._levels)
    Ifac = factor_indicator_matrix(fac, levels)
    n_levels = len(levels)

    rp = nat_param_type1(B0, S0, rank=base_rank, unit_fnorm=True)
    Xb = np.asarray(rp["X"], dtype=np.float64)
    P = np.asarray(rp["P"], dtype=np.float64)
    r = int(rp["rank"])
    D = np.asarray(rp["D"], dtype=np.float64)

    X_full = rowwise_kronecker([Ifac, Xb])
    d_vec = np.concatenate([D, np.zeros(base_null, dtype=np.float64)])
    penalties = [np.diag(np.tile(d_vec, n_levels))]
    for i in range(base_null):
        um = np.zeros(Xb.shape[1], dtype=np.float64)
        um[r + i] = 1.0
        penalties.append(np.diag(np.tile(um, n_levels)))

    ranks = [int(n_levels * r)] + [int(n_levels)] * base_null
    return _common_raw_state(
        "fs.interaction",
        X_full,
        penalties,
        rank=_scalar_or_list(ranks),
        null_space_dim=0,
        extra={
            "base": _serialize_base_summary(base_term, X),
            "P": P,
            "fterm": str(term._factor_feature_names[0]),
            "flev": _scalar_or_list(levels),
            "Xb": Xb,
            "C": np.zeros((0, X_full.shape[1]), dtype=np.float64),
            "te_ok": 0,
            "side_constrain": False,
        },
    )


def _serialize_sz_raw(term, X):
    base_term = term._base_term
    base_raw = _serialize_term_raw(base_term, X)
    base_summary = _serialize_base_summary(base_term, X)
    base_summary["dim"] = len(base_term.resolved_feature_names_list())
    B0 = np.asarray(base_raw["X"], dtype=np.float64)
    S0 = np.asarray(base_raw["S"][0], dtype=np.float64)
    base_rank = int(base_raw["rank"])
    base_null = int(base_raw["null_space_dim"])

    indicator_mats = []
    level_sizes = []
    for idx, lev in zip(term._factor_feature_indices, term._factor_levels):
        indicator_mats.append(factor_indicator_matrix(as_object_1d(X[:, idx]), lev))
        level_sizes.append(len(lev))

    X_raw = rowwise_kronecker(indicator_mats + [B0])

    n_groups = int(np.prod(level_sizes, dtype=np.int64))
    if term.smoothing_id is None:
        penalties = [_block_penalty_for_group(g, n_groups, S0) for g in range(n_groups)]
        rank = _scalar_or_list([base_rank] * n_groups)
    else:
        P_sum = np.zeros(
            (n_groups * B0.shape[1], n_groups * B0.shape[1]), dtype=np.float64
        )
        for g in range(n_groups):
            P_sum += _block_penalty_for_group(g, n_groups, S0)
        penalties = [P_sum]
        effective_groups = int(np.prod(np.asarray(level_sizes) - 1, dtype=np.int64))
        rank = int(base_rank * effective_groups)

    null_dim = int(base_null * np.prod(np.asarray(level_sizes) - 1, dtype=np.int64))
    return _common_raw_state(
        "sz.interaction",
        X_raw,
        penalties,
        rank=rank,
        null_space_dim=null_dim,
        extra={
            "base": base_summary,
            "fterm": _scalar_or_list([str(v) for v in term._factor_feature_names]),
            "flev": [[str(v) for v in lev] for lev in term._factor_levels],
            "Xb": B0,
            "C": _scalar_or_list([0] + level_sizes),
            "te_ok": 0,
            "side_constrain": False,
        },
    )


def _serialize_te_raw(term, X):
    n_marg = len(term._marginals)
    _, marginal_penalties, marginal_np_transforms, basis_dims, B_raw, _ = (
        build_tensor_product_components(
            term._marginals,
            X,
            use_centered=[False] * n_marg,
            apply_np=True,
        )
    )
    penalties = tensor_product_penalties(marginal_penalties, basis_dims=basis_dims)
    marginal_ranks = [int(_sym_rank(S)) for S in marginal_penalties]
    marginal_null = [int(d - r) for d, r in zip(basis_dims, marginal_ranks)]
    total_dim = int(np.prod(basis_dims, dtype=np.int64))
    ranks = [int(total_dim * r // d) for r, d in zip(marginal_ranks, basis_dims)]
    null_dim = int(np.prod(marginal_null, dtype=np.int64))
    return _common_raw_state(
        "tensor.smooth",
        B_raw,
        penalties,
        rank=_scalar_or_list(ranks),
        null_space_dim=null_dim,
        extra={
            "mc": [False] * n_marg,
            "XP": [
                np.asarray(xp, dtype=np.float64)
                for xp in marginal_np_transforms
                if xp is not None
            ],
            "C": None,
        },
    )


def _serialize_ti_raw(term, X):
    use_centered = list(term._marginal_is_centered)
    _, marginal_penalties, marginal_np_transforms, basis_dims, B_raw, _ = (
        build_tensor_product_components(
            term._marginals,
            X,
            use_centered=use_centered,
            apply_np=True,
        )
    )
    penalties = tensor_product_penalties(marginal_penalties, basis_dims=basis_dims)
    marginal_ranks = [int(_sym_rank(S)) for S in marginal_penalties]
    marginal_null = [int(d - r) for d, r in zip(basis_dims, marginal_ranks)]
    total_dim = int(np.prod(basis_dims, dtype=np.int64))
    ranks = [int(total_dim * r // d) for r, d in zip(marginal_ranks, basis_dims)]
    null_dim = int(np.prod(marginal_null, dtype=np.int64))
    return _common_raw_state(
        "tensor.smooth",
        B_raw,
        penalties,
        rank=_scalar_or_list(ranks),
        null_space_dim=null_dim,
        extra={
            "mc": [bool(v) for v in term._mc],
            "XP": (
                []
                if all(xp is None for xp in marginal_np_transforms)
                else [
                    None if xp is None else np.asarray(xp, dtype=np.float64)
                    for xp in marginal_np_transforms
                ]
            ),
            "C": np.zeros((0, 0), dtype=np.float64),
        },
    )


def _serialize_t2_raw(term):
    marginal_decompositions = []
    P_list = []
    for basis_name, marginal in zip(term.basis, term._marginals):
        B_i, S_i, _ = marginal.tensor_marginal_fit_matrices(centered=False)
        dec = t2_marginal_reparameterization(B_i, S_i, basis_name=basis_name)
        marginal_decompositions.append(dec)
        P_list.append(
            np.column_stack(
                [
                    np.asarray(dec["T_range"], dtype=np.float64),
                    np.asarray(dec["T_null"], dtype=np.float64),
                ]
            )
        )

    t2_obj = build_t2_basis_and_penalties(
        marginal_decompositions,
        full=bool(term.full),
        ord=term.ord,
        remove_constant_from_null_block=False,
    )
    B = np.asarray(t2_obj["basis"], dtype=np.float64)
    penalties = [np.asarray(S, dtype=np.float64) for S in t2_obj["penalties"]]
    ranks = [int(np.linalg.matrix_rank(S)) for S in penalties]
    nup = int(sum(ranks))
    null_dim = int(B.shape[1] - nup)
    if null_dim == 0:
        C = np.zeros((0, 0), dtype=np.float64)
        Cp = None
    elif null_dim == 1:
        C = int(B.shape[1])
        Cp = np.sum(B, axis=0, keepdims=True)
    else:
        C = np.zeros((1, B.shape[1]), dtype=np.float64)
        C[0, nup:] = np.sum(B[:, nup:], axis=0)
        Cp = np.sum(B, axis=0, keepdims=True)

    ord_value = None
    if term.ord is not None:
        ord_vals = [int(v) for v in np.asarray(term.ord).ravel().tolist()]
        ord_value = _scalar_or_list(ord_vals)

    labels = [str(spec["label"]) for spec in t2_obj["penalized_specs"]]
    return _common_raw_state(
        "t2.smooth",
        B,
        penalties,
        rank=_scalar_or_list(ranks),
        null_space_dim=null_dim,
        extra={
            "full": bool(term.full),
            "ord": ord_value,
            "C": C,
            "Cp": None if Cp is None else np.asarray(Cp, dtype=np.float64),
            "P": [np.asarray(P, dtype=np.float64) for P in P_list],
            "penalty_labels": _scalar_or_list(labels),
        },
    )


def _serialize_term_raw(term, X):
    if isinstance(term, SplineTerm1D):
        return _serialize_cubic_raw(term, X)
    if isinstance(term, PSplineTerm1D):
        return _serialize_ps_raw(term)
    if isinstance(term, ThinPlateSplineTerm):
        return _serialize_tprs_raw(term)
    if isinstance(term, GPSmoothTerm):
        return _serialize_gp_raw(term)
    if isinstance(term, MarkovRandomFieldTerm):
        return _serialize_mrf_raw(term, X)
    if isinstance(term, RandomEffectTerm):
        return _serialize_re_raw(term)
    if isinstance(term, FSmoothInteractionTerm):
        return _serialize_fs_raw(term, X)
    if isinstance(term, SZSmoothInteractionTerm):
        return _serialize_sz_raw(term, X)
    if isinstance(term, TensorProductSplineTerm):
        return _serialize_te_raw(term, X)
    if isinstance(term, InteractionTensorProductSplineTerm):
        return _serialize_ti_raw(term, X)
    if isinstance(term, TensorANOVASplineTerm):
        return _serialize_t2_raw(term)
    raise TypeError(f"Unsupported runtime term type {type(term).__name__}.")


def _assert_raw_state_equal(actual, expected, *, atol, path="state"):
    actual_numeric = None
    expected_numeric = None
    if isinstance(actual, np.ndarray):
        actual_numeric = np.asarray(actual, dtype=np.float64)
    elif isinstance(actual, (list, tuple)) and all(
        isinstance(v, Number) and not isinstance(v, bool) for v in actual
    ):
        actual_numeric = np.asarray(actual, dtype=np.float64)

    if isinstance(expected, np.ndarray):
        expected_numeric = np.asarray(expected, dtype=np.float64)
    elif isinstance(expected, (list, tuple)) and all(
        isinstance(v, Number) and not isinstance(v, bool) for v in expected
    ):
        expected_numeric = np.asarray(expected, dtype=np.float64)

    if actual_numeric is not None and expected_numeric is not None:
        np.testing.assert_allclose(
            actual_numeric,
            expected_numeric,
            atol=atol,
            rtol=0.0,
            err_msg=path,
        )
        return

    if isinstance(expected, np.ndarray):
        np.testing.assert_allclose(
            np.asarray(actual, dtype=np.float64),
            expected,
            atol=atol,
            rtol=0.0,
            err_msg=path,
        )
        return

    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"{path}: expected dict, got {type(actual)}"
        assert set(actual) == set(
            expected
        ), f"{path}: key mismatch {set(actual)} != {set(expected)}"
        for key in expected:
            _assert_raw_state_equal(
                actual[key], expected[key], atol=atol, path=f"{path}.{key}"
            )
        return

    if isinstance(expected, list):
        assert isinstance(actual, list), f"{path}: expected list, got {type(actual)}"
        assert len(actual) == len(
            expected
        ), f"{path}: length mismatch {len(actual)} != {len(expected)}"
        for idx, (got, want) in enumerate(zip(actual, expected)):
            _assert_raw_state_equal(got, want, atol=atol, path=f"{path}[{idx}]")
        return

    if expected is None:
        assert actual is None, f"{path}: expected None, got {actual!r}"
        return

    if isinstance(expected, Number) and not isinstance(expected, bool):
        assert actual == pytest.approx(expected, abs=atol, rel=0.0), path
        return

    assert actual == expected, f"{path}: {actual!r} != {expected!r}"


@pytest.mark.parametrize("case", CASE_PARAMS)
def test_raw_constructor_matches_mgcv(case: RawConstructorCase):
    data = case.data_factory()
    term, X, _feature_names = _build_runtime_term(data, case.formula)
    actual = _serialize_term_raw(term, X)
    smooth_expr = _normalize_python_formula_text(case.formula.split("~", 1)[1].strip())
    expected = _run_mgcv_raw_constructor(data, smooth_expr)
    _assert_raw_state_equal(actual, expected, atol=case.atol)
