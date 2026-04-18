from __future__ import annotations

import itertools
from dataclasses import dataclass
from numbers import Number

import numpy as np
import pandas as pd
import pytest

from nampy._column_orientation import apply_column_signs, canonical_column_signs
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
from nampy.splines.cubic_basis import cr_exact_null_basis_from_knots
from nampy.splines.mrf import nat_param_type0, nat_param_type1
from nampy.splines.univariate_bases import (
    add_full_rank_shrinkage,
    cyclic_cubic_bd,
    cyclic_cubic_predict_matrix,
    pspline_knots,
)
from tests.mgcv_parity_utils import (
    _make_fs_data,
    _make_fs_data_4levels,
    _make_gaussian_data,
    _make_gaussian_data_3col,
    _make_mrf_data,
    _make_mrf_low_rank_data,
    _make_random_effect_data,
    _make_random_effect_data_noisy,
    _make_sz_data,
    _make_sz_data_3x3,
    _normalize_python_formula_text,
    _run_mgcv_raw_constructor,
)


@dataclass(frozen=True)
class RawConstructorCase:
    case_id: str
    data_factory: object
    formula: str
    atol: float = 1e-10
    knots_factory: object | None = None


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


def _make_large_univariate_data(seed=111, n=2205):
    return _make_univariate_data(seed=seed, n=n)


def _make_large_gaussian_data(seed=112, n=2205):
    return _make_gaussian_data(seed=seed, n=n)


def _make_factorized_gaussian_data(seed=101, n=96):
    df = _make_gaussian_data(seed=seed, n=n).copy()
    levels = np.array(["a", "b", "c"], dtype=object)
    df["f"] = levels[np.arange(n) % levels.size]
    return df


def _make_sz_metric2d_data(seed=102, n=54):
    df = _make_gaussian_data(seed=seed, n=n).copy()
    f1_levels = np.array(["a", "b", "c"], dtype=object)
    f2_levels = np.array(["u", "v", "w"], dtype=object)
    df["f1"] = f1_levels[np.arange(n) % f1_levels.size]
    df["f2"] = f2_levels[(np.arange(n) // f1_levels.size) % f2_levels.size]
    return df


def _make_random_effect_pair_data():
    f1 = np.array(["a", "b", "c", "a", "b", "c"], dtype=object)
    f2 = np.array(["u", "u", "u", "v", "v", "v"], dtype=object)
    y = np.array([1.0, -0.5, 0.2, 1.4, -0.1, 0.6], dtype=np.float64)
    return pd.DataFrame({"y": y, "f1": f1, "f2": f2})


def _make_random_effect_numeric_pair_data():
    return pd.DataFrame(
        {
            "y": [0.1, 0.3, 0.9, 1.2, 1.7, 2.0],
            "x0": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "x1": [0.5, 0.75, 1.5, 1.0, 1.25, 1.75],
        }
    )


def _factory(fn, **kwargs):
    return lambda fn=fn, kwargs=kwargs: fn(**kwargs)


def _equally_spaced_knots(column: str, n_knots: int):
    def _build(data):
        vals = np.asarray(data[column], dtype=np.float64)
        return {
            str(column): np.linspace(
                float(np.min(vals)),
                float(np.max(vals)),
                num=int(n_knots),
            )
        }

    return _build


def _pspline_supplied_knots(column: str, bs_dim: int, basis_order: int):
    def _build(data):
        vals = np.asarray(data[column], dtype=np.float64)
        return {
            str(column): pspline_knots(
                vals,
                bs_dim=int(bs_dim),
                basis_order=int(basis_order),
                supplied_knots=None,
            )
        }

    return _build


def _paired_feature_knots(columns, n_knots: int):
    cols = [str(col) for col in columns]

    def _build(data):
        out = {}
        for col in cols:
            vals = np.asarray(data[col], dtype=np.float64)
            out[col] = np.linspace(
                float(np.min(vals)),
                float(np.max(vals)),
                num=int(n_knots),
            )
        return out

    return _build


def _merge_knots_factories(*builders):
    def _build(data):
        out = {}
        for builder in builders:
            if builder is None:
                continue
            out.update(builder(data))
        return out

    return _build


def _feature_specific_knots(counts):
    counts = {str(key): int(value) for key, value in dict(counts).items()}

    def _build(data):
        out = {}
        for col, n_knots in counts.items():
            vals = np.asarray(data[col], dtype=np.float64)
            out[col] = np.linspace(
                float(np.min(vals)),
                float(np.max(vals)),
                num=n_knots,
            )
        return out

    return _build


def _cyclic_endpoint_knots(column: str):
    def _build(data):
        vals = np.asarray(data[column], dtype=np.float64)
        return {str(column): [float(np.min(vals)), float(np.max(vals))]}

    return _build


def _mrf_nb3():
    return {"A": ["B"], "B": ["A", "C"], "C": ["B"]}


def _mrf_nb4():
    return {"A": ["B"], "B": ["A", "C"], "C": ["B", "D"], "D": ["C"]}


def _mrf_penalty3():
    return [[1.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 1.0]]


def _mrf_polys3():
    return {
        "A": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]],
        "B": [[1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0], [1.0, 0.0]],
        "C": [[2.0, 0.0], [3.0, 0.0], [3.0, 1.0], [2.0, 1.0], [2.0, 0.0]],
    }


def _mrf_region_knots(*levels):
    return lambda _data: {"region": list(levels)}


def _tensor_case_atol(special, bases):
    basis_set = {str(b).lower() for b in bases}
    if special == "ti" and basis_set == {"gp"}:
        return 1e-4
    if special == "ti" and basis_set.intersection({"ps", "tp", "ts", "gp"}):
        return 1e-8
    if basis_set.intersection({"tp", "ts", "gp"}):
        return 1e-8
    return 1e-10


def _case(
    case_id: str,
    data_factory,
    formula: str,
    atol: float = 1e-10,
    knots_factory=None,
):
    return RawConstructorCase(
        case_id=case_id,
        data_factory=data_factory,
        formula=formula,
        atol=atol,
        knots_factory=knots_factory,
    )


def _build_cubic_case_matrix():
    cases = []
    for basis, seed_default, seed_shared, seed_knots in [
        ("cr", 31, 43, 57),
        ("cs", 32, 44, 58),
    ]:
        cases.extend(
            [
                _case(
                    f"{basis}_default_k",
                    _factory(_make_univariate_data, seed=seed_default),
                    f'y ~ s(x, bs="{basis}")',
                ),
                _case(
                    f"{basis}_k8",
                    _factory(_make_univariate_data, seed=seed_default + 100),
                    f'y ~ s(x, bs="{basis}", k=8)',
                ),
                _case(
                    f"{basis}_shared_id",
                    _factory(_make_univariate_data, seed=seed_shared),
                    f'y ~ s(x, bs="{basis}", k=8, id="shared")',
                ),
                _case(
                    f"{basis}_supplied_knots",
                    _factory(_make_univariate_data, seed=seed_knots),
                    f'y ~ s(x, bs="{basis}", k=8)',
                    knots_factory=_equally_spaced_knots("x", 8),
                ),
            ]
        )

    cases.extend(
        [
            _case(
                "cc_default_k",
                _factory(_make_cyclic_data, seed=77),
                'y ~ s(x, bs="cc")',
            ),
            _case(
                "cc_k10",
                _factory(_make_cyclic_data, seed=78),
                'y ~ s(x, bs="cc", k=10)',
            ),
            _case(
                "cc_endpoint_knots",
                _factory(_make_cyclic_data, seed=79),
                'y ~ s(x, bs="cc", k=8)',
                knots_factory=_cyclic_endpoint_knots("x"),
            ),
            _case(
                "cc_full_knots",
                _factory(_make_cyclic_data, seed=80),
                'y ~ s(x, bs="cc", k=8)',
                knots_factory=_equally_spaced_knots("x", 8),
            ),
        ]
    )
    return cases


def _build_ps_case_matrix():
    return [
        _case(
            "ps_default_k_default_m",
            _factory(_make_univariate_data, seed=33),
            'y ~ s(x, bs="ps")',
        ),
        _case(
            "ps_k9_default_m",
            _factory(_make_univariate_data, seed=34),
            'y ~ s(x, bs="ps", k=9)',
        ),
        _case(
            "ps_m_scalar_1",
            _factory(_make_univariate_data, seed=35),
            'y ~ s(x, bs="ps", k=9, m=1)',
        ),
        _case(
            "ps_m_vec_len1",
            _factory(_make_univariate_data, seed=36),
            'y ~ s(x, bs="ps", k=9, m=[2])',
        ),
        _case(
            "ps_m_balanced",
            _factory(_make_univariate_data, seed=45),
            'y ~ s(x, bs="ps", k=9, m=[2, 2])',
        ),
        _case(
            "ps_m_ordered",
            _factory(_make_univariate_data, seed=46),
            'y ~ s(x, bs="ps", k=10, m=[2, 3])',
        ),
        _case(
            "ps_shared_id",
            _factory(_make_univariate_data, seed=47),
            'y ~ s(x, bs="ps", k=10, id="shared")',
        ),
        _case(
            "ps_supplied_knots",
            _factory(_make_univariate_data, seed=48),
            'y ~ s(x, bs="ps", k=10, m=[2, 2])',
            knots_factory=_pspline_supplied_knots("x", bs_dim=10, basis_order=2),
        ),
    ]


def _build_tprs_case_matrix():
    cases = []
    for basis, seed_base in [("tp", 60), ("ts", 80)]:
        cases.extend(
            [
                _case(
                    f"{basis}_1d_default_k",
                    _factory(_make_univariate_data, seed=seed_base),
                    f'y ~ s(x, bs="{basis}")',
                    atol=1e-7,
                ),
                _case(
                    f"{basis}_1d_k9",
                    _factory(_make_univariate_data, seed=seed_base + 1),
                    f'y ~ s(x, bs="{basis}", k=9)',
                    atol=1e-7,
                ),
                _case(
                    f"{basis}_1d_shared_id",
                    _factory(_make_univariate_data, seed=seed_base + 2),
                    f'y ~ s(x, bs="{basis}", k=10, id="shared")',
                    atol=1e-7,
                ),
                _case(
                    f"{basis}_1d_supplied_knots",
                    _factory(_make_univariate_data, seed=seed_base + 3),
                    f'y ~ s(x, bs="{basis}", k=9)',
                    atol=1e-7,
                    knots_factory=_equally_spaced_knots("x", 9),
                ),
                _case(
                    f"{basis}_2d_default_m",
                    _factory(_make_gaussian_data, seed=seed_base + 4, n=120),
                    f'y ~ s(x0, x1, bs="{basis}", k=10)',
                    atol=1e-7,
                ),
                _case(
                    f"{basis}_2d_m3",
                    _factory(_make_gaussian_data, seed=seed_base + 5, n=120),
                    f'y ~ s(x0, x1, bs="{basis}", k=12, m=3)',
                    atol=1e-7,
                ),
                _case(
                    f"{basis}_2d_supplied_knots",
                    _factory(_make_gaussian_data, seed=seed_base + 6, n=120),
                    f'y ~ s(x0, x1, bs="{basis}", k=10)',
                    atol=1e-7,
                    knots_factory=_paired_feature_knots(["x0", "x1"], 12),
                ),
                _case(
                    f"{basis}_3d_basic",
                    _factory(_make_gaussian_data_3col, seed=seed_base + 7, n=90),
                    f'y ~ s(x0, x1, x2, bs="{basis}", k=15)',
                    atol=5e-7 if basis == "tp" else 1e-7,
                ),
                _case(
                    f"{basis}_max_knots_xt",
                    _factory(_make_large_gaussian_data, seed=seed_base + 8, n=2205),
                    f'y ~ s(x0, x1, bs="{basis}", k=14, xt={{"max.knots": 60, "seed": 2}})',
                    atol=1e-7,
                ),
            ]
        )
    cases.append(
        _case(
            "tp_drop_null",
            _factory(_make_gaussian_data, seed=34, n=120),
            'y ~ s(x0, x1, bs="tp", k=10, m=[2, 0])',
            atol=1e-7,
        )
    )
    return cases


def _build_gp_case_matrix():
    cases = [
        _case(
            "gp_default_k",
            _factory(_make_gp_data, seed=90),
            'y ~ s(x, bs="gp")',
        ),
        _case(
            "gp_shared_id",
            _factory(_make_gp_data, seed=91),
            'y ~ s(x, bs="gp", k=9, id="shared")',
        ),
        _case(
            "gp_supplied_knots",
            _factory(_make_gp_data, seed=92),
            'y ~ s(x, bs="gp", k=8)',
            knots_factory=_equally_spaced_knots("x", 8),
        ),
        _case(
            "gp_2d_default",
            _factory(_make_gaussian_data, seed=93, n=120),
            'y ~ s(x0, x1, bs="gp", k=12)',
        ),
        _case(
            "gp_3d_default",
            _factory(_make_gaussian_data_3col, seed=94, n=100),
            'y ~ s(x0, x1, x2, bs="gp", k=14)',
        ),
        _case(
            "gp_max_knots_xt",
            _factory(_make_large_gaussian_data, seed=95, n=2205),
            'y ~ s(x0, x1, bs="gp", k=12, xt={"max.knots": 60, "seed": 3})',
        ),
    ]
    gp_m_specs = [
        ("spherical", [1, 1.0], 1e-10),
        ("stationary_spherical", [-1, 1.0], 1e-10),
        ("powerexp", [2, 0.8, 1.2], 1e-8),
        ("stationary_powerexp", [-2, 0.6, 1.7], 1e-10),
        ("matern15", [3, 1.0], 1e-10),
        ("stationary_matern15", [-3, 1.0], 1e-10),
        ("matern25", [4, 1.0], 1e-10),
        ("stationary_matern25", [-4, 1.0], 1e-10),
        ("matern35", [5, 1.1], 1e-10),
        ("stationary_matern35", [-5, 1.1], 1e-10),
    ]
    for idx, (label, m_spec, atol) in enumerate(gp_m_specs):
        cases.append(
            _case(
                f"gp_{label}",
                _factory(_make_gp_data, seed=100 + idx),
                f'y ~ s(x, bs="gp", k=10, m={repr(m_spec)})',
                atol=atol,
            )
        )
    return cases


def _build_mrf_case_matrix():
    nb3 = _mrf_nb3()
    nb4 = _mrf_nb4()
    penalty3 = _mrf_penalty3()
    polys3 = _mrf_polys3()
    return [
        _case(
            "mrf_nb_full_rank",
            _make_mrf_data,
            f'y ~ s(region, bs="mrf", xt={repr({"nb": nb3})})',
        ),
        _case(
            "mrf_penalty_full_rank",
            _make_mrf_data,
            f'y ~ s(region, bs="mrf", xt={repr({"penalty": penalty3})})',
        ),
        _case(
            "mrf_nb_plus_penalty",
            _make_mrf_data,
            f'y ~ s(region, bs="mrf", xt={repr({"nb": nb3, "penalty": penalty3})})',
        ),
        _case(
            "mrf_polys_full_rank",
            _make_mrf_data,
            f'y ~ s(region, bs="mrf", xt={repr({"polys": polys3})})',
        ),
        _case(
            "mrf_knots_explicit",
            _make_mrf_data,
            f'y ~ s(region, bs="mrf", xt={repr({"nb": nb3})})',
            knots_factory=_mrf_region_knots("A", "B", "C"),
        ),
        _case(
            "mrf_low_rank",
            _make_mrf_low_rank_data,
            f'y ~ s(region, bs="mrf", k=3, xt={repr({"nb": nb4})})',
        ),
    ]


def _build_re_case_matrix():
    penalty_multi = {
        "S": [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[1.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 1.0]],
        ],
        "rank": [3, 2],
    }
    return [
        _case("re_factor", _make_random_effect_data, 'y ~ s(f, bs="re")'),
        _case(
            "re_factor_pair",
            _make_random_effect_pair_data,
            'y ~ s(f1, f2, bs="re")',
        ),
        _case(
            "re_numeric_factor",
            lambda: pd.DataFrame(
                {
                    "y": [0.0, 1.0, 2.0, 3.0],
                    "x": [1.0, 2.0, 3.0, 4.0],
                    "f": ["b", "a", "c", "a"],
                }
            ),
            'y ~ s(x, f, bs="re")',
        ),
        _case(
            "re_numeric_pair",
            _make_random_effect_numeric_pair_data,
            'y ~ s(x0, x1, bs="re")',
        ),
        _case(
            "re_factor_custom_xt",
            _make_random_effect_data,
            f'y ~ s(f, bs="re", xt={repr(penalty_multi)})',
        ),
        _case(
            "re_factor_noisy",
            _factory(_make_random_effect_data_noisy, seed=21, n_draws=36, sigma=0.35),
            'y ~ s(f, bs="re")',
        ),
    ]


def _build_factor_smooth_case_matrix():
    cases = [
        _case("fs_default_tp", _make_fs_data, 'y ~ s(f, x, bs="fs")'),
        _case(
            "fs_default_tp_shared_id",
            _make_fs_data,
            'y ~ s(f, x, bs="fs", id="shared")',
        ),
        _case(
            "fs_tp_4levels",
            _factory(_make_fs_data_4levels, seed=77, n=24),
            'y ~ s(f, x, bs="fs", k=8)',
        ),
        _case("sz_default", _make_sz_data, 'y ~ s(f1, f2, x, bs="sz", k=6)'),
        _case(
            "sz_shared_id",
            _make_sz_data,
            'y ~ s(f1, f2, x, bs="sz", k=6, id="shared")',
        ),
        _case(
            "sz_grid_3x3",
            _factory(_make_sz_data_3x3, seed=83),
            'y ~ s(f1, f2, x, bs="sz", k=6)',
            atol=1e-9,
        ),
        _case(
            "sz_grid_3x3_shared_id",
            _factory(_make_sz_data_3x3, seed=84),
            'y ~ s(f1, f2, x, bs="sz", k=6, id="shared")',
            atol=2e-8,
        ),
    ]
    fs_xt_cases = [
        ("cr", "cr"),
        ("cs", "cs"),
        ("cc", "cc"),
        ("ps", {"bs": "ps", "m": 2, "k": 7}),
        ("ts", "ts"),
        ("gp", "gp"),
    ]
    for label, xt_spec in fs_xt_cases:
        cases.append(
            _case(
                f"fs_base_{label}",
                _make_fs_data,
                f'y ~ s(f, x, bs="fs", xt={repr(xt_spec)})',
                atol=1e-8 if label in {"ps", "ts", "gp"} else 1e-10,
            )
        )
        cases.append(
            _case(
                f"sz_base_{label}",
                _make_sz_data,
                f'y ~ s(f1, f2, x, bs="sz", k=6, xt={repr(xt_spec)})',
                atol=1e-8 if label in {"ps", "ts", "gp"} else 1e-10,
            )
        )

    for base_bs in ["tp", "ts", "gp"]:
        cases.append(
            _case(
                f"fs_2d_base_{base_bs}",
                _factory(_make_factorized_gaussian_data, seed=120 + len(cases), n=96),
                f'y ~ s(f, x0, x1, bs="fs", xt={repr(base_bs)}, k=10)',
                atol=1e-7,
            )
        )
        cases.append(
            _case(
                f"sz_2d_base_{base_bs}",
                _factory(_make_sz_metric2d_data, seed=140 + len(cases), n=54),
                f'y ~ s(f1, f2, x0, x1, bs="sz", k=8, xt={repr(base_bs)})',
                atol=1e-7,
            )
        )

    return cases


def _build_tensor_case_matrix():
    tensor_bases = ("cr", "cs", "cc", "ps", "tp", "ts", "gp")
    cases = []

    for special in ("te", "ti", "t2"):
        for b0, b1 in itertools.product(tensor_bases, repeat=2):
            cases.append(
                _case(
                    f"{special}_2d_{b0}_{b1}",
                    _factory(_make_gaussian_data, seed=200 + len(cases), n=90),
                    f'y ~ {special}(x0, x1, bs=["{b0}", "{b1}"], k=[5, 6])',
                    atol=_tensor_case_atol(special, (b0, b1)),
                )
            )

    for special in ("te", "ti", "t2"):
        for basis in tensor_bases:
            cases.append(
                _case(
                    f"{special}_3d_{basis}",
                    _factory(_make_gaussian_data_3col, seed=500 + len(cases), n=90),
                    f'y ~ {special}(x0, x1, x2, bs=["{basis}", "{basis}", "{basis}"], k=[4, 4, 4])',
                    atol=_tensor_case_atol(special, (basis, basis, basis)),
                )
            )

    cases.extend(
        [
            _case(
                "te_default_k",
                _factory(_make_gaussian_data, seed=800, n=90),
                'y ~ te(x0, x1, bs=["cr", "cr"])',
            ),
            _case(
                "ti_default_k",
                _factory(_make_gaussian_data, seed=801, n=90),
                'y ~ ti(x0, x1, bs=["cr", "cr"])',
            ),
            _case(
                "t2_default_k",
                _factory(_make_gaussian_data, seed=802, n=90),
                'y ~ t2(x0, x1, bs=["cr", "cr"])',
            ),
            _case(
                "te_ps_ps_m",
                _factory(_make_gaussian_data, seed=803, n=90),
                'y ~ te(x0, x1, bs=["ps", "ps"], k=[6, 6], m=[[2, 2], [2, 3]])',
            ),
            _case(
                "te_tp_ts_m",
                _factory(_make_gaussian_data, seed=804, n=90),
                'y ~ te(x0, x1, bs=["tp", "ts"], k=[10, 10], m=[3, 3])',
                atol=1e-8,
            ),
            _case(
                "te_gp_gp_m",
                _factory(_make_gaussian_data, seed=805, n=90),
                'y ~ te(x0, x1, bs=["gp", "gp"], k=[8, 8], m=[[2, 0.8, 1.2], [-3, 1.0]])',
                atol=1e-8,
            ),
            _case(
                "ti_mc_true_false",
                _factory(_make_gaussian_data, seed=806, n=90),
                'y ~ ti(x0, x1, bs=["cr", "ps"], k=[5, 6], m=[2, 2], mc=[True, False])',
                atol=1e-8,
            ),
            _case(
                "ti_mc_false_true",
                _factory(_make_gaussian_data, seed=807, n=90),
                'y ~ ti(x0, x1, bs=["cr", "ps"], k=[5, 6], m=[2, 2], mc=[False, True])',
                atol=1e-8,
            ),
            _case(
                "ti_mc_false_false",
                _factory(_make_gaussian_data, seed=808, n=90),
                'y ~ ti(x0, x1, bs=["cr", "ps"], k=[5, 6], m=[2, 2], mc=[False, False])',
                atol=1e-8,
            ),
            _case(
                "t2_full_true",
                _factory(_make_gaussian_data, seed=809, n=90),
                'y ~ t2(x0, x1, bs=["cr", "cr"], k=[5, 6], full=True)',
            ),
            _case(
                "t2_ord_1",
                _factory(_make_gaussian_data, seed=810, n=90),
                'y ~ t2(x0, x1, bs=["cr", "cr"], k=[5, 6], ord=[1])',
            ),
            _case(
                "te_knots_cr_cs",
                _factory(_make_gaussian_data, seed=811, n=90),
                'y ~ te(x0, x1, bs=["cr", "cs"], k=[5, 6])',
                atol=2e-3,
                knots_factory=_feature_specific_knots({"x0": 5, "x1": 6}),
            ),
            _case(
                "ti_knots_tp_gp",
                _factory(_make_gaussian_data, seed=812, n=90),
                'y ~ ti(x0, x1, bs=["tp", "gp"], k=[8, 8])',
                atol=1e-8,
                knots_factory=_paired_feature_knots(["x0", "x1"], 9),
            ),
            _case(
                "t2_knots_ps_cc",
                _factory(_make_gaussian_data, seed=813, n=90),
                'y ~ t2(x0, x1, bs=["ps", "cc"], k=[6, 6])',
                knots_factory=_merge_knots_factories(
                    _pspline_supplied_knots("x0", bs_dim=6, basis_order=2),
                    _equally_spaced_knots("x1", 6),
                ),
            ),
        ]
    )

    return cases


CASES = [
    *_build_cubic_case_matrix(),
    *_build_ps_case_matrix(),
    *_build_tprs_case_matrix(),
    *_build_gp_case_matrix(),
    *_build_mrf_case_matrix(),
    *_build_re_case_matrix(),
    *_build_factor_smooth_case_matrix(),
    *_build_tensor_case_matrix(),
]

# Triage categories from a fixed-sp fit parity sweep against mgcv REML
# reference smoothing parameters. This separates unsupported branches from
# raw-only representation mismatches and branches that already leak into
# downstream fitted behavior.
_KNOWN_RAW_GAPS_UNSUPPORTED_BY_MGCV = {
    "tp_max_knots_xt",
    "ts_max_knots_xt",
    "gp_max_knots_xt",
    "fs_base_cs",
    "fs_base_ts",
    "fs_2d_base_ts",
}

_KNOWN_RAW_GAPS_FIXED_SP_RAW_ONLY = {
    "tp_1d_supplied_knots",
    "tp_2d_supplied_knots",
    "ts_1d_supplied_knots",
    "ts_2d_supplied_knots",
    "gp_supplied_knots",
    "gp_2d_default",
    "gp_3d_default",
    "gp_spherical",
    "gp_powerexp",
    "re_factor_pair",
    "fs_base_cc",
    "sz_base_cc",
    "fs_base_gp",
    "te_2d_ps_cr",
    "te_2d_ps_cc",
    "te_2d_tp_cr",
    "te_2d_tp_cc",
    "te_2d_ts_cr",
    "te_2d_ts_cc",
    "te_2d_gp_cr",
    "te_2d_gp_cc",
    "ti_2d_cs_cs",
    "ti_2d_cs_ps",
    "ti_2d_ps_cr",
    "ti_2d_ps_cs",
    "ti_2d_ps_cc",
    "ti_2d_tp_cr",
    "ti_2d_tp_cs",
    "ti_2d_tp_cc",
    "ti_2d_ts_cr",
    "ti_2d_ts_cs",
    "ti_2d_ts_cc",
    "ti_2d_gp_cr",
    "ti_2d_gp_cs",
    "ti_2d_gp_cc",
    "ti_3d_cs",
    "ti_knots_tp_gp",
    "t2_3d_cs",
    "te_2d_ps_cs",
    "te_2d_tp_cs",
    "te_2d_ts_cs",
    "te_2d_gp_cs",
}

_KNOWN_RAW_GAPS_FIXED_SP_BEHAVIOR = {
    "t2_2d_cs_cr",
    "t2_2d_cs_tp",
    "t2_2d_cs_gp",
    "t2_2d_ts_cs",
    "t2_2d_gp_cs",
}

KNOWN_GAP_REASONS = {
    **dict.fromkeys(
        sorted(_KNOWN_RAW_GAPS_UNSUPPORTED_BY_MGCV),
        "mgcv itself does not support fitting this branch; leave unsupported rather than porting raw constructor behavior.",
    ),
    **dict.fromkeys(
        sorted(_KNOWN_RAW_GAPS_FIXED_SP_RAW_ONLY),
        "raw constructor mismatch is present, but fixed-sp fit parity holds; representation-only or optimizer-only de-prioritized gap.",
    ),
    **dict.fromkeys(
        sorted(_KNOWN_RAW_GAPS_FIXED_SP_BEHAVIOR),
        "raw constructor mismatch already leaks into fixed-sp fit parity; priority behavior-affecting gap.",
    ),
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


def _build_runtime_term(data: pd.DataFrame, formula: str, knots=None):
    parsed = parse_gam_formula(formula)
    extracted = extract_formula_terms(parsed)
    built = build_formula_model(
        extracted, data=data, y=np.zeros(len(data)), knots=knots
    )
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
    return out


def _serialize_cubic_raw(term, X):
    basis_name = str(term.basis_name).lower()
    if basis_name in {"cr", "cs"}:
        B = np.asarray(term._spline.raw_basis, dtype=np.float64)
        S = np.asarray(term._spline.raw_penalty_unscaled, dtype=np.float64)
        if basis_name == "cs":
            S = add_full_rank_shrinkage(
                S,
                shrink=0.1,
                null_basis=cr_exact_null_basis_from_knots(term._spline.knots),
                knots=term._spline.knots,
            )
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
    fac_idx = term._factor_feature_indices[0]
    fac = as_object_1d(X[:, fac_idx])
    levels = list(term._levels)
    Ifac = factor_indicator_matrix(fac, levels)
    n_levels = len(levels)

    if (
        term._base_transform is not None
        and term._base_range_penalty_diag is not None
        and term._range_rank is not None
        and term._null_dim is not None
    ):
        Xb = np.asarray(term._base_constructor_predict_matrix(X), dtype=np.float64)
        P = np.asarray(term._base_transform, dtype=np.float64)
        Xb = Xb @ P
        r = int(term._range_rank)
        base_null = int(term._null_dim)
        d_vec = np.asarray(term._base_range_penalty_diag, dtype=np.float64)
        D = d_vec[:r].copy()
    else:
        B0 = np.asarray(base_raw["X"], dtype=np.float64)
        S0 = np.asarray(base_raw["S"][0], dtype=np.float64)
        base_rank = int(base_raw["rank"])
        base_null = int(base_raw["null_space_dim"])

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
            "XP": (
                []
                if all(xp is None for xp in marginal_np_transforms)
                else [
                    None if xp is None else np.asarray(xp, dtype=np.float64)
                    for xp in marginal_np_transforms
                ]
            ),
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


def _matrix_self_gram(matrix):
    mat = np.asarray(matrix, dtype=np.float64)
    return np.asarray(mat @ mat.T, dtype=np.float64)


def _column_space_projector(matrix):
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError("matrix must be 2D.")
    if mat.shape[1] == 0:
        return np.zeros((mat.shape[0], mat.shape[0]), dtype=np.float64)
    return np.asarray(mat @ np.linalg.pinv(mat), dtype=np.float64)


def _row_space_projector(matrix):
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError("matrix must be 2D.")
    if mat.shape[0] == 0:
        return np.zeros((mat.shape[1], mat.shape[1]), dtype=np.float64)
    return np.asarray(np.linalg.pinv(mat) @ mat, dtype=np.float64)


def _penalty_spectrum(matrix):
    mat = np.asarray(matrix, dtype=np.float64)
    sym = 0.5 * (mat + mat.T)
    return np.asarray(np.sort(np.linalg.eigvalsh(sym)), dtype=np.float64)


def _matrix_summary(matrix):
    mat = np.asarray(matrix, dtype=np.float64)
    return {
        "shape": tuple(int(v) for v in mat.shape),
        "rank": int(
            0 if mat.size == 0 or 0 in mat.shape else np.linalg.matrix_rank(mat)
        ),
    }


def _canonicalize_tprs_raw_state(state):
    state["S"] = [_penalty_spectrum(S) for S in state["S"]]
    state["X"] = _matrix_self_gram(state["X"])
    extra = state["extra"]
    extra["UZ"] = _column_space_projector(extra["UZ"])
    return state


def _canonicalize_cs_raw_state(state):
    state["S"] = [_penalty_spectrum(S) for S in state["S"]]
    return state


def _canonicalize_gp_raw_state(state):
    state["S"] = [_penalty_spectrum(S) for S in state["S"]]
    state["X"] = _matrix_self_gram(state["X"])
    extra = state["extra"]
    extra["UZ"] = _column_space_projector(extra["UZ"])
    return state


def _canonicalize_mrf_raw_state(state):
    extra = state["extra"]
    if extra["P"] is None:
        return state
    P = np.asarray(extra["P"], dtype=np.float64)
    col_signs = canonical_column_signs(P)
    extra["P"] = apply_column_signs(P, col_signs)
    state["X"] = apply_column_signs(np.asarray(state["X"], dtype=np.float64), col_signs)
    return state


def _canonicalize_fs_raw_state(state):
    state["S"] = [_penalty_spectrum(S) for S in state["S"]]
    state["X"] = _matrix_self_gram(state["X"])
    extra = state["extra"]
    extra["P"] = _column_space_projector(extra["P"])
    extra["Xb"] = _matrix_self_gram(extra["Xb"])
    return state


def _canonicalize_sz_raw_state(state):
    state["S"] = [_penalty_spectrum(S) for S in state["S"]]
    state["X"] = _matrix_self_gram(state["X"])
    extra = state["extra"]
    extra["Xb"] = _matrix_self_gram(extra["Xb"])
    return state


def _canonicalize_tensor_raw_state(state):
    state["S"] = [_penalty_spectrum(S) for S in state["S"]]
    # mgcv's tensor `np=TRUE` reparameterization only fixes the function space,
    # not a unique basis scaling for ill-conditioned marginal inverses. Compare
    # the tensor column space invariantly instead of amplifying that scaling
    # drift through `X @ X.T`.
    state["X"] = _column_space_projector(state["X"])
    extra = state["extra"]

    XP = extra.get("XP", None)
    if isinstance(XP, list):
        extra["XP"] = [None if xp is None else _row_space_projector(xp) for xp in XP]

    C = extra.get("C", None)
    if isinstance(C, np.ndarray) and C.ndim == 2:
        extra["C"] = _matrix_summary(C)

    return state


def _canonicalize_t2_raw_state(state):
    state["S"] = [_penalty_spectrum(S) for S in state["S"]]
    state["X"] = _matrix_self_gram(state["X"])
    extra = state["extra"]
    extra["P"] = [_column_space_projector(P) for P in extra["P"]]

    Cp = extra["Cp"]
    if isinstance(Cp, np.ndarray) and Cp.ndim == 2:
        extra["Cp"] = _matrix_summary(Cp)

    C = extra["C"]
    if isinstance(C, np.ndarray) and C.ndim == 2:
        extra["C"] = _matrix_summary(C)

    return state


def _canonicalize_raw_state(state):
    state = _copy_raw_value(state)
    state["S"] = _normalized_penalties(state["S"])
    class_name = str(state["class_name"])

    if class_name == "cs.smooth":
        return _canonicalize_cs_raw_state(state)
    if class_name in {"tprs.smooth", "ts.smooth"}:
        return _canonicalize_tprs_raw_state(state)
    if class_name == "gp.smooth":
        return _canonicalize_gp_raw_state(state)
    if class_name == "mrf.smooth":
        return _canonicalize_mrf_raw_state(state)
    if class_name == "fs.interaction":
        return _canonicalize_fs_raw_state(state)
    if class_name == "sz.interaction":
        return _canonicalize_sz_raw_state(state)
    if class_name == "tensor.smooth":
        return _canonicalize_tensor_raw_state(state)
    if class_name == "t2.smooth":
        return _canonicalize_t2_raw_state(state)
    return state


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
    knots = None if case.knots_factory is None else case.knots_factory(data)
    term, X, _feature_names = _build_runtime_term(data, case.formula, knots=knots)
    actual = _canonicalize_raw_state(_serialize_term_raw(term, X))
    smooth_expr = _normalize_python_formula_text(case.formula.split("~", 1)[1].strip())
    expected = _canonicalize_raw_state(
        _run_mgcv_raw_constructor(data, smooth_expr, knots=knots)
    )
    _assert_raw_state_equal(actual, expected, atol=case.atol)
