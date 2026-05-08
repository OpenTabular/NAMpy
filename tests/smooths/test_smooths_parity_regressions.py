from __future__ import annotations

import numpy as np
import pandas as pd

from nampy.gam.compiler.factory import instantiate_term
from nampy.gam.formula import extract_formula_terms, parse_gam_formula
from nampy.gam.smooths.categorical.fs import FSmoothInteractionTerm
from nampy.gam.smooths.categorical.re import RandomEffectTerm
from nampy.gam.smooths.univariate.tp import ThinPlateSplineTerm
from nampy.gam.specs.build import build_formula_model


def _numeric_tensor_data(n=80, seed=101):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-1.0, 1.0, n)
    x1 = rng.uniform(-0.5, 0.7, n)
    x2 = rng.uniform(-2.0, 2.0, n)
    z = 0.5 + rng.uniform(0.2, 1.0, n)
    y = np.sin(x0) + x1 * x2 + 0.1 * rng.normal(size=n)
    return pd.DataFrame({"y": y, "x0": x0, "x1": x1, "x2": x2, "z": z})


def test_categorical_metadata_preserves_unused_factor_levels_for_re_and_fs():
    levels = ["b", "a", "unused"]
    meta = {"factor_levels_by_feature": {"f": {"levels": levels, "ordered": False}}}
    X = np.asarray(
        np.column_stack(
            [
                np.linspace(0.0, 1.0, 12),
                np.array(
                    ["b", "a", "b", "a", "b", "a", "b", "a", "b", "a", "b", "a"],
                    dtype=object,
                ),
            ]
        ),
        dtype=object,
    )

    re = RandomEffectTerm(feature="f", metadata=meta)
    re.fit(X, ["x", "f"])
    assert re.basis_train.shape[1] == 3
    assert re._component_specs[0].levels == levels

    fs = FSmoothInteractionTerm(feature=["x", "f"], k=5, xt="cr", metadata=meta)
    fs.fit(X, ["x", "f"])
    assert fs._levels == levels
    assert fs.basis_train.shape[1] == 3 * fs._range_rank + 3 * fs._null_dim


def test_tensor_d_groups_multivariate_margin_and_coerces_cr_to_tp():
    data = _numeric_tensor_data(n=70)
    formula = 'y ~ te(x0, x1, x2, d=[2, 1], bs=["cr", "cr"], k=[6, 5])'
    extracted = extract_formula_terms(parse_gam_formula(formula))
    built = build_formula_model(extracted, data)
    spec = built.predictor_specs[0].terms[0]

    term = instantiate_term(spec)
    term.fit(built.X, built.feature_names)

    assert term.basis == ["tp", "cr"]
    assert term._marginals[0].resolved_feature_names_list() == ["x0", "x1"]
    assert term._marginals[1].resolved_feature_names_list() == ["x2"]


def test_tp_low_order_m_resets_to_mgcv_default_instead_of_error():
    data = _numeric_tensor_data(n=50)
    term = ThinPlateSplineTerm(feature=["x0", "x1"], k=8, basis="tp", m=1)
    term.fit(data[["x0", "x1"]].to_numpy(dtype=np.float64), ["x0", "x1"])
    assert term._setup.penalty_order == 2
