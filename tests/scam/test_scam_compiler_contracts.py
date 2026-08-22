"""Compiler ownership contracts for shape-constrained smooth terms."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from nampy.gam.compiler.compile_model import compile_model
from nampy.gam.specs.modeling import prepare_formula_inputs
from nampy.gam.splines.shape import build_bivariate_shape_setup
from tests.scam.scam_reference_utils import (
    run_scam_linear_functional_constructor,
    run_scam_raw_constructor,
)

_BY_BASIS_CODES = {
    "mpiby": "mpiBy",
    "mpdby": "mpdBy",
    "mdcvby": "mdcvBy",
    "mdcxby": "mdcxBy",
    "micvby": "micvBy",
    "micxby": "micxBy",
    "cvby": "cvBy",
    "cxby": "cxBy",
}


@pytest.mark.parametrize(
    "basis_code",
    [
        "mpi",
        "mpd",
        "mdcv",
        "mdcx",
        "micv",
        "micx",
        "cv",
        "cx",
        "po",
        "dpo",
        "ipo",
        "miso",
        "mifo",
        "cpop",
    ],
)
def test_shape_term_compilation_matches_scam_smoothcon_and_assembles_mask(basis_code):
    rng = np.random.default_rng(447)
    x = rng.uniform(-1.8, 2.6, size=72)
    data = pd.DataFrame({"y": np.sin(x), "x": x})
    model_like = SimpleNamespace(k=10, basis="tp", select=False)
    (
        _parsed,
        predictor_specs,
        X,
        feature_names,
        _response,
        _used_columns,
        _offsets,
        _preprocess_state,
    ) = prepare_formula_inputs(
        model_like,
        data=data,
        formula=f'y ~ s(x, bs="{basis_code}", k=8, m=2)',
        y=np.asarray(data["y"], dtype=np.float64),
    )
    compiled = compile_model(
        X,
        feature_names,
        predictor_specs,
        fit_intercept=True,
    )
    expected = run_scam_raw_constructor(
        data[["x"]],
        f"s(x, bs='{basis_code}', k=8, m=2)",
        smoothcon=True,
    )

    assert len(compiled.compiled_terms) == 1
    term = compiled.compiled_terms[0]
    assert term.basis_name == basis_code
    assert term.term_type == "shape_constrained_smooth"
    np.testing.assert_allclose(
        compiled.design_matrix, expected["X"], rtol=0.0, atol=3e-14
    )
    np.testing.assert_allclose(
        compiled.compiled_penalties[0].matrix,
        expected["S"][0],
        rtol=0.0,
        atol=3e-14,
    )
    np.testing.assert_array_equal(
        term.positive_coefficient_mask, expected["p_ident"]
    )
    np.testing.assert_array_equal(
        compiled.positive_coefficient_mask,
        np.concatenate(
            [np.zeros(1, dtype=bool), np.asarray(expected["p_ident"], dtype=bool)]
        ),
    )
    assert term.side_condition_policy.skip_centering is True


def test_shape_basis_rejects_ordinary_by_until_matching_by_constructor_exists():
    data = pd.DataFrame(
        {
            "y": np.linspace(0.0, 1.0, 30),
            "x": np.linspace(-1.0, 1.0, 30),
            "z": np.linspace(0.5, 1.5, 30),
        }
    )
    model_like = SimpleNamespace(k=10, basis="tp", select=False)
    inputs = prepare_formula_inputs(
        model_like,
        data=data,
        formula='y ~ s(x, by=z, bs="mpi", k=7)',
        y=np.asarray(data["y"], dtype=np.float64),
    )
    with pytest.raises(NotImplementedError, match="mpiBy"):
        compile_model(
            inputs[2],
            inputs[3],
            inputs[1],
            fit_intercept=True,
        )


@pytest.mark.parametrize("basis_code,upstream_code", _BY_BASIS_CODES.items())
def test_numeric_by_shape_compilation_matches_scam(
    basis_code, upstream_code
):
    rng = np.random.default_rng(684)
    x = rng.uniform(-1.7, 2.9, size=79)
    z = rng.uniform(-2.0, 2.5, size=x.size)
    data = pd.DataFrame({"y": np.sin(x) * z, "x": x, "z": z})
    formula = f'y ~ s(x, by=z, bs="{basis_code}", k=8, m=2)'
    model_like = SimpleNamespace(k=10, basis="tp", select=False)
    inputs = prepare_formula_inputs(
        model_like,
        data=data,
        formula=formula,
        y=np.asarray(data["y"], dtype=np.float64),
    )
    compiled = compile_model(
        inputs[2], inputs[3], inputs[1], fit_intercept=True
    )
    expected = run_scam_raw_constructor(
        data[["x", "z"]],
        f"s(x, by=z, bs='{upstream_code}', k=8, m=2)",
        smoothcon=True,
    )

    term = compiled.compiled_terms[0]
    np.testing.assert_allclose(
        compiled.design_matrix, expected["X"], rtol=0.0, atol=3e-14
    )
    np.testing.assert_allclose(
        compiled.compiled_penalties[0].matrix,
        expected["S"][0],
        rtol=0.0,
        atol=3e-14,
    )
    np.testing.assert_array_equal(
        term.positive_coefficient_mask, expected["p_ident"]
    )
    np.testing.assert_array_equal(
        compiled.positive_coefficient_mask,
        np.concatenate(
            [np.zeros(1, dtype=bool), np.asarray(expected["p_ident"], dtype=bool)]
        ),
    )


@pytest.mark.parametrize(
    "basis_code,upstream_code", [("mpdby", "mpdBy"), ("cxby", "cxBy")]
)
def test_shape_linear_functional_compilation_and_prediction_match_scam(
    basis_code, upstream_code
):
    rng = np.random.default_rng(685)
    n, points = 31, 17
    grid = np.linspace(-1.4, 2.2, points)
    locations = np.tile(grid, (n, 1))
    weights = rng.normal(size=(n, points))
    new_locations = np.tile(np.linspace(-1.2, 2.0, 13), (7, 1))
    new_weights = rng.normal(size=new_locations.shape)
    data = pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "X": list(locations),
            "L": list(weights),
        }
    )
    formula = f'y ~ s(X, by=L, bs="{basis_code}", k=8, m=2)'
    model_like = SimpleNamespace(k=10, basis="tp", select=False)
    inputs = prepare_formula_inputs(
        model_like,
        data=data,
        formula=formula,
        y=np.asarray(data["y"], dtype=np.float64),
    )
    compiled = compile_model(inputs[2], inputs[3], inputs[1], fit_intercept=True)
    expected = run_scam_linear_functional_constructor(
        locations,
        weights,
        basis_code=upstream_code,
        k=8,
        m=2,
        new_locations=new_locations,
        new_weights=new_weights,
    )

    term = compiled.compiled_terms[0]
    prediction_data = np.empty((7, 2), dtype=object)
    prediction_data[:, 0] = list(new_locations)
    prediction_data[:, 1] = list(new_weights)
    np.testing.assert_allclose(compiled.design_matrix, expected["X"], atol=4e-13)
    np.testing.assert_allclose(
        compiled.compiled_penalties[0].matrix, expected["S"][0], atol=4e-13
    )
    np.testing.assert_array_equal(
        term.positive_coefficient_mask, expected["p_ident"]
    )
    np.testing.assert_allclose(
        term.predict_fn(prediction_data), expected["prediction"], atol=4e-13
    )


@pytest.mark.parametrize(
    "basis_code",
    [
        "tedmi",
        "tedmd",
        "temicx",
        "temicv",
        "tedecv",
        "tedecx",
        "tecvcv",
        "tecxcx",
        "tecxcv",
        "tescv",
        "tescx",
        "tesmi1",
        "tesmd1",
        "tesmi2",
        "tesmd2",
        "tismi",
        "tismd",
    ],
)
def test_bivariate_shape_compilation_and_prediction_match_scam(basis_code):
    rng = np.random.default_rng(686)
    data = pd.DataFrame(
        {
            "y": rng.normal(size=73),
            "x": rng.uniform(-1.8, 2.3, size=73),
            "z": rng.uniform(-2.2, 1.6, size=73),
        }
    )
    new_data = pd.DataFrame(
        {
            "x": np.linspace(-2.0, 2.5, 21),
            "z": np.linspace(1.8, -2.4, 21),
        }
    )
    formula = f'y ~ s(x, z, bs="{basis_code}", k=c(6, 7), m=c(2, 1))'
    model_like = SimpleNamespace(k=10, basis="tp", select=False)
    inputs = prepare_formula_inputs(
        model_like,
        data=data,
        formula=formula,
        y=np.asarray(data["y"], dtype=np.float64),
    )
    compiled = compile_model(inputs[2], inputs[3], inputs[1], fit_intercept=True)
    expected = run_scam_raw_constructor(
        data[["x", "z"]],
        f"s(x, z, bs='{basis_code}', k=c(6, 7), m=c(2, 1))",
        new_data=new_data,
        smoothcon=True,
    )

    term = compiled.compiled_terms[0]
    np.testing.assert_allclose(compiled.design_matrix, expected["X"], atol=6e-14)
    for actual, reference in zip(
        compiled.compiled_penalties, expected["S"], strict=True
    ):
        np.testing.assert_allclose(actual.matrix, reference, atol=6e-14)
    np.testing.assert_array_equal(
        term.positive_coefficient_mask, expected["p_ident"]
    )
    np.testing.assert_array_equal(
        compiled.positive_coefficient_mask,
        np.concatenate([np.zeros(1, dtype=bool), expected["p_ident"]]),
    )
    prediction_data = np.asarray(new_data[["x", "z"]], dtype=np.float64)
    setup = build_bivariate_shape_setup(
        data["x"],
        data["z"],
        basis_code=basis_code,
        bs_dim=(6, 7),
        spline_order=(2, 1),
    )
    np.testing.assert_allclose(
        term.predict_fn(prediction_data),
        expected["prediction"] @ setup.constraint_matrix,
        atol=8e-14,
    )


@pytest.mark.parametrize("basis_code", ["lmpi", "lipl"])
def test_local_shape_compilation_matches_scam(basis_code):
    rng = np.random.default_rng(742)
    x = rng.uniform(-2.1, 3.2, size=81)
    change_point = 0.35
    data = pd.DataFrame({"y": np.sin(x), "x": x})
    formula = (
        f'y ~ s(x, bs="{basis_code}", k=12, m=2, '
        f"xt=list(xc={change_point}))"
    )
    model_like = SimpleNamespace(k=10, basis="tp", select=False)
    inputs = prepare_formula_inputs(
        model_like,
        data=data,
        formula=formula,
        y=np.asarray(data["y"], dtype=np.float64),
    )
    compiled = compile_model(
        inputs[2], inputs[3], inputs[1], fit_intercept=True
    )
    expected = run_scam_raw_constructor(
        data[["x"]],
        f"s(x, bs='{basis_code}', k=12, m=2, xt=list(xc={change_point}))",
        smoothcon=True,
    )

    term = compiled.compiled_terms[0]
    np.testing.assert_allclose(
        compiled.design_matrix, expected["X"], rtol=0.0, atol=4e-14
    )
    np.testing.assert_allclose(
        compiled.compiled_penalties[0].matrix,
        expected["S"][0],
        rtol=0.0,
        atol=4e-14,
    )
    np.testing.assert_array_equal(
        term.positive_coefficient_mask, expected["p_ident"]
    )
