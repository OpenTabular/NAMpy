from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from nampy.gam.data import coerce_formula_predict_inputs
from nampy.gam.formula import extract_formula_terms, parse_gam_formula
from nampy.gam.specs.build import build_formula_model

pytestmark = [pytest.mark.surface_regression]


def _build_from_formula(formula, data: pd.DataFrame):
    parsed = parse_gam_formula(formula)
    extracted = extract_formula_terms(parsed)
    return build_formula_model(extracted, data=data)


def test_build_formula_model_routes_multi_predictor_offsets_in_declared_order():
    """
    Owner-contract coverage verifying that build formula model routes multi predictor
    offsets in declared order.
    """
    data = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0],
            "x": [0.0, 0.5, 1.0, 1.5],
            "z": [1.5, 1.0, 0.5, 0.0],
            "o1": [0.2, 0.4, 0.6, 0.8],
            "o2": [1.0, 1.5, 2.0, 2.5],
        }
    )

    built = _build_from_formula(
        [
            'y ~ s(x, bs="cr", k=5) + offset(o1)',
            '~ s(z, bs="cr", k=5) + offset(log(o2 + 1))',
        ],
        data,
    )

    assert len(built.predictor_specs) == 2
    assert [pred.offset_name for pred in built.predictor_specs] == [
        "o1",
        built.predictor_specs[1].offset_name,
    ]
    assert built.preprocess_state["offset_names"] == (
        "o1",
        built.predictor_specs[1].offset_name,
    )
    assert built.predictor_specs[0].terms[0].features == ("x",)
    assert built.predictor_specs[1].terms[0].features == ("z",)

    hidden_offset = built.predictor_specs[1].offset_name
    assert hidden_offset in built.working_data.columns
    np.testing.assert_allclose(
        np.asarray(built.offsets[0], dtype=np.float64),
        data["o1"].to_numpy(dtype=np.float64),
    )
    np.testing.assert_allclose(
        np.asarray(built.offsets[1], dtype=np.float64),
        np.log(data["o2"].to_numpy(dtype=np.float64) + 1.0),
    )
    assert built.preprocess_state["formula_expression_columns"] == [
        {
            "hidden_name": hidden_offset,
            "expr": "log(o2 + 1)",
            "source_variables": ["o2"],
        }
    ]


def test_build_formula_model_reuses_transformed_offset_columns_across_predictors():
    """
    Owner-contract coverage verifying that build formula model reuses transformed offset
    columns across predictors.
    """
    data = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0],
            "x": [0.0, 0.5, 1.0, 1.5],
            "z": [1.5, 1.0, 0.5, 0.0],
            "o": [1.0, 1.5, 2.0, 2.5],
        }
    )

    built = _build_from_formula(
        [
            'y ~ s(x, bs="cr", k=5) + offset(log(o + 1))',
            '~ s(z, bs="cr", k=5) + offset(log(o + 1))',
        ],
        data,
    )

    offset_names = [pred.offset_name for pred in built.predictor_specs]
    assert len(offset_names) == 2
    assert offset_names[0] == offset_names[1]
    assert built.preprocess_state["offset_names"] == tuple(offset_names)
    assert built.preprocess_state["formula_expression_columns"] == [
        {
            "hidden_name": offset_names[0],
            "expr": "log(o + 1)",
            "source_variables": ["o"],
        }
    ]

    expected = np.log(data["o"].to_numpy(dtype=np.float64) + 1.0)
    np.testing.assert_allclose(
        built.working_data[offset_names[0]].to_numpy(dtype=np.float64),
        expected,
    )
    for off in built.offsets:
        np.testing.assert_allclose(np.asarray(off, dtype=np.float64), expected)


def test_build_formula_model_expands_shared_component_into_matching_predictor_specs():
    """
    Owner-contract coverage verifying that build formula model expands shared component
    into matching predictor specs.
    """
    data = pd.DataFrame(
        {
            "y1": [1.0, 1.5, 2.0, 2.5],
            "y2": [0.5, 1.0, 1.5, 2.0],
            "x": [0.0, 0.5, 1.0, 1.5],
            "z": [2.0, 1.5, 1.0, 0.5],
        }
    )

    built = _build_from_formula(
        [
            "y1 ~ -1",
            "y2 ~ -1",
            '1 + 2 ~ s(x, k=5, bs="cr") + I(z**2)',
        ],
        data,
    )

    assert len(built.predictor_specs) == 2
    assert [pred.has_intercept for pred in built.predictor_specs] == [False, False]

    first_labels = [term.label for term in built.predictor_specs[0].terms]
    second_labels = [term.label for term in built.predictor_specs[1].terms]
    assert first_labels == second_labels

    first_hidden = built.predictor_specs[0].terms[0].features[0]
    second_hidden = built.predictor_specs[1].terms[0].features[0]
    assert first_hidden != second_hidden
    assert first_hidden in built.working_data.columns
    assert second_hidden in built.working_data.columns

    expected = data["z"].to_numpy(dtype=np.float64) ** 2
    np.testing.assert_allclose(
        built.working_data[first_hidden].to_numpy(dtype=np.float64),
        expected,
    )
    np.testing.assert_allclose(
        built.working_data[second_hidden].to_numpy(dtype=np.float64),
        expected,
    )
    assert built.preprocess_state["formula_expression_columns"] == [
        {
            "hidden_name": built.preprocess_state["formula_expression_columns"][0][
                "hidden_name"
            ],
            "expr": "I(z**2)",
            "source_variables": ["z"],
        }
    ]


def test_extract_formula_terms_rejects_multiple_offsets_for_one_predictor():
    """
    Owner-contract coverage verifying that extract formula terms rejects multiple
    offsets for one predictor.
    """
    formula = [
        "y ~ -1",
        "1 ~ x + offset(o1)",
        "1 ~ z + offset(o2)",
    ]

    parsed = parse_gam_formula(formula)

    assert parsed.predictors[0].offset_names == ("o1", "o2")
    with pytest.raises(
        NotImplementedError,
        match="Multiple offset\\(\\.\\.\\.\\) terms per predictor are not yet supported",
    ):
        extract_formula_terms(parsed)


def test_build_formula_model_rejects_transformed_smooth_by_expressions():
    """
    Owner-contract coverage verifying that build formula model rejects transformed
    smooth by expressions.
    """
    data = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0],
            "x": [0.0, 0.5, 1.0, 1.5],
            "z": [0.2, 0.4, 0.8, 1.6],
        }
    )

    with pytest.raises(
        NotImplementedError,
        match="Transformed smooth `by` expressions are parsed exactly",
    ):
        _build_from_formula('y ~ s(x, by=log(z + 1), bs="cr", k=5)', data)


def test_formula_predict_inputs_rebuild_multi_predictor_offsets_in_declared_order():
    """
    Owner-contract coverage verifying that formula predict inputs rebuild multi
    predictor offsets in declared order.
    """
    fit_data = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0],
            "x": [0.0, 0.5, 1.0, 1.5],
            "z": [1.5, 1.0, 0.5, 0.0],
            "o1": [0.2, 0.4, 0.6, 0.8],
            "o2": [1.0, 1.5, 2.0, 2.5],
        }
    )
    new_data = pd.DataFrame(
        {
            "x": [2.0, 2.5],
            "z": [0.25, -0.25],
            "o1": [1.1, 1.3],
            "o2": [3.0, 4.0],
        }
    )
    built = _build_from_formula(
        [
            'y ~ s(x, bs="cr", k=5) + offset(o1)',
            '~ s(z, bs="cr", k=5) + offset(log(o2 + 1))',
        ],
        fit_data,
    )
    model = SimpleNamespace(
        formula_mode_=True,
        formula_preprocess_state_=built.preprocess_state,
        formula_used_columns_=list(built.used_columns),
        formula_offset_names_=tuple(built.preprocess_state["offset_names"]),
    )

    X_np, feature_names, offset = coerce_formula_predict_inputs(model, new_data)

    assert feature_names == list(built.used_columns)
    np.testing.assert_allclose(X_np[:, 0], new_data["x"].to_numpy(dtype=np.float64))
    np.testing.assert_allclose(X_np[:, 1], new_data["z"].to_numpy(dtype=np.float64))
    assert isinstance(offset, list)
    assert len(offset) == 2
    np.testing.assert_allclose(offset[0], new_data["o1"].to_numpy(dtype=np.float64))
    np.testing.assert_allclose(
        offset[1],
        np.log(new_data["o2"].to_numpy(dtype=np.float64) + 1.0),
    )


def test_formula_predict_inputs_reject_non_numeric_formula_offset_columns():
    """
    Owner-contract coverage verifying that formula predict inputs reject non numeric
    formula offset columns.
    """
    fit_data = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0],
            "x": [0.0, 0.5, 1.0, 1.5],
            "o": [0.2, 0.4, 0.6, 0.8],
        }
    )
    new_data = pd.DataFrame(
        {
            "x": [2.0, 2.5],
            "o": ["bad", "worse"],
        }
    )
    built = _build_from_formula("y ~ x + offset(o)", fit_data)
    model = SimpleNamespace(
        formula_mode_=True,
        formula_preprocess_state_=built.preprocess_state,
        formula_used_columns_=list(built.used_columns),
        formula_offset_names_=tuple(built.preprocess_state["offset_names"]),
    )

    with pytest.raises(
        NotImplementedError,
        match="Current formula-based prediction supports numeric offsets only",
    ):
        coerce_formula_predict_inputs(model, new_data)
