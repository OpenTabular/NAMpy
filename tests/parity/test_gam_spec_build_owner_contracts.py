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


def test_single_formula_multiple_offsets_keep_first_with_mgcv_warning():
    """
    mgcv::interpret.gam0 (mgcv/R/mgcv.r:387-389) assigns all offset labels into
    one slot, so base R keeps only the first offset and warns; verified against
    mgcv 1.9-4 in debug/multi_offset_probe.R.
    """
    data = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0],
            "x": [0.0, 0.5, 1.0, 1.5],
            "a": [0.2, 0.4, 0.6, 0.8],
            "b": [1.0, 1.5, 2.0, 2.5],
        }
    )

    with pytest.warns(
        UserWarning,
        match="number of items to replace is not a multiple of replacement length",
    ):
        built = _build_from_formula(
            'y ~ offset(a) + offset(b) + s(x, bs="cr", k=4)', data
        )

    assert built.predictor_specs[0].offset_name == "a"
    assert built.preprocess_state["offset_names"] == ("a",)


def test_formula_multivariate_tp_default_k_defers_to_mgcv_constructor_rule():
    """
    mgcv::s() leaves k = -1 and smooth.construct.tp.smooth.spec resolves the
    d-dependent default M + c(8, 27, 100)[min(d, 3)] (mgcv/R/smooth.r:1316-1318).
    A flat spec-level default of 10 silently changed d > 1 models.
    """
    data = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0],
            "x": [0.0, 0.5, 1.0, 1.5],
            "z": [1.5, 1.0, 0.5, 0.0],
        }
    )

    built = _build_from_formula("y ~ s(x, z)", data)
    assert built.predictor_specs[0].terms[0].smooth_spec.k == -1

    built_1d = _build_from_formula("y ~ s(x)", data)
    assert built_1d.predictor_specs[0].terms[0].smooth_spec.k == -1

    built_explicit = _build_from_formula("y ~ s(x, z, k=20)", data)
    assert built_explicit.predictor_specs[0].terms[0].smooth_spec.k == 20


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


def test_build_formula_model_materializes_transformed_smooth_by_expressions():
    """Verify transformed smooth by-expressions use a paired fit/predict recipe."""
    data = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0],
            "x": [0.0, 0.5, 1.0, 1.5],
            "z": [0.2, 0.4, 0.8, 1.6],
        }
    )

    built = _build_from_formula('y ~ s(x, by=log(z + 1), bs="cr", k=5)', data)

    term = built.predictor_specs[0].terms[0]
    hidden = term.by_variable
    assert hidden in built.working_data
    np.testing.assert_allclose(
        built.working_data[hidden].to_numpy(dtype=np.float64),
        np.log(data["z"].to_numpy(dtype=np.float64) + 1.0),
    )
    assert term.metadata["formula_by_expansion"] == {
        "hidden_name": hidden,
        "expr": "log(z + 1)",
        "source_variables": ["z"],
    }
    assert built.preprocess_state["formula_expression_columns"] == [
        {
            "hidden_name": hidden,
            "expr": "log(z + 1)",
            "source_variables": ["z"],
        }
    ]


def test_formula_smooth_args_mirror_mgcv_k_rounding_and_id_truncation():
    """
    Regression coverage for mgcv/R/smooth.r::s() argument normalization.
    """
    data = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0, 5.0],
            "x": [0.0, 0.25, 0.5, 0.75, 1.0],
        }
    )

    with pytest.warns(UserWarning) as caught:
        built = _build_from_formula('y ~ s(x, bs="cr", k=4.6, id=c("a", "b"))', data)

    messages = [str(item.message) for item in caught]
    assert "argument k of s() should be integer and has been rounded" in messages
    assert "only first element of `id' used" in messages
    term = built.predictor_specs[0].terms[0]
    assert term.smooth_spec.k == 5
    assert term.smoothing_id == "a"


def test_formula_tensor_k_too_small_resets_to_mgcv_default():
    """
    Regression coverage for mgcv/R/smooth.r::te() tensor k validation.
    """
    data = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0, 5.0],
            "x": [0.0, 0.25, 0.5, 0.75, 1.0],
            "z": [1.0, 0.75, 0.5, 0.25, 0.0],
        }
    )

    with pytest.warns(
        UserWarning,
        match="one or more supplied k too small - reset to default",
    ):
        built = _build_from_formula('y ~ te(x, z, bs=["cr", "cr"], k=[2, 8])', data)

    term = built.predictor_specs[0].terms[0]
    assert term.smooth_spec.k == [5, 5]


def test_build_formula_model_rejects_ordered_parametric_factor_without_r_contrasts():
    """
    Owner-contract coverage verifying that ordered parametric factors stay unsupported
    until mgcv/R ordered contrasts are mirrored.
    """
    data = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0],
            "f": pd.Categorical(
                ["lo", "mid", "hi", "mid"],
                categories=["lo", "mid", "hi"],
                ordered=True,
            ),
        }
    )

    with pytest.raises(
        NotImplementedError,
        match="Ordered parametric factors require mgcv/R ordered contrasts",
    ):
        _build_from_formula("y ~ f", data)


@pytest.mark.parametrize(
    "formula",
    [
        'y ~ te(x, z, bs=["cr", "cr"], k=[5, 5], fx=[True, False])',
        'y ~ ti(x, z, bs=["cr", "cr"], k=[5, 5], fx=[True, False], mc=[True, False])',
    ],
    ids=["te_vector_fx", "ti_vector_fx"],
)
def test_build_formula_model_preserves_tensor_vector_fx(formula):
    """Verify te/ti retain mgcv's one fixed flag per marginal basis."""
    data = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0],
            "x": [0.0, 0.5, 1.0, 1.5],
            "z": [1.5, 1.0, 0.5, 0.0],
        }
    )

    built = _build_from_formula(formula, data)

    term = built.predictor_specs[0].terms[0]
    assert term.smooth_spec.fx == [True, False]


def test_build_formula_model_resets_wrong_length_tensor_fx_like_mgcv():
    """Mirror mgcv::te() warning and all-penalized fallback for malformed fx."""
    data = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0],
            "x": [0.0, 0.5, 1.0, 1.5],
            "z": [1.5, 1.0, 0.5, 0.0],
        }
    )

    with pytest.warns(UserWarning, match="dimension of fx is wrong"):
        built = _build_from_formula(
            'y ~ te(x, z, bs=["cr", "cr"], k=[5, 5], fx=[True, False, True])',
            data,
        )

    assert built.predictor_specs[0].terms[0].smooth_spec.fx == [False, False]


@pytest.mark.parametrize(
    "formula",
    [
        'y ~ te(x, z, bs=["cr", "cr"], k=[5, 5], fx=True)',
        'y ~ ti(x, z, bs=["cr", "cr"], k=[5, 5], fx=True, mc=[True, False])',
    ],
    ids=["te_scalar_fx", "ti_scalar_fx"],
)
def test_build_formula_model_accepts_tensor_scalar_fx(formula):
    """Owner-contract coverage verifying that scalar tensor fx remains supported."""
    data = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0],
            "x": [0.0, 0.5, 1.0, 1.5],
            "z": [1.5, 1.0, 0.5, 0.0],
        }
    )

    built = _build_from_formula(formula, data)
    smooth = built.predictor_specs[0].terms[0].smooth_spec
    assert smooth is not None
    assert smooth.fx is True


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
