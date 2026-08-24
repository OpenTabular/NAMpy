from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from nampy.gam.compiler.compile_model import compile_model
from nampy.gam.data import coerce_formula_predict_inputs
from nampy.gam.formula import extract_formula_terms, parse_gam_formula
from nampy.gam.specs.build import build_formula_model
from nampy.gam.specs.modeling import prepare_formula_inputs

pytestmark = [pytest.mark.surface_regression]


def _build_from_formula(formula, data: pd.DataFrame):
    parsed = parse_gam_formula(formula)
    extracted = extract_formula_terms(parsed)
    return build_formula_model(extracted, data=data)


def _compile_from_formula(formula, data: pd.DataFrame):
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
        formula=formula,
        y=np.zeros(len(data), dtype=np.float64),
    )
    return compile_model(
        X=X,
        feature_names=feature_names,
        predictor_specs=predictor_specs,
        fit_intercept=bool(predictor_specs[0].has_intercept),
    )


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


def test_build_formula_model_preserves_one_shared_component_block():
    """
    Component specs own coefficients and carry their overlapping LP indices.
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
    assert built.n_linear_predictors == 2
    assert built.component_lpi == ((1,), (2,), (1, 2))
    assert len(built.predictor_specs) == 3
    assert [term.kind for term in built.predictor_specs[2].terms] == [
        "parametric",
        "smooth",
    ]


def test_single_formula_multiple_offsets_keep_first_with_mgcv_warning():
    """
    mgcv::interpret.gam0 (mgcv/R/mgcv.r:387-389) assigns all offset labels into
    one slot, so base R keeps only the first offset and warns; verified against
    mgcv 1.9-4 in the retained local multi-offset probe.
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
    with pytest.raises(ValueError, match="shared offsets not allowed"):
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


@pytest.mark.parametrize(
    ("formula", "exc_type", "match"),
    [
        (
            'y ~ s(x, bs="cr", k=matrix(c(1,2,3), nrow=2))',
            ValueError,
            r"matrix\(\.\.\.\) nrow must divide",
        ),
        (
            'y ~ s(x, k=matrix(c(1,2,3,4), ncol=3))',
            ValueError,
            r"matrix\(\.\.\.\) ncol must divide",
        ),
        (
            'y ~ te(x, z, k=matrix(c(1,2,3,4), foo=2))',
            NotImplementedError,
            r"Unsupported matrix\(\.\.\.\) argument",
        ),
        (
            'y ~ s(x, xt=diag(2, 3))',
            NotImplementedError,
            r"Only diag\(x\) formula values are supported",
        ),
        (
            'y ~ s(x, k=c(a=5))',
            NotImplementedError,
            r"keyword arguments to c\(\.\.\.\) are not supported",
        ),
        (
            "y ~ x:s(z)",
            NotImplementedError,
            "Interactions involving smooth specials",
        ),
        (
            "y ~ offset(x, z)",
            NotImplementedError,
            r"offset\(\.\.\.\) currently supports one expression",
        ),
        ("y ~ x**0", ValueError, "invalid power in formula"),
        ("y ~ x**1", ValueError, "invalid power in formula"),
        ("y ~ x**z", ValueError, "invalid power in formula"),
        ("y ~ 2", ValueError, "invalid model formula in ExtractVars"),
        (
            "y ~ s(x, k=set())",
            NotImplementedError,
            "Unsupported formula value expression",
        ),
    ],
    ids=[
        "matrix_nrow",
        "matrix_ncol",
        "matrix_kwarg",
        "diag_args",
        "c_kwargs",
        "smooth_interaction",
        "offset_multi_expr",
        "power_zero",
        "power_one",
        "power_symbol",
        "bare_numeric",
        "unsupported_value_expr",
    ],
)
def test_formula_error_branches_fail_loudly(formula, exc_type, match):
    """Every user-reachable parse/build rejection raises its documented error.

    These branches existed with zero coverage; each is the loud-failure
    counterpart of an mgcv-side R error or an explicitly unsupported literal.
    """
    data = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0],
            "x": [0.0, 1.0, 2.0, 3.0],
            "z": [1.0, 2.0, 1.0, 2.0],
        }
    )
    with pytest.raises(exc_type, match=match):
        _build_from_formula(formula, data)


def test_formula_warning_branches_recover_like_mgcv():
    """Warned-and-recovered branches keep mgcv's fallback behavior."""
    data = pd.DataFrame(
        {
            "y": np.linspace(0.0, 1.0, 12),
            "x": np.linspace(-1.0, 1.0, 12),
            "z": np.linspace(0.5, 1.5, 12),
            "w": np.linspace(-0.5, 0.5, 12),
        }
    )

    # mgcv warns "bs wrong length and ignored." and falls back to defaults.
    with pytest.warns(UserWarning, match="bs wrong length and ignored"):
        built = _build_from_formula(
            'y ~ te(x, z, bs=["cr","cr","cr"], k=[5,5])', data
        )
    spec = built.predictor_specs[0].terms[0].smooth_spec
    assert spec is not None

    # mgcv warns "something wrong with argument d." and resets to all-1D
    # marginals (specs/build.py mirror of smooth.construct dispatch).
    with pytest.warns(UserWarning, match=r"something wrong with argument d\."):
        built = _build_from_formula(
            'y ~ te(x, z, w, d=[2,2], bs=["cr","cr","cr"], k=[5,5,5])', data
        )
    spec = built.predictor_specs[0].terms[0].smooth_spec
    assert list(spec.d) == [1, 1, 1]

    # "single linear predictor indices are ignored" (parse.py) for a
    # one-element label list.
    with pytest.warns(UserWarning, match="single linear predictor indices"):
        parse_gam_formula(["y ~ x", "~ 1", "1 ~ z"])


def test_tensor_m_wrong_length_warns_and_uses_mgcv_zero_fallback():
    """``te()`` wrong-length ``m`` warns and resets every margin to zero.

    This mirrors ``mgcv/R/smooth.r::te`` lines 442-448.  Comparing the compiled
    design and penalties against an explicit ``m=[0, 0]`` contract verifies
    that the formula path reaches the upstream fallback, not just the private
    normalization helper.
    """
    x = np.linspace(-1.0, 1.0, 48)
    data = pd.DataFrame(
        {
            "y": np.sin(2.0 * x),
            "x": x,
            "z": np.cos(1.3 * x),
        }
    )
    common = 'y ~ te(x, z, bs=["tp", "tp"], k=[6, 6], sp=[0.7, 1.1]'

    with pytest.warns(UserWarning, match="m wrong length and ignored"):
        wrong_length = _compile_from_formula(f"{common}, m=[1, 2, 3])", data)
    explicit_fallback = _compile_from_formula(f"{common}, m=[0, 0])", data)

    np.testing.assert_array_equal(
        np.asarray(wrong_length.design_matrix, dtype=np.float64),
        np.asarray(explicit_fallback.design_matrix, dtype=np.float64),
    )
    assert len(wrong_length.compiled_penalties) == len(
        explicit_fallback.compiled_penalties
    )
    for actual_penalty, expected_penalty in zip(
        wrong_length.compiled_penalties,
        explicit_fallback.compiled_penalties,
        strict=True,
    ):
        np.testing.assert_array_equal(
            np.asarray(actual_penalty.matrix, dtype=np.float64),
            np.asarray(expected_penalty.matrix, dtype=np.float64),
        )


def test_m_argument_is_silently_ignored_on_cubic_bases_like_mgcv():
    """s(..., m=) on cr is accepted and ignored, exactly like upstream.

    mgcv documents m as ps/tp-only; smooth.construct.cr.smooth.spec never
    reads it. The fit with m supplied must be byte-identical to the fit
    without it.
    """
    rng = np.random.default_rng(31)
    data = pd.DataFrame({"x": np.linspace(-1, 1, 60)})
    data["y"] = np.sin(2.0 * data["x"]) + rng.normal(scale=0.1, size=60)

    from nampy.gam import GAM

    def _fit(formula):
        return GAM(
            family="gaussian",
            formula=formula,
            optimize_smoothing=False,
            smoothing_method="fixed",
            smoothing_params=[1.0],
        ).fit(data=data)

    plain = _fit('y ~ s(x, bs="cr", k=6)')
    with_m = _fit('y ~ s(x, bs="cr", k=6, m=3)')
    np.testing.assert_array_equal(
        np.asarray(plain.fit_result().coef_full, dtype=np.float64),
        np.asarray(with_m.fit_result().coef_full, dtype=np.float64),
    )


def test_drop_intercept_list_routes_per_predictor_and_validates_length():
    """List-valued drop_intercept maps per predictor; wrong length raises."""
    parsed = parse_gam_formula(['y ~ s(x, bs="cr", k=5)', "~ z"])
    extracted = extract_formula_terms(parsed, drop_intercept=[True, False])
    assert [pred.intercept for pred in extracted] == [False, True]

    with pytest.raises(ValueError, match="drop_intercept must have length 2"):
        extract_formula_terms(parsed, drop_intercept=[True, False, True])
