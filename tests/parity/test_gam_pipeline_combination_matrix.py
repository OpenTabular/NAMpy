"""Cross-feature parity contracts for all seven GAM pipeline stages."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from nampy.gam.compiler.compile_model import compile_model
from nampy.gam.compiler.contracts import compose_coefficient_maps
from nampy.gam.formula import extract_formula_terms, parse_gam_formula
from nampy.gam.specs.build import build_formula_model
from nampy.gam.specs.modeling import prepare_formula_inputs
from tests.families.test_general_family_mgcv_parity import (
    _gammals_data,
    _gaulss_data,
)
from tests.mgcv_invariant_policy import stable_column_space_projector
from tests.mgcv_parity_utils import (
    _assert_basic_mgcv_parity,
    _make_fs_data,
    _make_gamma_data,
    _make_gaussian_data,
    _make_negbin_data,
    _make_poisson_data,
    _make_random_effect_data,
    _make_sz_data,
    _normalize_python_formula_text,
    _run_mgcv_anova,
    _run_mgcv_predict_on_newdata,
    _run_mgcv_snapshot,
)

pytestmark = [pytest.mark.surface_regression]


def _formula_data(seed: int = 901, n: int = 96) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x0 = np.linspace(-1.25, 1.25, n)
    x1 = rng.uniform(-1.0, 1.0, size=n)
    z = 0.75 + 0.2 * np.cos(1.7 * x0)
    off = 0.12 * np.sin(0.8 * x0)
    f = pd.Categorical(np.resize(np.array(["a", "b", "c"], dtype=object), n))
    f_effect = np.array([{"a": -0.2, "b": 0.1, "c": 0.35}[str(v)] for v in f])
    y = off + 0.45 * z + np.sin(np.pi * x0) + 0.2 * x1**2 + f_effect
    y += rng.normal(scale=0.035, size=n)
    w = 0.4 + np.linspace(0.0, 1.0, n)
    return pd.DataFrame(
        {"y": y, "x0": x0, "x1": x1, "z": z, "off": off, "f": f, "w": w}
    )


def _compile_formula(data: pd.DataFrame, formula, *, select: bool = False):
    model_like = SimpleNamespace(k=10, basis="tp", select=bool(select))
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
    ), X, predictor_specs


def test_stage_1_formula_list_combines_shared_terms_transforms_offsets_and_intercepts():
    """Formula-list preprocessing keeps predictor-local contracts; shared guarded.

    Per-predictor offsets, transformed offsets, and intercept policies are the
    supported formula-list surface. Shared '1 + 2 ~ ...' components are guarded
    at extraction: mgcv shares one coefficient block across the labelled
    predictors, which the former duplicate-terms expansion did not implement.
    """
    data = _formula_data()
    data = data.assign(o2=np.exp(0.2 + data["off"]))
    formula = [
        'y ~ -1 + offset(off) + I(z**2) + s(x0, bs="cr", k=6, sp=0.7)',
        '~ 1 + offset(log(o2 + 1)) + I(z**2) + s(x0, bs="cr", k=6, sp=0.7)',
    ]
    parsed = parse_gam_formula(formula)
    extracted = extract_formula_terms(parsed)
    built = build_formula_model(extracted, data=data)

    assert len(built.predictor_specs) == 2
    assert [pred.has_intercept for pred in built.predictor_specs] == [False, True]
    assert built.predictor_specs[0].offset_name == "off"
    assert built.predictor_specs[1].offset_name != "o2"
    assert built.preprocess_state["offset_names"] == tuple(
        pred.offset_name for pred in built.predictor_specs
    )
    assert len(built.preprocess_state["formula_expression_columns"]) == 2
    assert [term.label for term in built.predictor_specs[0].terms] == [
        term.label for term in built.predictor_specs[1].terms
    ]
    assert [(term.kind, term.smooth_spec is not None) for term in built.predictor_specs[0].terms] == [
        ("parametric", False),
        ("smooth", True),
    ]
    assert isinstance(built.offsets, list)
    np.testing.assert_allclose(built.offsets[0], data["off"].to_numpy())
    np.testing.assert_allclose(built.offsets[1], np.log(data["o2"].to_numpy() + 1.0))

    shared_formula = [
        "y ~ -1 + offset(off)",
        "~ 1",
        '1 + 2 ~ I(z**2) + s(x0, bs="cr", k=6, sp=0.7)',
    ]
    with pytest.raises(
        NotImplementedError,
        match="Shared linear-predictor components",
    ):
        extract_formula_terms(parse_gam_formula(shared_formula))


def test_stage_1_api_routes_weights_knots_min_sp_drop_intercept_and_fit_offset():
    """All non-formula fit inputs arrive at their canonical stage-one owners."""
    data = _formula_data(seed=902, n=84)
    knots = np.linspace(-1.25, 1.25, 6)
    fit_offset = 0.03 * np.cos(data["x0"].to_numpy(dtype=np.float64))
    gam = GAM(
        family="gaussian",
        formula='y ~ s(x0, bs="cr", k=6, sp=0.8)',
        optimize_smoothing=False,
        smoothing_method="fixed",
        min_sp=[0.2],
        knots={"x0": knots},
        drop_intercept=True,
    )
    gam.fit(data=data, sample_weight="w", offset=fit_offset)

    runtime = gam.gam_result_.compiled_model.compiled_terms[0].predict_fn.__self__
    assert gam.fit_intercept is False
    np.testing.assert_allclose(gam.prior_weights_, data["w"].to_numpy())
    np.testing.assert_allclose(gam.offset_train_, fit_offset)
    assert gam.offset_predict_default_ is None
    np.testing.assert_allclose(gam.min_sp_, [0.2])
    np.testing.assert_allclose(gam.smoothing_params, [0.8])
    np.testing.assert_allclose(runtime._spline.knots, knots)


def test_stage_1_supported_numeric_factor_interactions_rebuild_on_newdata_like_mgcv():
    """Supported interactions, transforms, factor-by, and offsets share one recipe."""
    data = _formula_data(seed=916, n=96)
    formula = (
        'y ~ z * f + x0:f + I(x1**2)'
        ' + s(x0, by=f, bs="cr", k=5, id="factor_curve", sp=0.8)'
        " + offset(off)"
    )
    built = build_formula_model(extract_formula_terms(parse_gam_formula(formula)), data=data)

    assert built.preprocess_state["formula_expression_columns"]
    assert any(
        term.kind == "parametric" and ":" in term.label
        for term in built.predictor_specs[0].terms
    )
    assert any(
        term.smooth_spec is not None and term.by_variable is not None
        for term in built.predictor_specs[0].terms
    )

    gam = GAM(
        family="gaussian",
        formula=formula,
        optimize_smoothing=False,
        smoothing_method="fixed",
    ).fit(data=data)
    newdata = data.iloc[4::13].drop(columns=["y"]).copy()
    actual, actual_se = gam.predict(newdata, type="response", return_se=True)
    expected = _run_mgcv_predict_on_newdata(
        data,
        newdata,
        formula,
        family="gaussian",
        method="fixed",
        type="response",
        return_se=True,
        allow_live_run=True,
    )
    np.testing.assert_allclose(actual, np.asarray(expected["pred"]).ravel(), atol=2e-8, rtol=2e-8)
    np.testing.assert_allclose(actual_se, np.asarray(expected["se"]).ravel(), atol=2e-8, rtol=2e-8)


@pytest.mark.parametrize("basis", ["cr", "cs", "cc", "ps", "tp", "ts"])
def test_stage_2_univariate_runtime_boundary_predictions_and_se_match_mgcv(basis):
    """Every supported univariate runtime matches behavior at and beyond fit bounds."""
    data = _formula_data(seed=903, n=90)[["y", "x0"]].rename(columns={"x0": "x"})
    newdata = pd.DataFrame(
        {"x": np.array([-1.5, -1.25, -0.9, 0.0, 0.9, 1.25, 1.5])}
    )
    formula = f'y ~ s(x, bs="{basis}", k=7, sp=0.8)'
    gam = GAM(
        family="gaussian",
        formula=formula,
        optimize_smoothing=False,
        smoothing_method="fixed",
    ).fit(data=data)

    term = gam.gam_result_.compiled_model.compiled_terms[0]
    np.testing.assert_allclose(
        term.predict_matrix(gam.X_), term.basis_train, rtol=0.0, atol=2e-12
    )
    actual_fit, actual_se = gam.predict(newdata, type="response", return_se=True)
    expected = _run_mgcv_predict_on_newdata(
        data,
        newdata,
        formula,
        family="gaussian",
        method="fixed",
        type="response",
        return_se=True,
    )
    # `cs` replaces a two-dimensional numerical null eigenspace with a
    # full-rank shrinkage penalty.  Its raw eigenvectors are not identified;
    # the invariant penalty spectrum is covered by the raw-constructor suite,
    # while response and SE behavior remain tight here.
    behavior_tol = 2e-5 if basis == "cs" else 2e-7
    np.testing.assert_allclose(
        np.asarray(actual_fit, dtype=np.float64),
        np.asarray(expected["pred"], dtype=np.float64).ravel(),
        rtol=behavior_tol,
        atol=behavior_tol,
    )
    np.testing.assert_allclose(
        np.asarray(actual_se, dtype=np.float64),
        np.asarray(expected["se"], dtype=np.float64).ravel(),
        rtol=behavior_tol,
        atol=behavior_tol,
    )


@pytest.mark.parametrize(
    "formula",
    [
        'y ~ s(f, bs="re", sp=0.8)',
        'y ~ s(f, x0, bs="fs", k=5, xt="cr", sp=[0.8, 0.8, 0.8])',
        'y ~ s(f, x0, bs="sz", k=5, xt="cr")',
        'y ~ te(x0, x1, bs=["cr", "ps"], k=[5, 6], sp=[0.7, 1.1])',
        'y ~ ti(x0, x1, bs=["tp", "cr"], k=[6, 5], sp=[0.7, 1.1])',
    ],
    ids=["re", "fs", "sz", "te", "ti"],
)
def test_stage_2_categorical_and_tensor_runtimes_pair_training_prediction_and_penalties(
    formula,
):
    """Categorical and tensor runtimes retain paired bases and finite penalties."""
    data = _formula_data(seed=904, n=90)
    compiled, X, _predictor_specs = _compile_formula(data, formula)

    assert compiled.compiled_terms
    assert compiled.compiled_penalties
    for term in compiled.compiled_terms:
        np.testing.assert_allclose(
            term.predict_matrix(X), term.basis_train, rtol=0.0, atol=2e-11
        )
    for penalty in compiled.compiled_penalties:
        matrix = np.asarray(penalty.matrix, dtype=np.float64)
        assert np.all(np.isfinite(matrix))
        np.testing.assert_allclose(matrix, matrix.T, rtol=0.0, atol=2e-12)


@pytest.mark.parametrize(
    ("case_id", "data_factory", "formula", "method", "pred_tol", "se_tol"),
    [
        (
            "re",
            _make_random_effect_data,
            'y ~ s(f, bs="re", sp=1.0)',
            "fixed",
            2e-9,
            2e-9,
        ),
        (
            "fs",
            _make_fs_data,
            'y ~ s(f, x, bs="fs", k=6)',
            "REML",
            6e-3,
            8e-4,
        ),
        (
            "sz",
            _make_sz_data,
            'y ~ s(f1, f2, x, bs="sz", k=6)',
            "REML",
            6e-3,
            2e-2,
        ),
    ],
    ids=["re", "fs", "sz"],
)
def test_stage_2_structured_runtime_boundary_prediction_and_se_match_mgcv(
    case_id, data_factory, formula, method, pred_tol, se_tol
):
    """Structured runtimes retain behavior on known levels beyond metric bounds."""
    data = data_factory()
    if case_id == "re":
        newdata = pd.DataFrame({"f": ["a", "c", "b", "a"]})
    elif case_id == "fs":
        newdata = pd.DataFrame(
            {
                "f": ["a", "b", "c", "a", "c"],
                "x": [-0.2, 0.1, 0.9, 1.7, 2.0],
            }
        )
    else:
        newdata = pd.DataFrame(
            {
                "f1": ["a", "b", "c", "a", "c"],
                "f2": ["x", "y", "x", "y", "y"],
                "x": [-0.2, 0.0, 0.9, 2.0, 2.2],
            }
        )

    gam = GAM(
        family="gaussian",
        formula=formula,
        optimize_smoothing=method.lower() != "fixed",
        smoothing_method=method,
    ).fit(data=data)
    actual, actual_se = gam.predict(newdata, type="response", return_se=True)
    expected = _run_mgcv_predict_on_newdata(
        data,
        newdata,
        formula,
        family="gaussian",
        method=method,
        type="response",
        return_se=True,
        allow_live_run=True,
    )
    np.testing.assert_allclose(actual, np.asarray(expected["pred"]).ravel(), atol=pred_tol, rtol=pred_tol)
    np.testing.assert_allclose(actual_se, np.asarray(expected["se"]).ravel(), atol=se_tol, rtol=se_tol)


@pytest.mark.parametrize(
    ("data_factory", "formula", "geometry_tol", "behavior_tol"),
    [
        (
            lambda: _formula_data(seed=917, n=96),
            'y ~ s(x0, bs="cr", k=6, id="pooled", sp=0.8)'
            ' + s(x1, bs="cr", k=6, id="pooled")',
            2e-10,
            2e-9,
        ),
        (
            _make_fs_data,
            'y ~ s(f, x, bs="fs", k=6, xt="cr", sp=[0.8, 0.8, 0.8])',
            2e-10,
            2e-8,
        ),
        (
            lambda: _formula_data(seed=936, n=96),
            'y ~ s(x0, bs="ps", k=7, sp=0.8)',
            5e-10,
            3e-7,
        ),
        (
            lambda: _formula_data(seed=937, n=96),
            'y ~ s(x0, bs="tp", k=7, sp=0.8)',
            5e-9,
            2e-6,
        ),
        (
            lambda: _formula_data(seed=938, n=96),
            'y ~ s(x0, bs="ts", k=7, sp=0.8)',
            5e-9,
            2e-6,
        ),
        (
            lambda: _formula_data(seed=939, n=96),
            'y ~ te(x0, x1, bs=["cr", "cr"], k=[5, 5], sp=[0.7, 1.1])',
            5e-9,
            2e-6,
        ),
        (
            lambda: _formula_data(seed=940, n=96),
            'y ~ ti(x0, x1, bs=["cr", "cr"], k=[5, 5], sp=[0.7, 1.1])',
            5e-9,
            3e-4,
        ),
        (
            _make_sz_data,
            'y ~ s(f1, f2, x, bs="sz", k=6, id="shared", sp=0.8)',
            2e-8,
            3e-2,
        ),
        (
            lambda: _formula_data(seed=941, n=96),
            'y ~ f + s(x0, by=f, bs="cr", k=6, id="factor_curve", sp=0.8)',
            5e-9,
            3e-5,
        ),
    ],
    ids=[
        "linked_pooled",
        "fs_exchangeable_null_space",
        "ps",
        "tp_indeterminate_eigenspace",
        "ts_indeterminate_eigenspace",
        "tensor_te",
        "tensor_ti",
        "sz",
        "factor_by",
    ],
)
def test_stage_2_row_permutation_preserves_basis_space_penalties_and_behavior(
    data_factory, formula, geometry_tol, behavior_tol
):
    """Row order preserves identified geometry and each ordering matches mgcv."""
    data = data_factory()
    permutation = np.random.default_rng(918).permutation(len(data))
    inverse = np.argsort(permutation)
    permuted = data.iloc[permutation].reset_index(drop=True)

    compiled, _X, _specs = _compile_formula(data, formula)
    compiled_permuted, _Xp, _specs_p = _compile_formula(permuted, formula)
    np.testing.assert_allclose(
        stable_column_space_projector(compiled.design_matrix),
        stable_column_space_projector(compiled_permuted.design_matrix[inverse]),
        atol=geometry_tol,
        rtol=0.0,
    )
    actual_design = np.asarray(compiled.design_matrix, dtype=np.float64)
    permuted_design = np.asarray(
        compiled_permuted.design_matrix[inverse], dtype=np.float64
    )
    transform, *_ = np.linalg.lstsq(actual_design, permuted_design, rcond=None)
    np.testing.assert_allclose(
        actual_design @ transform,
        permuted_design,
        atol=geometry_tol,
        rtol=0.0,
    )

    def total_penalty(model):
        total = np.zeros((model.n_coef, model.n_coef), dtype=np.float64)
        for penalty in model.compiled_penalties:
            sl = penalty.coef_slice
            total[sl, sl] += np.asarray(penalty.matrix, dtype=np.float64)
        return total

    def penalized_response_operator(model):
        design = np.asarray(model.design_matrix, dtype=np.float64)
        penalized_crossproduct = design.T @ design + total_penalty(model)
        return design @ np.linalg.pinv(penalized_crossproduct) @ design.T

    actual_operator = penalized_response_operator(compiled)
    permuted_operator = penalized_response_operator(compiled_permuted)
    permuted_operator = permuted_operator[inverse][:, inverse]
    np.testing.assert_allclose(
        actual_operator,
        permuted_operator,
        atol=geometry_tol,
        rtol=0.0,
    )

    original_fit = GAM(
        family="gaussian",
        formula=formula,
        optimize_smoothing=False,
        smoothing_method="fixed",
    ).fit(data=data)
    permuted_fit = GAM(
        family="gaussian",
        formula=formula,
        optimize_smoothing=False,
        smoothing_method="fixed",
    ).fit(data=permuted)
    prediction_data = data.drop(columns=["y"])
    original_prediction, original_se = original_fit.predict(
        prediction_data, type="response", return_se=True
    )
    permuted_prediction, permuted_se = permuted_fit.predict(
        prediction_data, type="response", return_se=True
    )
    np.testing.assert_allclose(
        original_prediction,
        permuted_prediction,
        atol=behavior_tol,
        rtol=behavior_tol,
    )
    np.testing.assert_allclose(
        original_se,
        permuted_se,
        atol=behavior_tol,
        rtol=behavior_tol,
    )
    for training_data, actual_prediction, actual_se in (
        (data, original_prediction, original_se),
        (permuted, permuted_prediction, permuted_se),
    ):
        expected = _run_mgcv_predict_on_newdata(
            training_data,
            prediction_data,
            formula,
            family="gaussian",
            method="fixed",
            type="response",
            return_se=True,
            allow_live_run=True,
        )
        np.testing.assert_allclose(
            actual_prediction,
            np.asarray(expected["pred"]).ravel(),
            atol=behavior_tol,
            rtol=behavior_tol,
        )
        np.testing.assert_allclose(
            actual_se,
            np.asarray(expected["se"]).ravel(),
            atol=behavior_tol,
            rtol=behavior_tol,
        )

@pytest.mark.parametrize(
    ("formula", "select"),
    [
        (
            'y ~ s(x0, by=z, bs="cr", k=6, id="shared", sp=0.7)'
            ' + s(x1, by=z, bs="cr", k=6, id="shared")',
            False,
        ),
        ('y ~ f + s(x0, by=f, bs="cr", k=6, id="shared")', False),
        (
            'y ~ te(x0, x1, by=z, bs=["cr", "cr"], k=[5, 5], sp=[0.7, 1.1])',
            False,
        ),
        ('y ~ s(x0, bs="cr", k=6, sp=[0.7, -1]) + s(x1, bs="ps", k=7)', True),
    ],
    ids=["numeric_by_linked", "factor_by_linked", "tensor_by", "mixed_select"],
)
def test_stage_3_wrapping_combinations_preserve_maps_and_penalty_ownership(
    formula, select
):
    """By/link/select combinations keep coefficient maps and penalties term-local."""
    data = _formula_data(seed=905, n=90)
    compiled, X, _predictor_specs = _compile_formula(data, formula, select=select)

    last_term_index = -1
    for penalty in compiled.compiled_penalties:
        assert penalty.term_index >= last_term_index
        last_term_index = penalty.term_index
        term = compiled.compiled_terms[penalty.term_index]
        assert penalty.coef_slice == term.coef_slice
        assert penalty.smoothing_index in term.smoothing_indices
        assert penalty.matrix.shape == (
            term.coef_slice.stop - term.coef_slice.start,
            term.coef_slice.stop - term.coef_slice.start,
        )
    for term in compiled.compiled_terms:
        np.testing.assert_allclose(
            term.predict_matrix(X), term.basis_train, rtol=0.0, atol=2e-11
        )
        for coefficient_map in term.coefficient_maps:
            assert coefficient_map.matrix.ndim == 2


def test_stage_3_composed_coefficient_maps_reproduce_each_constructed_basis():
    """Local and predictor maps compose numerically in their declared order."""
    data = _formula_data(seed=919, n=92)
    formula = (
        'y ~ s(x0, bs="tp", k=6, sp=0.8)'
        ' + s(x0, x1, bs="tp", k=8, sp=0.9)'
    )
    compiled, X, _predictor_specs = _compile_formula(data, formula)

    mapped_terms = [term for term in compiled.compiled_terms if term.coefficient_maps]
    assert mapped_terms
    for term in mapped_terms:
        composed = compose_coefficient_maps(term.coefficient_maps)
        assert composed is not None
        raw_prediction_basis = np.asarray(term.predict_fn(X), dtype=np.float64)
        np.testing.assert_allclose(
            raw_prediction_basis @ composed,
            term.basis_train,
            atol=2e-11,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            term.predict_matrix(X), term.basis_train, atol=2e-11, rtol=0.0
        )


@pytest.mark.parametrize(
    ("formula", "select", "method", "pred_tol", "se_tol"),
    [
        (
            'y ~ s(x0, by=z, bs="cr", k=6, id="shared", sp=0.7)'
            ' + s(x1, by=z, bs="cr", k=6, id="shared")',
            False,
            "fixed",
            2e-8,
            2e-8,
        ),
        (
            'y ~ f + s(x0, by=f, bs="cr", k=6, id="shared", sp=0.8)',
            False,
            "fixed",
            2e-8,
            2e-8,
        ),
        (
            'y ~ te(x0, x1, by=z, bs=["cr", "cr"], k=[5, 5], sp=[0.7, 1.1])',
            False,
            "fixed",
            2e-7,
            2e-7,
        ),
        (
            'y ~ s(x0, bs="cr", k=6, sp=[0.7, -1])'
            ' + s(x1, bs="ps", k=7)',
            True,
            "REML",
            3e-5,
            3e-5,
        ),
    ],
    ids=["numeric_by_linked", "factor_by_linked", "tensor_by", "mixed_select"],
)
def test_stage_3_wrapping_combinations_match_mgcv_fitted_behavior(
    formula, select, method, pred_tol, se_tol
):
    """Every supported wrapper combination preserves fitted response and SE."""
    data = _formula_data(seed=920, n=110)
    optimize = method.lower() != "fixed"
    gam = GAM(
        family="gaussian",
        formula=formula,
        select=select,
        optimize_smoothing=optimize,
        smoothing_method=method,
        smoothing_optimizer="outer_newton",
    ).fit(data=data)
    newdata = data.iloc[3::12].drop(columns=["y"]).copy()
    actual, actual_se = gam.predict(newdata, type="response", return_se=True)
    expected = _run_mgcv_predict_on_newdata(
        data,
        newdata,
        formula,
        family="gaussian",
        method=method,
        type="response",
        return_se=True,
        select=select,
        optimizer="newton" if optimize else None,
        allow_live_run=True,
    )
    np.testing.assert_allclose(
        actual, np.asarray(expected["pred"]).ravel(), atol=pred_tol, rtol=pred_tol
    )
    np.testing.assert_allclose(
        actual_se, np.asarray(expected["se"]).ravel(), atol=se_tol, rtol=se_tol
    )


def test_stage_4_multi_predictor_compilation_preserves_slices_offsets_and_intercepts():
    """Multi-predictor compilation assembles unequal predictor contracts in order."""
    data = _formula_data(seed=906, n=88).assign(
        o2=lambda frame: np.exp(0.15 + frame["off"])
    )
    formula = [
        'y ~ 1 + s(x0, bs="cr", k=6, sp=0.7) + offset(off)',
        '~ -1 + z + s(x1, bs="ps", k=7, sp=1.1) + offset(log(o2 + 1))',
    ]
    compiled, _X, predictor_specs = _compile_formula(data, formula)

    assert len(compiled.predictors) == 2
    assert [pred.has_intercept for pred in compiled.predictors] == [True, False]
    assert [spec.offset_name for spec in predictor_specs] == [
        "off",
        predictor_specs[1].offset_name,
    ]
    assert compiled.predictor_full_slices[0].start == 0
    assert compiled.predictor_full_slices[0].stop == (
        compiled.predictors[0].n_coef + 1
    )
    assert compiled.predictor_full_slices[1].start == compiled.predictor_full_slices[0].stop
    assert compiled.predictor_full_slices[1].stop == (
        compiled.predictor_full_slices[1].start + compiled.predictors[1].n_coef
    )
    assert compiled.n_coef == sum(pred.n_coef for pred in compiled.predictors)
    assert compiled.n_smoothing_params == sum(
        pred.n_smoothing_params for pred in compiled.predictors
    )
    assert [penalty.term_index for penalty in compiled.compiled_penalties] == sorted(
        penalty.term_index for penalty in compiled.compiled_penalties
    )


def test_stage_4_three_term_linked_fixed_free_and_rank_deficient_assembly():
    """Compiler ordering survives linked, fixed/free, and aliased parametric terms."""
    data = _formula_data(seed=907, n=92)
    formula = (
        'y ~ x0 + I(2*x0) + s(x0, bs="cr", k=6, id="shared")'
        ' + s(x1, bs="cr", k=6, id="shared")'
        ' + s(z, bs="ps", k=7, sp=0.9)'
    )
    compiled, _X, _predictor_specs = _compile_formula(data, formula)
    predictor = compiled.predictors[0]

    assert len(predictor.compiled_terms) == 5
    assert [term.term_type for term in predictor.compiled_terms[:2]] == [
        "parametric",
        "parametric",
    ]
    assert predictor.compiled_terms[2].smoothing_indices == predictor.compiled_terms[3].smoothing_indices
    assert predictor.smoothing_override_modes[-1] == "fixed"
    assert predictor.smoothing_override_values[-1] == 0.9
    assert [term.coef_slice.start for term in predictor.compiled_terms] == sorted(
        term.coef_slice.start for term in predictor.compiled_terms
    )


def _stage_4_general_data(
    data_factory, seed: int = 921, n: int = 130
) -> pd.DataFrame:
    data = data_factory(seed=seed, n=n)
    x = data["x"].to_numpy(dtype=np.float64)
    data["z"] = 0.35 + np.cos(1.4 * x)
    data["o1"] = 0.04 * np.sin(0.8 * x)
    data["o2"] = 0.03 * np.cos(0.6 * x)
    return data


@pytest.mark.parametrize(
    ("family", "data_factory", "formula", "expected_intercepts"),
    [
        (
            "gaulss",
            _gaulss_data,
            [
                'y ~ 1 + s(x, bs="cr", k=6, sp=0.8) + offset(o1)',
                "~ -1 + z + offset(o2)",
            ],
            [True, False],
        ),
        (
            "gammals",
            _gammals_data,
            [
                "y ~ -1 + x + offset(o1)",
                '~ 1 + s(z, bs="cr", k=6, sp=0.9) + offset(o2)',
            ],
            [False, True],
        ),
    ],
    ids=["gaulss_first_predictor_smooth", "gammals_second_predictor_smooth"],
)
def test_stage_4_multi_predictor_fitted_assembly_matches_mgcv(
    family, data_factory, formula, expected_intercepts
):
    """Supported unequal predictor blocks retain fitted response and SE parity."""
    data = _stage_4_general_data(data_factory)
    gam = GAM(
        family=family,
        formula=formula,
        optimize_smoothing=False,
        smoothing_method="fixed",
    ).fit(data=data)
    newdata = data.iloc[4::13].drop(columns=["y"]).copy()
    actual, actual_se = gam.predict(newdata, type="link", return_se=True)
    expected = _run_mgcv_predict_on_newdata(
        data,
        newdata,
        formula,
        family=family,
        method="fixed",
        type="link",
        return_se=True,
        allow_live_run=True,
    )

    assert [predictor.has_intercept for predictor in gam.predictor_specs] == expected_intercepts
    np.testing.assert_allclose(
        actual, np.asarray(expected["pred"], dtype=np.float64), atol=2e-5, rtol=2e-5
    )
    np.testing.assert_allclose(
        actual_se,
        np.asarray(expected["se"], dtype=np.float64),
        atol=2e-5,
        rtol=2e-5,
    )


@dataclass(frozen=True)
class _FitCombination:
    case_id: str
    data_factory: object
    formula: str
    family: object
    method: str
    optimizer: str
    select: bool = False
    weights_column: str | None = None
    pred_atol: float = 2e-5
    sp_log_atol: float = 3e-3
    boundary_sp_indices: tuple[int, ...] = ()


def _gaussian_combination_data() -> pd.DataFrame:
    return _formula_data(seed=908, n=110)


def _binomial_combination_data() -> pd.DataFrame:
    data = _formula_data(seed=909, n=130)
    rng = np.random.default_rng(909)
    eta = -0.2 + 0.75 * data["x0"].to_numpy() + data["off"].to_numpy()
    prob = 1.0 / (1.0 + np.exp(-eta))
    data["y"] = rng.binomial(1, prob)
    data["w"] = 1.0 + (np.arange(len(data)) % 3)
    return data


def _gamma_combination_data() -> pd.DataFrame:
    data = _make_gamma_data(seed=910, n=220)
    data["off"] = 0.03 * np.sin(data["x0"].to_numpy(dtype=np.float64))
    data["w"] = 0.6 + np.linspace(0.0, 0.8, len(data))
    data.loc[data.index[::29], "w"] = 0.0
    return data


def _negbin_combination_data() -> pd.DataFrame:
    data = _make_negbin_data(seed=911, n=140, theta=1.8).rename(
        columns={"x0": "x0", "x1": "x1"}
    )
    data["off"] = 0.08 * np.sin(data["x0"].to_numpy(dtype=np.float64))
    data["w"] = 0.5 + np.linspace(0.0, 1.0, len(data))
    return data


def _poisson_combination_data() -> pd.DataFrame:
    data = _make_poisson_data(seed=922, n=180)
    data["off"] = 0.06 * np.cos(data["x0"].to_numpy(dtype=np.float64))
    data["w"] = 0.7 + np.linspace(0.0, 0.9, len(data))
    return data


def _gaulss_combination_data() -> pd.DataFrame:
    return _gaulss_data(seed=923, n=130)


def _gammals_combination_data() -> pd.DataFrame:
    return _gammals_data(seed=924, n=130)


_FIT_COMBINATIONS = [
    _FitCombination(
        "gaussian_gcv_weight_offset_select",
        _gaussian_combination_data,
        'y ~ offset(off) + s(x0, bs="cr", k=7) + s(x1, bs="cr", k=7)',
        "gaussian",
        "GCV",
        "outer_newton",
        select=True,
        weights_column="w",
        pred_atol=2e-6,
        sp_log_atol=2e-3,
        boundary_sp_indices=(3,),
    ),
    _FitCombination(
        "binomial_probit_reml_bfgs_weight_offset_select",
        _binomial_combination_data,
        'y ~ offset(off) + s(x0, bs="cr", k=7)',
        {"name": "binomial", "link": "probit"},
        "REML",
        "bfgs",
        select=True,
        weights_column="w",
        pred_atol=2e-5,
        sp_log_atol=5e-3,
    ),
    _FitCombination(
        "gamma_log_ml_bfgs_zero_weight",
        _gamma_combination_data,
        'y ~ offset(off) + s(x0, bs="cr", k=8)',
        {"name": "gamma", "link": "log"},
        "ML",
        "bfgs",
        weights_column="w",
        pred_atol=3e-5,
        sp_log_atol=5e-3,
    ),
    _FitCombination(
        "negbin_est_reml_optim_weight_offset",
        _negbin_combination_data,
        'y ~ offset(off) + s(x0, bs="cr", k=7)',
        {"name": "negbin", "theta": 1.8, "estimate_theta": True},
        "REML",
        "optim",
        weights_column="w",
        pred_atol=5e-5,
        sp_log_atol=5e-2,
    ),
    _FitCombination(
        "poisson_reml_newton_weight_offset",
        _poisson_combination_data,
        'y ~ offset(off) + s(x0, bs="cr", k=7)',
        "poisson",
        "REML",
        "outer_newton",
        weights_column="w",
        pred_atol=2e-5,
        sp_log_atol=5e-3,
    ),
    _FitCombination(
        "gaulss_ml_newton",
        _gaulss_combination_data,
        ['y ~ s(x, bs="cr", k=6)', "~ 1"],
        "gaulss",
        "ML",
        "outer_newton",
        pred_atol=6e-6,
        sp_log_atol=2e-2,
    ),
    _FitCombination(
        "gammals_ml_newton",
        _gammals_combination_data,
        ['y ~ s(x, bs="cr", k=6)', "~ 1"],
        "gammals",
        "ML",
        "outer_newton",
        pred_atol=1e-5,
        sp_log_atol=2e-2,
    ),
]


@pytest.mark.parametrize("case", _FIT_COMBINATIONS, ids=lambda case: case.case_id)
def test_stage_6_combined_family_link_criterion_optimizer_fit_matches_mgcv(case):
    """Representative cross-products cover fit inputs that isolated tests do not."""
    data = case.data_factory()
    weights = (
        None
        if case.weights_column is None
        else data[case.weights_column].to_numpy(dtype=np.float64)
    )
    gam = GAM(
        family=case.family,
        formula=case.formula,
        select=case.select,
        optimize_smoothing=True,
        smoothing_method=case.method,
        smoothing_optimizer=case.optimizer,
    ).fit(data=data, sample_weight=weights)
    actual = gam.parity_snapshot(X=data, include_covariances=True)
    expected = _run_mgcv_snapshot(
        data,
        _normalize_python_formula_text(case.formula),
        case.family,
        "GCV.Cp" if case.method.upper() == "GCV" else case.method,
        select=case.select,
        weights_column=case.weights_column,
        optimizer="newton" if case.optimizer == "outer_newton" else case.optimizer,
    )

    if case.boundary_sp_indices:
        actual_log_sp = np.log(
            np.asarray(actual["fit"]["smoothing_params"], dtype=np.float64)
        )
        expected_log_sp = np.log(
            np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64)
        )
        assert actual_log_sp.shape == expected_log_sp.shape
        boundary = np.zeros(actual_log_sp.size, dtype=bool)
        boundary[list(case.boundary_sp_indices)] = True
        np.testing.assert_allclose(
            actual_log_sp[~boundary],
            expected_log_sp[~boundary],
            rtol=0.0,
            atol=case.sp_log_atol,
        )
        # At effective infinity, the exact optimizer endpoint is not identified:
        # both penalties remove the same component.  Compare the boundary state
        # and fitted behavior, not arbitrary endpoint magnitudes.
        assert np.all(actual_log_sp[boundary] > 10.0)
        assert np.all(expected_log_sp[boundary] > 10.0)

    _assert_basic_mgcv_parity(
        actual,
        expected,
        pred_atol=case.pred_atol,
        pred_rtol=case.pred_atol,
        sp_log_atol=case.sp_log_atol,
        check_sp=not bool(case.boundary_sp_indices),
        criterion_atol=max(2e-4, case.pred_atol * 10.0),
    )


def test_stage_6_near_separated_binomial_fixed_fit_matches_mgcv():
    """P-IRLS remains stable near separation and retains prediction uncertainty."""
    x = np.linspace(-2.0, 2.0, 180)
    y = (x > 0.0).astype(np.float64)
    y[[43, 136]] = 1.0 - y[[43, 136]]
    data = pd.DataFrame({"y": y, "x": x})
    formula = 'y ~ s(x, bs="cr", k=8, sp=0.8)'
    gam = GAM(
        family={"name": "binomial", "link": "logit"},
        formula=formula,
        optimize_smoothing=False,
        smoothing_method="fixed",
    ).fit(data=data)
    newdata = pd.DataFrame({"x": np.linspace(-1.8, 1.8, 17)})
    actual, actual_se = gam.predict(newdata, type="response", return_se=True)
    expected = _run_mgcv_predict_on_newdata(
        data,
        newdata,
        formula,
        family={"name": "binomial", "link": "logit"},
        method="fixed",
        type="response",
        return_se=True,
        allow_live_run=True,
    )
    np.testing.assert_allclose(
        actual, np.asarray(expected["pred"]).ravel(), atol=2e-6, rtol=2e-6
    )
    np.testing.assert_allclose(
        actual_se, np.asarray(expected["se"]).ravel(), atol=2e-6, rtol=2e-6
    )


@pytest.mark.parametrize(
    ("family", "data_factory", "formula"),
    [
        ("gaussian", _gaussian_combination_data, 'y ~ s(x0, bs="cr", k=6)'),
        ("poisson", _poisson_combination_data, 'y ~ s(x0, bs="cr", k=6)'),
        ("gaulss", _gaulss_combination_data, ['y ~ s(x, bs="cr", k=6)', "~ 1"]),
    ],
    ids=["gaussian", "poisson", "gaulss"],
)
def test_stage_6_all_zero_prior_weights_fail_explicitly(family, data_factory, formula):
    """Every fit backend rejects a data set with no positively weighted row."""
    data = data_factory()
    with pytest.raises(ValueError, match="sample_weight must sum to a positive value"):
        GAM(family=family, formula=formula).fit(
            data=data,
            sample_weight=np.zeros(len(data), dtype=np.float64),
        )


def test_stage_7_terms_filter_unconditional_se_matches_mgcv():
    """Term filtering and smoothing uncertainty compose on one newdata surface."""
    data = _make_gaussian_data(seed=912, n=120)
    formula = 'y ~ s(x0, bs="cr", k=7) + s(x1, bs="cr", k=7)'
    gam = GAM(
        family="gaussian",
        formula=formula,
        optimize_smoothing=True,
        smoothing_method="REML",
        smoothing_optimizer="outer_newton",
    ).fit(data=data)
    newdata = data.iloc[::7].drop(columns=["y"]).copy()
    actual_fit, actual_se = gam.predict(
        newdata,
        type="terms",
        return_se=True,
        terms=["s(x0)"],
        cov=gam.vcov(unconditional=True),
    )
    expected = _run_mgcv_predict_on_newdata(
        data,
        newdata,
        formula,
        family="gaussian",
        method="REML",
        type="terms",
        return_se=True,
        unconditional=True,
        terms=["s(x0)"],
        optimizer="newton",
    )

    np.testing.assert_allclose(actual_fit, expected["pred"], rtol=2e-7, atol=2e-7)
    np.testing.assert_allclose(actual_se, expected["se"], rtol=2e-7, atol=2e-7)


@pytest.mark.parametrize(
    ("data_factory", "formula", "tol"),
    [
        (lambda: _formula_data(seed=925, n=120), 'y ~ s(f, bs="re")', 2e-5),
        (
            lambda: _formula_data(seed=925, n=120),
            'y ~ te(x0, x1, bs=["cr", "cr"], k=[5, 5])',
            2e-5,
        ),
        (
            lambda: _formula_data(seed=925, n=120),
            'y ~ s(x0, bs="cr", k=6, id="shared")'
            ' + s(x1, bs="cr", k=6, id="shared")',
            3e-6,
        ),
        (_make_fs_data, 'y ~ s(f, x, bs="fs", k=6)', 1e-2),
        (_make_sz_data, 'y ~ s(f1, f2, x, bs="sz", k=6)', 3e-2),
        (
            lambda: _formula_data(seed=931, n=120),
            'y ~ f + s(x0, by=f, bs="cr", k=6)',
            3e-5,
        ),
        (
            lambda: _formula_data(seed=932, n=120),
            'y ~ ti(x0, x1, bs=["cr", "cr"], k=[5, 5])',
            3e-4,
        ),
    ],
    ids=[
        "random_effect",
        "tensor",
        "linked_id",
        "fs",
        "sz",
        "factor_by",
        "ti",
    ],
)
def test_stage_7_structured_terms_unconditional_se_matches_mgcv(
    data_factory, formula, tol
):
    """Structured coefficient blocks retain term inference with SP uncertainty."""
    data = data_factory()
    gam = GAM(
        family="gaussian",
        formula=formula,
        optimize_smoothing=True,
        smoothing_method="REML",
        smoothing_optimizer="outer_newton",
    ).fit(data=data)
    newdata = data.iloc[6::13].drop(columns=["y"]).copy()
    actual, actual_se = gam.predict(
        newdata,
        type="terms",
        return_se=True,
        cov=gam.vcov(unconditional=True),
    )
    expected = _run_mgcv_predict_on_newdata(
        data,
        newdata,
        formula,
        family="gaussian",
        method="REML",
        type="terms",
        return_se=True,
        unconditional=True,
        optimizer="newton",
        allow_live_run=True,
    )
    np.testing.assert_allclose(actual, expected["pred"], atol=tol, rtol=tol)
    np.testing.assert_allclose(actual_se, expected["se"], atol=tol, rtol=tol)


@pytest.mark.parametrize(
    "formula",
    [
        'y ~ s(f, bs="re", sp=0.8)',
        'y ~ f + s(x0, by=f, bs="cr", k=6, sp=0.8)',
        'y ~ te(x0, x1, bs=["cr", "cr"], k=[5, 5], sp=[0.7, 1.1])',
        'y ~ s(x0, bs="cr", k=6, id="shared", sp=0.8) + s(x1, bs="cr", k=6, id="shared")',
    ],
    ids=["random_effect", "factor_by", "tensor", "linked_id"],
)
def test_stage_7_structured_term_newdata_response_matches_mgcv(formula):
    """Structured term families share the same fit/newdata prediction contract."""
    data = _formula_data(seed=913, n=90)
    newdata = data.iloc[5::11].drop(columns=["y"]).copy()
    gam = GAM(
        family="gaussian",
        formula=formula,
        optimize_smoothing=False,
        smoothing_method="fixed",
    ).fit(data=data)
    actual, actual_se = gam.predict(newdata, type="response", return_se=True)
    expected = _run_mgcv_predict_on_newdata(
        data,
        newdata,
        formula,
        family="gaussian",
        method="fixed",
        type="response",
        return_se=True,
    )
    np.testing.assert_allclose(
        actual, np.asarray(expected["pred"]).ravel(), rtol=2e-7, atol=2e-7
    )
    np.testing.assert_allclose(
        actual_se, np.asarray(expected["se"]).ravel(), rtol=2e-7, atol=2e-7
    )


def test_stage_7_prediction_rejects_unseen_factor_levels_and_nan_explicitly():
    """Unsupported newdata values fail at prediction rather than degrading silently."""
    data = _formula_data(seed=914, n=75)
    formula = 'y ~ f + s(x0, by=f, bs="cr", k=6, sp=0.8)'
    gam = GAM(family="gaussian", formula=formula).fit(data=data)

    unseen = data.iloc[:3].drop(columns=["y"]).copy()
    unseen["f"] = pd.Categorical(["new", "new", "new"])
    with pytest.raises((KeyError, ValueError), match="level|factor|column"):
        gam.predict(unseen, type="response")

    with_nan = data.iloc[:3].drop(columns=["y"]).copy()
    with_nan.loc[with_nan.index[0], "x0"] = np.nan
    with pytest.raises(ValueError, match="NaN|Inf"):
        gam.predict(with_nan, type="response")


def test_stage_7_fit_diagnostics_snapshot_and_persistence_roundtrip(tmp_path):
    """Diagnostics, parity serialization, and model persistence consume one fit state."""
    data = _make_gaussian_data(seed=915, n=100)
    formula = 'y ~ s(x0, bs="cr", k=7) + s(x1, bs="cr", k=7)'
    gam = GAM(
        family="gaussian",
        formula=formula,
        optimize_smoothing=True,
        smoothing_method="REML",
    ).fit(data=data)
    prediction = gam.predict(data, type="response")
    snapshot = gam.parity_snapshot(X=data, include_covariances=True)
    concurvity = gam.concurvity(full=False)
    k_table = gam.k_check(subsample=80, n_rep=20, seed=17)
    check = gam.gam_check(k_sample=80, k_rep=20, seed=17)

    assert snapshot["fit"]["coef_full"] is not None
    assert concurvity["labels"][0] == "para"
    assert concurvity["labels"][1:] == [
        's(x0, bs="cr", k=7)',
        's(x1, bs="cr", k=7)',
    ]
    for measure in concurvity["measure_names"]:
        assert np.asarray(concurvity["values"][measure]).shape == (3, 3)
    assert len(k_table) == 2
    assert check["mgcv_comparable"] is not None

    path = tmp_path / "gam.pkl"
    assert gam.save_model(path) == path
    restored = GAM.load_model(path)
    np.testing.assert_allclose(
        restored.predict(data, type="response"), prediction, rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        restored.vcov(unconditional=True), gam.vcov(unconditional=True), rtol=0.0, atol=0.0
    )


def test_stage_7_persistence_rejects_corrupt_and_non_model_payloads(tmp_path):
    """Persistence failures are explicit for damaged or wrong-type state."""
    corrupt = tmp_path / "corrupt.pkl"
    corrupt.write_bytes(b"not a pickle payload")
    with pytest.raises((pickle.UnpicklingError, EOFError)):
        GAM.load_model(corrupt)

    wrong_type = tmp_path / "wrong-type.pkl"
    with wrong_type.open("wb") as stream:
        pickle.dump({"_fitted": True}, stream)
    with pytest.raises(TypeError, match=r"contains dict, not GAM\.$"):
        GAM.load_model(wrong_type)


@pytest.mark.parametrize(
    ("case_id", "family", "data_factory", "formula"),
    [
        (
            "random_effect",
            "gaussian",
            lambda: _formula_data(seed=926, n=84),
            'y ~ s(f, bs="re", sp=0.8)',
        ),
        (
            "tensor",
            "gaussian",
            lambda: _formula_data(seed=927, n=84),
            'y ~ te(x0, x1, bs=["cr", "cr"], k=[5, 5], sp=[0.7, 1.1])',
        ),
        (
            "linked_id",
            "gaussian",
            lambda: _formula_data(seed=928, n=84),
            'y ~ s(x0, bs="cr", k=6, id="shared", sp=0.8)'
            ' + s(x1, bs="cr", k=6, id="shared")',
        ),
        (
            "gaulss",
            "gaulss",
            lambda: _gaulss_data(seed=929, n=90),
            ['y ~ s(x, bs="cr", k=6, sp=0.8)', "~ 1"],
        ),
        (
            "fs",
            "gaussian",
            _make_fs_data,
            'y ~ s(f, x, bs="fs", k=6, xt="cr", sp=[0.8, 0.8, 0.8])',
        ),
        (
            "sz",
            "gaussian",
            _make_sz_data,
            'y ~ s(f1, f2, x, bs="sz", k=6, id="shared", sp=0.8)',
        ),
        (
            "ti",
            "gaussian",
            lambda: _formula_data(seed=933, n=84),
            'y ~ ti(x0, x1, bs=["cr", "cr"], k=[5, 5], sp=[0.7, 1.1])',
        ),
        (
            "ti_mixed_shrinkage",
            "gaussian",
            lambda: _formula_data(seed=936, n=84),
            'y ~ ti(x0, x1, bs=["cs", "ps"], k=[5, 6], sp=[0.7, 1.1])',
        ),
        (
            "factor_by",
            "gaussian",
            lambda: _formula_data(seed=934, n=84),
            'y ~ f + s(x0, by=f, bs="cr", k=6, id="factor_curve", sp=0.8)',
        ),
        (
            "gammals",
            "gammals",
            lambda: _gammals_data(seed=935, n=90),
            ['y ~ s(x, bs="cr", k=6, sp=0.8)', "~ 1"],
        ),
        (
            "binomial_pirls",
            {"name": "binomial", "link": "probit"},
            _binomial_combination_data,
            'y ~ s(x0, bs="cr", k=7, sp=0.8)',
        ),
        (
            "poisson_pirls",
            "poisson",
            _poisson_combination_data,
            'y ~ s(x0, bs="cr", k=7, sp=0.8)',
        ),
        (
            "gamma_pirls",
            {"name": "gamma", "link": "log"},
            _gamma_combination_data,
            'y ~ s(x0, bs="cr", k=7, sp=0.8)',
        ),
    ],
    ids=[
        "random_effect",
        "tensor",
        "linked_id",
        "gaulss",
        "fs",
        "sz",
        "ti",
        "ti_mixed_shrinkage",
        "factor_by",
        "gammals",
        "binomial_pirls",
        "poisson_pirls",
        "gamma_pirls",
    ],
)
def test_stage_7_structured_and_general_persistence_roundtrip(
    tmp_path, case_id, family, data_factory, formula
):
    """Persistence retains every supported structured prediction parameterization."""
    data = data_factory()
    gam = GAM(
        family=family,
        formula=formula,
        optimize_smoothing=False,
        smoothing_method="fixed",
    ).fit(data=data)
    expected_response = gam.predict(data, type="response")
    expected_lpmatrix = gam.predict(data, type="lpmatrix")
    expected_cov = gam.vcov(unconditional=True)

    path = tmp_path / f"{case_id}.pkl"
    assert gam.save_model(path) == path
    restored = GAM.load_model(path)
    np.testing.assert_allclose(
        restored.predict(data, type="response"),
        expected_response,
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        restored.predict(data, type="lpmatrix"),
        expected_lpmatrix,
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        restored.vcov(unconditional=True), expected_cov, atol=0.0, rtol=0.0
    )


@pytest.mark.parametrize(
    ("data_factory", "family", "formulas", "atol"),
    [
        (
            lambda: _formula_data(seed=930, n=120),
            "gaussian",
            [
                "y ~ x0 + I(2*x0)",
                'y ~ x0 + I(2*x0) + s(x1, bs="cr", k=6, sp=0.8)',
                (
                    'y ~ x0 + I(2*x0) + s(x1, bs="cr", k=6, sp=0.8)'
                    ' + I(x1**2)'
                ),
            ],
            8e-6,
        ),
        (
            _binomial_combination_data,
            "binomial",
            [
                "y ~ x0 + I(2*x0)",
                'y ~ x0 + I(2*x0) + s(x1, bs="cr", k=6, sp=0.8)',
                (
                    'y ~ x0 + I(2*x0) + s(x1, bs="cr", k=6, sp=0.8)'
                    ' + I(x1**2)'
                ),
            ],
            2e-5,
        ),
        (
            _poisson_combination_data,
            "poisson",
            [
                "y ~ x0 + I(2*x0)",
                'y ~ x0 + I(2*x0) + s(x1, bs="cr", k=6, sp=0.8)',
                (
                    'y ~ x0 + I(2*x0) + s(x1, bs="cr", k=6, sp=0.8)'
                    ' + I(x1**2)'
                ),
            ],
            2e-5,
        ),
        (
            _gamma_combination_data,
            "gamma",
            [
                "y ~ x0 + I(2*x0)",
                'y ~ x0 + I(2*x0) + s(x1, bs="cr", k=6, sp=0.8)',
                (
                    'y ~ x0 + I(2*x0) + s(x1, bs="cr", k=6, sp=0.8)'
                    ' + I(x1**2)'
                ),
            ],
            2e-4,
        ),
        (
            lambda: _gaulss_data(seed=942, n=120),
            "gaulss",
            [
                ["y ~ 1", "~ 1"],
                ['y ~ s(x, bs="cr", k=6, sp=0.8)', "~ 1"],
                [
                    'y ~ s(x, bs="cr", k=6, sp=0.8)',
                    '~ s(x, bs="cr", k=6, sp=1.1)',
                ],
            ],
            2e-5,
        ),
        (
            lambda: _gammals_data(seed=943, n=120),
            "gammals",
            [
                ["y ~ 1", "~ 1"],
                ['y ~ s(x, bs="cr", k=6, sp=0.8)', "~ 1"],
                [
                    'y ~ s(x, bs="cr", k=6, sp=0.8)',
                    '~ s(x, bs="cr", k=6, sp=1.1)',
                ],
            ],
            2e-4,
        ),
    ],
    ids=["gaussian", "binomial", "poisson", "gamma", "gaulss", "gammals"],
)
def test_stage_7_three_model_rank_deficient_anova_matches_mgcv(
    data_factory, family, formulas, atol
):
    """Three-model ANOVA matches mgcv across supported fit engines."""
    data = data_factory()
    models = [
        GAM(
            family=family,
            formula=formula,
            optimize_smoothing=False,
            smoothing_method="fixed",
        ).fit(data=data)
        for formula in formulas
    ]
    actual = models[0].anova(*models[1:], test="Chisq")
    expected = _run_mgcv_anova(data, formulas, family, "fixed", test="Chisq")
    expected_values = np.asarray(
        [
            [np.nan if value in {None, "NA"} else float(value) for value in row]
            for row in expected["table"]["values"]
        ],
        dtype=np.float64,
    )

    assert actual.table.columns.tolist() == expected["table"]["columns"]
    np.testing.assert_allclose(
        actual.table.to_numpy(dtype=np.float64),
        expected_values,
        atol=atol,
        rtol=atol,
        equal_nan=True,
    )
