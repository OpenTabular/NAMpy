"""Smooth-owned offsets follow mgcv predict.gam term filtering semantics."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from nampy.gam.fit.offsets import resolve_prediction_offset, smooth_offsets
from nampy.gam.model_state import _coef_full, _term_blocks_seq
from tests.smooths.test_mgcv_soap_parity import _case

pytestmark = pytest.mark.surface_regression


@pytest.fixture(params=[False, True], scope="module")
def known_boundary_model(request):
    case = _case(geometry="known", by=request.param)
    formula = f"y ~ s(x,z,bs='so',k=8,xt={case['xt']!r}" + (
        ",by=v)" if request.param else ")"
    )
    model = GAM(
        formula=formula,
        knots=case["knots"],
        optimize_smoothing=False,
        smoothing_method="fixed",
        smoothing_params=[0.37],
    )
    data = pd.DataFrame(case["data"])
    model.fit(data=data)
    term = next(t for t in _term_blocks_seq(model) if t.offset_train is not None)
    return model, data, pd.DataFrame(case["newdata"]), term


@pytest.mark.parametrize("in_sample", [False, True])
def test_excluding_soap_removes_its_known_boundary_offset(
    known_boundary_model, in_sample
):
    model, data, newdata, term = known_boundary_model
    X = None if in_sample else newdata
    count = len(data) if in_sample else len(newdata)
    explicit = np.linspace(0.2, 0.4, count)
    intercept = model.predict(X, type="lpmatrix", terms=["(Intercept)"]) @ _coef_full(
        model
    )
    for options in ({"exclude": [term.label]}, {"terms": ["(Intercept)"]}):
        actual = model.predict(X, type="link", offset=explicit, **options)
        np.testing.assert_allclose(actual, intercept + explicit, atol=1e-12)


@pytest.mark.parametrize("kind", ["terms", "iterms"])
@pytest.mark.parametrize("in_sample", [False, True])
def test_term_predictions_include_known_boundary_offset(
    known_boundary_model, kind, in_sample
):
    model, _, newdata, term = known_boundary_model
    X = None if in_sample else newdata
    values = model.predict(X, type=kind)
    intercept = model.predict(X, type="lpmatrix", terms=["(Intercept)"]) @ _coef_full(
        model
    )
    np.testing.assert_allclose(
        values.sum(axis=1) + intercept, model.predict(X, type="link"), atol=1e-12
    )
    explicit = np.full(len(values), 0.5)
    np.testing.assert_allclose(
        model.predict(X, type=kind, offset=explicit), values, atol=1e-12
    )


def test_excluded_soap_does_not_require_missing_coordinates(known_boundary_model):
    model, _, newdata, term = known_boundary_model
    result = model.predict(
        pd.DataFrame(index=newdata.index),
        type="link",
        exclude=[term.label],
        newdata_guaranteed=True,
    )
    expected = model.predict(
        newdata, type="lpmatrix", terms=["(Intercept)"]
    ) @ _coef_full(model)
    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_multi_predictor_smooth_offset_accumulation_and_skip():
    first = np.array([1.0, 2.0, 3.0])
    second = np.array([4.0, 5.0, 6.0])
    shared = np.array([0.2, 0.3, 0.4])
    terms = [
        SimpleNamespace(
            term_id=name,
            offset_train=value,
            offset_fn=lambda X, value=value: value[: len(X)],
            predictor_indices=indices,
        )
        for name, value, indices in [
            ("a", first, (0,)),
            ("b", second, (1,)),
            ("c", shared, (0, 1)),
        ]
    ]
    model = SimpleNamespace(
        gam_result_=SimpleNamespace(
            compiled_model=SimpleNamespace(
                compiled_terms=terms, predictors=[None, None]
            )
        ),
        n_samples_=3,
        offset_predict_default_=[np.ones(3), None],
    )
    np.testing.assert_allclose(smooth_offsets(model), [first + shared, second + shared])
    np.testing.assert_allclose(
        smooth_offsets(model, skip_term_ids={"c"}), [first, second]
    )
    np.testing.assert_allclose(
        resolve_prediction_offset(model, None, None),
        [first + shared + 1, second + shared],
    )
    np.testing.assert_allclose(
        resolve_prediction_offset(model, np.ones((2, 1)), None),
        [(first + shared)[:2], (second + shared)[:2]],
    )


def test_ordinary_offsets_keep_fit_and_prediction_semantics():
    x = np.linspace(0, 1, 30)
    data = pd.DataFrame({"x": x, "o": x * x, "y": 1 + 2 * x + x * x})
    model = GAM(formula="y ~ x + offset(o)", optimize_smoothing=False)
    model.fit(data=data)
    np.testing.assert_allclose(model.predict(), data.y, atol=1e-12)
    np.testing.assert_allclose(model.predict(data), data.y, atol=1e-12)
    np.testing.assert_allclose(
        model.predict(type="terms").sum(axis=1), 2 * x, atol=1e-12
    )
    np.testing.assert_allclose(
        model.predict(data, exclude=["x"]), 1 + x * x, atol=1e-12
    )


def test_general_family_known_boundary_offset_prediction_contract():
    from nampy.gam.model_state import _predictor_full_indices

    case = _case(geometry="known")
    model = GAM(
        formula=[f"y ~ s(x,z,bs='so',k=8,xt={case['xt']!r})", "~ 1"],
        family="gaulss",
        knots=case["knots"],
        optimize_smoothing=False,
        smoothing_method="fixed",
        smoothing_params=[0.37],
    )
    model.fit(data=pd.DataFrame(case["data"]))
    newdata = pd.DataFrame(case["newdata"])
    term = next(t for t in _term_blocks_seq(model) if t.offset_train is not None)
    matrix = model.predict(newdata, type="lpmatrix")
    index = _predictor_full_indices(model)
    linear = np.column_stack([matrix[:, i] @ _coef_full(model)[i] for i in index])
    expected_offset = term.offset_fn(newdata[model.feature_names].to_numpy())
    link = model.predict(newdata, type="link")
    np.testing.assert_allclose(link[:, 0], linear[:, 0] + expected_offset, atol=1e-12)
    np.testing.assert_allclose(link[:, 1], linear[:, 1], atol=1e-12)
    contribution = model.predict(newdata, type="terms")[:, 0]
    excluded = model.predict(newdata, type="link", exclude=[term.label])
    np.testing.assert_allclose(link[:, 0] - excluded[:, 0], contribution, atol=1e-12)
    np.testing.assert_allclose(link[:, 1], excluded[:, 1], atol=1e-12)
    outside = newdata.copy()
    outside.loc[0, "x"] = -1.0
    outside_link = model.predict(outside, type="link")
    assert np.isnan(outside_link[0, 0])
    assert np.isfinite(outside_link[0, 1])
    np.testing.assert_allclose(outside_link[1:], link[1:], atol=1e-12)
    with pytest.raises(ValueError, match="NaN or Inf"):
        resolve_prediction_offset(
            model, newdata[model.feature_names].to_numpy(),
            np.full(len(newdata), np.nan),
        )
