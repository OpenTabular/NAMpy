"""Integrated parity coverage for derivative-penalized B-splines (``bs='bs'``)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from nampy.gam.splines.univariate.bs import build_derivative_bspline_setup
from tests.mgcv_parity_utils import (
    _assert_basic_mgcv_parity,
    _fit_nampy_model,
    _fit_nampy_snapshot,
    _run_mgcv_snapshot,
)


def _bs_data(seed=251, n=190):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-2.0, 2.0, size=n)
    x1 = rng.uniform(-1.5, 2.5, size=n)
    z = 0.8 + rng.uniform(-0.4, 0.7, size=n)
    f = np.asarray(["a", "b", "c"], dtype=object)[np.arange(n) % 3]
    f1 = np.asarray(["u", "v"], dtype=object)[np.arange(n) % 2]
    y = (
        0.2
        + z * np.sin(1.2 * x0)
        + 0.35 * x1**2
        + 0.2 * (f == "b")
        - 0.15 * (f1 == "v")
        + rng.normal(scale=0.12, size=n)
    )
    return pd.DataFrame({"y": y, "x0": x0, "x1": x1, "z": z, "f": f, "f1": f1})


def _assert_snapshot_fit(actual, expected, *, atol=4e-8):
    for key in ("response", "link"):
        np.testing.assert_allclose(
            actual["predictions"][key],
            expected["predictions"][key],
            atol=atol,
            rtol=atol,
        )
    np.testing.assert_allclose(
        actual["fit"]["edf_total"], expected["fit"]["edf_total"], atol=atol, rtol=atol
    )


def test_bs_numeric_by_select_true_matches_mgcv():
    data = _bs_data(seed=252)
    formula = 'y ~ s(x0, by=z, bs="bs", k=9)'
    actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML", select=True)
    expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML", select=True)
    assert len(actual["fit"]["smoothing_params"]) == 2
    _assert_basic_mgcv_parity(
        actual,
        expected,
        pred_atol=3e-6,
        pred_rtol=3e-6,
        sp_log_atol=5e-5,
        criterion_atol=4e-8,
    )


def test_bs_factor_by_fixed_sp_matches_mgcv():
    data = _bs_data(seed=253)
    formula = 'y ~ s(x0, by=f, bs="bs", k=8, sp=0.7)'
    actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
    expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
    _assert_snapshot_fit(actual, expected)


def test_bs_linked_multi_penalty_terms_pool_basis_and_share_two_sp():
    data = _bs_data(seed=254)
    formula = (
        'y ~ s(x0, bs="bs", k=9, m=[3,2,0], id="derivative", sp=[0.6,0.8])'
        ' + s(x1, bs="bs", k=9, m=[3,2,0], id="derivative")'
    )
    model = _fit_nampy_model(data, formula, "gaussian", "fixed")
    expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
    assert len(model.smoothing_params) == 2
    runtimes = [
        term.predict_fn.__self__
        for term in model.gam_result_.compiled_model.compiled_terms
        if term.term_type == "smooth"
    ]
    assert len(runtimes) == 2
    np.testing.assert_allclose(runtimes[0]._setup.knots, runtimes[1]._setup.knots)
    pooled = np.concatenate([data["x0"].to_numpy(), data["x1"].to_numpy()])
    assert runtimes[0]._setup.knots[3] <= np.min(pooled)
    assert runtimes[0]._setup.knots[-4] >= np.max(pooled)
    actual = model.parity_snapshot(X=data, include_covariances=True)
    _assert_snapshot_fit(actual, expected, atol=8e-8)


def test_bs_fixed_term_has_no_penalty_and_matches_mgcv():
    data = _bs_data(seed=255)
    formula = 'y ~ s(x0, bs="bs", k=8, fx=True)'
    model = _fit_nampy_model(data, formula, "gaussian", "fixed")
    assert model.gam_result_.compiled_model.compiled_penalties == ()
    actual = model.parity_snapshot(X=data, include_covariances=True)
    expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
    _assert_snapshot_fit(actual, expected)


@pytest.mark.parametrize(
    "formula",
    [
        'y ~ te(x0, x1, bs=["bs","bs"], k=[5,6], m=[2,1], sp=[0.6,0.8])',
        'y ~ ti(x0, x1, bs=["bs","bs"], k=[5,6], m=[2,1], sp=[0.6,0.8])',
    ],
    ids=["te", "ti"],
)
def test_bs_tensor_marginal_fixed_sp_fit_matches_mgcv(formula):
    data = _bs_data(seed=256, n=170)
    actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
    expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
    _assert_snapshot_fit(actual, expected, atol=3e-7)


def test_bs_multiply_penalized_tensor_margin_is_rejected():
    data = _bs_data(seed=257, n=80)
    formula = 'y ~ te(x0, x1, bs=["bs","bs"], k=[6,6], m=[[3,2,1],[3,2]])'
    with pytest.raises(NotImplementedError, match="multiple penalties"):
        GAM(formula=formula).fit(data=data)


@pytest.mark.parametrize(
    "formula",
    [
        'y ~ s(f, x0, bs="fs", k=6, xt="bs", m=[3,2], sp=[0.7,0.9,1.1])',
        'y ~ s(f, f1, x0, bs="sz", k=6, xt="bs", m=[3,2], id="shared", sp=0.7)',
    ],
    ids=["fs", "sz"],
)
def test_bs_factor_smooth_base_fixed_sp_fit_matches_mgcv(formula):
    data = _bs_data(seed=258, n=180)
    actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
    expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
    _assert_snapshot_fit(actual, expected, atol=4e-7)


@pytest.mark.parametrize("basis", ["fs", "sz"])
def test_bs_multiply_penalized_factor_smooth_base_is_rejected(basis):
    data = _bs_data(seed=259, n=80)
    formula = f'y ~ s(f, x0, bs="{basis}", k=6, xt="bs", m=[3,2,1])'
    with pytest.raises(NotImplementedError, match="multiply penalized basis"):
        GAM(formula=formula).fit(data=data)


def test_bs_array_api_and_persistence_preserve_extrapolated_prediction(tmp_path):
    data = _bs_data(seed=260, n=150)
    features = data[["x0", "x1"]]
    model = GAM(
        family="gaussian",
        basis="bs",
        k=8,
        optimize_smoothing=False,
        smoothing_params=[0.5, 0.8],
    ).fit(X=features, y=data["y"].to_numpy(dtype=np.float64))
    newdata = pd.DataFrame({"x0": [-4.0, 0.0, 4.0], "x1": [-3.0, 1.0, 5.0]})
    expected = model.predict(newdata, type="link")
    path = tmp_path / "bs.pkl"
    model.save_model(path)
    restored = GAM.load_model(path)
    np.testing.assert_allclose(restored.predict(newdata, type="link"), expected)


@pytest.mark.parametrize(
    "formula,message",
    [
        ('y ~ s(x0, bs="bs", k=2, m=[3,2])', "basis dimension too small"),
        ('y ~ s(x0, bs="bs", k=6, m=[3,4])', "non-existent derivative"),
        ('y ~ s(x0, bs="bs", k=6, m=[3,2,2])', "multiple penalties"),
        ('y ~ s(x0, bs="bs", k=6, m=[3.5,2])', "non-negative integers"),
    ],
)
def test_bs_invalid_orders_fail_loudly(formula, message):
    data = _bs_data(seed=261, n=40)
    with pytest.raises(ValueError, match=message):
        GAM(formula=formula).fit(data=data)


def test_bs_knot_validation_and_unique_covariate_warning():
    x = np.linspace(0.0, 1.0, 30)
    kwargs = {
        "feature_index": 0,
        "feature_name": "x",
        "bs_dim": 8,
        "m": (3, 2),
    }
    with pytest.raises(ValueError, match="knot range does not include data"):
        build_derivative_bspline_setup(x, knots=[0.2, 0.8], **kwargs)
    with pytest.raises(ValueError, match="there should be 12 supplied knots"):
        build_derivative_bspline_setup(x, knots=np.linspace(0.0, 1.0, 11), **kwargs)
    with pytest.warns(UserWarning, match="larger than number of unique"):
        build_derivative_bspline_setup(np.repeat([0.0, 0.5, 1.0], 10), **kwargs)
