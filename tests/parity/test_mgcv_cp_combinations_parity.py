"""Integrated parity coverage for cyclic P-splines (``bs='cp'``)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from nampy.gam.splines.univariate.ps import build_pspline_term_setup
from tests.mgcv_parity_utils import (
    _assert_basic_mgcv_parity,
    _fit_nampy_model,
    _fit_nampy_snapshot,
    _run_mgcv_snapshot,
)


def _cp_data(seed=231, n=190):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(0.0, 2.0 * np.pi, size=n)
    x1 = rng.uniform(-1.0, 5.0, size=n)
    z = 0.7 + rng.uniform(-0.5, 0.8, size=n)
    f = np.asarray(["a", "b", "c"], dtype=object)[np.arange(n) % 3]
    f1 = np.asarray(["u", "v"], dtype=object)[np.arange(n) % 2]
    y = (
        0.3
        + z * np.sin(x0)
        + 0.45 * np.cos(x1)
        + 0.2 * (f == "b")
        - 0.15 * (f1 == "v")
        + rng.normal(scale=0.12, size=n)
    )
    return pd.DataFrame({"y": y, "x0": x0, "x1": x1, "z": z, "f": f, "f1": f1})


def _assert_snapshot_fit(actual, expected, *, atol=3e-8):
    for key in ("response", "link"):
        np.testing.assert_allclose(
            actual["predictions"][key], expected["predictions"][key], atol=atol, rtol=atol
        )
    np.testing.assert_allclose(
        actual["fit"]["edf_total"], expected["fit"]["edf_total"], atol=atol, rtol=atol
    )


def test_cp_numeric_by_select_true_matches_mgcv():
    data = _cp_data(seed=232)
    formula = 'y ~ s(x0, by=z, bs="cp", k=9)'
    actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML", select=True)
    expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML", select=True)
    assert len(actual["fit"]["smoothing_params"]) == 2
    _assert_basic_mgcv_parity(
        actual,
        expected,
        pred_atol=2e-8,
        pred_rtol=2e-8,
        sp_log_atol=2e-7,
        criterion_atol=2e-8,
    )


def test_cp_centered_select_true_keeps_one_penalty_like_mgcv():
    data = _cp_data(seed=240)
    formula = 'y ~ s(x0, bs="cp", k=9, sp=0.7)'
    actual = _fit_nampy_snapshot(
        data, formula, "gaussian", "fixed", select=True
    )
    expected = _run_mgcv_snapshot(
        data, formula, "gaussian", "REML", select=True
    )
    assert len(actual["fit"]["smoothing_params"]) == 1
    _assert_snapshot_fit(actual, expected)


def test_cp_factor_by_fixed_sp_matches_mgcv():
    data = _cp_data(seed=233)
    formula = 'y ~ s(x0, by=f, bs="cp", k=8, sp=0.7)'
    actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
    expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
    _assert_snapshot_fit(actual, expected)


def test_cp_linked_terms_pool_basis_and_share_sp_like_mgcv():
    data = _cp_data(seed=234)
    formula = (
        'y ~ s(x0, bs="cp", k=9, id="periodic")'
        ' + s(x1, bs="cp", k=9, id="periodic")'
    )
    model = _fit_nampy_model(data, formula, "gaussian", "REML")
    expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
    assert len(model.smoothing_params) == 1
    runtimes = [
        term.predict_fn.__self__
        for term in model.gam_result_.compiled_model.compiled_terms
        if term.term_type == "smooth"
    ]
    assert len(runtimes) == 2
    np.testing.assert_allclose(runtimes[0]._setup.knots, runtimes[1]._setup.knots)
    pooled = np.concatenate([data["x0"].to_numpy(), data["x1"].to_numpy()])
    assert np.min(runtimes[0]._setup.knots) == pytest.approx(np.min(pooled))
    assert np.max(runtimes[0]._setup.knots) == pytest.approx(np.max(pooled))
    actual = model.parity_snapshot(X=data, include_covariances=True)
    _assert_basic_mgcv_parity(
        actual,
        expected,
        pred_atol=2e-8,
        pred_rtol=2e-8,
        sp_log_atol=2e-7,
        criterion_atol=2e-8,
    )


def test_cp_fixed_term_has_no_penalty_and_matches_mgcv():
    data = _cp_data(seed=235)
    formula = 'y ~ s(x0, bs="cp", k=8, fx=True)'
    model = _fit_nampy_model(data, formula, "gaussian", "fixed")
    assert model.gam_result_.compiled_model.compiled_penalties == ()
    actual = model.parity_snapshot(X=data, include_covariances=True)
    expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
    _assert_snapshot_fit(actual, expected)


@pytest.mark.parametrize(
    "formula",
    [
        'y ~ te(x0, x1, bs=["cp","cp"], k=[5,6], m=[1,2], sp=[0.6,0.8])',
        'y ~ ti(x0, x1, bs=["cp","cp"], k=[5,6], m=[1,2], sp=[0.6,0.8])',
    ],
    ids=["te", "ti"],
)
def test_cp_tensor_marginal_fixed_sp_fit_matches_mgcv(formula):
    data = _cp_data(seed=236, n=170)
    actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
    expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
    _assert_snapshot_fit(actual, expected, atol=2e-7)


@pytest.mark.parametrize(
    "formula",
    [
        'y ~ s(f, x0, bs="fs", k=6, xt="cp", sp=[0.7,0.9])',
        'y ~ s(f, f1, x0, bs="sz", k=6, xt="cp", id="shared", sp=0.7)',
    ],
    ids=["fs", "sz"],
)
def test_cp_factor_smooth_base_fixed_sp_fit_matches_mgcv(formula):
    data = _cp_data(seed=237, n=180)
    actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
    expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
    _assert_snapshot_fit(actual, expected, atol=3e-7)


def test_cp_array_api_and_persistence_preserve_wrapped_prediction(tmp_path):
    data = _cp_data(seed=238, n=150)
    features = data[["x0", "x1"]]
    model = GAM(
        family="gaussian",
        basis="cp",
        k=8,
        optimize_smoothing=False,
        smoothing_params=[0.5, 0.8],
    ).fit(X=features, y=data["y"].to_numpy(dtype=np.float64))
    newdata = pd.DataFrame({"x0": [-7.0, 0.0, 7.0], "x1": [-8.0, 1.0, 9.0]})
    expected = model.predict(newdata, type="link")
    path = tmp_path / "cp.pkl"
    model.save_model(path)
    restored = GAM.load_model(path)
    np.testing.assert_allclose(restored.predict(newdata, type="link"), expected)


@pytest.mark.parametrize(
    "formula,message",
    [
        ('y ~ s(x0, bs="cp", k=1, m=[2,1])', "basis dimension too small"),
        ('y ~ s(x0, bs="cp", k=3, m=[2,3])', "penalty order too high"),
    ],
)
def test_cp_invalid_orders_fail_loudly(formula, message):
    data = _cp_data(seed=239, n=40)
    with pytest.raises(ValueError, match=message):
        GAM(formula=formula).fit(data=data)


def test_cp_knot_validation_and_uninformed_coefficient_warning():
    x = np.linspace(0.0, 1.0, 30)
    kwargs = {
        "feature_index": 0,
        "feature_name": "x",
        "bs_dim": 8,
        "m": (2, 2),
        "basis": "cp",
    }
    with pytest.raises(ValueError, match="knot range does not include data"):
        build_pspline_term_setup(x, knots=[0.2, 0.8], **kwargs)
    with pytest.raises(ValueError, match="there should be 9 supplied knots"):
        build_pspline_term_setup(x, knots=np.linspace(0.0, 1.0, 8), **kwargs)
    with pytest.warns(UserWarning, match="no.*information"):
        build_pspline_term_setup(x, knots=[-100.0, 100.0], **kwargs)
