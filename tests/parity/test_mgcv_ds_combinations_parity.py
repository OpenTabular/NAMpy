"""Integrated parity coverage for Duchon regression splines (``bs='ds'``)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from nampy.gam.splines.univariate.ds import build_duchon_spline_setup
from tests.mgcv_parity_utils import (
    _assert_basic_mgcv_parity,
    _fit_nampy_model,
    _fit_nampy_snapshot,
    _run_mgcv_snapshot,
)


def _ds_data(seed=271, n=180):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-2.0, 2.0, size=n)
    x1 = rng.uniform(-1.5, 1.5, size=n)
    x2 = rng.uniform(-1.8, 1.8, size=n)
    x3 = rng.uniform(-1.2, 1.7, size=n)
    z = 0.8 + rng.uniform(-0.4, 0.7, size=n)
    f = np.asarray(["a", "b", "c"], dtype=object)[np.arange(n) % 3]
    f1 = np.asarray(["u", "v"], dtype=object)[np.arange(n) % 2]
    y = (
        0.2
        + z * np.sin(1.2 * x0)
        + 0.3 * x1**2
        - 0.25 * np.cos(x2)
        + 0.15 * x3
        + 0.2 * (f == "b")
        - 0.15 * (f1 == "v")
        + rng.normal(scale=0.12, size=n)
    )
    return pd.DataFrame(
        {
            "y": y,
            "x0": x0,
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "z": z,
            "f": f,
            "f1": f1,
        }
    )


def _assert_snapshot_fit(actual, expected, *, atol=2e-7):
    for key in ("response", "link"):
        np.testing.assert_allclose(
            actual["predictions"][key],
            expected["predictions"][key],
            atol=atol,
            rtol=atol,
        )
    np.testing.assert_allclose(
        actual["fit"]["edf_total"],
        expected["fit"]["edf_total"],
        atol=atol,
        rtol=atol,
    )


def test_ds_numeric_by_select_true_matches_mgcv():
    data = _ds_data(seed=272)
    formula = 'y ~ s(x0, x1, by=z, bs="ds", k=10, m=[1,.5])'
    actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML", select=True)
    expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML", select=True)
    assert len(actual["fit"]["smoothing_params"]) == 2
    _assert_basic_mgcv_parity(
        actual,
        expected,
        pred_atol=4e-7,
        pred_rtol=4e-7,
        sp_log_atol=5e-6,
        criterion_atol=2e-7,
    )


def test_ds_factor_by_fixed_sp_matches_mgcv():
    data = _ds_data(seed=273)
    formula = 'y ~ s(x0, x1, by=f, bs="ds", k=10, m=[1,.5], sp=.7)'
    actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
    expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
    _assert_snapshot_fit(actual, expected)


def test_ds_linked_multivariate_terms_pool_basis_and_share_sp():
    data = _ds_data(seed=274)
    formula = (
        'y ~ s(x0, x1, bs="ds", k=10, m=[1,.5], id="duchon", sp=.7)'
        ' + s(x2, x3, bs="ds", k=10, m=[1,.5], id="duchon")'
    )
    model = _fit_nampy_model(data, formula, "gaussian", "fixed")
    expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
    assert len(model.smoothing_params) == 1
    runtimes = [
        term.predict_fn.__self__
        for term in model.gam_result_.compiled_model.compiled_terms
        if term.term_type == "smooth"
    ]
    np.testing.assert_allclose(runtimes[0]._setup.knots, runtimes[1]._setup.knots)
    actual = model.parity_snapshot(X=data, include_covariances=True)
    _assert_snapshot_fit(actual, expected, atol=5e-7)


def test_ds_point_constraint_and_fixed_basis_match_mgcv():
    data = _ds_data(seed=275)
    formula = 'y ~ s(x0, x1, bs="ds", k=10, m=[1,.5], pc=[.2,-.3], sp=.8)'
    actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
    expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
    _assert_snapshot_fit(actual, expected)

    fixed_formula = 'y ~ s(x2, x3, bs="ds", k=10, m=[1,.5], fx=True)'
    model = _fit_nampy_model(data, fixed_formula, "gaussian", "fixed")
    assert model.gam_result_.compiled_model.compiled_penalties == ()
    _assert_snapshot_fit(
        model.parity_snapshot(X=data, include_covariances=True),
        _run_mgcv_snapshot(data, fixed_formula, "gaussian", "REML"),
    )
    selected = _fit_nampy_model(
        data, fixed_formula, "gaussian", "fixed", select=True
    )
    assert selected.gam_result_.compiled_model.compiled_penalties == ()
    _assert_snapshot_fit(
        selected.parity_snapshot(X=data, include_covariances=True),
        _run_mgcv_snapshot(
            data, fixed_formula, "gaussian", "REML", select=True
        ),
    )


@pytest.mark.parametrize(
    "special",
    ["te", "ti"],
)
def test_ds_multivariate_tensor_margin_fixed_sp_matches_mgcv(special):
    data = _ds_data(seed=276, n=150)
    formula = (
        f'y ~ {special}(x0, x1, x2, d=[2,1], bs=["ds","cr"], '
        "k=[10,5], m=[[1,.5], None], sp=[.6,.8])"
    )
    actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
    expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
    _assert_snapshot_fit(actual, expected, atol=8e-7)


def test_ds_linked_multivariate_tensor_margins_share_two_sp():
    data = _ds_data(seed=280, n=140)
    data["w0"] = 0.7 * data["x0"] - 0.2 * data["x3"]
    data["w1"] = 0.6 * data["x1"] + 0.3 * data["x2"]
    formula = (
        'y ~ te(x0, x1, x2, d=[2,1], bs=["ds","cr"], k=[10,5], '
        'm=[[1,.5], None], id="tensor_ds", sp=[.6,.8])'
        ' + te(x3, w0, w1, d=[2,1], bs=["ds","cr"], k=[10,5], '
        'm=[[1,.5], None], id="tensor_ds")'
    )
    model = _fit_nampy_model(data, formula, "gaussian", "fixed")
    assert len(model.smoothing_params) == 2
    _assert_snapshot_fit(
        model.parity_snapshot(X=data, include_covariances=True),
        _run_mgcv_snapshot(data, formula, "gaussian", "REML"),
        atol=1e-6,
    )


@pytest.mark.parametrize(
    "formula",
    [
        'y ~ s(f, x0, x1, bs="fs", xt="ds", k=10, m=[1,.5], sp=[.7,.9])',
        'y ~ s(f, f1, x0, x1, bs="sz", xt="ds", k=10, m=[1,.5], id="shared", sp=.7)',
    ],
    ids=["fs", "sz"],
)
def test_ds_multivariate_factor_smooth_base_fixed_sp_matches_mgcv(formula):
    data = _ds_data(seed=277, n=150)
    actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
    expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
    _assert_snapshot_fit(actual, expected, atol=1e-6)


def test_ds_identified_fs_base_uses_dynamic_penalty_vector():
    data = _ds_data(seed=281, n=140)
    formula = (
        'y ~ s(f, x0, x1, bs="fs", xt="ds", k=10, m=[1,.5], '
        'id="fs_ds", sp=[.7,.9])'
    )
    model = _fit_nampy_model(data, formula, "gaussian", "fixed")
    assert len(model.smoothing_params) == 2
    _assert_snapshot_fit(
        model.parity_snapshot(X=data, include_covariances=True),
        _run_mgcv_snapshot(data, formula, "gaussian", "REML"),
        atol=1e-6,
    )


def test_ds_array_api_persistence_and_blocked_extrapolation(tmp_path):
    data = _ds_data(seed=278, n=130)
    features = data[["x0", "x1"]]
    model = GAM(
        family="gaussian",
        basis="ds",
        k=10,
        optimize_smoothing=False,
        smoothing_params=[0.5, 0.8],
    ).fit(X=features, y=data["y"].to_numpy(dtype=np.float64))
    newdata = pd.DataFrame({"x0": [-4.0, 0.0, 4.0], "x1": [-3.0, 0.5, 3.5]})
    expected = model.predict(newdata, type="link", block_size=1)
    path = tmp_path / "ds.pkl"
    model.save_model(path)
    restored = GAM.load_model(path)
    np.testing.assert_allclose(
        restored.predict(newdata, type="link", block_size=1), expected
    )


def test_ds_order_normalization_warnings_and_validation():
    x = np.linspace(-1.0, 1.0, 30)
    X = np.column_stack([x, np.sin(x)])
    with pytest.warns(UserWarning, match="s value reduced"):
        setup = build_duchon_spline_setup(X, k=8, m=[1, 3])
    assert setup.shift_order == 0.5
    with pytest.warns(UserWarning) as caught:
        build_duchon_spline_setup(X, k=8, m=[1, -3])
    messages = [str(item.message) for item in caught]
    assert "s value increased" in messages
    assert "s value modified to give continuous function" in messages
    with pytest.warns(UserWarning, match="s value modified"):
        build_duchon_spline_setup(X, k=8, m=[1, 0])
    with pytest.warns(UserWarning, match="basis dimension reset"):
        assert build_duchon_spline_setup(X, k=1, m=[1, 0.5]).bs_dim == 2
    with pytest.warns(UserWarning, match="more knots than data"):
        ignored = build_duchon_spline_setup(
            X, k=8, m=[1, 0.5], knots=[np.arange(31), np.arange(31)]
        )
    assert not ignored.used_supplied_knots

    with pytest.raises(ValueError, match="same length"):
        build_duchon_spline_setup(
            X,
            k=8,
            m=[1, 0.5],
            knots=[np.linspace(-1, 1, 10), np.linspace(-1, 1, 9)],
        )
    with pytest.raises(ValueError, match="fewer unique covariate combinations"):
        build_duchon_spline_setup(np.repeat(X[:4], 5, axis=0), k=8, m=[1, 0.5])
    with pytest.raises(ValueError, match="at least as many knot locations"):
        build_duchon_spline_setup(
            X,
            k=8,
            m=[1, 0.5],
            knots=[np.arange(7), np.arange(7)],
        )


def test_ds_public_derivative_is_explicitly_unsupported():
    data = _ds_data(seed=279, n=80)
    model = GAM(
        formula='y ~ s(x0, bs="ds", k=10, sp=.7)',
        optimize_smoothing=False,
    ).fit(data=data)
    with pytest.raises(NotImplementedError, match="derivative provider"):
        model.derivative(data, smooth_number=1)
