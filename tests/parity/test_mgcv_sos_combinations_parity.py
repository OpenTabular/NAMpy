"""Integrated parity coverage for spherical splines (``bs='sos'``)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from nampy.gam.splines.univariate.sos import (
    build_spherical_spline_setup,
    predict_spherical_spline,
)
from tests.mgcv_parity_utils import (
    _assert_basic_mgcv_parity,
    _fit_nampy_model,
    _fit_nampy_snapshot,
    _run_mgcv_snapshot,
)


def _sos_data(seed=941, n=180):
    rng = np.random.default_rng(seed)
    lo = rng.uniform(-180.0, 180.0, size=n)
    la = np.rad2deg(np.arcsin(rng.uniform(-1.0, 1.0, size=n)))
    x = rng.uniform(-1.5, 1.5, size=n)
    z = 0.8 + rng.uniform(-0.3, 0.5, size=n)
    g = np.asarray(["a", "b", "c"], dtype=object)[np.arange(n) % 3]
    y = (
        0.2
        + z * np.sin(np.deg2rad(lo)) * np.cos(np.deg2rad(la - 12.0))
        + 0.2 * x
        + 0.15 * (g == "b")
        - 0.1 * (g == "c")
        + rng.normal(scale=0.12, size=n)
    )
    return pd.DataFrame({"y": y, "la": la, "lo": lo, "x": x, "z": z, "g": g})


def _assert_snapshot_fit(actual, expected, *, atol=1e-6):
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


def test_sos_default_reml_matches_mgcv():
    data = _sos_data(seed=942)
    formula = 'y ~ s(la, lo, bs="sos", k=12)'
    _assert_basic_mgcv_parity(
        _fit_nampy_snapshot(data, formula, "gaussian", "REML"),
        _run_mgcv_snapshot(data, formula, "gaussian", "REML"),
        pred_atol=1e-6,
        pred_rtol=1e-6,
        sp_log_atol=2e-5,
        criterion_atol=2e-6,
    )


def test_sos_duchon_null_space_select_matches_mgcv():
    data = _sos_data(seed=943)
    formula = 'y ~ s(la, lo, bs="sos", k=12, m=-1)'
    model = _fit_nampy_model(data, formula, "gaussian", "REML", select=True)
    runtime = [
        term.predict_fn.__self__
        for term in model.gam_result_.compiled_model.compiled_terms
        if term.term_type == "smooth"
    ][0]
    assert runtime._setup.null_space_dim == 4
    assert len(model.smoothing_params) == 2
    _assert_basic_mgcv_parity(
        model.parity_snapshot(X=data, include_covariances=True),
        _run_mgcv_snapshot(data, formula, "gaussian", "REML", select=True),
        pred_atol=1e-6,
        pred_rtol=1e-6,
        sp_log_atol=3e-5,
        criterion_atol=3e-6,
    )


def test_sos_numeric_and_factor_by_fixed_sp_match_mgcv():
    data = _sos_data(seed=944)
    numeric = 'y ~ s(la, lo, by=z, bs="sos", k=12, m=2, sp=.7)'
    _assert_snapshot_fit(
        _fit_nampy_snapshot(data, numeric, "gaussian", "fixed"),
        _run_mgcv_snapshot(data, numeric, "gaussian", "REML"),
    )

    factor = 'y ~ s(la, lo, by=g, bs="sos", k=12, m=-2, sp=.7)'
    _assert_snapshot_fit(
        _fit_nampy_snapshot(data, factor, "gaussian", "fixed"),
        _run_mgcv_snapshot(data, factor, "gaussian", "REML"),
        atol=2e-6,
    )


def test_sos_linked_id_and_point_constraint_match_mgcv():
    data = _sos_data(seed=945, n=150)
    data["la2"] = np.roll(data["la"].to_numpy(), 7)
    data["lo2"] = np.roll(data["lo"].to_numpy(), 11)
    linked = (
        'y ~ s(la, lo, bs="sos", k=12, m=3, id="sphere", sp=.7)'
        ' + s(la2, lo2, bs="sos", k=14, m=-1, id="sphere")'
    )
    _assert_snapshot_fit(
        _fit_nampy_snapshot(data, linked, "gaussian", "fixed"),
        _run_mgcv_snapshot(data, linked, "gaussian", "REML"),
        atol=2e-6,
    )

    point = 'y ~ s(la, lo, bs="sos", k=12, pc=[0,0], sp=.7)'
    _assert_snapshot_fit(
        _fit_nampy_snapshot(data, point, "gaussian", "fixed"),
        _run_mgcv_snapshot(data, point, "gaussian", "REML"),
    )


def test_sos_dynamic_null_penalties_link_and_fs_boundary():
    data = _sos_data(seed=955, n=150)
    data["la2"] = np.roll(data["la"].to_numpy(), 5)
    data["lo2"] = np.roll(data["lo"].to_numpy(), 9)
    linked = (
        'y ~ s(la, lo, bs="sos", k=12, m=-1, id="sphere_null", sp=[.7,.9])'
        ' + s(la2, lo2, bs="sos", k=14, m=3, id="sphere_null")'
    )
    _assert_snapshot_fit(
        _fit_nampy_snapshot(data, linked, "gaussian", "fixed", select=True),
        _run_mgcv_snapshot(data, linked, "gaussian", "REML", select=True),
        atol=3e-6,
    )

    factor = 'y ~ s(g, la, lo, bs="fs", xt="sos", k=8, m=-1, sp=[.5,.6,.7,.8,.9])'
    with pytest.raises(NotImplementedError, match="LAPACK-dependent"):
        _fit_nampy_snapshot(data, factor, "gaussian", "fixed")


@pytest.mark.parametrize(
    "formula",
    [
        'y ~ te(la, lo, x, d=[2,1], bs=["sos","cr"], k=[8,4], sp=[.6,.8])',
        'y ~ ti(la, lo, x, d=[2,1], bs=["sos","cr"], k=[8,4], sp=[.6,.8])',
    ],
    ids=["te", "ti"],
)
def test_sos_tensor_margin_fixed_sp_matches_mgcv(formula):
    data = _sos_data(seed=946, n=150)
    _assert_snapshot_fit(
        _fit_nampy_snapshot(data, formula, "gaussian", "fixed"),
        _run_mgcv_snapshot(data, formula, "gaussian", "REML"),
        atol=3e-6,
    )


@pytest.mark.parametrize(
    "formula",
    [
        'y ~ s(g, la, lo, bs="fs", xt="sos", k=8, sp=[.6,.8])',
        'y ~ s(g, la, lo, bs="sz", xt="sos", k=8, sp=[.7,.7,.7])',
    ],
    ids=["fs", "sz"],
)
def test_sos_factor_smooth_base_fixed_sp_matches_mgcv(formula):
    data = _sos_data(seed=947, n=150)
    _assert_snapshot_fit(
        _fit_nampy_snapshot(data, formula, "gaussian", "fixed"),
        _run_mgcv_snapshot(data, formula, "gaussian", "REML"),
        atol=3e-6,
    )


def test_sos_prediction_periodicity_poles_and_persistence(tmp_path):
    data = _sos_data(seed=948, n=130)
    formula = 'y ~ s(la, lo, bs="sos", k=12, m=-1, sp=.7)'
    model = _fit_nampy_model(data, formula, "gaussian", "fixed")
    newdata = pd.DataFrame(
        {
            "la": [-90.0, -45.0, 0.0, 45.0, 90.0],
            "lo": [-170.0, -80.0, 10.0, 100.0, 170.0],
        }
    )
    wrapped = newdata.copy()
    wrapped["lo"] += 360.0
    expected = model.predict(newdata, type="link", block_size=1)
    np.testing.assert_allclose(
        model.predict(wrapped, type="link", block_size=1), expected, atol=2e-13
    )
    poles = pd.DataFrame({"la": [-90.0] * 3 + [90.0] * 3, "lo": [-120, 0, 120] * 2})
    pole_predictions = model.predict(poles, type="link")
    np.testing.assert_allclose(pole_predictions[:3], pole_predictions[0], atol=2e-13)
    np.testing.assert_allclose(pole_predictions[3:], pole_predictions[3], atol=2e-13)

    path = tmp_path / "sos.pkl"
    model.save_model(path)
    restored = GAM.load_model(path)
    np.testing.assert_allclose(restored.predict(newdata, type="link"), expected)


def test_sos_order_knots_and_dimension_guards():
    data = _sos_data(seed=949, n=60)
    X = data[["la", "lo"]].to_numpy()
    assert build_spherical_spline_setup(X, k=12, m=-3).order == -1
    assert build_spherical_spline_setup(X, k=12, m=1.6).order == 2
    assert build_spherical_spline_setup(X, k=12, m=5).order == 4
    assert build_spherical_spline_setup(X, k=12, xt={"ignored": True}).bs_dim == 12
    with pytest.warns(UserWarning, match="more knots than data in an sos term"):
        ignored = build_spherical_spline_setup(
            X, k=12, knots=[np.arange(61), np.arange(61)]
        )
    assert not ignored.used_supplied_knots
    with pytest.raises(ValueError, match="at least 6"):
        build_spherical_spline_setup(X, k=5, m=-1)
    with pytest.raises(ValueError, match="at least as many unique knot locations"):
        build_spherical_spline_setup(X, k=12, knots=[np.arange(10), np.arange(10)])
    with pytest.raises(ValueError, match="single numeric value"):
        build_spherical_spline_setup(X, k=12, m=[1, 2])

    setup = build_spherical_spline_setup(X, k=12, m=0)
    np.testing.assert_allclose(
        predict_spherical_spline(X, setup), setup.basis_train, atol=2e-13
    )
    np.testing.assert_allclose(
        setup.UZ.T @ np.ones(setup.knots.shape[0]), 0.0, atol=2e-12
    )


def test_sos_array_and_derivative_surfaces_are_explicitly_unsupported():
    data = _sos_data(seed=950, n=80)
    with pytest.raises(NotImplementedError, match="formula with both latitude"):
        GAM(
            family="gaussian",
            basis="sos",
            k=12,
            optimize_smoothing=False,
            smoothing_params=[0.7],
        ).fit(X=data[["la"]], y=data["y"].to_numpy())

    model = GAM(
        formula='y ~ s(la, lo, bs="sos", k=12, sp=.7)',
        optimize_smoothing=False,
    ).fit(data=data)
    with pytest.raises(NotImplementedError, match="only 1D smooths"):
        model.derivative(data, smooth_number=1)
    with pytest.raises(NotImplementedError, match="rotated hemisphere"):
        model.plot()
