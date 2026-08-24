"""Integrated parity coverage for Gaussian-process smooths (``bs='gp'``)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from nampy.gam.splines.univariate.gp import build_gaussian_process_setup
from tests.mgcv_parity_utils import (
    _assert_basic_mgcv_parity,
    _fit_nampy_model,
    _fit_nampy_snapshot,
    _run_mgcv_snapshot,
)


def _gp_data(seed=301, n=180):
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


def _assert_snapshot_fit(actual, expected, *, atol=3e-7):
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


def test_gp_numeric_by_select_true_matches_mgcv():
    data = _gp_data(seed=302)
    formula = 'y ~ s(x0, x1, by=z, bs="gp", k=10, m=[4,.8])'
    actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML", select=True)
    expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML", select=True)
    assert len(actual["fit"]["smoothing_params"]) == 2
    _assert_basic_mgcv_parity(
        actual,
        expected,
        pred_atol=5e-7,
        pred_rtol=5e-7,
        sp_log_atol=8e-6,
        criterion_atol=3e-7,
    )


def test_gp_stationary_select_has_no_extra_null_penalty_and_matches_mgcv():
    data = _gp_data(seed=303)
    formula = 'y ~ s(x0, x1, bs="gp", k=10, m=[-5,.9])'
    model = _fit_nampy_model(data, formula, "gaussian", "REML", select=True)
    assert len(model.smoothing_params) == 1
    _assert_basic_mgcv_parity(
        model.parity_snapshot(X=data, include_covariances=True),
        _run_mgcv_snapshot(data, formula, "gaussian", "REML", select=True),
        pred_atol=5e-7,
        pred_rtol=5e-7,
        sp_log_atol=8e-6,
        criterion_atol=3e-7,
    )


def test_gp_factor_by_fixed_sp_matches_mgcv():
    data = _gp_data(seed=304)
    formula = 'y ~ s(x0, x1, by=f, bs="gp", k=10, m=[2,.7,1.4], sp=.7)'
    actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
    expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
    _assert_snapshot_fit(actual, expected)


def test_gp_linked_terms_clone_first_definition_but_retain_each_xt():
    data = _gp_data(seed=305, n=150)
    formula = (
        'y ~ s(x0, x1, bs="gp", k=10, m=[4,.8], '
        'xt={"max.knots":14,"seed":7}, id="shared_gp", sp=[.7,.9])'
        ' + s(x2, x3, bs="gp", k=12, m=[1,.5], '
        'xt={"max.knots":16,"seed":9}, id="shared_gp")'
    )
    model = _fit_nampy_model(data, formula, "gaussian", "fixed", select=True)
    assert len(model.smoothing_params) == 2
    runtimes = [
        term.predict_fn.__self__
        for term in model.gam_result_.compiled_model.compiled_terms
        if term.term_type == "smooth"
    ]
    assert [runtime._setup.knots.shape[0] for runtime in runtimes] == [14, 16]
    for runtime in runtimes:
        np.testing.assert_allclose(runtime._setup.definition, [4.0, 0.8, 1.0])
    _assert_snapshot_fit(
        model.parity_snapshot(X=data, include_covariances=True),
        _run_mgcv_snapshot(data, formula, "gaussian", "REML", select=True),
        atol=8e-7,
    )


def test_gp_point_constraint_and_fixed_basis_match_mgcv():
    data = _gp_data(seed=306)
    formula = 'y ~ s(x0, x1, bs="gp", k=10, m=[3,.8], pc=[.2,-.3], sp=.8)'
    actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
    expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
    _assert_snapshot_fit(actual, expected)

    fixed_formula = 'y ~ s(x2, x3, bs="gp", k=10, m=[-3,.8], fx=True)'
    model = _fit_nampy_model(data, fixed_formula, "gaussian", "fixed", select=True)
    assert model.gam_result_.compiled_model.compiled_penalties == ()
    _assert_snapshot_fit(
        model.parity_snapshot(X=data, include_covariances=True),
        _run_mgcv_snapshot(
            data, fixed_formula, "gaussian", "REML", select=True
        ),
    )


@pytest.mark.parametrize(
    "formula",
    [
        'y ~ te(x0, x2, bs=["gp","cr"], k=[8,5], '
        'm=[[2,.7,1.5],None], sp=[.6,.8])',
        'y ~ ti(x0, x1, x2, d=[2,1], bs=["gp","cr"], k=[10,5], '
        'm=[[-5,.9],None], sp=[.6,.8])',
    ],
    ids=["te_univariate", "ti_multivariate_stationary"],
)
def test_gp_tensor_margins_fixed_sp_match_mgcv(formula):
    data = _gp_data(seed=307, n=150)
    actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
    expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
    _assert_snapshot_fit(actual, expected, atol=1e-6)


@pytest.mark.parametrize(
    "formula",
    [
        'y ~ s(f, x0, x1, bs="fs", xt="gp", k=10, m=[-1,.6], '
        'sp=[.7,.9])',
        'y ~ s(f, f1, x0, x1, bs="sz", xt="gp", k=10, '
        'm=[1,.6], id="shared_gp_sz", sp=.7)',
    ],
    ids=["fs", "sz"],
)
def test_gp_multivariate_factor_smooth_base_fixed_sp_matches_mgcv(formula):
    data = _gp_data(seed=308, n=150)
    actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
    expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
    _assert_snapshot_fit(actual, expected, atol=1e-6)


def test_gp_persistence_preserves_chunked_extrapolation(tmp_path):
    data = _gp_data(seed=309, n=130)
    formula = (
        'y ~ s(x0, x1, bs="gp", k=10, m=[-1,.6], '
        'xt={"max.knots":14,"seed":7}, sp=.7)'
    )
    model = _fit_nampy_model(data, formula, "gaussian", "fixed")
    newdata = pd.DataFrame(
        {
            "x0": np.linspace(-4.0, 4.0, 31),
            "x1": np.linspace(-3.0, 3.5, 31),
        }
    )
    expected = model.predict(newdata, type="link", block_size=1)
    path = tmp_path / "gp.pkl"
    model.save_model(path)
    restored = GAM.load_model(path)
    np.testing.assert_allclose(
        restored.predict(newdata, type="link", block_size=1), expected
    )
    _assert_snapshot_fit(
        model.parity_snapshot(X=data, include_covariances=True),
        _run_mgcv_snapshot(data, formula, "gaussian", "REML"),
    )


def test_gp_array_api_with_explicit_k_builds_one_term_per_feature():
    data = _gp_data(seed=312, n=100)
    features = data[["x0", "x1"]]
    model = GAM(
        family="gaussian",
        basis="gp",
        k=10,
        optimize_smoothing=False,
        smoothing_params=[0.5, 0.8],
    ).fit(X=features, y=data["y"].to_numpy(dtype=np.float64))
    assert len(model.smoothing_params) == 2
    assert model.predict(features.iloc[:7]).shape == (7,)


def test_gp_definition_warnings_and_validation():
    x = np.linspace(-1.0, 1.0, 30)
    X = np.column_stack([x, np.sin(x)])
    rounded = build_gaussian_process_setup(X, k=8, m=[1.6, 0.7, 1.5])
    np.testing.assert_allclose(rounded.definition, [2.0, 0.7, 1.5])

    with pytest.warns(UserWarning, match="basis dimension reset"):
        assert build_gaussian_process_setup(X, k=2, m=[-3, 0.7]).bs_dim == 4
    with pytest.warns(UserWarning, match="more knots than data in an ms term"):
        ignored = build_gaussian_process_setup(
            X, k=8, m=[3, 0.7], knots=[np.arange(31), np.arange(31)]
        )
    assert not ignored.used_supplied_knots

    with pytest.raises(ValueError, match="incorrect arguments"):
        build_gaussian_process_setup(X, k=8, m=[0, 0.7])
    with pytest.raises(ValueError, match="incorrect arguments"):
        build_gaussian_process_setup(X, k=8, m=[1, 0.7, 2.1])
    with pytest.raises(ValueError, match="same length"):
        build_gaussian_process_setup(
            X,
            k=8,
            m=[3, 0.7],
            knots=[np.linspace(-1, 1, 10), np.linspace(-1, 1, 9)],
        )
    with pytest.raises(ValueError, match="fewer unique covariate combinations"):
        build_gaussian_process_setup(np.repeat(X[:4], 5, axis=0), k=8)
    with pytest.raises(ValueError, match="at least as many knot locations"):
        build_gaussian_process_setup(
            X,
            k=10,
            m=[3, 0.7],
            knots=[np.arange(6), np.arange(6)],
        )
    with pytest.raises(ValueError, match="supply k explicitly"):
        build_gaussian_process_setup(np.column_stack([X, X]), k=-1)


def test_gp_linked_pc_different_features_and_derivative_are_explicitly_unsupported():
    data = _gp_data(seed=310, n=80)
    with pytest.raises(NotImplementedError, match="different feature sets"):
        GAM(
            formula=(
                'y ~ s(x0, bs="gp", k=10, pc=[0], id="gp_pc")'
                ' + s(x1, bs="gp", k=10, pc=[0], id="gp_pc")'
            ),
            optimize_smoothing=False,
        ).fit(data=data)

    model = GAM(
        formula='y ~ s(x0, bs="gp", k=10, sp=.7)',
        optimize_smoothing=False,
    ).fit(data=data)
    with pytest.raises(NotImplementedError, match="derivative provider"):
        model.derivative(data, smooth_number=1)


def test_gp_tensor_stationarity_requires_nested_m():
    data = _gp_data(seed=311, n=90)
    with pytest.raises(ValueError, match="incorrect arguments to GP smoother"):
        GAM(
            formula='y ~ te(x0, x1, bs=["gp","gp"], k=[8,8], m=[-3,-3])',
            optimize_smoothing=False,
        ).fit(data=data)

    nested = GAM(
        formula='y ~ te(x0, x1, bs=["gp","gp"], k=[8,8], m=[[-3],[-3]])',
        optimize_smoothing=False,
    ).fit(data=data)
    assert nested.predict(data).shape == (len(data),)
