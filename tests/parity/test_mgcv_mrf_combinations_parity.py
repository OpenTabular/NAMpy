"""Integrated parity coverage for Markov-random-field smooths (``bs='mrf'``)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from nampy.gam.smooths.categorical.mrf import build_markov_random_field_setup
from tests.mgcv_parity_utils import (
    _assert_basic_mgcv_parity,
    _fit_nampy_model,
    _fit_nampy_snapshot,
    _run_mgcv_snapshot,
)

_PATH_NB = '{"a":["b"],"b":["a","c"],"c":["b","d"],"d":["c","e"],"e":["d"]}'
_DISCONNECTED_NB = '{"a":["b"],"b":["a"],"c":["d"],"d":["c"],"e":[]}'
_SPD_PENALTY = "[[2,-1,0,0,0],[-1,3,-1,0,0],[0,-1,3,-1,0],[0,0,-1,3,-1],[0,0,0,-1,2]]"


def _mrf_data(seed=921, n=180):
    rng = np.random.default_rng(seed)
    region = np.resize(np.asarray(["a", "b", "c", "d", "e"], dtype=object), n)
    group = np.resize(np.asarray(["u", "v"], dtype=object), n)
    x = rng.uniform(-1.5, 1.5, size=n)
    z = 0.8 + rng.uniform(-0.3, 0.5, size=n)
    effect = {"a": -0.6, "b": -0.2, "c": 0.15, "d": 0.45, "e": 0.75}
    y = (
        np.asarray([effect[value] for value in region])
        + 0.35 * np.sin(1.3 * x)
        + 0.2 * (group == "v")
        + rng.normal(scale=0.12, size=n)
    )
    return pd.DataFrame({"y": y, "region": region, "group": group, "x": x, "z": z})


def _numeric_mrf_data(seed=930, n=150):
    data = _mrf_data(seed=seed, n=n)
    data["region"] = np.resize(np.arange(1, 6, dtype=np.float64), n)
    return data


def _assert_snapshot_fit(actual, expected, *, atol=4e-7):
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


def test_mrf_full_rank_reml_matches_mgcv():
    data = _mrf_data(seed=922)
    formula = f'y ~ s(region, bs="mrf", xt={{"nb":{_PATH_NB}}})'
    _assert_basic_mgcv_parity(
        _fit_nampy_snapshot(data, formula, "gaussian", "REML"),
        _run_mgcv_snapshot(data, formula, "gaussian", "REML"),
        pred_atol=5e-7,
        pred_rtol=5e-7,
        sp_log_atol=8e-6,
        criterion_atol=5e-7,
    )


def test_mrf_numeric_by_select_adds_null_penalty_and_matches_mgcv():
    data = _mrf_data(seed=923)
    formula = f'y ~ s(region, by=z, bs="mrf", k=3, xt={{"nb":{_PATH_NB}}})'
    model = _fit_nampy_model(data, formula, "gaussian", "REML", select=True)
    assert len(model.smoothing_params) == 2
    _assert_basic_mgcv_parity(
        model.parity_snapshot(X=data, include_covariances=True),
        _run_mgcv_snapshot(data, formula, "gaussian", "REML", select=True),
        pred_atol=7e-7,
        pred_rtol=7e-7,
        sp_log_atol=5e-5,
        criterion_atol=6e-7,
    )


def test_mrf_factor_by_fixed_sp_matches_mgcv():
    data = _mrf_data(seed=924)
    formula = f'y ~ s(region, by=group, bs="mrf", k=3, xt={{"nb":{_PATH_NB}}}, sp=.7)'
    _assert_snapshot_fit(
        _fit_nampy_snapshot(data, formula, "gaussian", "fixed"),
        _run_mgcv_snapshot(data, formula, "gaussian", "REML"),
        atol=8e-7,
    )


def test_mrf_factor_by_linked_id_shares_one_sp_and_matches_mgcv():
    data = _mrf_data(seed=931)
    formula = (
        f'y ~ s(region, by=group, bs="mrf", k=3, '
        f'xt={{"nb":{_PATH_NB}}}, id="shared_mrf", sp=.7)'
    )
    model = _fit_nampy_model(data, formula, "gaussian", "fixed")
    assert len(model.smoothing_params) == 1
    _assert_snapshot_fit(
        model.parity_snapshot(X=data, include_covariances=True),
        _run_mgcv_snapshot(data, formula, "gaussian", "REML"),
        atol=8e-7,
    )


def test_mrf_disconnected_select_retains_one_null_penalty_after_centering():
    data = _mrf_data(seed=925)
    formula = f'y ~ s(region, bs="mrf", xt={{"nb":{_DISCONNECTED_NB}}})'
    model = _fit_nampy_model(data, formula, "gaussian", "REML", select=True)
    assert len(model.smoothing_params) == 2
    _assert_basic_mgcv_parity(
        model.parity_snapshot(X=data, include_covariances=True),
        _run_mgcv_snapshot(data, formula, "gaussian", "REML", select=True),
        pred_atol=7e-7,
        pred_rtol=7e-7,
        sp_log_atol=1e-5,
        criterion_atol=6e-7,
    )


def test_mrf_positive_definite_penalty_and_fx_match_mgcv():
    data = _mrf_data(seed=926)
    formula = f'y ~ s(region, bs="mrf", xt={{"penalty":{_SPD_PENALTY}}}, sp=.7)'
    model = _fit_nampy_model(data, formula, "gaussian", "fixed")
    penalty = model.gam_result_.compiled_model.compiled_penalties[0]
    assert penalty.rank == penalty.matrix.shape[0]
    _assert_snapshot_fit(
        model.parity_snapshot(X=data, include_covariances=True),
        _run_mgcv_snapshot(data, formula, "gaussian", "REML"),
    )

    fixed_formula = f'y ~ s(region, bs="mrf", xt={{"nb":{_PATH_NB}}}, fx=True)'
    fixed = _fit_nampy_model(data, fixed_formula, "gaussian", "fixed")
    assert fixed.gam_result_.compiled_model.compiled_penalties == ()
    _assert_snapshot_fit(
        fixed.parity_snapshot(X=data, include_covariances=True),
        _run_mgcv_snapshot(data, fixed_formula, "gaussian", "REML"),
    )


@pytest.mark.parametrize(
    "special",
    ["te", "ti"],
)
def test_mrf_tensor_margin_fixed_sp_matches_mgcv(special):
    data = _mrf_data(seed=927, n=150)
    formula = (
        f'y ~ {special}(region, x, bs=["mrf","cr"], k=[3,5], '
        f'xt=[{{"nb":{_PATH_NB}}},None], sp=[.6,.8])'
    )
    _assert_snapshot_fit(
        _fit_nampy_snapshot(data, formula, "gaussian", "fixed"),
        _run_mgcv_snapshot(data, formula, "gaussian", "REML"),
        atol=1e-6,
    )


@pytest.mark.parametrize("basis", ["fs", "sz"])
def test_mrf_numeric_factor_smooth_base_fixed_sp_matches_mgcv(basis):
    data = _numeric_mrf_data(seed=932)
    numeric_nb = '{"1":["2"],"2":["1","3"],"3":["2","4"],"4":["3","5"],"5":["4"]}'
    formula = (
        f'y ~ s(group, region, bs="{basis}", k=5, '
        f'xt={{"bs":"mrf","nb":{numeric_nb}}}, sp=[.6,.8])'
    )
    _assert_snapshot_fit(
        _fit_nampy_snapshot(data, formula, "gaussian", "fixed"),
        _run_mgcv_snapshot(data, formula, "gaussian", "REML"),
        atol=1e-6,
    )


def test_mrf_array_api_and_persistence(tmp_path):
    data = _mrf_data(seed=928, n=100)
    nb = {
        "a": ["b"],
        "b": ["a", "c"],
        "c": ["b", "d"],
        "d": ["c", "e"],
        "e": ["d"],
    }
    X = data[["region"]]
    model = GAM(
        family="gaussian",
        basis="mrf",
        k=3,
        xt={"nb": nb},
        optimize_smoothing=False,
        smoothing_params=[0.7],
    ).fit(X=X, y=data["y"].to_numpy(dtype=np.float64))
    expected = model.predict(X.iloc[:8])
    path = tmp_path / "mrf.pkl"
    model.save_model(path)
    restored = GAM.load_model(path)
    np.testing.assert_allclose(restored.predict(X.iloc[:8]), expected)


def test_mrf_validation_and_linked_boundaries_are_explicit():
    values = np.resize(np.asarray(["a", "b", "c"], dtype=object), 30)
    nb = {"a": ["b"], "b": ["a", "c"], "c": ["b"]}
    with pytest.raises(ValueError, match="must be supplied in xt"):
        build_markov_random_field_setup(values)
    with pytest.raises(ValueError, match="dimension set too high"):
        build_markov_random_field_setup(values, k=4, xt={"nb": nb})
    with pytest.raises(ValueError, match="k<=2"):
        build_markov_random_field_setup(values, k=1, xt={"nb": nb})
    with pytest.raises(ValueError, match="k<=2"):
        build_markov_random_field_setup(values, k=2, xt={"nb": nb})
    with pytest.raises(ValueError, match="auto- penalty construction"):
        build_markov_random_field_setup(
            values,
            xt={"nb": {"a": ["b"], "b": [], "c": []}},
        )
    with pytest.raises(TypeError, match="uniform representation"):
        build_markov_random_field_setup(
            values,
            xt={"nb": {"a": [2], "b": ["a", "c"], "c": [2]}},
        )
    duplicate_named = build_markov_random_field_setup(
        values,
        xt={
            "nb": {
                "a": ["b", "b", "unknown"],
                "b": ["a", "c"],
                "c": ["b"],
            }
        },
    )
    assert duplicate_named.raw_penalty[0, 0] == 1.0
    with pytest.raises(ValueError, match="not contained in the knot specification"):
        build_markov_random_field_setup(
            np.asarray(["a", "stale"], dtype=object),
            xt={"nb": nb},
            factor_levels=["a", "b", "c"],
        )

    data = _mrf_data(seed=929, n=80)
    with pytest.raises(NotImplementedError, match="different feature sets"):
        GAM(
            formula=(
                f'y ~ s(region, bs="mrf", xt={{"nb":{_PATH_NB}}}, id="g")'
                f' + s(group, bs="mrf", xt={{"nb":{{"u":["v"],"v":["u"]}}}}, id="g")'
            ),
            optimize_smoothing=False,
        ).fit(data=data)

    formula = f'y ~ s(region, bs="mrf", xt={{"nb":{_PATH_NB}}}, sp=.7)'
    model = _fit_nampy_model(data, formula, "gaussian", "fixed")
    bad = data.iloc[:3].copy()
    bad.loc[bad.index[0], "region"] = "unknown"
    with pytest.raises(ValueError, match="unseen levels|unknown regions"):
        model.predict(bad)

    with pytest.raises(ValueError, match="incorrect number of smoothing parameters"):
        GAM(
            formula=(
                f'y ~ s(region, bs="mrf", xt={{"nb":{_PATH_NB}}}, fx=True, sp=.7)'
            ),
            optimize_smoothing=False,
        ).fit(data=data)
    with pytest.raises(NotImplementedError, match="pc"):
        GAM(
            formula=(f'y ~ s(region, bs="mrf", xt={{"nb":{_PATH_NB}}}, pc="a")'),
            optimize_smoothing=False,
        ).fit(data=data)


def test_mrf_boolean_regions_and_array_xt_namespace_are_safe():
    boolean_data = pd.DataFrame(
        {
            "y": np.linspace(-0.2, 0.4, 20),
            "region": np.resize(np.asarray([False, True]), 20),
        }
    )
    boolean_model = GAM(
        formula=(
            'y ~ s(region, bs="mrf", '
            'xt={"nb":{"FALSE":["TRUE"],"TRUE":["FALSE"]}}, sp=.7)'
        ),
        optimize_smoothing=False,
    ).fit(data=boolean_data)
    prediction = boolean_model.predict(boolean_data.iloc[:4])
    assert np.isfinite(prediction).all()
    assert (
        np.linalg.norm(
            boolean_model.gam_result_.compiled_model.compiled_terms[0].basis_train
        )
        > 0.0
    )

    array_data = pd.DataFrame(
        {"nb": np.resize(np.asarray(["a", "b", "c"], dtype=object), 30)}
    )
    array_model = GAM(
        family="gaussian",
        basis="mrf",
        k=3,
        xt={"nb": {"a": ["b"], "b": ["a", "c"], "c": ["b"]}},
        optimize_smoothing=False,
        smoothing_params=[0.7],
    ).fit(X=array_data, y=np.linspace(-0.4, 0.5, len(array_data)))
    assert np.isfinite(array_model.predict(array_data.iloc[:5])).all()


@pytest.mark.parametrize("special", ["te", "ti"])
def test_mrf_tensor_knots_promote_unobserved_region_for_prediction(special):
    data = _mrf_data(seed=934, n=80)
    data = data.loc[data["region"] != "e"].reset_index(drop=True)
    knots = pd.Categorical(
        ["a", "b", "c", "d", "e"],
        categories=["a", "b", "c", "d", "e"],
    )
    formula = (
        f'y ~ {special}(region, x, bs=["mrf","cr"], k=[3,5], '
        f'xt=[{{"nb":{_PATH_NB}}},None], sp=[.6,.8])'
    )
    model = GAM(formula=formula, optimize_smoothing=False).fit(
        data=data, knots={"region": knots}
    )
    newdata = data.iloc[:2].copy()
    newdata.loc[newdata.index[0], "region"] = "e"
    assert np.isfinite(model.predict(newdata)).all()
