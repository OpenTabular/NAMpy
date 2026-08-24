"""Direct predict.gam argument parity for memory, rows, and covariance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from tests.mgcv_parity_utils import _run_mgcv_predict_on_newdata


def _data(seed=2401, n=110):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.4, 1.4, size=n)
    z = rng.normal(size=n)
    f = pd.Categorical(np.asarray(["a", "b", "c"])[np.arange(n) % 3])
    f_effect = np.asarray([{"a": 0.2, "b": -0.15, "c": 0.05}[v] for v in f])
    y = 0.4 + 0.35 * z + np.sin(1.3 * x) + f_effect + rng.normal(0, 0.12, n)
    return pd.DataFrame({"y": y, "x": x, "z": z, "f": f})


def _numeric(value):
    array = np.asarray(value, dtype=object)
    return np.vectorize(
        lambda item: np.nan if item is None or item == "NA" else float(item),
        otypes=[np.float64],
    )(array)


def _fixed_model(data):
    return GAM(
        family="gaussian",
        formula=(
            'y ~ z + s(x, bs="cr", k=7, sp=0.8)'
            ' + s(f, bs="re", sp=0.9)'
        ),
        optimize_smoothing=False,
        smoothing_method="fixed",
    ).fit(data=data)


def test_block_size_matches_mgcv_and_really_bounds_constructor_rows():
    data = _data()
    model = _fixed_model(data)
    newdata = data.drop(columns="y").iloc[4:23].reset_index(drop=True)

    actual, actual_se = model.predict(
        newdata, type="link", return_se=True, block_size=3
    )
    expected = _run_mgcv_predict_on_newdata(
        data,
        newdata,
        model.formula,
        method="fixed",
        type="link",
        return_se=True,
        block_size=3,
        allow_live_run=True,
    )
    np.testing.assert_allclose(actual, _numeric(expected["pred"]).ravel(), atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(actual_se, _numeric(expected["se"]).ravel(), atol=1e-10, rtol=1e-10)

    calls = []
    compiled = model.gam_result_.require_compiled_model()
    originals = []
    for predictor in compiled.predictors:
        for term in predictor.compiled_terms:
            original = term.predict_fn
            originals.append((term, original))

            def wrapped(X, _original=original):
                calls.append(len(X))
                return _original(X)

            term.predict_fn = wrapped
    try:
        baseline = model.predict(newdata, type="terms", return_se=True, block_size=0)
        blocked = model.predict(newdata, type="terms", return_se=True, block_size=2)
        np.testing.assert_allclose(blocked[0], baseline[0], atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(blocked[1], baseline[1], atol=1e-12, rtol=1e-12)
        blocked_calls = list(calls)
        assert 2 in blocked_calls
        assert max(blocked_calls[-30:]) <= 2
        calls.clear()
        model.predict(newdata, type="lpmatrix", block_size=1)
        assert max(calls) == len(newdata)
    finally:
        for term, original in originals:
            term.predict_fn = original


@pytest.mark.parametrize("na_action", ["pass", "omit", "exclude"])
def test_na_action_link_and_se_match_mgcv(na_action):
    data = _data(seed=2402)
    model = _fixed_model(data)
    newdata = data.drop(columns="y").iloc[:9].reset_index(drop=True)
    newdata.loc[2, "x"] = np.nan
    newdata.loc[7, "z"] = np.nan

    actual, actual_se = model.predict(
        newdata, type="link", return_se=True, na_action=na_action, block_size=2
    )
    expected = _run_mgcv_predict_on_newdata(
        data,
        newdata,
        model.formula,
        method="fixed",
        type="link",
        return_se=True,
        na_action=na_action,
        allow_live_run=True,
    )
    np.testing.assert_allclose(actual, _numeric(expected["pred"]).ravel(), equal_nan=True, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(actual_se, _numeric(expected["se"]).ravel(), equal_nan=True, atol=1e-10, rtol=1e-10)


def test_na_action_fail_and_all_missing_row_restoration():
    data = _data(seed=2403)
    model = _fixed_model(data)
    newdata = data.drop(columns="y").iloc[:4].reset_index(drop=True)
    newdata.loc[:, ["x", "z", "f"]] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        model.predict(newdata, na_action="fail")
    passed = model.predict(newdata, type="terms", return_se=True, na_action="pass")
    assert passed[0].shape == passed[1].shape == (4, 3)
    assert np.isnan(passed[0]).all()
    assert np.isnan(passed[1]).all()
    assert model.predict(newdata, na_action="omit").shape == (0,)


def test_na_pass_and_blocking_preserve_general_family_rows():
    data = _data(seed=2406)
    model = GAM(
        family="gaulss",
        formula=["y ~ z", "~ 1"],
        optimize_smoothing=False,
        smoothing_method="fixed",
    ).fit(data=data)
    newdata = data[["z"]].iloc[:7].reset_index(drop=True)
    newdata.loc[3, "z"] = np.nan

    link, link_se = model.predict(
        newdata, type="link", return_se=True, block_size=2
    )
    assert link.shape == link_se.shape == (7, 2)
    assert np.isnan(link[3]).all()
    assert np.isnan(link_se[3]).all()
    assert np.isfinite(np.delete(link, 3, axis=0)).all()

    lpmatrix = model.predict(newdata, type="lpmatrix", block_size=1)
    assert lpmatrix.shape[0] == 7
    assert np.isnan(lpmatrix[3]).all()


def test_newdata_guaranteed_skips_only_an_excluded_smooth_constructor():
    data = _data(seed=2404)
    formula = 'y ~ z + s(x, bs="cr", k=7, sp=0.8)'
    model = GAM(formula=formula, optimize_smoothing=False).fit(data=data)
    newdata = data[["z"]].iloc[:8].reset_index(drop=True)

    actual, actual_se = model.predict(
        newdata,
        type="link",
        return_se=True,
        exclude=["s(x)"],
        newdata_guaranteed=True,
        na_action="fail",
    )
    expected = _run_mgcv_predict_on_newdata(
        data,
        newdata,
        formula,
        method="fixed",
        type="link",
        return_se=True,
        exclude=["s(x)"],
        newdata_guaranteed=True,
        allow_live_run=True,
    )
    np.testing.assert_allclose(actual, _numeric(expected["pred"]).ravel(), atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(actual_se, _numeric(expected["se"]).ravel(), atol=1e-10, rtol=1e-10)

    with pytest.raises(KeyError, match="missing formula columns"):
        model.predict(
            data[["x"]].iloc[:4],
            exclude=["s(x)"],
            newdata_guaranteed=True,
        )
    with pytest.raises(ValueError, match="requires explicit newdata"):
        model.predict(None, newdata_guaranteed=True)

    factor_model = GAM(
        formula='y ~ z + s(f, bs="re", sp=0.9)',
        optimize_smoothing=False,
    ).fit(data=data)
    factor_missing = data[["z"]].iloc[:5]
    assert factor_model.predict(
        factor_missing,
        exclude=["s(f)"],
        newdata_guaranteed=True,
    ).shape == (5,)
    factor_unseen = data[["z", "f"]].iloc[:5].copy()
    factor_unseen["f"] = "unseen"
    with pytest.raises(ValueError, match="unseen levels"):
        factor_model.predict(
            factor_unseen,
            exclude=["s(f)"],
            newdata_guaranteed=True,
        )


def test_direct_unconditional_and_iterms_type_two_match_mgcv():
    data = _data(seed=2405)
    model = _fixed_model(data)
    newdata = data.drop(columns="y").iloc[2:15].reset_index(drop=True)

    conditional = model.predict(newdata, type="link", return_se=True)
    with pytest.warns(
        UserWarning,
        match="Smoothness uncertainty corrected covariance not available",
    ):
        unconditional = model.predict(
            newdata, type="link", return_se=True, unconditional=True
        )
    explicit = model.predict(
        newdata,
        type="link",
        return_se=True,
        cov=model.vcov(unconditional=True),
    )
    np.testing.assert_array_equal(unconditional[0], conditional[0])
    np.testing.assert_allclose(unconditional[1], explicit[1], atol=0.0, rtol=0.0)
    with pytest.raises(ValueError, match="cannot be used together"):
        model.predict(newdata, cov="bayes", unconditional=True)

    actual, actual_se = model.predict(
        newdata, type="iterms", return_se=True, iterms_type=2
    )
    expected = _run_mgcv_predict_on_newdata(
        data,
        newdata,
        model.formula,
        method="fixed",
        type="iterms",
        return_se=True,
        iterms_type=2,
        allow_live_run=True,
    )
    np.testing.assert_allclose(actual, _numeric(expected["pred"]), atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(actual_se, _numeric(expected["se"]), atol=1e-10, rtol=1e-10)
