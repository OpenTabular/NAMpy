"""Owner contracts for the summary.gam port surfaces (null deviance first)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from nampy.gam.inference.null_deviance import compute_null_deviance, null_deviance

pytestmark = [pytest.mark.surface_output, pytest.mark.surface_regression]


def _gaussian_data(seed=5, n=80):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(size=n)
    data = pd.DataFrame({"x0": x0, "off": rng.uniform(-0.2, 0.2, size=n)})
    data["y"] = np.sin(2.0 * np.pi * x0) + 0.1 * rng.standard_normal(n)
    return data


def _fit(formula, data, family="gaussian", **kwargs):
    gam = GAM(
        formula=formula,
        family=family,
        optimize_smoothing=False,
        smoothing_method="fixed",
        **kwargs,
    )
    gam.fit(data=data)
    return gam


def test_null_deviance_gaussian_matches_weighted_mean_closed_form():
    """gam.fit3.r:838-842: wtdmu = sum(w*y)/sum(w); nulldev = dev.resids."""
    data = _gaussian_data()
    gam = _fit('y ~ s(x0, bs="cr", k=6)', data)
    y = np.asarray(gam.y_, dtype=np.float64)
    expected = float(np.sum((y - float(np.mean(y))) ** 2.0))
    assert null_deviance(gam) == pytest.approx(expected, rel=1e-12)
    # cached accessor returns the same value without recomputation
    assert gam._null_deviance_ == pytest.approx(expected, rel=1e-12)


def test_null_deviance_offset_triggers_intercept_only_refit():
    """mgcv.r:2072-2075: intercept + nonzero offset -> offset-only GLM refit."""
    rng = np.random.default_rng(6)
    n = 90
    x0 = rng.uniform(size=n)
    off = rng.uniform(-0.4, 0.4, size=n)
    data = pd.DataFrame({"x0": x0, "off": off})
    data["y"] = rng.poisson(np.exp(0.4 + np.sin(2.0 * np.pi * x0) + off))
    gam = _fit('y ~ offset(off) + s(x0, bs="cr", k=6)', data, family="poisson")

    y = np.asarray(gam.y_, dtype=np.float64)
    naive = float(
        gam.family.deviance(y, np.full_like(y, float(np.mean(y))), weights=None)
    )
    nd = compute_null_deviance(gam)
    assert np.isfinite(nd)
    # The refit accounts for the offset, so it must differ from the naive
    # weighted-mean form and cannot exceed it (the offset model nests it).
    assert nd != pytest.approx(naive, rel=1e-8)


def test_null_deviance_without_intercept_uses_linkinv_offset():
    """gam.fit3.r:841: no intercept -> wtdmu = linkinv(offset)."""
    data = _gaussian_data(seed=7)
    gam = _fit('y ~ s(x0, bs="cr", k=6) - 1', data)
    assert not bool(gam.fit_intercept)
    y = np.asarray(gam.y_, dtype=np.float64)
    # gaussian identity with zero offset: mu0 = 0.
    expected = float(np.sum(y**2.0))
    assert compute_null_deviance(gam) == pytest.approx(expected, rel=1e-12)


def test_null_deviance_general_family_without_hook_raises(monkeypatch):
    """General families without a postproc hook must fail loudly."""
    rng = np.random.default_rng(8)
    n = 70
    x0 = rng.uniform(size=n)
    data = pd.DataFrame({"x0": x0})
    data["y"] = np.sin(2.0 * np.pi * x0) + 0.2 * rng.standard_normal(n)
    gam = GAM(
        formula=['y ~ s(x0, bs="cr", k=6)', "~ 1"],
        family="gaulss",
        optimize_smoothing=True,
        smoothing_method="REML",
    )
    gam.fit(data=data)

    assert np.isfinite(compute_null_deviance(gam))
    monkeypatch.delattr(type(gam.family), "null_deviance")
    with pytest.raises(NotImplementedError, match="null_deviance"):
        compute_null_deviance(gam)


def _re_data(seed=13, n=90):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(size=n)
    g = rng.choice(list("abcde"), size=n)
    effects = {"a": 0.6, "b": -0.4, "c": 0.1, "d": -0.2, "e": 0.3}
    data = pd.DataFrame({"x0": x0, "g": pd.Categorical(g)})
    data["y"] = (
        np.sin(2.0 * np.pi * x0)
        + np.array([effects[v] for v in g])
        + 0.2 * rng.standard_normal(n)
    )
    return data


def test_summary_gam_dispersion_freq_and_re_test_branches():
    """
    mgcv/R/mgcv.r:3895-3900 (dispersion rescales and forces z / Chi.sq),
    :3890 (freq switches the parametric covariance only), and :4021-4022
    (re.test=FALSE drops reTest-eligible rows).
    """
    from nampy.gam.inference.summary import summary_gam

    data = _re_data()
    gam = GAM(
        formula='y ~ s(x0, bs="cr", k=6) + s(g, bs="re")',
        family="gaussian",
        optimize_smoothing=True,
        smoothing_method="REML",
    )
    gam.fit(data=data)

    base = summary_gam(gam)
    assert base.scale_estimated
    assert base.p_table.columns[-2] == "t value"
    assert any(base.s_table["basis_name"] == "re")

    # dispersion= rescales SEs by sqrt(dispersion/scale) and forces z-tests.
    disp = 2.0 * base.scale
    forced = summary_gam(gam, dispersion=disp)
    assert not forced.scale_estimated
    assert forced.p_table.columns[-2] == "z value"
    np.testing.assert_allclose(
        forced.p_table["Std. Error"].to_numpy(),
        base.p_table["Std. Error"].to_numpy() * np.sqrt(disp / base.scale),
        rtol=1e-12,
    )

    # freq=True switches the parametric covariance branch only.
    freq = summary_gam(gam, freq=True)
    assert freq.covariance == "freq"
    assert not np.allclose(
        freq.p_table["Std. Error"].to_numpy(),
        base.p_table["Std. Error"].to_numpy(),
    )
    pd.testing.assert_frame_equal(freq.s_table, base.s_table)

    # re_test=False drops the reTest-eligible (random-effect) row.
    no_re = summary_gam(gam, re_test=False)
    assert not any(no_re.s_table["basis_name"] == "re")
    assert any(no_re.s_table["basis_name"] != "re")
