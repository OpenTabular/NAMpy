"""
Parity tests for all predict() output types against mgcv.

Covers:
  - type="terms"    : per-term contribution matrix
  - type="lpmatrix" : full linear predictor matrix
  - return_se=True  : standard errors for type="response" and type="link"
  - Prediction on held-out (new) data for type="link" and type="terms"
  - Multiple families: Gaussian, Poisson, Binomial
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from mgcv_parity_utils import (
    R_SCRIPT,
    _fit_nampy_model,
    _fit_nampy_snapshot,
    _make_binomial_data,
    _make_gaussian_data,
    _make_poisson_data,
    _run_mgcv_predict_on_newdata,
    _run_mgcv_snapshot,
)


def _norm_labels(labels) -> list[str]:
    vals = [labels] if isinstance(labels, str) else list(labels)
    out = []
    for lab in vals:
        s = str(lab)
        s = re.sub(r',\s*bs\s*=\s*(?:"[^"]*"|\[[^\]]*\])', "", s)
        s = re.sub(r',\s*k\s*=\s*(?:[^,\)]+|\[[^\]]*\])', "", s)
        s = re.sub(r"\s+", "", s)
        s = s.replace("])", ")")
        s = re.sub(r",\d+\)$", ")", s)
        out.append(s)
    return out


# ---------------------------------------------------------------------------
# predict type="terms" parity
# ---------------------------------------------------------------------------

@pytest.mark.skipif(R_SCRIPT is None, reason="Rscript is not available; mgcv parity tests are skipped.")
def test_predict_terms_parity_gaussian_reml():
    """predict(type='terms') column values match mgcv for Gaussian REML."""
    data = _make_gaussian_data(seed=201, n=160)
    formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

    actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
    expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")

    a_terms = np.asarray(actual["predictions"]["terms"], dtype=np.float64)
    e_terms = np.asarray(expected["predictions"]["terms"], dtype=np.float64)

    assert a_terms.shape == e_terms.shape
    # Each column is a smooth contribution; sum should equal eta - intercept.
    # Tolerances match the REML smoothing parameter agreement (~5e-3).
    np.testing.assert_allclose(a_terms, e_terms, atol=5e-3, rtol=0.0)


@pytest.mark.skipif(R_SCRIPT is None, reason="Rscript is not available; mgcv parity tests are skipped.")
def test_predict_terms_parity_gaussian_fixed_sp():
    """predict(type='terms') matches mgcv exactly at fixed smoothing parameters."""
    data = _make_gaussian_data(seed=202, n=160)
    formula = 'y ~ s(x0, bs="cr", k=8, sp=0.8) + s(x1, bs="cr", k=8, sp=1.5)'

    actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
    expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")

    a_terms = np.asarray(actual["predictions"]["terms"], dtype=np.float64)
    e_terms = np.asarray(expected["predictions"]["terms"], dtype=np.float64)

    assert a_terms.shape == e_terms.shape
    np.testing.assert_allclose(a_terms, e_terms, atol=1e-10, rtol=1e-10)


@pytest.mark.skipif(R_SCRIPT is None, reason="Rscript is not available; mgcv parity tests are skipped.")
def test_predict_terms_sum_equals_eta_minus_intercept():
    """Column sum of predict(type='terms') == eta - intercept for Gaussian."""
    data = _make_gaussian_data(seed=203, n=160)
    formula = 'y ~ s(x0, bs="cr", k=8, sp=0.8) + s(x1, bs="cr", k=8, sp=1.5)'

    actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")

    a_terms = np.asarray(actual["predictions"]["terms"], dtype=np.float64)
    a_link = np.asarray(actual["predictions"]["link"], dtype=np.float64)
    a_intercept = float(actual["fit"]["intercept"])

    # sum of per-term contributions == eta - intercept
    np.testing.assert_allclose(
        a_terms.sum(axis=1),
        a_link - a_intercept,
        atol=1e-10,
        rtol=1e-10,
    )


@pytest.mark.skipif(R_SCRIPT is None, reason="Rscript is not available; mgcv parity tests are skipped.")
def test_predict_terms_parity_with_parametric_term():
    """predict(type='terms') with a parametric covariate in the formula."""
    data = _make_gaussian_data(seed=204, n=160)
    formula = "y ~ x0 + s(x1, bs=\"cr\", k=8, sp=1.2)"

    actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
    expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")

    a_terms = np.asarray(actual["predictions"]["terms"], dtype=np.float64)
    e_terms = np.asarray(expected["predictions"]["terms"], dtype=np.float64)

    assert a_terms.shape == e_terms.shape
    np.testing.assert_allclose(a_terms, e_terms, atol=1e-10, rtol=1e-10)


@pytest.mark.skipif(R_SCRIPT is None, reason="Rscript is not available; mgcv parity tests are skipped.")
def test_predict_terms_parity_poisson_reml():
    """predict(type='terms') matches mgcv for Poisson REML."""
    data = _make_poisson_data(seed=205)
    formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

    actual = _fit_nampy_snapshot(data, formula, "poisson", "REML")
    expected = _run_mgcv_snapshot(data, formula, "poisson", "REML")

    a_terms = np.asarray(actual["predictions"]["terms"], dtype=np.float64)
    e_terms = np.asarray(expected["predictions"]["terms"], dtype=np.float64)

    assert a_terms.shape == e_terms.shape
    np.testing.assert_allclose(a_terms, e_terms, atol=1e-2, rtol=0.0)


@pytest.mark.skipif(R_SCRIPT is None, reason="Rscript is not available; mgcv parity tests are skipped.")
def test_predict_terms_parity_binomial_reml():
    """predict(type='terms') matches mgcv for Binomial REML."""
    data = _make_binomial_data(seed=206)
    formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

    actual = _fit_nampy_snapshot(data, formula, "binomial", "REML")
    expected = _run_mgcv_snapshot(data, formula, "binomial", "REML")

    a_terms = np.asarray(actual["predictions"]["terms"], dtype=np.float64)
    e_terms = np.asarray(expected["predictions"]["terms"], dtype=np.float64)

    assert a_terms.shape == e_terms.shape
    np.testing.assert_allclose(a_terms, e_terms, atol=1e-2, rtol=0.0)


# ---------------------------------------------------------------------------
# predict type="lpmatrix" parity
# ---------------------------------------------------------------------------

@pytest.mark.skipif(R_SCRIPT is None, reason="Rscript is not available; mgcv parity tests are skipped.")
def test_predict_lpmatrix_parity_gaussian_fixed_sp():
    """predict(type='lpmatrix') matches mgcv exactly at fixed smoothing parameters."""
    data = _make_gaussian_data(seed=211, n=160)
    formula = 'y ~ s(x0, bs="cr", k=8, sp=0.8) + s(x1, bs="cr", k=8, sp=1.5)'

    actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
    expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")

    a_lp = np.asarray(actual["predictions"]["lpmatrix"], dtype=np.float64)
    e_lp = np.asarray(expected["predictions"]["lpmatrix"], dtype=np.float64)

    assert a_lp.shape == e_lp.shape
    np.testing.assert_allclose(a_lp, e_lp, atol=1e-10, rtol=1e-10)


@pytest.mark.skipif(R_SCRIPT is None, reason="Rscript is not available; mgcv parity tests are skipped.")
def test_predict_lpmatrix_parity_gaussian_reml():
    """predict(type='lpmatrix') @ coef == eta for Gaussian REML."""
    data = _make_gaussian_data(seed=212, n=160)
    formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

    actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
    expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")

    a_lp = np.asarray(actual["predictions"]["lpmatrix"], dtype=np.float64)
    e_lp = np.asarray(expected["predictions"]["lpmatrix"], dtype=np.float64)
    e_coef = np.asarray(expected["fit"]["coef_full"], dtype=np.float64)

    assert a_lp.shape == e_lp.shape
    # At the same smoothing parameters, the lp-matrix should match tightly.
    np.testing.assert_allclose(a_lp, e_lp, atol=5e-3, rtol=0.0)

    # Cross-check: R's lpmatrix @ R's coefficients == R's link predictions.
    e_link = np.asarray(expected["predictions"]["link"], dtype=np.float64)
    np.testing.assert_allclose(e_lp @ e_coef, e_link, atol=1e-10, rtol=1e-10)


@pytest.mark.skipif(R_SCRIPT is None, reason="Rscript is not available; mgcv parity tests are skipped.")
def test_predict_lpmatrix_parity_poisson_fixed_sp():
    """predict(type='lpmatrix') matches mgcv at fixed smoothing parameters (Poisson)."""
    data = _make_poisson_data(seed=213)
    formula = 'y ~ s(x0, bs="cr", k=8, sp=0.5) + s(x1, bs="cr", k=8, sp=0.9)'

    actual = _fit_nampy_snapshot(data, formula, "poisson", "fixed")
    expected = _run_mgcv_snapshot(data, formula, "poisson", "REML")

    a_lp = np.asarray(actual["predictions"]["lpmatrix"], dtype=np.float64)
    e_lp = np.asarray(expected["predictions"]["lpmatrix"], dtype=np.float64)

    assert a_lp.shape == e_lp.shape
    np.testing.assert_allclose(a_lp, e_lp, atol=1e-10, rtol=1e-10)


@pytest.mark.skipif(R_SCRIPT is None, reason="Rscript is not available; mgcv parity tests are skipped.")
def test_predict_lpmatrix_times_coef_equals_link():
    """lpmatrix @ coef_full == link predictions (both Python and R)."""
    data = _make_gaussian_data(seed=214, n=160)
    formula = 'y ~ s(x0, bs="cr", k=8, sp=0.8) + s(x1, bs="cr", k=8, sp=1.5)'

    actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
    expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")

    for snap in (actual, expected):
        lp = np.asarray(snap["predictions"]["lpmatrix"], dtype=np.float64)
        coef = np.asarray(snap["fit"]["coef_full"], dtype=np.float64)
        link = np.asarray(snap["predictions"]["link"], dtype=np.float64)
        np.testing.assert_allclose(lp @ coef, link, atol=1e-10, rtol=1e-10)


# ---------------------------------------------------------------------------
# predict with return_se=True parity
# ---------------------------------------------------------------------------

@pytest.mark.skipif(R_SCRIPT is None, reason="Rscript is not available; mgcv parity tests are skipped.")
def test_predict_se_link_parity_gaussian_fixed_sp():
    """SE for type='link' matches mgcv at fixed smoothing parameters."""
    data = _make_gaussian_data(seed=221, n=160)
    formula = 'y ~ s(x0, bs="cr", k=8, sp=0.8) + s(x1, bs="cr", k=8, sp=1.5)'

    actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
    expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")

    a_se = np.asarray(actual["predictions"]["se_link"], dtype=np.float64)
    e_se = np.asarray(expected["predictions"]["se_link"], dtype=np.float64)

    assert a_se.shape == e_se.shape
    np.testing.assert_allclose(a_se, e_se, atol=1e-10, rtol=1e-10)


@pytest.mark.skipif(R_SCRIPT is None, reason="Rscript is not available; mgcv parity tests are skipped.")
def test_predict_se_response_parity_gaussian_fixed_sp():
    """SE for type='response' matches mgcv. For Gaussian (identity link) SE equals SE_link."""
    data = _make_gaussian_data(seed=222, n=160)
    formula = 'y ~ s(x0, bs="cr", k=8, sp=0.8) + s(x1, bs="cr", k=8, sp=1.5)'

    actual = _fit_nampy_snapshot(data, formula, "gaussian", "fixed")
    expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")

    a_se = np.asarray(actual["predictions"]["se_response"], dtype=np.float64)
    e_se = np.asarray(expected["predictions"]["se_response"], dtype=np.float64)

    assert a_se.shape == e_se.shape
    np.testing.assert_allclose(a_se, e_se, atol=1e-10, rtol=1e-10)


@pytest.mark.skipif(R_SCRIPT is None, reason="Rscript is not available; mgcv parity tests are skipped.")
def test_predict_se_link_parity_gaussian_reml():
    """SE for type='link' matches mgcv after REML smoothing parameter optimisation."""
    data = _make_gaussian_data(seed=223, n=160)
    formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

    actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
    expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")

    a_se = np.asarray(actual["predictions"]["se_link"], dtype=np.float64)
    e_se = np.asarray(expected["predictions"]["se_link"], dtype=np.float64)

    assert a_se.shape == e_se.shape
    np.testing.assert_allclose(a_se, e_se, atol=5e-3, rtol=0.0)


@pytest.mark.skipif(R_SCRIPT is None, reason="Rscript is not available; mgcv parity tests are skipped.")
def test_predict_se_link_parity_poisson_fixed_sp():
    """SE for type='link' matches mgcv for Poisson at fixed smoothing parameters."""
    data = _make_poisson_data(seed=224)
    formula = 'y ~ s(x0, bs="cr", k=8, sp=0.5) + s(x1, bs="cr", k=8, sp=0.9)'

    actual = _fit_nampy_snapshot(data, formula, "poisson", "fixed")
    expected = _run_mgcv_snapshot(data, formula, "poisson", "REML")

    a_se = np.asarray(actual["predictions"]["se_link"], dtype=np.float64)
    e_se = np.asarray(expected["predictions"]["se_link"], dtype=np.float64)

    assert a_se.shape == e_se.shape
    np.testing.assert_allclose(a_se, e_se, atol=1e-8, rtol=1e-8)


@pytest.mark.skipif(R_SCRIPT is None, reason="Rscript is not available; mgcv parity tests are skipped.")
def test_predict_se_response_parity_poisson_fixed_sp():
    """SE for type='response' matches mgcv for Poisson (delta method, non-trivial)."""
    data = _make_poisson_data(seed=225)
    formula = 'y ~ s(x0, bs="cr", k=8, sp=0.5) + s(x1, bs="cr", k=8, sp=0.9)'

    actual = _fit_nampy_snapshot(data, formula, "poisson", "fixed")
    expected = _run_mgcv_snapshot(data, formula, "poisson", "REML")

    a_se = np.asarray(actual["predictions"]["se_response"], dtype=np.float64)
    e_se = np.asarray(expected["predictions"]["se_response"], dtype=np.float64)

    assert a_se.shape == e_se.shape
    np.testing.assert_allclose(a_se, e_se, atol=1e-8, rtol=1e-8)


@pytest.mark.skipif(R_SCRIPT is None, reason="Rscript is not available; mgcv parity tests are skipped.")
def test_predict_se_link_parity_binomial_fixed_sp():
    """SE for type='link' matches mgcv for Binomial at fixed smoothing parameters."""
    data = _make_binomial_data(seed=226)
    formula = 'y ~ s(x0, bs="cr", k=8, sp=0.6) + s(x1, bs="cr", k=8, sp=0.8)'

    actual = _fit_nampy_snapshot(data, formula, "binomial", "fixed")
    expected = _run_mgcv_snapshot(data, formula, "binomial", "REML")

    a_se = np.asarray(actual["predictions"]["se_link"], dtype=np.float64)
    e_se = np.asarray(expected["predictions"]["se_link"], dtype=np.float64)

    assert a_se.shape == e_se.shape
    np.testing.assert_allclose(a_se, e_se, atol=1e-8, rtol=1e-8)


# ---------------------------------------------------------------------------
# Prediction on new (held-out) data
# ---------------------------------------------------------------------------

@pytest.mark.skipif(R_SCRIPT is None, reason="Rscript is not available; mgcv parity tests are skipped.")
def test_predict_newdata_link_parity_gaussian():
    """predict(newdata, type='link') matches R on held-out data."""
    rng = np.random.default_rng(231)
    n_train, n_new = 160, 40
    x0_tr = rng.uniform(-2.0, 2.0, size=n_train)
    x1_tr = rng.uniform(-1.5, 1.5, size=n_train)
    y_tr = np.sin(x0_tr) + 0.3 * x1_tr**2 + rng.normal(scale=0.15, size=n_train)
    train = pd.DataFrame({"y": y_tr, "x0": x0_tr, "x1": x1_tr})

    x0_new = rng.uniform(-2.0, 2.0, size=n_new)
    x1_new = rng.uniform(-1.5, 1.5, size=n_new)
    y_new = np.zeros(n_new)
    newdata = pd.DataFrame({"y": y_new, "x0": x0_new, "x1": x1_new})

    formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

    model = _fit_nampy_model(train, formula, "gaussian", "REML")
    a_link = model.predict(X=newdata, type="link")

    r_result = _run_mgcv_predict_on_newdata(
        train, newdata, formula, family="gaussian", method="REML", type="link"
    )
    e_link = np.asarray(r_result["pred"], dtype=np.float64)

    np.testing.assert_allclose(a_link, e_link, atol=5e-3, rtol=0.0)


@pytest.mark.skipif(R_SCRIPT is None, reason="Rscript is not available; mgcv parity tests are skipped.")
def test_predict_newdata_response_parity_poisson():
    """predict(newdata, type='response') matches R on held-out data (Poisson)."""
    rng = np.random.default_rng(232)
    n_train, n_new = 200, 40
    x0_tr = rng.normal(size=n_train)
    x1_tr = rng.normal(size=n_train)
    mu_tr = np.exp(0.2 + 0.7 * np.sin(x0_tr) - 0.25 * x1_tr)
    y_tr = rng.poisson(mu_tr)
    train = pd.DataFrame({"y": y_tr, "x0": x0_tr, "x1": x1_tr})

    x0_new = rng.normal(size=n_new)
    x1_new = rng.normal(size=n_new)
    newdata = pd.DataFrame({"y": np.zeros(n_new), "x0": x0_new, "x1": x1_new})

    formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

    model = _fit_nampy_model(train, formula, "poisson", "REML")
    a_resp = model.predict(X=newdata, type="response")

    r_result = _run_mgcv_predict_on_newdata(
        train, newdata, formula, family="poisson", method="REML", type="response"
    )
    e_resp = np.asarray(r_result["pred"], dtype=np.float64)

    np.testing.assert_allclose(a_resp, e_resp, atol=1e-2, rtol=0.0)


@pytest.mark.skipif(R_SCRIPT is None, reason="Rscript is not available; mgcv parity tests are skipped.")
def test_predict_newdata_link_parity_binomial():
    """predict(newdata, type='link') matches R on held-out data (Binomial)."""
    rng = np.random.default_rng(233)
    n_train, n_new = 200, 40
    x0_tr = rng.normal(size=n_train)
    x1_tr = rng.normal(size=n_train)
    eta_tr = 0.9 * np.sin(x0_tr) - 0.45 * x1_tr
    p_tr = 1.0 / (1.0 + np.exp(-eta_tr))
    y_tr = rng.binomial(1, p_tr)
    train = pd.DataFrame({"y": y_tr, "x0": x0_tr, "x1": x1_tr})

    x0_new = rng.normal(size=n_new)
    x1_new = rng.normal(size=n_new)
    newdata = pd.DataFrame({"y": np.zeros(n_new), "x0": x0_new, "x1": x1_new})

    formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

    model = _fit_nampy_model(train, formula, "binomial", "REML")
    a_link = model.predict(X=newdata, type="link")

    r_result = _run_mgcv_predict_on_newdata(
        train, newdata, formula, family="binomial", method="REML", type="link"
    )
    e_link = np.asarray(r_result["pred"], dtype=np.float64)

    np.testing.assert_allclose(a_link, e_link, atol=1e-2, rtol=0.0)


@pytest.mark.skipif(R_SCRIPT is None, reason="Rscript is not available; mgcv parity tests are skipped.")
def test_predict_newdata_link_fixed_sp_matches_exactly():
    """predict(newdata, type='link') at fixed SP should match R to near machine precision."""
    rng = np.random.default_rng(234)
    n_train, n_new = 160, 40
    x0_tr = rng.uniform(-2.0, 2.0, size=n_train)
    x1_tr = rng.uniform(-1.5, 1.5, size=n_train)
    y_tr = np.sin(x0_tr) + 0.3 * x1_tr**2 + rng.normal(scale=0.15, size=n_train)
    train = pd.DataFrame({"y": y_tr, "x0": x0_tr, "x1": x1_tr})

    x0_new = rng.uniform(-2.0, 2.0, size=n_new)
    x1_new = rng.uniform(-1.5, 1.5, size=n_new)
    newdata = pd.DataFrame({"y": np.zeros(n_new), "x0": x0_new, "x1": x1_new})

    formula = 'y ~ s(x0, bs="cr", k=8, sp=0.9) + s(x1, bs="cr", k=8, sp=1.1)'

    model = _fit_nampy_model(train, formula, "gaussian", "fixed")
    a_link = model.predict(X=newdata, type="link")

    r_result = _run_mgcv_predict_on_newdata(
        train, newdata, formula, family="gaussian", method="REML", type="link"
    )
    e_link = np.asarray(r_result["pred"], dtype=np.float64)

    np.testing.assert_allclose(a_link, e_link, atol=1e-10, rtol=1e-10)
