"""
Phase 0.3 — minimal mgcv parity corpus (safety rail for cleanup).

Uses the R snapshot machinery in ``mgcv_parity_utils``; skipped when
``Rscript`` is unavailable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mgcv_parity_utils import (
    R_SCRIPT,
    _fit_nampy_snapshot,
    _make_binomial_data,
    _make_fs_data,
    _make_gamma_data,
    _make_gaussian_data,
    _make_mrf_data,
    _make_negbin_data,
    _make_poisson_data,
    _make_random_effect_data_noisy,
    _run_mgcv_snapshot,
)

from gam_phase0_utils import parity_snapshot_structure


pytestmark = pytest.mark.skipif(R_SCRIPT is None, reason="Rscript is not available")


def _assert_corpus_shape_alignment(actual, expected):
    for k in ("response", "link", "terms", "lpmatrix"):
        assert np.asarray(actual["predictions"][k]).shape == np.asarray(
            expected["predictions"][k]
        ).shape


def _assert_corpus_predictions_and_edf(actual, expected, *, pred_atol, pred_rtol, edf_atol=0.5):
    """Looser than full snapshot equality: predictions + EDF, no REML score."""
    a_fit = actual["fit"]
    e_fit = expected["fit"]
    a_pred = actual["predictions"]
    e_pred = expected["predictions"]
    np.testing.assert_allclose(
        np.asarray(a_pred["response"], dtype=np.float64),
        np.asarray(e_pred["response"], dtype=np.float64),
        atol=pred_atol,
        rtol=pred_rtol,
    )
    np.testing.assert_allclose(
        np.asarray(a_pred["link"], dtype=np.float64),
        np.asarray(e_pred["link"], dtype=np.float64),
        atol=pred_atol,
        rtol=pred_rtol,
    )
    np.testing.assert_allclose(
        np.asarray(a_fit["edf_total"], dtype=np.float64),
        np.asarray(e_fit["edf_total"], dtype=np.float64),
        atol=edf_atol,
        rtol=0.12,
    )
    np.testing.assert_allclose(
        np.asarray(a_fit["edf_by_term"], dtype=np.float64),
        np.asarray(e_fit["edf_by_term"], dtype=np.float64),
        atol=edf_atol,
        rtol=0.15,
    )


def test_corpus_gaussian_univariate_smooths():
    """Single predictor: two univariate ``cr`` smooths at REML."""
    data = _make_gaussian_data(seed=11, n=120)
    formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'
    actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
    expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
    _assert_corpus_predictions_and_edf(
        actual, expected, pred_atol=1.2e-2, pred_rtol=1.2e-2, edf_atol=0.55
    )
    assert parity_snapshot_structure(actual)["predictions"]["lpmatrix"] == tuple(
        np.asarray(expected["predictions"]["lpmatrix"]).shape
    )


def test_corpus_gaussian_tp_univariate_smooths():
    data = _make_gaussian_data(seed=13, n=120)
    formula = 'y ~ s(x0, bs="tp", k=8) + s(x1, bs="tp", k=8)'
    actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
    expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
    _assert_corpus_predictions_and_edf(
        actual, expected, pred_atol=1.5e-2, pred_rtol=1.5e-2, edf_atol=0.6
    )


def test_corpus_gaussian_ts_univariate_smooths():
    data = _make_gaussian_data(seed=17, n=120)
    formula = 'y ~ s(x0, bs="ts", k=8) + s(x1, bs="ts", k=8)'
    actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
    expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
    _assert_corpus_predictions_and_edf(
        actual, expected, pred_atol=2.0e-2, pred_rtol=2.0e-2, edf_atol=0.7
    )


def test_corpus_tensor_product_te():
    data = _make_gaussian_data(seed=29, n=140)
    formula = "y ~ te(x0, x1, k=[6, 6])"
    actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
    expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
    _assert_corpus_predictions_and_edf(
        actual, expected, pred_atol=1.2e-2, pred_rtol=1.2e-2, edf_atol=0.55
    )


def test_corpus_poisson_tensor_te():
    data = _make_poisson_data(seed=53, n=220)
    formula = 'y ~ te(x0, x1, bs=["cr", "cr"], k=[6, 6])'
    actual = _fit_nampy_snapshot(data, formula, "poisson", "REML")
    expected = _run_mgcv_snapshot(data, formula, "poisson", "REML")
    _assert_corpus_predictions_and_edf(
        actual, expected, pred_atol=7e-2, pred_rtol=7e-2, edf_atol=1.0
    )


def test_corpus_gamma_tensor_te():
    data = _make_gamma_data(seed=59, n=220)
    formula = 'y ~ te(x0, x1, bs=["cr", "cr"], k=[6, 6])'
    actual = _fit_nampy_snapshot(data, formula, "gamma", "REML")
    expected = _run_mgcv_snapshot(data, formula, "gamma", "REML")
    _assert_corpus_predictions_and_edf(
        actual, expected, pred_atol=3.0e-1, pred_rtol=2.6e-1, edf_atol=1.4
    )


def test_corpus_negbin_tensor_te():
    data = _make_negbin_data(seed=61, n=240, theta=1.0)
    family = {"name": "negbin", "theta": 1.0}
    formula = 'y ~ te(x0, x1, bs=["cr", "cr"], k=[6, 6])'
    actual = _fit_nampy_snapshot(data, formula, family, "REML")
    expected = _run_mgcv_snapshot(data, formula, family, "REML")
    _assert_corpus_predictions_and_edf(
        actual, expected, pred_atol=1.0e-1, pred_rtol=1.0e-1, edf_atol=1.1
    )


def test_corpus_tensor_interaction_ti():
    data = _make_gaussian_data(seed=33, n=140)
    formula = 'y ~ ti(x0, x1, bs=["cr", "cr"], k=[6, 6])'
    actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
    expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
    _assert_corpus_predictions_and_edf(
        actual, expected, pred_atol=1.6e-2, pred_rtol=1.6e-2, edf_atol=0.6
    )


def test_corpus_tensor_anova_t2():
    data = _make_gaussian_data(seed=37, n=140)
    formula = 'y ~ t2(x0, x1, bs=["cr", "cr"], k=[6, 6])'
    actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
    expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
    _assert_corpus_predictions_and_edf(
        actual, expected, pred_atol=2.5e-2, pred_rtol=2.5e-2, edf_atol=0.8
    )


def test_corpus_poisson_tensor_anova_t2():
    data = _make_poisson_data(seed=71, n=220)
    formula = 'y ~ t2(x0, x1, bs=["cr", "cr"], k=[6, 6])'
    actual = _fit_nampy_snapshot(data, formula, "poisson", "REML")
    expected = _run_mgcv_snapshot(data, formula, "poisson", "REML")
    _assert_corpus_predictions_and_edf(
        actual, expected, pred_atol=1e-10, pred_rtol=1e-10, edf_atol=1e-10
    )


def test_corpus_binomial_tensor_anova_t2():
    data = _make_binomial_data(seed=73, n=220)
    formula = 'y ~ t2(x0, x1, bs=["cr", "cr"], k=[6, 6])'
    actual = _fit_nampy_snapshot(data, formula, "binomial", "REML")
    expected = _run_mgcv_snapshot(data, formula, "binomial", "REML")
    _assert_corpus_predictions_and_edf(
        actual, expected, pred_atol=1e-10, pred_rtol=1e-10, edf_atol=1e-10
    )


def test_corpus_gamma_tensor_anova_t2():
    data = _make_gamma_data(seed=101, n=220)
    formula = 'y ~ t2(x0, x1, bs=["cr", "cr"], k=[6, 6])'
    actual = _fit_nampy_snapshot(data, formula, "gamma", "REML")
    expected = _run_mgcv_snapshot(data, formula, "gamma", "REML")
    _assert_corpus_predictions_and_edf(
        actual, expected, pred_atol=1e-5, pred_rtol=1e-5, edf_atol=1e-1
    )
    np.testing.assert_allclose(
        float(actual["fit"]["criterion_value"]),
        float(expected["fit"]["criterion_value"]),
        atol=1e-8,
        rtol=1e-8,
    )
    np.testing.assert_allclose(
        np.asarray(actual["fit"]["smoothing_params"], dtype=np.float64),
        np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64),
        atol=1e-6,
        rtol=5e-4,
    )


def test_corpus_factor_by_s():
    rng = np.random.default_rng(31)
    n = 60
    x = rng.normal(size=n)
    fac = np.array(["p", "q"] * (n // 2), dtype=object)
    y = np.sin(x) + 0.4 * (fac == "q").astype(float) + rng.normal(0, 0.15, n)
    data = pd.DataFrame({"y": y, "x": x, "fac": fac})
    formula = 'y ~ s(x, by=fac, bs="cr", k=8)'
    actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
    expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
    _assert_corpus_shape_alignment(actual, expected)
    a_link = np.asarray(actual["predictions"]["link"], dtype=np.float64).ravel()
    e_link = np.asarray(expected["predictions"]["link"], dtype=np.float64).ravel()
    rmse = float(np.sqrt(np.mean((a_link - e_link) ** 2)))
    assert rmse < 0.22, rmse
    if np.std(a_link) > 1e-12 and np.std(e_link) > 1e-12:
        corr = float(np.corrcoef(a_link, e_link)[0, 1])
        assert corr > 0.92, corr
    a_c = a_link.copy()
    e_c = e_link.copy()
    for lev in np.unique(fac):
        m = fac == lev
        if np.sum(m) < 2:
            continue
        a_c[m] -= float(np.mean(a_c[m]))
        e_c[m] -= float(np.mean(e_c[m]))
    rmse_dm = float(np.sqrt(np.mean((a_c - e_c) ** 2)))
    assert rmse_dm < 0.18, rmse_dm


def test_corpus_re_noisy():
    re_data = _make_random_effect_data_noisy()
    re_form = 'y ~ s(f, bs="re")'
    a_re = _fit_nampy_snapshot(re_data, re_form, "gaussian", "REML")
    e_re = _run_mgcv_snapshot(re_data, re_form, "gaussian", "REML")
    _assert_corpus_predictions_and_edf(
        a_re, e_re, pred_atol=1.2e-2, pred_rtol=1.2e-2, edf_atol=0.55
    )


def test_corpus_mrf():
    mrf_data = _make_mrf_data()
    mrf_form = (
        'y ~ s(region, bs="mrf", k=3, '
        'xt=list(nb=list(A=c("B"), B=c("A","C"), C=c("B"))))'
    )
    a_mrf = _fit_nampy_snapshot(mrf_data, mrf_form, "gaussian", "REML")
    e_mrf = _run_mgcv_snapshot(mrf_data, mrf_form, "gaussian", "REML")
    _assert_corpus_predictions_and_edf(
        a_mrf, e_mrf, pred_atol=1.2e-2, pred_rtol=1.2e-2, edf_atol=0.55
    )


def test_corpus_fs():
    fs_data = _make_fs_data()
    fs_form = 'y ~ s(f, x, bs="fs", k=6)'
    a_fs = _fit_nampy_snapshot(fs_data, fs_form, "gaussian", "REML")
    e_fs = _run_mgcv_snapshot(fs_data, fs_form, "gaussian", "REML")
    _assert_corpus_predictions_and_edf(
        a_fs, e_fs, pred_atol=1.5e-2, pred_rtol=1.5e-2, edf_atol=0.65
    )


def test_corpus_poisson_tp_univariate_smooths():
    data = _make_poisson_data(seed=41, n=220)
    formula = 'y ~ s(x0, bs="tp", k=8) + s(x1, bs="tp", k=8)'
    actual = _fit_nampy_snapshot(data, formula, "poisson", "REML")
    expected = _run_mgcv_snapshot(data, formula, "poisson", "REML")
    _assert_corpus_predictions_and_edf(
        actual, expected, pred_atol=6e-2, pred_rtol=6e-2, edf_atol=0.8
    )


def test_corpus_binomial_tp_univariate_smooths():
    data = _make_binomial_data(seed=53, n=220)
    formula = 'y ~ s(x0, bs="tp", k=8) + s(x1, bs="tp", k=8)'
    actual = _fit_nampy_snapshot(data, formula, "binomial", "REML")
    expected = _run_mgcv_snapshot(data, formula, "binomial", "REML")
    _assert_corpus_predictions_and_edf(
        actual, expected, pred_atol=7e-2, pred_rtol=7e-2, edf_atol=0.9
    )


def test_corpus_gamma_tp_univariate_smooths():
    data = _make_gamma_data(seed=43, n=220)
    formula = 'y ~ s(x0, bs="tp", k=8) + s(x1, bs="tp", k=8)'
    actual = _fit_nampy_snapshot(data, formula, "gamma", "REML")
    expected = _run_mgcv_snapshot(data, formula, "gamma", "REML")
    _assert_corpus_predictions_and_edf(
        actual, expected, pred_atol=2.8e-1, pred_rtol=2.5e-1, edf_atol=1.2
    )


def test_corpus_negbin_tp_univariate_smooths():
    data = _make_negbin_data(seed=47, n=240, theta=1.0)
    family = {"name": "negbin", "theta": 1.0}
    formula = 'y ~ s(x0, bs="tp", k=8) + s(x1, bs="tp", k=8)'
    actual = _fit_nampy_snapshot(data, formula, family, "REML")
    expected = _run_mgcv_snapshot(data, formula, family, "REML")
    _assert_corpus_predictions_and_edf(
        actual, expected, pred_atol=9e-2, pred_rtol=9e-2, edf_atol=0.95
    )
