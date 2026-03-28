from __future__ import annotations

import numpy as np
import pytest

from nampy.gam.inference.anova import _smooth_test_stat

from mgcv_parity_utils import (
    R_SCRIPT,
    _fit_nampy_model,
    _fit_nampy_snapshot,
    _make_gaussian_data,
    _make_poisson_data,
    _run_mgcv_anova,
    _run_mgcv_snapshot,
)


def _labels(x):
    return [x] if isinstance(x, str) else list(x)


@pytest.mark.skipif(R_SCRIPT is None, reason="Rscript is not available")
def test_anova_single_model_matches_mgcv_term_tables():
    data = _make_gaussian_data(seed=902, n=160)
    formula = "y ~ s(x0, k=8) + s(x1, k=8)"

    actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
    expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")

    a_diag = actual["parity"]["diagnostics"]
    e_diag = expected["parity"]["diagnostics"]

    assert _labels(a_diag["anova_smooth"]["labels"]) == _labels(e_diag["anova_smooth"]["labels"])

    a_smooth = np.asarray(a_diag["anova_smooth"]["values"], dtype=np.float64)
    e_smooth = np.asarray(e_diag["anova_smooth"]["values"], dtype=np.float64)

    assert a_smooth.shape == e_smooth.shape
    np.testing.assert_allclose(a_smooth[:, 0], e_smooth[:, 0], atol=0.12, rtol=0.0)
    assert np.all(a_smooth[:, 1] > 0.0)
    assert np.all(e_smooth[:, 1] > 0.0)
    assert np.all(a_smooth[:, 2] > 0.0)
    assert np.all(e_smooth[:, 2] > 0.0)
    assert np.array_equal(np.argsort(a_smooth[:, 2]), np.argsort(e_smooth[:, 2]))
    assert np.all(a_smooth[:, 3] < 1e-12)


@pytest.mark.skipif(R_SCRIPT is None, reason="Rscript is not available")
def test_anova_single_model_parametric_term_tracks_mgcv_signal():
    data = _make_gaussian_data(seed=921, n=160)
    formula = "y ~ x0 + s(x1, k=8)"

    actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
    expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")

    a_param = actual["parity"]["diagnostics"]["anova_parametric"]
    e_param = expected["parity"]["diagnostics"]["anova_parametric"]
    assert _labels(a_param["labels"]) == _labels(e_param["labels"])

    a_vals = np.asarray(a_param["values"], dtype=np.float64)
    e_vals = np.asarray(e_param["values"], dtype=np.float64)
    np.testing.assert_allclose(a_vals[:, 0], e_vals[:, 0], atol=1e-10, rtol=0.0)
    stat_ratio = a_vals[:, 1] / e_vals[:, 1]
    assert np.all((stat_ratio > 0.5) & (stat_ratio < 1.5))
    assert np.all(a_vals[:, 2] < 1e-20)


@pytest.mark.skipif(R_SCRIPT is None, reason="Rscript is not available")
def test_python_smooth_test_translation_matches_mgcv_on_mgcv_inputs():
    data = _make_gaussian_data(seed=921, n=160)
    formula = "y ~ x0 + s(x1, k=8)"

    expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
    diag = expected["parity"]["diagnostics"]
    inputs = diag["smooth_test_inputs"]
    smooth = diag["anova_smooth"]
    cov = diag["smooth_cov_bayes"]

    labels = _labels(inputs["labels"])
    assert labels == _labels(smooth["labels"]) == _labels(cov["labels"])
    coef_blocks = inputs["coef_blocks"]
    if not isinstance(coef_blocks, list) or (
        len(labels) == 1 and coef_blocks and not isinstance(coef_blocks[0], (list, tuple))
    ):
        coef_blocks = [coef_blocks]
    r_blocks = inputs["r_blocks"]
    if not isinstance(r_blocks, list) or (
        len(labels) == 1 and r_blocks and not isinstance(r_blocks[0], (list, tuple))
    ):
        r_blocks = [r_blocks]
    cov_blocks = cov["blocks"]
    if not isinstance(cov_blocks, list) or (
        len(labels) == 1 and cov_blocks and not isinstance(cov_blocks[0], (list, tuple))
    ):
        cov_blocks = [cov_blocks]
    edf1_vals = np.atleast_1d(np.asarray(inputs["edf1"], dtype=np.float64))
    smooth_vals = np.atleast_2d(np.asarray(smooth["values"], dtype=np.float64))

    for i, label in enumerate(labels):
        stat, ref_df, p_value = _smooth_test_stat(
            np.asarray(coef_blocks[i], dtype=np.float64),
            np.asarray(r_blocks[i], dtype=np.float64),
            np.asarray(cov_blocks[i], dtype=np.float64),
            rank=float(edf1_vals[i]),
            residual_df=float(inputs["residual_df"]),
        )
        expected_row = smooth_vals[i]
        np.testing.assert_allclose(ref_df, expected_row[1], atol=1e-10, rtol=0.0)
        np.testing.assert_allclose(stat / ref_df, expected_row[2], atol=1e-10, rtol=0.0)
        assert 0.0 <= p_value <= 1.0


@pytest.mark.skipif(R_SCRIPT is None, reason="Rscript is not available")
def test_python_smooth_test_matches_local_snapshot_inputs_exactly():
    data = _make_gaussian_data(seed=921, n=160)
    formula = "y ~ x0 + s(x1, k=8)"

    actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
    diag = actual["parity"]["diagnostics"]
    inputs = diag["smooth_test_inputs"]
    smooth = diag["anova_smooth"]
    cov = diag["smooth_cov_bayes"]

    labels = _labels(inputs["labels"])
    assert labels == _labels(smooth["labels"]) == _labels(cov["labels"])
    coef_blocks = inputs["coef_blocks"]
    if not isinstance(coef_blocks, list) or (
        len(labels) == 1 and coef_blocks and not isinstance(coef_blocks[0], (list, tuple))
    ):
        coef_blocks = [coef_blocks]
    r_blocks = inputs["r_blocks"]
    if not isinstance(r_blocks, list) or (
        len(labels) == 1 and r_blocks and not isinstance(r_blocks[0], (list, tuple))
    ):
        r_blocks = [r_blocks]
    cov_blocks = cov["blocks"]
    if not isinstance(cov_blocks, list) or (
        len(labels) == 1 and cov_blocks and not isinstance(cov_blocks[0], (list, tuple))
    ):
        cov_blocks = [cov_blocks]
    edf1_vals = np.atleast_1d(np.asarray(inputs["edf1"], dtype=np.float64))
    smooth_vals = np.atleast_2d(np.asarray(smooth["values"], dtype=np.float64))

    for i in range(len(labels)):
        stat, ref_df, _ = _smooth_test_stat(
            np.asarray(coef_blocks[i], dtype=np.float64),
            np.asarray(r_blocks[i], dtype=np.float64),
            np.asarray(cov_blocks[i], dtype=np.float64),
            rank=float(edf1_vals[i]),
            residual_df=float(inputs["residual_df"]),
        )
        expected_row = smooth_vals[i]
        np.testing.assert_allclose(ref_df, expected_row[1], atol=1e-10, rtol=0.0)
        np.testing.assert_allclose(stat / ref_df, expected_row[2], atol=1e-10, rtol=0.0)


@pytest.mark.skipif(R_SCRIPT is None, reason="Rscript is not available")
def test_anova_multi_model_matches_mgcv_analysis_of_deviance():
    data = _make_poisson_data(seed=923, n=220)
    formulas = [
        'y ~ s(x0, bs="cr", k=8)',
        'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)',
    ]

    reduced = _fit_nampy_model(data, formulas[0], "poisson", "REML")
    full = _fit_nampy_model(data, formulas[1], "poisson", "REML")
    actual = reduced.anova(full, test="Chisq")
    expected = _run_mgcv_anova(data, formulas, "poisson", "REML", test="Chisq")

    a_tbl = actual.table.copy()
    e_cols = list(expected["table"]["columns"])
    e_vals = np.asarray(expected["table"]["values"], dtype=object)

    assert e_cols[:5] == ["Resid. Df", "Resid. Dev", "Df", "Deviance", "Pr(>Chi)"]

    np.testing.assert_allclose(
        a_tbl["residual_df"].to_numpy(dtype=np.float64),
        np.asarray(e_vals[:, 0], dtype=np.float64),
        atol=0.35,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        a_tbl["deviance"].to_numpy(dtype=np.float64),
        np.asarray(e_vals[:, 1], dtype=np.float64),
        atol=0.5,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        a_tbl["edf_diff"].to_numpy(dtype=np.float64)[1:],
        np.asarray(e_vals[1:, 2], dtype=np.float64),
        atol=0.25,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        a_tbl["deviance_diff"].to_numpy(dtype=np.float64)[1:],
        np.asarray(e_vals[1:, 3], dtype=np.float64),
        atol=0.5,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        a_tbl["p_value"].to_numpy(dtype=np.float64)[1:],
        np.asarray(e_vals[1:, 4], dtype=np.float64),
        atol=2e-5,
        rtol=0.0,
    )


def test_anova_multi_model_requires_compatible_model_family():
    data = _make_gaussian_data(seed=924, n=140)
    g = _fit_nampy_model(data, "y ~ s(x0, k=8)", "gaussian", "REML")
    p = _fit_nampy_model(
        (data.assign(y=np.round(np.exp(data.y - data.y.min())))).astype({"y": int}),
        "y ~ s(x0, k=8)",
        "poisson",
        "REML",
    )

    try:
        g.anova(p)
    except ValueError as exc:
        assert "same family" in str(exc)
    else:
        raise AssertionError("Expected family mismatch to raise ValueError.")
