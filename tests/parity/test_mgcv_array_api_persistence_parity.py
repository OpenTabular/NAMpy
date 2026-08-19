"""Array (non-formula) fitting API and post-save persistence vs live mgcv.

The array API builds one s(column, bs=basis, k=k) per feature column
(nampy/gam/specs/modeling.py::make_predictor_specs), so its mgcv reference is
the equivalent explicit formula.  Persistence is checked against mgcv itself,
not merely against the pre-save Python object.
"""

from __future__ import annotations

import numpy as np
import pytest

from nampy.gam import GAM
from tests.mgcv_parity_utils import (
    _make_gaussian_data,
    _make_poisson_data,
    _run_mgcv_snapshot,
)

pytestmark = [pytest.mark.surface_regression]

_EQUIVALENT_FORMULA = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'


def test_array_api_gaussian_reml_matches_mgcv_formula_fit():
    """fit(X, y) with basis=/k= reproduces the equivalent mgcv formula fit."""
    data = _make_gaussian_data()
    gam = GAM(
        family="gaussian",
        basis="cr",
        k=8,
        optimize_smoothing=True,
        smoothing_method="REML",
        smoothing_optimizer="outer_newton",
    ).fit(X=data[["x0", "x1"]], y=data["y"].to_numpy(dtype=np.float64))
    expected = _run_mgcv_snapshot(
        data,
        _EQUIVALENT_FORMULA,
        "gaussian",
        "REML",
        optimizer="newton",
        allow_live_run=True,
    )
    result = gam.fit_result(include_covariances=True)
    np.testing.assert_allclose(
        np.log(np.asarray(result.smoothing_params, dtype=np.float64)),
        np.asarray(expected["fit"]["log_smoothing_params"], dtype=np.float64),
        atol=5e-6,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        result.criterion_value,
        float(np.ravel(expected["fit"]["criterion_value"])[0]),
        rtol=1e-9,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        np.asarray(result.edf_by_term, dtype=np.float64),
        np.asarray(expected["fit"]["edf_by_term"], dtype=np.float64),
        atol=5e-6,
        rtol=1e-7,
    )
    actual_link, actual_se = gam.predict(
        data[["x0", "x1"]], type="link", return_se=True
    )
    np.testing.assert_allclose(
        actual_link,
        np.asarray(expected["predictions"]["link"], dtype=np.float64),
        atol=1e-8,
        rtol=1e-8,
    )
    np.testing.assert_allclose(
        actual_se,
        np.asarray(expected["predictions"]["se_link"], dtype=np.float64),
        atol=1e-7,
        rtol=1e-7,
    )
    expected_cov = np.asarray(expected["fit"]["cov_bayes"], dtype=np.float64)
    actual_cov = np.asarray(gam.vcov(), dtype=np.float64)
    scale_ref = max(float(np.max(np.abs(expected_cov))), 1e-12)
    np.testing.assert_allclose(
        actual_cov, expected_cov, atol=1e-6 * scale_ref, rtol=1e-6
    )


def test_array_api_poisson_weights_offset_matches_mgcv_formula_fit():
    """Array-API offset= and sample_weight= reproduce the mgcv fit."""
    data = _make_poisson_data()
    rng = np.random.default_rng(452)
    data = data.copy()
    data["off"] = 0.1 * np.sin(data["x1"].to_numpy(dtype=np.float64))
    data["w"] = np.exp(rng.uniform(-0.5, 0.5, size=len(data)))
    formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8) + offset(off)'
    gam = GAM(
        family="poisson",
        basis="cr",
        k=8,
        optimize_smoothing=True,
        smoothing_method="REML",
        smoothing_optimizer="outer_newton",
    ).fit(
        X=data[["x0", "x1"]],
        y=data["y"].to_numpy(dtype=np.float64),
        offset=data["off"].to_numpy(dtype=np.float64),
        sample_weight=data["w"].to_numpy(dtype=np.float64),
    )
    expected = _run_mgcv_snapshot(
        data,
        formula,
        "poisson",
        "REML",
        weights_column="w",
        optimizer="newton",
        allow_live_run=True,
    )
    result = gam.fit_result(include_covariances=False)
    np.testing.assert_allclose(
        np.log(np.asarray(result.smoothing_params, dtype=np.float64)),
        np.asarray(expected["fit"]["log_smoothing_params"], dtype=np.float64),
        atol=5e-6,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        result.criterion_value,
        float(np.ravel(expected["fit"]["criterion_value"])[0]),
        rtol=1e-9,
        atol=1e-9,
    )
    actual_link = gam.predict(
        data[["x0", "x1"]],
        type="link",
        offset=data["off"].to_numpy(dtype=np.float64),
    )
    np.testing.assert_allclose(
        actual_link,
        np.asarray(expected["predictions"]["link"], dtype=np.float64),
        atol=1e-7,
        rtol=1e-7,
    )


def test_persistence_after_optimized_fit_matches_mgcv(tmp_path):
    """A saved+reloaded optimized model still matches mgcv directly."""
    data = _make_gaussian_data()
    gam = GAM(
        family="gaussian",
        formula=_EQUIVALENT_FORMULA,
        optimize_smoothing=True,
        smoothing_method="REML",
        smoothing_optimizer="outer_newton",
    ).fit(data=data)
    path = tmp_path / "gam_reml.pkl"
    assert gam.save_model(path) == path
    restored = GAM.load_model(path)

    expected = _run_mgcv_snapshot(
        data,
        _EQUIVALENT_FORMULA,
        "gaussian",
        "REML",
        optimizer="newton",
        allow_live_run=True,
    )
    e_fit = expected["fit"]

    result = restored.fit_result(include_covariances=True)
    np.testing.assert_allclose(
        np.log(np.asarray(result.smoothing_params, dtype=np.float64)),
        np.asarray(e_fit["log_smoothing_params"], dtype=np.float64),
        atol=5e-6,
        rtol=0.0,
    )
    actual_link, actual_se = restored.predict(
        data.drop(columns=["y"]), type="link", return_se=True
    )
    np.testing.assert_allclose(
        actual_link,
        np.asarray(expected["predictions"]["link"], dtype=np.float64),
        atol=1e-8,
        rtol=1e-8,
    )
    np.testing.assert_allclose(
        actual_se,
        np.asarray(expected["predictions"]["se_link"], dtype=np.float64),
        atol=1e-7,
        rtol=1e-7,
    )

    expected_cov = np.asarray(e_fit["cov_bayes"], dtype=np.float64)
    actual_cov = np.asarray(restored.vcov(), dtype=np.float64)
    scale_ref = max(float(np.max(np.abs(expected_cov))), 1e-12)
    np.testing.assert_allclose(
        actual_cov, expected_cov, atol=1e-6 * scale_ref, rtol=1e-6
    )

    # Restored inference/diagnostic surfaces stay mgcv-consistent.
    summary = restored.summary()
    block = expected["parity"]["diagnostics"]["summary"]
    assert summary.residual_df == pytest.approx(
        float(np.ravel(block["residual_df"])[0]), rel=1e-6
    )
    assert summary.scale == pytest.approx(float(np.ravel(block["scale"])[0]), rel=1e-6)
    anova = restored.anova(freq=False)
    expected_smooth = np.asarray(
        expected["parity"]["diagnostics"]["anova_smooth"]["values"],
        dtype=np.float64,
    )
    actual_smooth = anova.smooth_table[
        ["edf", "ref_df", "wald_stat", "p_value"]
    ].to_numpy(dtype=np.float64)
    np.testing.assert_allclose(actual_smooth, expected_smooth, rtol=1e-5, atol=1e-8)
