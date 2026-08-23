from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from nampy.gam.parity import build_optimizer_trace
from tests._paths import PARITY_DIR, REPO_ROOT
from tests.reference_fixtures import (
    load_reference,
    portable_dataframe_identity,
    reference_key,
    save_reference,
)

R_SCRIPT = shutil.which("Rscript")
MGCV_TRACE_SCRIPT = PARITY_DIR / "mgcv_trace.R"


def _make_gaussian_data(seed=321, n=180):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-2, 2, size=n)
    x1 = rng.uniform(-1.5, 1.5, size=n)
    y = np.sin(1.1 * x0) + 0.35 * x1**2 + rng.normal(scale=0.15, size=n)
    return pd.DataFrame({"y": y, "x0": x0, "x1": x1})


def _make_binomial_data(seed=456, n=220):
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n)
    x1 = rng.normal(size=n)
    eta = 0.9 * np.sin(x0) - 0.45 * x1
    p = 1.0 / (1.0 + np.exp(-eta))
    y = rng.binomial(1, p, size=n).astype(np.float64)
    return pd.DataFrame({"y": y, "x0": x0, "x1": x1})


def _make_poisson_data(seed=789, n=220):
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n)
    x1 = rng.normal(size=n)
    mu = np.exp(0.2 + 0.7 * np.sin(x0) - 0.25 * x1)
    y = rng.poisson(mu).astype(np.float64)
    return pd.DataFrame({"y": y, "x0": x0, "x1": x1})


def _run_mgcv_trace(
    data: pd.DataFrame,
    formula: str,
    family: str,
    method: str,
):
    key = reference_key(
        "optimizer_trace",
        {
            "data": portable_dataframe_identity(data),
            "formula": formula,
            "family": family,
            "method": method,
            "select": False,
        },
    )
    cached = load_reference("mgcv", key)
    if cached is not None:
        return cached
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        csv_path = tmpdir_path / "data.csv"
        json_path = tmpdir_path / "trace.json"
        data.to_csv(csv_path, index=False)
        subprocess.run(
            [
                R_SCRIPT,
                str(MGCV_TRACE_SCRIPT),
                str(csv_path),
                str(json_path),
                formula,
                family,
                method,
                "false",
            ],
            check=True,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        result = json.loads(json_path.read_text(encoding="utf-8"))
        save_reference("mgcv", key, result)
        return result


def _fit_nampy_trace(
    data: pd.DataFrame,
    formula: str,
    family: str,
    method: str,
):
    gam = GAM(
        family=family,
        formula=formula,
        optimize_smoothing=True,
        smoothing_method=method,
    )
    gam.fit(data=data)
    return build_optimizer_trace(gam), gam


def _assert_strict_score_hist_exact(model, expected, *, atol=0.0, rtol=0.0):
    expected_scores = np.asarray(
        expected["fit"]["outer_info"]["score_hist"], dtype=np.float64
    )
    actual_result = getattr(model, "_optim_result", None)

    assert actual_result is not None
    assert hasattr(actual_result, "strict_score_hist")

    actual_scores = np.asarray(actual_result.strict_score_hist, dtype=np.float64)

    assert actual_scores.shape == expected_scores.shape
    np.testing.assert_allclose(
        actual_scores,
        expected_scores,
        rtol=rtol,
        atol=atol,
    )


class TestMgcvNewtonParity:
    """
    Newton score-history parity checks against mgcv for representative Gaussian and non-
    Gaussian REML fits.
    """
    def test_newton_score_hist_gaussian_reml_matches_r_exact(self):
        """Verify that newton score hist gaussian REML matches r exact."""
        data = _make_gaussian_data(seed=321)
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'
        expected = _run_mgcv_trace(data, formula, "gaussian", "REML")
        _, gam = _fit_nampy_trace(data, formula, "gaussian", "REML")
        _assert_strict_score_hist_exact(gam, expected, atol=1e-12)

    @pytest.mark.parametrize("family", ["binomial", "poisson"])
    def test_newton_score_hist_non_gaussian_reml_matches_r(self, family):
        """Verify that newton score hist non gaussian REML matches r."""
        data = (
            _make_binomial_data(seed=456)
            if family == "binomial"
            else _make_poisson_data(seed=789)
        )

        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'
        expected = _run_mgcv_trace(data, formula, family, "REML")
        _, gam = _fit_nampy_trace(data, formula, family, "REML")
        _assert_strict_score_hist_exact(gam, expected, atol=1e-6)
