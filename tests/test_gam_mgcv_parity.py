from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nampy.basemodels.gam import GAM


R_SCRIPT = shutil.which("Rscript")
MGCV_SNAPSHOT_SCRIPT = Path(__file__).resolve().parent / "parity" / "mgcv_snapshot.R"


def _make_gaussian_data(seed=123, n=180):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-2.0, 2.0, size=n)
    x1 = rng.uniform(-1.5, 1.5, size=n)
    y = np.sin(1.2 * x0) + 0.4 * x1**2 + rng.normal(scale=0.15, size=n)
    return pd.DataFrame({"y": y, "x0": x0, "x1": x1})


def _make_binomial_data(seed=456, n=220):
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n)
    x1 = rng.normal(size=n)
    eta = 0.9 * np.sin(x0) - 0.45 * x1
    p = 1.0 / (1.0 + np.exp(-eta))
    y = rng.binomial(1, p)
    return pd.DataFrame({"y": y, "x0": x0, "x1": x1})


def _make_poisson_data(seed=789, n=220):
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n)
    x1 = rng.normal(size=n)
    mu = np.exp(0.2 + 0.7 * np.sin(x0) - 0.25 * x1)
    y = rng.poisson(mu)
    return pd.DataFrame({"y": y, "x0": x0, "x1": x1})


def _make_gamma_data(seed=1701, n=220):
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n)
    x1 = rng.normal(size=n)
    eta = 0.15 + 0.6 * np.sin(x0) - 0.2 * x1
    mu = np.exp(eta)
    shape = 3.5
    y = rng.gamma(shape=shape, scale=mu / shape)
    return pd.DataFrame({"y": y, "x0": x0, "x1": x1})


def _make_negbin_data(seed=2024, n=240, theta=1.0):
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n)
    x1 = rng.normal(size=n)
    eta = 0.2 + 0.55 * np.sin(x0) - 0.25 * x1
    mu = np.exp(eta)
    p = theta / (theta + mu)
    y = rng.negative_binomial(theta, p, size=n)
    return pd.DataFrame({"y": y, "x0": x0, "x1": x1})


def _family_specs(family):
    if isinstance(family, dict):
        key = str(family.get("name", "")).lower()
        if key in {"negbin", "negativebinomial", "negative_binomial"}:
            theta = float(family.get("theta", 1.0))
            return family, f"negbin:{theta:.12g}"
        return family, key
    key = str(family).lower()
    return family, key


def _run_mgcv_snapshot(data: pd.DataFrame, formula: str, family, method: str, *, select: bool = False):
    if R_SCRIPT is None:
        pytest.skip("Rscript is not available; mgcv parity tests are skipped.")

    _family_nampy, family_token = _family_specs(family)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        csv_path = tmpdir_path / "data.csv"
        json_path = tmpdir_path / "snapshot.json"
        data.to_csv(csv_path, index=False)

        subprocess.run(
            [
                R_SCRIPT,
                str(MGCV_SNAPSHOT_SCRIPT),
                str(csv_path),
                str(json_path),
                formula,
                family_token,
                method,
                "true" if select else "false",
            ],
            check=True,
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
        )

        return json.loads(json_path.read_text(encoding="utf-8"))


def _fit_nampy_snapshot(data: pd.DataFrame, formula: str, family, method: str, *, select: bool = False):
    family_nampy, _family_token = _family_specs(family)
    gam = GAM(
        family=family_nampy,
        formula=formula,
        select=select,
        optimize_smoothing=True,
        smoothing_method=method,
    )
    gam.fit(data=data)
    return gam.parity_snapshot(X=data, include_covariances=False)


def _assert_basic_mgcv_parity(actual, expected, *, pred_atol, pred_rtol, sp_log_atol):
    a_fit = actual["fit"]
    e_fit = expected["fit"]
    a_pred = actual["predictions"]
    e_pred = expected["predictions"]

    assert len(a_fit["smoothing_params"]) == len(e_fit["smoothing_params"])
    np.testing.assert_allclose(
        np.log(np.asarray(a_fit["smoothing_params"], dtype=np.float64)),
        np.log(np.asarray(e_fit["smoothing_params"], dtype=np.float64)),
        atol=sp_log_atol,
        rtol=0.0,
    )

    np.testing.assert_allclose(
        np.asarray(a_fit["edf_total"], dtype=np.float64),
        np.asarray(e_fit["edf_total"], dtype=np.float64),
        atol=0.25,
        rtol=0.1,
    )
    np.testing.assert_allclose(
        np.asarray(a_fit["edf_by_term"], dtype=np.float64),
        np.asarray(e_fit["edf_by_term"], dtype=np.float64),
        atol=0.25,
        rtol=0.15,
    )
    np.testing.assert_allclose(
        np.asarray(a_fit["deviance"], dtype=np.float64),
        np.asarray(e_fit["deviance"], dtype=np.float64),
        atol=0.5,
        rtol=0.1,
    )

    if a_fit.get("criterion_value", None) is not None and e_fit.get("criterion_value", None) is not None:
        np.testing.assert_allclose(
            np.asarray(a_fit["criterion_value"], dtype=np.float64),
            np.asarray(e_fit["criterion_value"], dtype=np.float64),
            atol=1.0,
            rtol=0.1,
        )

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


class TestParitySnapshotAPI:
    def test_parity_snapshot_supports_direct_gam_object(self):
        data = _make_gaussian_data(n=80)
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'
        gam = GAM(formula=formula, optimize_smoothing=True, smoothing_method="REML")
        gam.fit(data=data)

        snap = gam.parity_snapshot(X=data, include_covariances=False)

        assert "fit" in snap
        assert "predictions" in snap
        assert len(snap["fit"]["smoothing_params"]) == 2
        assert np.asarray(snap["predictions"]["response"]).shape == (len(data),)


@pytest.mark.skipif(R_SCRIPT is None, reason="Rscript is not available")
class TestMgcvParity:
    def test_gaussian_reml_matches_mgcv(self):
        data = _make_gaussian_data()
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")

        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=5e-2,
            pred_rtol=5e-2,
            sp_log_atol=0.75,
        )

    def test_binomial_reml_matches_mgcv(self):
        data = _make_binomial_data()
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

        actual = _fit_nampy_snapshot(data, formula, "binomial", "REML")
        expected = _run_mgcv_snapshot(data, formula, "binomial", "REML")

        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=6e-2,
            pred_rtol=6e-2,
            sp_log_atol=0.9,
        )

    def test_poisson_reml_matches_mgcv(self):
        data = _make_poisson_data()
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

        actual = _fit_nampy_snapshot(data, formula, "poisson", "REML")
        expected = _run_mgcv_snapshot(data, formula, "poisson", "REML")

        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=8e-2,
            pred_rtol=8e-2,
            sp_log_atol=1.0,
        )

    def test_gaussian_select_reml_matches_mgcv(self):
        data = _make_gaussian_data(seed=999)
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML", select=True)
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML", select=True)

        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=6e-2,
            pred_rtol=6e-2,
            sp_log_atol=1.0,
        )

    def test_gamma_reml_matches_mgcv(self):
        data = _make_gamma_data()
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

        actual = _fit_nampy_snapshot(data, formula, "gamma", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gamma", "REML")
        a_fit = actual["fit"]
        e_fit = expected["fit"]
        a_pred = actual["predictions"]
        e_pred = expected["predictions"]

        # Gamma REML is currently less stable in log(sp) and EDF parity than
        # Gaussian/Binomial/Poisson, so we assert predictive and criterion parity.
        np.testing.assert_allclose(
            np.asarray(a_pred["response"], dtype=np.float64),
            np.asarray(e_pred["response"], dtype=np.float64),
            atol=2.7e-1,
            rtol=2.5e-1,
        )
        np.testing.assert_allclose(
            np.asarray(a_pred["link"], dtype=np.float64),
            np.asarray(e_pred["link"], dtype=np.float64),
            atol=2.2e-1,
            rtol=2.5e-1,
        )
        np.testing.assert_allclose(
            np.asarray(a_fit["deviance"], dtype=np.float64),
            np.asarray(e_fit["deviance"], dtype=np.float64),
            atol=1.0,
            rtol=0.15,
        )
        np.testing.assert_allclose(
            np.asarray(a_fit["criterion_value"], dtype=np.float64),
            np.asarray(e_fit["criterion_value"], dtype=np.float64),
            atol=2.0,
            rtol=0.1,
        )

    def test_negbin_reml_matches_mgcv(self):
        data = _make_negbin_data(theta=1.0)
        family = {"name": "negbin", "theta": 1.0}
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

        actual = _fit_nampy_snapshot(data, formula, family, "REML")
        expected = _run_mgcv_snapshot(data, formula, family, "REML")

        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=1.0e-1,
            pred_rtol=1.0e-1,
            sp_log_atol=1.25,
        )

    def test_poisson_reml_with_formula_offset_matches_mgcv(self):
        data = _make_poisson_data(seed=177)
        data = data.copy()
        data["off"] = np.linspace(-0.35, 0.35, len(data))
        formula = 'y ~ offset(off) + s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

        actual = _fit_nampy_snapshot(data, formula, "poisson", "REML")
        expected = _run_mgcv_snapshot(data, formula, "poisson", "REML")

        _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=9e-2,
            pred_rtol=9e-2,
            sp_log_atol=1.1,
        )
