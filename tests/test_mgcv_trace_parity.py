from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mgcv_parity_utils import _family_specs, _make_negbin_data

from nampy.basemodels.gam import GAM
from nampy.gam.parity import (
    build_optimizer_trace,
    load_optimizer_trace,
    save_optimizer_trace,
)
from nampy.gam.smoothing_selection.criteria import criterion_gradient, criterion_hessian

R_SCRIPT = shutil.which("Rscript")
MGCV_TRACE_SCRIPT = Path(__file__).resolve().parent / "parity" / "mgcv_trace.R"
MGCV_NEGBIN_INNER_TRACE_SCRIPT = (
    Path(__file__).resolve().parent / "parity" / "mgcv_negbin_inner_trace.R"
)


def _make_gaussian_data(seed=321, n=180):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-2.0, 2.0, size=n)
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
    *,
    select: bool = False,
):
    if R_SCRIPT is None:
        pytest.skip("Rscript is not available; mgcv trace parity tests are skipped.")
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
                "true" if select else "false",
            ],
            check=True,
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
        )
        return json.loads(json_path.read_text(encoding="utf-8"))


def _fit_nampy_trace(
    data: pd.DataFrame,
    formula: str,
    family: str,
    method: str,
    *,
    select: bool = False,
):
    gam = GAM(
        family=family,
        formula=formula,
        select=select,
        optimize_smoothing=True,
        smoothing_method=method,
    )
    gam.fit(data=data)
    return build_optimizer_trace(gam)


def _fit_nampy_model_and_trace(
    data: pd.DataFrame,
    formula: str,
    family: str,
    method: str,
    *,
    select: bool = False,
):
    gam = GAM(
        family=family,
        formula=formula,
        select=select,
        optimize_smoothing=True,
        smoothing_method=method,
    )
    gam.fit(data=data)
    return gam, build_optimizer_trace(gam)


def _run_mgcv_negbin_inner_trace(
    data: pd.DataFrame,
    formula: str,
    family,
):
    if R_SCRIPT is None:
        pytest.skip("Rscript is not available; mgcv trace parity tests are skipped.")
    _family_obj, family_token = _family_specs(family)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        csv_path = tmpdir_path / "data.csv"
        json_path = tmpdir_path / "trace.json"
        data.to_csv(csv_path, index=False)
        subprocess.run(
            [
                R_SCRIPT,
                str(MGCV_NEGBIN_INNER_TRACE_SCRIPT),
                str(csv_path),
                str(json_path),
                formula,
                family_token,
            ],
            check=True,
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
        )
        return json.loads(json_path.read_text(encoding="utf-8"))


def _fit_nampy_negbin_inner_trace(data: pd.DataFrame, formula: str, family):
    gam = GAM(
        family=family,
        formula=formula,
        optimize_smoothing=False,
        smoothing_method="fixed",
    )
    gam.fit(data=data)
    return list(getattr(gam, "_pirls_last_inner_trace_", []) or []), gam


def _criterion_series(trace_obj):
    out = []
    for row in trace_obj.get("trace", []):
        c = row.get("criterion", None)
        if c is not None:
            out.append(float(c))
    return np.asarray(out, dtype=np.float64)


def _tail_criterion_series(trace_obj, n_tail: int):
    crit = _criterion_series(trace_obj)
    if crit.size <= n_tail:
        return crit
    return crit[-int(n_tail) :]


def _assert_gaussian_outer_trace_parity(actual, expected):
    e_outer = expected["fit"]["outer_info"]
    e_n_iters = int(e_outer["iter"])
    e_score_hist = np.asarray(e_outer["score_hist"], dtype=np.float64)

    assert 1 <= e_n_iters <= 10, f"Unexpected mgcv outer iter count: {e_n_iters}"
    assert np.isfinite(e_score_hist).all()
    assert np.all(
        np.diff(e_score_hist) <= 1e-6
    ), "mgcv score.hist is not monotonically non-increasing"

    a_crit = _criterion_series(actual)
    assert a_crit.size >= 3
    assert np.isfinite(a_crit).all()
    assert np.all(
        np.diff(a_crit) <= 1e-6
    ), "nampy accepted-step criterion history is not monotonically non-increasing"

    np.testing.assert_allclose(
        float(a_crit[-1]),
        float(e_score_hist[-1]),
        atol=1e-3,
        rtol=0.0,
        err_msg="Nampy final REML criterion diverges from mgcv score.hist endpoint",
    )

    a_tail = _tail_criterion_series(actual, 3)
    e_tail = e_score_hist[-3:]
    np.testing.assert_allclose(a_tail, e_tail, atol=1e-3, rtol=0.0)


def _assert_non_gaussian_outer_diagnostics(actual, expected):
    e_outer = expected["fit"].get("outer_info", {})
    e_iters = e_outer.get("iter", None)
    e_scores = e_outer.get("score_hist", None)
    assert e_iters is not None
    assert int(e_iters) >= 1
    assert isinstance(e_scores, list)
    assert len(e_scores) == int(e_iters)

    a_crit = _criterion_series(actual)
    assert a_crit.size >= len(e_scores)
    assert np.isfinite(a_crit).all()
    assert np.isfinite(np.asarray(e_scores, dtype=np.float64)).all()

    # Both optimizers should land at effectively the same final objective value.
    np.testing.assert_allclose(
        float(a_crit[-1]),
        float(e_scores[-1]),
        atol=2e-3,
        rtol=0.0,
    )


@pytest.mark.skipif(R_SCRIPT is None, reason="Rscript is not available")
class TestMgcvTraceParity:
    def test_trace_io_roundtrip_preserves_schema(self):
        data = _make_gaussian_data(seed=42)
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'
        actual = _fit_nampy_trace(data, formula, "gaussian", "REML")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trace.json"
            save_optimizer_trace(actual, path)
            loaded = load_optimizer_trace(path)
        assert set(loaded.keys()) == {"fit", "trace"}
        assert set(loaded["fit"].keys()) >= {"criterion_name", "smoothing_params"}
        assert isinstance(loaded["trace"], list)
        assert len(loaded["trace"]) >= 1

    def test_mgcv_outer_info_schema_available(self):
        data = _make_gaussian_data(seed=101)
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'
        expected = _run_mgcv_trace(data, formula, "gaussian", "REML")
        outer = expected["fit"].get("outer_info", {})
        assert isinstance(outer.get("conv", None), str)
        assert int(outer.get("iter", 0)) >= 1
        score_hist = outer.get("score_hist", None)
        assert isinstance(score_hist, list)
        assert len(score_hist) == int(outer["iter"])

    @pytest.mark.parametrize(
        "family, maker",
        [
            ("binomial", _make_binomial_data),
            ("poisson", _make_poisson_data),
        ],
    )
    def test_non_gaussian_final_grad_hess_are_exposed(self, family, maker):
        data = maker()
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'
        expected = _run_mgcv_trace(data, formula, family, "REML")
        final = expected["trace"][-1]
        grad = np.asarray(final["gradient"], dtype=np.float64).ravel()
        hess = np.asarray(final["hessian"], dtype=np.float64)
        assert grad.ndim == 1
        assert grad.size == 2
        assert hess.shape == (2, 2)
        assert np.isfinite(grad).all()
        assert np.isfinite(hess).all()

    @pytest.mark.parametrize(
        "family, maker",
        [
            ("binomial", _make_binomial_data),
            ("poisson", _make_poisson_data),
        ],
    )
    def test_non_gaussian_final_gradient_matches_mgcv(self, family, maker):
        data = maker()
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'
        model, _trace = _fit_nampy_model_and_trace(data, formula, family, "REML")
        expected = _run_mgcv_trace(data, formula, family, "REML")

        y = np.asarray(data["y"], dtype=np.float64)
        log_sp = np.log(np.asarray(model.smoothing_params, dtype=np.float64))
        a_grad = np.asarray(
            criterion_gradient(model, y, log_sp, method="REML"), dtype=np.float64
        ).ravel()
        e_grad = np.asarray(expected["trace"][-1]["gradient"], dtype=np.float64).ravel()
        np.testing.assert_allclose(a_grad, e_grad, atol=5e-4, rtol=0.0)

    @pytest.mark.parametrize(
        "family, maker",
        [
            ("binomial", _make_binomial_data),
            ("poisson", _make_poisson_data),
        ],
    )
    def test_non_gaussian_final_hessian_matches_mgcv(self, family, maker):
        data = maker()
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'
        model, _trace = _fit_nampy_model_and_trace(data, formula, family, "REML")
        expected = _run_mgcv_trace(data, formula, family, "REML")

        y = np.asarray(data["y"], dtype=np.float64)
        log_sp = np.log(np.asarray(model.smoothing_params, dtype=np.float64))
        a_hess = np.asarray(
            criterion_hessian(model, y, log_sp, method="REML"), dtype=np.float64
        )
        e_hess = np.asarray(expected["trace"][-1]["hessian"], dtype=np.float64)
        np.testing.assert_allclose(a_hess, e_hess, atol=2e-3, rtol=0.0)

    def test_gaussian_reml_newton_trajectory_matches_mgcv(self):
        """Gaussian REML accepted-step tail should track mgcv outer score history."""
        data = _make_gaussian_data(seed=321)
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

        actual = _fit_nampy_trace(data, formula, "gaussian", "REML")
        expected = _run_mgcv_trace(data, formula, "gaussian", "REML")
        _assert_gaussian_outer_trace_parity(actual, expected)

        # Final log(sp) must be consistent with the endpoint criterion match.
        a_log_sp = np.log(
            np.asarray(actual["fit"]["smoothing_params"], dtype=np.float64)
        )
        e_log_sp = np.log(
            np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64)
        )
        np.testing.assert_allclose(a_log_sp, e_log_sp, atol=0.75, rtol=0.0)

    def test_gaussian_trace_final_row_matches_mgcv_endpoint(self):
        """Dedicated Gaussian trace parity: final row should expose mgcv endpoint."""
        data = _make_gaussian_data(seed=101)
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'

        actual = _fit_nampy_trace(data, formula, "gaussian", "REML")
        expected = _run_mgcv_trace(data, formula, "gaussian", "REML")

        final_row = actual["trace"][-1]
        mgcv_fit = expected["fit"]
        e_score_hist = np.asarray(
            expected["fit"]["outer_info"]["score_hist"], dtype=np.float64
        )

        assert final_row["iter"] >= len(e_score_hist)
        np.testing.assert_allclose(
            float(final_row["criterion"]),
            float(e_score_hist[-1]),
            atol=1e-3,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            np.log(np.asarray(mgcv_fit["smoothing_params"], dtype=np.float64)),
            np.asarray(final_row["log_sp"], dtype=np.float64),
            atol=0.75,
            rtol=0.0,
        )

    def test_negbin_estimated_theta_fixed_sp_inner_trace_is_exposed(self):
        data = _make_negbin_data(seed=2024, n=240, theta=1.0)
        formula = 'y ~ s(x0, bs="cr", k=8, sp=1.0)'
        family = {"name": "negbin", "theta": 2.0, "estimate_theta": True}

        actual_rows, model = _fit_nampy_negbin_inner_trace(data, formula, family)
        expected = _run_mgcv_negbin_inner_trace(data, formula, family)

        a_theta = np.asarray(
            [row["log_theta"] for row in actual_rows], dtype=np.float64
        )
        e_theta = np.asarray(
            [row["log_theta"] for row in expected["inner_trace"]], dtype=np.float64
        )

        assert a_theta.size >= 1
        assert e_theta.size >= 1
        assert e_theta[0] == pytest.approx(np.log(2.0), abs=1e-12)
        e_updates = e_theta[1:]
        assert a_theta.shape == e_updates.shape
        assert np.isfinite(a_theta).all()
        assert np.isfinite(e_theta).all()
        np.testing.assert_allclose(a_theta, e_updates, atol=1e-8, rtol=0.0)
        assert np.isfinite(float(model.family.theta))
        assert np.isfinite(float(expected["fit"]["family_theta"]))
        assert float(model.family.theta) == pytest.approx(
            float(expected["fit"]["family_theta"]), abs=1e-8
        )

    @pytest.mark.parametrize(
        "family, method, seed, atol",
        [
            ("gaussian", "REML", 321, 0.75),
            ("gaussian", "ML", 99, 0.8),
            ("binomial", "REML", 456, 1.05),
            ("poisson", "REML", 789, 1.0),
        ],
    )
    def test_endpoint_log_sp_seed_matrix(self, family, method, seed, atol):
        maker = {
            "gaussian": _make_gaussian_data,
            "binomial": _make_binomial_data,
            "poisson": _make_poisson_data,
        }[family]
        data = maker(seed=seed)
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'
        actual = _fit_nampy_trace(data, formula, family, method)
        expected = _run_mgcv_trace(data, formula, family, method)
        a_log = np.log(np.asarray(actual["fit"]["smoothing_params"], dtype=np.float64))
        e_log = np.log(
            np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64)
        )
        np.testing.assert_allclose(a_log, e_log, atol=atol, rtol=0.0)
