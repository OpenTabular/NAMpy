from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from nampy.gam.parity import load_optimizer_trace, save_optimizer_trace
from tests.optimization._trace_parity_helpers import (
    _assert_mgcv_score_hist_exact,
    _criterion_series,
    _fit_nampy_model_and_trace,
    _fit_nampy_trace,
    _make_binomial_data,
    _make_gaussian_data,
    _make_poisson_data,
    _run_mgcv_trace,
    _tail_criterion_series,
)


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


def _assert_gaussian_final_trace_row_matches_mgcv(actual, expected):
    final_row = actual["trace"][-1]
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
        np.log(np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64)),
        np.asarray(final_row["log_sp"], dtype=np.float64),
        atol=0.75,
        rtol=0.0,
    )


class TestMgcvScoreHistTraceParity:
    """
    Optimizer-trace parity checks for serialized traces, endpoints, and per-iteration
    score histories.
    """

    def test_trace_io_roundtrip_preserves_schema(self):
        """Verify that trace I/O roundtrip preserves schema."""
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

    def test_gaussian_reml_trace_matches_mgcv_endpoint(self):
        """Verify that gaussian REML trace matches mgcv endpoint."""
        data = _make_gaussian_data(seed=321)
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'
        actual = _fit_nampy_trace(data, formula, "gaussian", "REML")
        expected = _run_mgcv_trace(data, formula, "gaussian", "REML")
        outer = expected["fit"].get("outer_info", {})
        assert isinstance(outer.get("conv", None), str)
        assert int(outer.get("iter", 0)) >= 1
        score_hist = outer.get("score_hist", None)
        assert isinstance(score_hist, list)
        assert len(score_hist) == int(outer["iter"])
        _assert_gaussian_outer_trace_parity(actual, expected)
        _assert_gaussian_final_trace_row_matches_mgcv(actual, expected)

    def test_gaussian_reml_newton_score_hist_matches_exactly(self):
        """Verify that gaussian REML newton score hist matches exactly."""
        data = _make_gaussian_data(seed=321)
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'
        model, _ = _fit_nampy_model_and_trace(data, formula, "gaussian", "REML")
        expected = _run_mgcv_trace(data, formula, "gaussian", "REML")

        _assert_mgcv_score_hist_exact(model, expected)

    @pytest.mark.parametrize(
        "family,seed",
        [
            ("binomial", 456),
            ("poisson", 789),
        ],
    )
    def test_non_gaussian_newton_score_hist_matches_exactly(self, family, seed):
        """Verify that non gaussian newton score hist matches exactly."""
        maker = {
            "binomial": _make_binomial_data,
            "poisson": _make_poisson_data,
        }[family]
        data = maker(seed=seed)
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'
        model, _ = _fit_nampy_model_and_trace(data, formula, family, "REML")
        expected = _run_mgcv_trace(data, formula, family, "REML")

        _assert_mgcv_score_hist_exact(model, expected, atol=1e-6)


@pytest.mark.parametrize(
    "family, method, seed, atol",
    [
        ("gaussian", "REML", 321, 0.75),
        ("gaussian", "ML", 99, 0.8),
        ("binomial", "REML", 456, 1.05),
        ("poisson", "REML", 789, 1.0),
    ],
)
def test_endpoint_log_sp_seed_matrix(family, method, seed, atol):
    """Verify that endpoint log smoothing parameters stay close across seed matrix."""
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
    e_log = np.log(np.asarray(expected["fit"]["smoothing_params"], dtype=np.float64))
    np.testing.assert_allclose(a_log, e_log, atol=atol, rtol=0.0)
