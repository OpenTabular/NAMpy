from __future__ import annotations

import numpy as np
import pytest

from nampy.gam._model_state import _term_blocks_seq
from nampy.gam.parity import build_optimizer_trace
from nampy.gam.parity.snapshots import _normalize_mgcv_term_label
from tests.families.test_general_family_mgcv_parity import _gaulss_two_smooth_data
from tests.mgcv_parity_utils import (
    _fit_nampy_model,
    _make_gamma_data,
    _make_gaussian_data,
    _make_negbin_data,
)

pytestmark = [pytest.mark.surface_output, pytest.mark.surface_regression]


def _assert_metadata_equal(actual, expected):
    assert set(actual) == set(expected)
    for key in expected:
        got = actual[key]
        want = expected[key]
        if isinstance(want, np.ndarray):
            np.testing.assert_array_equal(np.asarray(got), want)
            continue
        if isinstance(want, (list, tuple)) and any(
            isinstance(v, np.ndarray) for v in want
        ):
            assert len(got) == len(want)
            for got_i, want_i in zip(got, want):
                if isinstance(want_i, np.ndarray):
                    np.testing.assert_array_equal(np.asarray(got_i), want_i)
                else:
                    assert got_i == want_i
            continue
        assert got == want


def test_fit_result_public_schema_tracks_term_results_without_duplicate_owners():
    """
    Owner-contract coverage verifying that fit result public schema tracks term results
    without duplicate owners.
    """
    data = _make_gaussian_data(seed=417, n=160)
    formula = 'y ~ x0 + s(x1, bs="cr", k=8)'
    gam = _fit_nampy_model(data, formula, "gaussian", "REML")

    fit_result = gam.fit_result(include_covariances=True)
    payload = fit_result.to_dict(include_covariances=True)
    term_blocks = tuple(_term_blocks_seq(gam))

    assert set(payload) == {
        "family_name",
        "link_name",
        "criterion_name",
        "criterion_value",
        "coef_full",
        "intercept",
        "smoothing_params",
        "edf_total",
        "edf_by_term",
        "trace_H",
        "scale",
        "rss",
        "deviance",
        "side_condition_reports",
        "term_results",
        "metadata",
        "cov_bayes",
        "cov_freq",
    }
    assert payload["metadata"] == {
        "n_samples": int(gam.n_samples_),
        "n_coef": int(gam.compiled_model_.n_coef),
        "fit_intercept": bool(gam.fit_intercept),
    }
    assert len(payload["term_results"]) == len(term_blocks)

    for item, tb in zip(payload["term_results"], term_blocks):
        assert set(item) == {
            "label",
            "term_type",
            "basis_name",
            "coef_slice",
            "n_coef",
            "edf",
            "smoothing_indices",
            "smoothing_ids",
            "smoothing_values",
            "deleted_columns",
            "kept_columns",
            "metadata",
        }
        assert item["label"] == tb.label
        assert item["term_type"] == tb.term_type
        assert item["basis_name"] == tb.basis_name
        assert item["coef_slice"] == [int(tb.coef_slice.start), int(tb.coef_slice.stop)]
        assert item["n_coef"] == int(tb.coef_slice.stop - tb.coef_slice.start)
        assert item["smoothing_indices"] == [int(v) for v in tb.smoothing_indices]
        assert item["smoothing_ids"] == list(tb.smoothing_ids)
        _assert_metadata_equal(item["metadata"], dict(tb.metadata or {}))


def test_general_family_parity_snapshot_diagnostic_labels_follow_public_apis():
    """
    Owner-contract coverage verifying that general family parity snapshot diagnostic
    labels follow public apis.
    """
    data = _gaulss_two_smooth_data(seed=49)
    formula = ['y ~ s(x, bs="cr", k=6) + s(z, bs="cr", k=6)', "~ 1"]
    gam = _fit_nampy_model(data, formula, "gaulss", "ML")

    snapshot = gam.parity_snapshot(include_covariances=True)
    diagnostics = snapshot["parity"]["diagnostics"]
    concurvity_full = gam.concurvity(full=True)
    k_check = gam.k_check(subsample=120, n_rep=8, seed=0)
    anova = gam.anova(freq=False)
    gam_vcomp = gam.gam_vcomp(rescale=False)

    assert set(snapshot) == {"fit", "predictions", "parity"}
    assert set(snapshot["predictions"]) == {
        "response",
        "link",
        "terms",
        "lpmatrix",
        "se_response",
        "se_link",
    }
    assert diagnostics["concurvity_labels"] == [
        _normalize_mgcv_term_label(v) for v in concurvity_full["labels"]
    ]
    assert diagnostics["k_check"]["labels"] == [
        _normalize_mgcv_term_label(v) for v in k_check.index.tolist()
    ]
    assert diagnostics["anova_smooth"]["labels"] == [
        _normalize_mgcv_term_label(v) for v in anova.smooth_table["label"].tolist()
    ]
    assert diagnostics["gam_vcomp_names"] == list(gam_vcomp.get("names", []))
    np.testing.assert_allclose(
        np.asarray(diagnostics["gam_vcomp"], dtype=np.float64),
        np.asarray(gam_vcomp["vc"], dtype=np.float64),
        atol=1e-12,
        rtol=0.0,
    )


def test_parity_snapshot_prediction_arrays_track_public_prediction_surfaces():
    """
    Owner-contract coverage verifying that parity snapshot prediction arrays track
    public prediction surfaces.
    """
    data = _gaulss_two_smooth_data(seed=53)
    formula = ['y ~ s(x, bs="cr", k=6) + s(z, bs="cr", k=6)', "~ 1"]
    gam = _fit_nampy_model(data, formula, "gaulss", "ML")

    snapshot = gam.parity_snapshot(X=data, include_covariances=True)
    expected_shapes = {
        "response": np.asarray(gam.predict(data, type="response"), dtype=np.float64),
        "link": np.asarray(gam.predict(data, type="link"), dtype=np.float64),
        "terms": np.asarray(gam.predict(data, type="terms"), dtype=np.float64),
        "lpmatrix": np.asarray(gam.predict(data, type="lpmatrix"), dtype=np.float64),
    }

    for key, value in expected_shapes.items():
        snap_arr = np.asarray(snapshot["predictions"][key], dtype=np.float64)
        expected = value.reshape(-1, order="F") if value.ndim == 2 else value
        np.testing.assert_allclose(snap_arr, expected, atol=1e-12, rtol=0.0)


def test_general_family_snapshot_schema_includes_smooth_covariance_and_function_space_blocks():
    """
    Owner-contract coverage verifying that general family snapshot schema includes
    smooth covariance and function space blocks.
    """
    data = _gaulss_two_smooth_data(seed=57)
    formula = ['y ~ s(x, bs="cr", k=6) + s(z, bs="cr", k=6)', "~ 1"]
    gam = _fit_nampy_model(data, formula, "gaulss", "ML")

    snapshot = gam.parity_snapshot(X=data, include_covariances=True)
    diagnostics = snapshot["parity"]["diagnostics"]

    for key in ("smooth_cov_bayes", "smooth_test_inputs", "smooth_function_space"):
        assert diagnostics[key] is not None
    assert (
        diagnostics["smooth_cov_bayes"]["labels"]
        == diagnostics["smooth_function_space"]["labels"]
    )
    assert (
        diagnostics["smooth_test_inputs"]["labels"]
        == diagnostics["smooth_function_space"]["labels"]
    )


def test_general_family_snapshot_term_order_preserves_predictor_partition_mapping():
    """
    Owner-contract coverage verifying that general family snapshot term order preserves
    predictor partition mapping.
    """
    import pandas as pd

    rng = np.random.default_rng(61)
    n = 120
    x = np.linspace(-1.0, 1.0, n)
    z = rng.uniform(-1.2, 1.2, size=n)
    mu = 0.25 + 0.3 * x - 0.2 * np.cos(1.3 * z)
    sigma = np.exp(-0.35 + 0.15 * x + 0.1 * z)
    y = rng.normal(mu, sigma, size=n)
    data = pd.DataFrame({"y": y, "x": x, "z": z})
    formula = [
        'y ~ x + s(z, bs="cr", k=6)',
        '~ z + s(x, bs="cr", k=6)',
    ]
    gam = _fit_nampy_model(data, formula, "gaulss", "ML")

    snapshot = gam.parity_snapshot(X=data, include_covariances=True)
    term_labels = [item["label"] for item in snapshot["fit"]["term_results"]]
    expected_term_labels = [tb.label for tb in _term_blocks_seq(gam)]
    assert term_labels == expected_term_labels
    assert snapshot["parity"]["diagnostics"]["smooth_function_space"]["labels"] == [
        _normalize_mgcv_term_label(tb.label)
        for tb in _term_blocks_seq(gam)
        if str(getattr(tb, "term_type", "")) != "parametric"
    ]


def test_optimizer_trace_schema_preserves_internal_joint_gamma_rows():
    """
    Owner-contract coverage verifying that optimizer trace schema preserves internal
    joint gamma rows.
    """
    data = _make_gamma_data(seed=123, n=180)
    gam = _fit_nampy_model(data, 'y ~ s(x0, bs="cr", k=8)', "gamma", "REML")

    trace = build_optimizer_trace(gam)
    internal_rows = list(getattr(gam, "_optim_trace", []) or [])

    assert set(trace) == {"fit", "trace"}
    assert trace["fit"]["criterion_name"] == gam._optim_method
    np.testing.assert_allclose(
        np.asarray(trace["fit"]["smoothing_params"], dtype=np.float64),
        np.asarray(gam.smoothing_params, dtype=np.float64),
        atol=0.0,
        rtol=0.0,
    )
    assert trace["fit"]["message"] == str(getattr(gam._optim_result, "message", ""))
    outer_info = dict(getattr(gam._optim_result, "outer_info", {}) or {})
    assert trace["fit"]["outer_info"]["conv"] == outer_info["conv"]
    np.testing.assert_allclose(
        np.asarray(trace["fit"]["outer_info"]["score_hist"], dtype=np.float64),
        np.asarray(outer_info["score_hist"], dtype=np.float64),
        atol=0.0,
        rtol=0.0,
    )
    assert len(trace["trace"]) == len(internal_rows)
    for serialized, internal in zip(trace["trace"], internal_rows):
        assert set(serialized) >= {
            "iter",
            "log_sp",
            "log_scale",
            "log_theta",
            "criterion",
            "gradient",
            "gradient_full",
            "hessian",
            "hessian_full",
            "accepted_step_norm",
            "n_fun",
            "n_jac",
            "n_hess",
            "rank_info",
        }
        assert serialized["iter"] == int(internal["iter"])
        np.testing.assert_allclose(
            np.asarray(serialized["log_sp"], dtype=np.float64),
            np.asarray(internal["log_sp"], dtype=np.float64),
            atol=0.0,
            rtol=0.0,
        )
        internal_log_scale = internal.get("log_scale", None)
        if internal_log_scale is None:
            assert serialized["log_scale"] is None
        else:
            np.testing.assert_allclose(
                float(serialized["log_scale"]),
                float(internal_log_scale),
                atol=0.0,
                rtol=0.0,
            )
        assert serialized["log_theta"] is None
        expected_gradient_full = internal.get("gradient_full", internal.get("gradient"))
        expected_hessian_full = internal.get("hessian_full", internal.get("hessian"))
        np.testing.assert_allclose(
            np.asarray(serialized["gradient_full"], dtype=np.float64),
            np.asarray(expected_gradient_full, dtype=np.float64),
            atol=0.0,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            np.asarray(serialized["hessian_full"], dtype=np.float64),
            np.asarray(expected_hessian_full, dtype=np.float64),
            atol=0.0,
            rtol=0.0,
        )
        assert serialized["rank_info"] == internal["rank_info"]


def test_optimizer_trace_schema_preserves_internal_joint_negbin_rows():
    """
    Owner-contract coverage verifying that optimizer trace schema preserves internal
    joint negative-binomial rows.
    """
    data = _make_negbin_data(seed=123, n=180)
    gam = _fit_nampy_model(
        data,
        'y ~ s(x0, bs="cr", k=8)',
        {"name": "negbin", "theta": 1.8, "estimate_theta": True},
        "REML",
    )

    trace = build_optimizer_trace(gam)
    internal_rows = list(getattr(gam, "_optim_trace", []) or [])

    assert len(trace["trace"]) == len(internal_rows)
    for serialized, internal in zip(trace["trace"], internal_rows):
        assert set(serialized) >= {
            "iter",
            "log_sp",
            "log_scale",
            "log_theta",
            "criterion",
            "gradient",
            "gradient_full",
            "hessian",
            "hessian_full",
            "accepted_step_norm",
            "n_fun",
            "n_jac",
            "n_hess",
            "rank_info",
        }
        assert serialized["iter"] == int(internal["iter"])
        assert serialized["log_scale"] is None
        np.testing.assert_allclose(
            float(serialized["log_theta"]),
            float(internal["log_theta"]),
            atol=0.0,
            rtol=0.0,
        )
        expected_gradient_full = internal.get("gradient_full", internal.get("gradient"))
        expected_hessian_full = internal.get("hessian_full", internal.get("hessian"))
        if expected_gradient_full is None:
            assert serialized["gradient_full"] is None
        else:
            np.testing.assert_allclose(
                np.asarray(serialized["gradient_full"], dtype=np.float64),
                np.asarray(expected_gradient_full, dtype=np.float64),
                atol=0.0,
                rtol=0.0,
            )
        if expected_hessian_full is None:
            assert serialized["hessian_full"] is None
        else:
            np.testing.assert_allclose(
                np.asarray(serialized["hessian_full"], dtype=np.float64),
                np.asarray(expected_hessian_full, dtype=np.float64),
                atol=0.0,
                rtol=0.0,
            )
