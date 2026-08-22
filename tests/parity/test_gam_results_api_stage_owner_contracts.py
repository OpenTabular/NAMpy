from __future__ import annotations

from dataclasses import fields, replace

import numpy as np
import pytest

from nampy.gam.model_state import _term_blocks_seq
from nampy.gam.parity import build_optimizer_trace
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
            for got_i, want_i in zip(got, want, strict=True):
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
        "edf2",
        "side_condition_reports",
        "term_results",
        "metadata",
        "cov_bayes",
        "cov_freq",
        "cov_unconditional",
        "cov_unconditional_space",
    }
    # ML/REML newton fits carry the smoothing-uncertainty state, so the
    # unconditional covariance and edf2 must survive into the public schema.
    assert payload["cov_unconditional"] is not None
    assert payload["edf2"] is not None
    assert np.asarray(gam.edf1(), dtype=np.float64).shape == (
        len(payload["coef_full"]),
    )
    assert payload["metadata"] == {
        "n_samples": int(gam.n_samples_),
        "n_coef": int(gam.gam_result_.require_compiled_model().n_coef),
        "fit_intercept": bool(gam.fit_intercept),
    }
    assert len(payload["term_results"]) == len(term_blocks)

    for item, tb in zip(payload["term_results"], term_blocks, strict=True):
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


def test_fit_result_is_a_fully_defensive_public_snapshot():
    """Nested public result mutation must not alter fitted model state."""
    data = _make_gaussian_data(seed=418, n=120)
    gam = _fit_nampy_model(
        data,
        'y ~ x0 + s(x1, bs="cr", k=7)',
        "gaussian",
        "REML",
    )
    internal = gam.gam_result_.require_fit_summary()
    internal.term_results[0].metadata["deep_copy_probe"] = {
        "values": [1, np.asarray([2.0, 3.0])]
    }
    internal.metadata["deep_copy_probe"] = {"values": [4, 5]}
    assert internal.side_condition_reports
    internal.side_condition_reports[0]["deep_copy_probe"] = {"values": [6, 7]}
    traced_core = replace(
        internal.core,
        inner_trace=[{"values": [8, np.asarray([9.0, 10.0])]}],
    )
    internal = replace(internal, core=traced_core)
    gam.gam_result_ = replace(gam.gam_result_, fit_summary=internal)
    prediction_before = gam.predict(data)

    public = gam.fit_result(include_covariances=True)
    for field in fields(public.core):
        public_value = getattr(public.core, field.name)
        internal_value = getattr(internal.core, field.name)
        if isinstance(public_value, np.ndarray):
            assert not np.shares_memory(public_value, internal_value)
            public_value[...] = 0
    public.smoothing_params[...] = 0
    public.core.inner_trace[0]["values"][1][...] = 0
    public.term_results[0].metadata["deep_copy_probe"]["values"][1][...] = 0
    public.metadata["deep_copy_probe"]["values"].append(99)
    public.side_condition_reports[0]["deep_copy_probe"]["values"].append(99)

    np.testing.assert_array_equal(
        internal.core.inner_trace[0]["values"][1],
        np.asarray([9.0, 10.0]),
    )
    np.testing.assert_array_equal(
        internal.term_results[0].metadata["deep_copy_probe"]["values"][1],
        np.asarray([2.0, 3.0]),
    )
    assert internal.metadata["deep_copy_probe"]["values"] == [4, 5]
    assert internal.side_condition_reports[0]["deep_copy_probe"]["values"] == [6, 7]
    np.testing.assert_array_equal(gam.predict(data), prediction_before)

    second = gam.fit_result(include_covariances=True)
    np.testing.assert_array_equal(second.core.coef_full, internal.core.coef_full)
    np.testing.assert_array_equal(second.smoothing_params, internal.smoothing_params)


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
    for serialized, internal in zip(trace["trace"], internal_rows, strict=True):
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
    for serialized, internal in zip(trace["trace"], internal_rows, strict=True):
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


def test_fit_result_fields_carry_owner_values_not_just_keys():
    """Value-level contracts for the fields that were only key-asserted.

    family/link identity, the unconditional-covariance space tag, side
    condition reports, per-term edf attribution, per-term smoothing values,
    kept/deleted column bookkeeping, and the public edf1() vector all have
    canonical owners; the payload must reproduce them, not merely carry keys.
    """
    data = _make_gaussian_data(seed=417, n=160)
    formula = 'y ~ x0 + s(x1, bs="cr", k=8)'
    gam = _fit_nampy_model(data, formula, "gaussian", "REML")

    payload = gam.fit_result(include_covariances=True).to_dict(
        include_covariances=True
    )
    term_blocks = tuple(_term_blocks_seq(gam))

    assert payload["family_name"] == gam.family.name
    assert payload["link_name"] == gam.family.link_name
    assert payload["cov_unconditional_space"] in {"fit", "prediction"}
    assert payload["side_condition_reports"] == (
        list(gam.gam_result_.require_compiled_model().side_condition_reports or ())
    )

    edfs = [float(item["edf"]) for item in payload["term_results"]]
    np.testing.assert_allclose(
        edfs, np.asarray(payload["edf_by_term"], dtype=np.float64), atol=1e-12
    )
    intercept_edf = 1.0 if payload["intercept"] is not None else 0.0
    np.testing.assert_allclose(
        float(np.sum(edfs)) + intercept_edf,
        float(payload["edf_total"]),
        atol=1e-8,
    )

    sp = np.asarray(payload["smoothing_params"], dtype=np.float64)
    report_by_label = {
        str(tr["label"]): tr
        for rep in payload["side_condition_reports"]
        for tr in rep["term_reports"]
    }
    for item, tb in zip(payload["term_results"], term_blocks, strict=True):
        np.testing.assert_allclose(
            np.asarray(item["smoothing_values"], dtype=np.float64),
            sp[[int(v) for v in tb.smoothing_indices]],
            atol=0.0,
            rtol=0.0,
        )
        report = report_by_label.get(str(item["label"]))
        assert report is not None, item["label"]
        assert item["kept_columns"] == report["kept_columns"]
        assert item["deleted_columns"] == report["deleted_columns"]

    from nampy.gam.inference.anova import _edf1_vector

    np.testing.assert_allclose(
        np.asarray(gam.edf1(), dtype=np.float64),
        np.asarray(_edf1_vector(gam), dtype=np.float64),
        atol=0.0,
        rtol=0.0,
    )
    # edf1 upper-bounds edf coefficientwise in mgcv's construction; its sum
    # must be at least the fitted total.
    assert float(np.sum(np.asarray(gam.edf1()))) >= float(payload["edf_total"]) - 1e-8
