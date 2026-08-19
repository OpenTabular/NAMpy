from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from nampy.gam.parity import snapshots as snapshots_module
from nampy.gam.parity.snapshots import _build_parity_criterion_view
from nampy.gam.parity.trace import build_optimizer_trace

pytestmark = [
    pytest.mark.surface_output,
    pytest.mark.surface_regression,
]


def test_build_optimizer_trace_serializes_core_rows_and_optimizer_metadata():
    """
    Owner-contract coverage verifying that build optimizer trace serializes core rows
    and optimizer metadata.
    """
    core = SimpleNamespace(
        _optim_trace=[
            {
                "iter": 3,
                "log_sp": np.array([0.1, -0.2], dtype=np.float64),
                "log_scale": -0.7,
                "criterion": 1.25,
                "gradient": np.array([4.0, 5.0], dtype=np.float64),
                "gradient_full": np.array([4.0, 5.0, 9.0], dtype=np.float64),
                "hessian": np.array([[6.0, 0.0], [0.0, 7.0]], dtype=np.float64),
                "hessian_full": np.array(
                    [[6.0, 0.0, 1.0], [0.0, 7.0, 2.0], [1.0, 2.0, 8.0]],
                    dtype=np.float64,
                ),
                "accepted_step_norm": 0.4,
                "n_fun": 11,
                "n_jac": 7,
                "n_hess": 3,
                "rank_info": {"step_halving_count": 1},
            }
        ],
        _optim_result=SimpleNamespace(
            success=True,
            message="ok",
            nit=4,
            edge_correction_requested=True,
            edge_correction_applied=False,
            outer_info={
                "conv": "ok",
                "iter": 4,
                "score_hist": np.array([1.8, 1.4, 1.25], dtype=np.float64),
                "counts": np.array([11, 7], dtype=np.int64),
            },
        ),
        _optim_method="REML",
        smoothing_params=np.array([1.5, 0.5], dtype=np.float64),
    )
    model = SimpleNamespace(core_=core)

    trace = build_optimizer_trace(model)

    assert trace["fit"]["criterion_name"] == "REML"
    assert trace["fit"]["converged"] is True
    assert trace["fit"]["message"] == "ok"
    assert trace["fit"]["optimizer_nit"] == 4
    assert trace["fit"]["edge_correct"] is True
    assert trace["fit"]["edge_correct_applied"] is False
    assert trace["fit"]["outer_info"]["conv"] == "ok"
    assert trace["fit"]["outer_info"]["counts"] == [11, 7]
    assert trace["fit"]["outer_info"]["db_drho1"] is None
    assert trace["fit"]["outer_info"]["dw_drho1"] is None
    assert trace["fit"]["smoothing_params"] == [1.5, 0.5]
    assert trace["trace"][0]["iter"] == 3
    assert trace["trace"][0]["log_sp"] == [0.1, -0.2]
    assert trace["trace"][0]["gradient"] == [4.0, 5.0]
    assert trace["trace"][0]["gradient_full"] == [4.0, 5.0, 9.0]
    assert trace["trace"][0]["hessian"] == [[6.0, 0.0], [0.0, 7.0]]
    assert trace["trace"][0]["hessian_full"] == [
        [6.0, 0.0, 1.0],
        [0.0, 7.0, 2.0],
        [1.0, 2.0, 8.0],
    ]
    assert trace["trace"][0]["n_fun"] == 11
    assert trace["trace"][0]["n_jac"] == 7
    assert trace["trace"][0]["n_hess"] == 3
    assert trace["trace"][0]["rank_info"] == {"step_halving_count": 1}


def test_build_parity_criterion_view_prefers_smoothing_score_source(monkeypatch):
    """
    Owner-contract coverage verifying that build parity criterion view prefers smoothing
    score source.
    """
    monkeypatch.setattr(snapshots_module, "_n_smoothing_params", lambda core: 1)
    monkeypatch.setattr(
        snapshots_module,
        "resolve_ml_reml_scoring_backend",
        lambda core, method="reml": "gaussian_dynamic",
    )
    monkeypatch.setattr(
        snapshots_module,
        "criterion_ml_reml",
        lambda core, y, log_sp, method: 2.25,
    )
    monkeypatch.setattr(
        snapshots_module,
        "criterion_ml_reml_gaussian_dynamic_joint",
        lambda core, y, log_sp, log_sigma2, method: 3.5,
    )

    core = SimpleNamespace(
        family=SimpleNamespace(name="gaussian"),
        smoothing_params=np.array([2.0], dtype=np.float64),
        smoothing_fixed_mask_=None,
        y_=np.array([1.0], dtype=np.float64),
        smoothing_score_=1.75,
        _gaussian_reml_sigma2_opt_=0.5,
    )
    fit_dict = {
        "criterion_name": "REML",
        "criterion_value": 1.1,
        "family_name": "gaussian",
    }

    view = _build_parity_criterion_view(core, fit_dict)

    assert view["criterion_backend"] == "gaussian_dynamic"
    assert view["stored_criterion_value"] == pytest.approx(1.1, abs=0.0)
    assert view["profiled_criterion_value"] == pytest.approx(2.25, abs=0.0)
    assert view["joint_criterion_value"] == pytest.approx(3.5, abs=0.0)
    assert view["recomputed_criterion_value"] == pytest.approx(1.75, abs=0.0)
    assert view["recomputed_criterion_source"] == "smoothing_score"
    assert view["joint_log_sigma2"] == pytest.approx(np.log(0.5), abs=0.0)


def test_build_parity_criterion_view_recomputes_from_joint_when_score_missing(
    monkeypatch,
):
    """
    Owner-contract coverage verifying that build parity criterion view recomputes from
    joint when score missing.
    """
    monkeypatch.setattr(snapshots_module, "_n_smoothing_params", lambda core: 1)
    monkeypatch.setattr(
        snapshots_module,
        "resolve_ml_reml_scoring_backend",
        lambda core, method="reml": "gaussian_dynamic",
    )
    monkeypatch.setattr(
        snapshots_module,
        "criterion_ml_reml",
        lambda core, y, log_sp, method: 2.0,
    )
    monkeypatch.setattr(
        snapshots_module,
        "criterion_ml_reml_gaussian_dynamic_joint",
        lambda core, y, log_sp, log_sigma2, method: 4.5,
    )

    core = SimpleNamespace(
        family=SimpleNamespace(name="gaussian"),
        smoothing_params=np.array([1.5], dtype=np.float64),
        smoothing_fixed_mask_=None,
        y_=np.array([1.0], dtype=np.float64),
        smoothing_score_=None,
        _gaussian_reml_sigma2_opt_=0.25,
    )
    fit_dict = {
        "criterion_name": "REML",
        "criterion_value": 0.9,
        "family_name": "gaussian",
    }

    view = _build_parity_criterion_view(core, fit_dict)

    assert view["profiled_criterion_value"] == pytest.approx(2.0, abs=0.0)
    assert view["joint_criterion_value"] == pytest.approx(4.5, abs=0.0)
    assert view["recomputed_criterion_value"] == pytest.approx(4.5, abs=0.0)
    assert view["recomputed_criterion_source"] == "gaussian_dynamic_joint"
