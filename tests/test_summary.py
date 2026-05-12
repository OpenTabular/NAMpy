from types import SimpleNamespace

import numpy as np
import torch

import nampy
from nampy.basemodels.spline_nam import SplineNAM
from nampy.configs.spline_nam_config import DefaultSplineNAMConfig
from nampy.models.spline_nam import SplineNAMRegressor


class DummyTaskModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.task_kind = "regression"
        self.output_dim = 1


def test_unfitted_estimator_summary_reports_config_without_error():
    estimator = SplineNAMRegressor(smoothing=0.1)

    info = estimator.summary(print_fn=None)

    assert info["fitted"] is False
    assert info["estimator"] == "SplineNAMRegressor"
    assert info["base_model"] == "SplineNAM"
    assert info["config"]["smoothing"] == 0.1


def test_summary_reports_terms_parameters_and_model_diagnostics():
    base_model = SplineNAM(
        cat_feature_info={},
        num_feature_info={"x1": {"dimension": 1}},
        config=DefaultSplineNAMConfig(n_knots=5, smoothing=0.1),
    )
    estimator = SplineNAMRegressor()
    estimator.model = DummyTaskModel(base_model)
    estimator.data_module = SimpleNamespace(
        num_feature_info={"x1": {"dimension": 1}},
        cat_feature_info={},
    )
    estimator.feature_names_in_ = np.asarray(["x1"], dtype=object)
    captured = []

    info = estimator.summary(print_fn=captured.append)

    assert info["fitted"] is True
    assert info["task"] == "regression"
    assert info["terms"]["numerical"] == ["x1"]
    assert info["parameters"]["trainable"] > 0
    assert "spline" in info["diagnostics"]
    assert "Spline:" in captured[0]


def test_top_level_summary_helpers_are_not_public_api():
    assert not hasattr(nampy, "summary")
    assert not hasattr(nampy, "diagnostics")
    assert not hasattr(nampy.utils, "summary")
    assert not hasattr(nampy.utils, "diagnostics")
