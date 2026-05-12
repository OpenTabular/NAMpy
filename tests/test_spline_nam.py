import torch

from nampy.arch_utils.neural_splines import (
    CubicSplineLayer,
    TensorProductCubicSplineLayer,
)
from nampy.basemodels.spline_nam import SplineNAM
from nampy.configs.spline_nam_config import DefaultSplineNAMConfig
from nampy.models.spline_nam import (
    SplineNAMClassifier,
    SplineNAMLSS,
    SplineNAMRegressor,
)


def test_cubic_spline_layer_preserves_batch_and_output_shape():
    layer = CubicSplineLayer(
        n_bases=5,
        input_dim=3,
        output_dim=2,
        learn_knots=True,
    )

    output = layer(torch.rand(7, 3))

    assert output.shape == (7, 2)
    assert layer.get_smooth_penalty().shape == ()
    assert layer.get_knot_distance_penalty().shape == ()
    assert layer.get_knot_locations().shape == (5,)


def test_cubic_spline_layer_rejects_wrong_input_dimension():
    layer = CubicSplineLayer(n_bases=5, input_dim=2)

    try:
        layer(torch.rand(4, 3))
    except ValueError as exc:
        assert "expected input dimension 2" in str(exc)
    else:
        raise AssertionError("Expected a ValueError for mismatched input dimension.")


def test_spline_nam_outputs_terms_and_scalar_regularization():
    config = DefaultSplineNAMConfig(
        interaction_degree=2,
        smoothing=0.1,
        learn_knots=True,
        n_knots=5,
    )
    model = SplineNAM(
        cat_feature_info={"region": {"dimension": 2}},
        num_feature_info={"age": {"dimension": 1}, "income": {"dimension": 3}},
        num_classes=2,
        config=config,
    )

    result = model(
        num_features={
            "age": torch.rand(4, 1),
            "income": torch.rand(4, 3),
        },
        cat_features={"region": torch.rand(4, 2)},
    )

    assert result["prediction"].shape == (4, 2)
    assert result["terms"]["age"].shape == (4, 2)
    assert result["terms"]["income"].shape == (4, 2)
    assert result["terms"]["region"].shape == (4, 2)
    assert result["terms"]["age:income"].shape == (4, 2)
    assert result["regularization"]["spline"].shape == ()


def test_spline_nam_additive_reconstruction_in_eval_mode():
    config = DefaultSplineNAMConfig(n_knots=5)
    model = SplineNAM(
        cat_feature_info={},
        num_feature_info={"x1": {"dimension": 1}, "x2": {"dimension": 2}},
        num_classes=1,
        config=config,
    )
    model.eval()

    result = model(
        num_features={"x1": torch.rand(6, 1), "x2": torch.rand(6, 2)},
        cat_features={},
    )

    reconstructed = result["terms"]["x1"] + result["terms"]["x2"] + result["intercept"]
    assert torch.allclose(result["prediction"], reconstructed)


def test_spline_nam_allows_terms_named_like_old_top_level_keys():
    config = DefaultSplineNAMConfig(n_knots=5)
    model = SplineNAM(
        cat_feature_info={},
        num_feature_info={"output": {"dimension": 1}, "intercept": {"dimension": 1}},
        num_classes=1,
        config=config,
    )

    result = model(
        num_features={
            "output": torch.rand(3, 1),
            "intercept": torch.rand(3, 1),
        },
        cat_features={},
    )

    assert result["prediction"].shape == (3, 1)
    assert set(result["terms"]) == {"output", "intercept"}
    assert result["intercept"].shape == (1,)


def test_spline_nam_rejects_colon_feature_names():
    config = DefaultSplineNAMConfig(n_knots=5)

    try:
        SplineNAM(
            cat_feature_info={},
            num_feature_info={"x:y": {"dimension": 1}},
            config=config,
        )
    except ValueError as exc:
        assert "cannot contain ':'" in str(exc)
    else:
        raise AssertionError("Expected a ValueError for colon-containing feature names.")


def test_spline_nam_reports_knot_and_penalty_diagnostics():
    config = DefaultSplineNAMConfig(
        interaction_degree=2,
        smoothing=0.01,
        knot_distance_penalty=0.02,
        learn_knots=True,
        n_knots=5,
    )
    model = SplineNAM(
        cat_feature_info={},
        num_feature_info={"x1": {"dimension": 1}, "x2": {"dimension": 1}},
        num_classes=1,
        config=config,
    )

    diagnostics = model.get_spline_diagnostics()

    assert diagnostics["n_knots"] == 5
    assert diagnostics["learn_knots"] is True
    assert diagnostics["terms"] == ["x1", "x2", "x1:x2"]
    assert diagnostics["interaction_layer_types"] == {"x1:x2": "tensor_product"}
    assert isinstance(
        model.interaction_networks["x1:x2"], TensorProductCubicSplineLayer
    )
    assert set(diagnostics["knot_locations"]) == {"x1", "x2", "x1:x2"}
    assert diagnostics["knot_locations"]["x1"].shape == (5,)
    assert set(diagnostics["penalties"]["x1"]) == {
        "smooth",
        "knot_distance",
        "weighted",
    }
    assert diagnostics["penalties"]["x1"]["weighted"] >= 0.0


def test_spline_nam_public_wrappers_are_importable():
    assert SplineNAMRegressor.__name__ == "SplineNAMRegressor"
    assert SplineNAMClassifier.__name__ == "SplineNAMClassifier"
    assert SplineNAMLSS.__name__ == "SplineNAMLSS"
