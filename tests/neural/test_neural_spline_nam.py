from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from nampy.models.spline_nam import SplineNAMRegressor
from nampy.neural.configs.spline_nam_config import DefaultSplineNAMConfig
from nampy.neural.modules.neural_splines import CubicSplineLayer
from nampy.neural.modules.spline_nam import SplineNAM


def test_cubic_spline_basis_handles_both_boundary_knots():
    layer = CubicSplineLayer(n_bases=5, min_val=0.0, max_val=1.0)

    basis = layer.apply_spline_basis(
        torch.tensor([0.0, 1.0]),
        layer.knots,
        layer.F,
    )

    assert torch.allclose(basis[0], torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]))
    assert torch.allclose(basis[1], torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]))


def test_cubic_spline_learnable_knots_receive_gradients_across_batches():
    layer = CubicSplineLayer(
        n_bases=5,
        min_val=0.0,
        max_val=1.0,
        learn_knots=True,
    )

    for values in (
        torch.tensor([[0.1], [0.4], [0.8]]),
        torch.tensor([[0.2], [0.5], [0.9]]),
    ):
        layer.zero_grad(set_to_none=True)
        layer(values).square().mean().backward()
        assert layer.relative_distances.grad is not None
        assert torch.isfinite(layer.relative_distances.grad).all()


def test_spline_nam_interactions_preserve_batch_and_additive_contract():
    config = DefaultSplineNAMConfig(
        interaction_degree=2,
        n_knots=6,
        smoothing=0.25,
    )
    model = SplineNAM(
        num_feature_info={
            "x": {"dimension": 1, "preprocessing": "minmax"},
            "z": {"dimension": 1, "preprocessing": "minmax"},
        },
        cat_feature_info={
            "group": {
                "dimension": 1,
                "categories": 3,
                "preprocessing": "continuous_ordinal",
            }
        },
        config=config,
    )
    model.eval()
    num_features = {
        "x": torch.tensor([[0.0], [0.25], [0.75], [1.0]]),
        "z": torch.tensor([[1.0], [0.75], [0.25], [0.0]]),
    }
    cat_features = {"group": torch.tensor([[0], [1], [2], [1]])}

    result = model(num_features=num_features, cat_features=cat_features)

    term_names = ("x", "z", "group", "x:z", "x:group", "z:group")
    for name in term_names:
        assert result[name].shape == (4, 1)
    expected = torch.stack([result[name] for name in term_names], dim=1).sum(dim=1)
    expected = expected + result["intercept"]
    assert torch.allclose(result["output"], expected)
    assert result["smoothness_penalty"].ndim == 0
    assert result["smoothness_penalty"] >= 0.0

    (result["output"].square().mean() + result["smoothness_penalty"]).backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


@pytest.mark.parametrize("feature_name", ["output", "intercept", "bad:name"])
def test_spline_nam_rejects_feature_names_that_break_result_keys(feature_name):
    with pytest.raises(ValueError, match="reserved"):
        SplineNAM(
            num_feature_info={feature_name: {"dimension": 1}},
            cat_feature_info={},
        )


def test_spline_nam_regressor_fits_and_predicts_with_default_preprocessing(tmp_path):
    x = np.linspace(-1.0, 1.0, 36)
    data = pd.DataFrame(
        {
            "x": x,
            "z": np.cos(x),
            "group": np.resize(np.array(["a", "b", "c"]), x.size),
        }
    )
    y = np.sin(2.0 * x) + 0.2 * (data["group"] == "b").to_numpy(dtype=float)
    model = SplineNAMRegressor(
        n_knots=6,
        interaction_degree=2,
        smoothing=1e-4,
    )

    fitted = model.fit(
        data,
        y,
        max_epochs=1,
        batch_size=12,
        checkpoint_path=tmp_path,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    predictions = fitted.predict(data)
    terms = fitted.predict_feature_vals(data)

    assert fitted is model
    assert model.config.n_knots == 6
    assert model.preprocessor.numerical_preprocessing == "minmax"
    assert predictions.shape == (len(data),)
    assert np.isfinite(predictions).all()
    assert terms["x:group"].shape == (len(data), 1)
