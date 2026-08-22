"""Focused contracts from the GP-NAM paper and released implementation."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch

from nampy.models.gpnam import GPNAMRegressor
from nampy.neural.architectures.gpnam import GPNAM
from nampy.neural.configs.gpnam_config import DefaultGPNAMConfig


def _architecture(config, n_features=2, n_outputs=1):
    info = {
        f"x{index}": {"dimension": 1, "preprocessing": "noop"}
        for index in range(n_features)
    }
    return GPNAM(
        cat_feature_info={},
        num_feature_info=info,
        num_classes=n_outputs,
        config=config,
    )


def test_gpnam_estimator_owns_reference_input_defaults():
    estimator = GPNAMRegressor()
    params = estimator.get_params(deep=False)
    assert params["numerical_method"] == "none"
    assert params["categorical_method"] == "one-hot"
    assert params["scaling"] is None

    explicit = GPNAMRegressor(
        numerical_method="standardization", scaling="minmax"
    )
    assert explicit.get_params(deep=False)["numerical_method"] == (
        "standardization"
    )
    assert explicit.get_params(deep=False)["scaling"] == "minmax"


def test_quasi_random_rff_uses_inverse_normal_and_per_feature_phase_grids():
    model = _architecture(
        DefaultGPNAMConfig(
            kernel_width=0.2,
            rff_num_feat=16,
            rff_scheme="quasi_random",
            rff_random_state=7,
        ),
        n_features=3,
    )
    probabilities = torch.arange(1, 17, dtype=torch.float32) / 17
    expected_z = torch.distributions.Normal(0.0, 1.0).icdf(probabilities)
    expected_phases = 2 * math.pi * torch.arange(16) / 16

    torch.testing.assert_close(model.z, expected_z)
    assert model.c.shape == (3, 16)
    for phases in model.c:
        torch.testing.assert_close(torch.sort(phases).values, expected_phases)
    assert not torch.equal(model.c[0], model.c[1])


def test_automatic_kernel_widths_match_training_sample_std_over_24():
    model = _architecture(
        DefaultGPNAMConfig(kernel_width="auto", rff_num_feat=8), n_features=2
    )
    features = {
        "x0": torch.tensor([[0.0], [1.0], [2.0], [3.0]]),
        "x1": torch.tensor([[1.0], [3.0], [5.0], [7.0]]),
    }
    model.initialize_from_training_data(features, {})
    expected = torch.std(torch.cat(list(features.values()), dim=1), dim=0) / 24
    torch.testing.assert_close(model.kernel_widths, expected)


def test_automatic_kernel_widths_reject_constant_architecture_inputs():
    model = _architecture(
        DefaultGPNAMConfig(kernel_width="auto", rff_num_feat=8), n_features=1
    )
    with pytest.raises(ValueError, match="non-constant"):
        model.initialize_from_training_data(
            {"x0": torch.ones(4, 1)}, {}
        )


def test_gpna2m_design_and_forward_include_only_requested_interactions():
    model = _architecture(
        DefaultGPNAMConfig(
            kernel_width=0.5,
            rff_num_feat=6,
            rff_random_state=3,
            interactions=(("x0", "x2"),),
        ),
        n_features=3,
    )
    features = {
        "x0": torch.tensor([[0.0], [0.5], [1.0]]),
        "x1": torch.tensor([[0.2], [0.4], [0.6]]),
        "x2": torch.tensor([[1.0], [0.0], [-1.0]]),
    }
    with torch.no_grad():
        model.weights.normal_()
        model.interaction_weights.normal_()
        model.intercept.fill_(0.25)
    result = model(features, {})
    design = model.linear_design(features, {})

    assert model.interaction_names == ["x0:x2"]
    assert design.shape == (3, 4 * 6)
    assert "x0:x2" in result
    reconstruction = result["intercept"]
    for name in (*model.atomic_feature_names, *model.interaction_names):
        reconstruction = reconstruction + result[name]
    torch.testing.assert_close(result["output"], reconstruction)


def test_cg_regression_matches_weighted_ridge_normal_equations():
    x = np.linspace(-1.0, 1.0, 30)
    X = pd.DataFrame({"x": x, "z": np.cos(2 * np.pi * x)})
    offset = np.linspace(-0.2, 0.2, len(X))
    y = np.column_stack(
        [np.sin(2 * np.pi * x) + offset, x**2 - 0.5 * offset]
    )
    weights = np.linspace(0.5, 2.0, len(X))
    ridge = 0.05
    estimator = GPNAMRegressor(
        kernel_width=0.4,
        rff_num_feat=12,
        rff_random_state=9,
        ridge=ridge,
    )
    estimator.fit(
        X,
        y,
        offset=offset,
        sample_weight=weights,
        random_state=4,
    )

    basis = estimator.basis_transform(X, batch_size=7)
    design = np.column_stack([basis, np.ones(len(X))])
    target = y - offset[:, None]
    weighted_design = design * np.sqrt(weights)[:, None]
    weighted_target = target * np.sqrt(weights)[:, None]
    normal = weighted_design.T @ weighted_design
    normal[np.diag_indices_from(normal)] += np.r_[
        np.full(basis.shape[1], ridge), 0.0
    ]
    expected = np.linalg.solve(normal, weighted_design.T @ weighted_target)

    np.testing.assert_allclose(
        estimator.predict(X, batch_size=5),
        design @ expected,
        atol=2e-5,
        rtol=2e-5,
    )
    assert estimator.linear_solver_info_["n_rows"] == len(X)
    assert estimator.linear_solver_info_["solver"] == "cg"
    assert estimator.kernel_widths_.shape == (2,)
    assert estimator.model_complexity()["trainable_parameters"] == 2 * 12 * 2 + 2


def test_gpnam_rff_seed_and_batched_predictions_are_reproducible():
    X = pd.DataFrame(
        {
            "x": np.linspace(-1.0, 1.0, 20),
            "z": np.linspace(1.0, -1.0, 20) ** 2,
        }
    )
    y = X["x"].to_numpy() + X["z"].to_numpy()
    kwargs = {"kernel_width": 0.3, "rff_num_feat": 10, "rff_random_state": 12}
    first = GPNAMRegressor(**kwargs).fit(X, y, random_state=8)
    second = GPNAMRegressor(**kwargs).fit(X, y, random_state=99)

    np.testing.assert_array_equal(
        first.basis_metadata()["phases"], second.basis_metadata()["phases"]
    )
    np.testing.assert_allclose(first.predict(X), first.predict(X, batch_size=3))
    np.testing.assert_allclose(
        first.basis_transform(X), first.basis_transform(X, batch_size=4)
    )
