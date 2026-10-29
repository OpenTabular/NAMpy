"""Normalization agrees with reference semantics, including singleton features."""

import pytest
import torch
import torch.nn.functional as F

from nampy.neural.architectures.components.mlp import make_norm
from nampy.neural.architectures.components.normalization import (
    BatchNorm,
    GroupNorm,
    InstanceNorm,
    LayerNorm,
)


@pytest.mark.parametrize("width", [1, 4])
@pytest.mark.parametrize("batch", [1, 5])
def test_layer_norm_values_and_gradients_match_torch(batch, width):
    x = torch.randn(batch, width, dtype=torch.float64, requires_grad=True)
    layer = LayerNorm(width).double()
    actual = layer(x)
    expected = F.layer_norm(x, (width,), layer.weight, layer.bias, layer.eps)
    torch.testing.assert_close(actual, expected)
    direction = torch.randn_like(actual)
    torch.testing.assert_close(
        torch.autograd.grad((actual * direction).sum(), x, retain_graph=True)[0],
        torch.autograd.grad((expected * direction).sum(), x)[0],
    )
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize("shape", [(1, 1), (5, 1), (5, 4), (3, 4, 7), (3, 4, 2, 7)])
def test_instance_norm_preserves_tabular_and_spatial_shape(shape):
    layer = InstanceNorm(shape[1]).double()
    x = torch.randn(shape, dtype=torch.float64, requires_grad=True)
    actual = layer(x)
    dims = (-1,) if x.ndim == 2 else tuple(range(2, x.ndim))
    expected = (x - x.mean(dims, keepdim=True)) / torch.sqrt(
        x.var(dims, correction=0, keepdim=True) + layer.eps
    )
    torch.testing.assert_close(actual, expected)
    assert actual.shape == x.shape
    actual.square().sum().backward()
    assert torch.isfinite(x.grad).all()


def test_mlp_instance_norm_has_per_instance_tabular_semantics():
    x = torch.tensor([[1.0, 2.0, 4.0], [10.0, 20.0, 40.0]])
    actual = make_norm("InstanceNorm", 3)(x)
    torch.testing.assert_close(actual[0], actual[1], atol=2e-5, rtol=2e-5)
    assert actual.shape == (2, 3)


@pytest.mark.parametrize("shape", [(1, 1), (3, 4), (3, 4, 7), (3, 4, 2, 7)])
def test_group_norm_agrees_with_population_moments(shape):
    x = torch.randn(shape, dtype=torch.float64, requires_grad=True)
    layer = GroupNorm(1, shape[1]).double()
    actual = layer(x)
    if x.numel() > 1:
        torch.testing.assert_close(
            actual, F.group_norm(x, 1, layer.weight, layer.bias, layer.eps)
        )
    else:
        torch.testing.assert_close(actual, torch.zeros_like(x))
    actual.square().sum().backward()
    assert torch.isfinite(x.grad).all()


def test_batch_norm_matches_native_running_statistics():
    actual = BatchNorm(3).double()
    expected = torch.nn.BatchNorm1d(3).double()
    for _ in range(3):
        x = torch.randn(5, 3, dtype=torch.float64)
        torch.testing.assert_close(actual(x), expected(x))
        torch.testing.assert_close(actual.running_mean, expected.running_mean)
        torch.testing.assert_close(actual.running_var, expected.running_var)
    actual.eval()
    expected.eval()
    torch.testing.assert_close(actual(x[:1]), expected(x[:1]))
