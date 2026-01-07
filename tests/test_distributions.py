import numpy as np
import torch

from nampy.utils.distributions import (
    BetaDistribution,
    CategoricalDistribution,
    DirichletDistribution,
    GammaDistribution,
    InverseGammaDistribution,
    NegativeBinomialDistribution,
    NormalDistribution,
    PoissonDistribution,
    Quantile,
    RobustNormalDistribution,
    StudentTDistribution,
)


def _assert_finite(value):
    assert torch.isfinite(value).all()


def test_normal_distribution_forward_and_metrics():
    dist = NormalDistribution()
    preds = torch.randn(8, 2)
    y = torch.randn(8)
    loss = dist.compute_loss(preds, y)
    _assert_finite(loss)

    transformed = dist(preds)
    assert transformed.shape == preds.shape
    assert torch.all(transformed[:, 1] > 0)

    metrics = dist.evaluate_nll(y.numpy(), preds.detach().numpy())
    assert "NLL" in metrics
    assert "mse" in metrics
    assert "rmse" in metrics
    assert "mae" in metrics


def test_poisson_distribution_loss():
    dist = PoissonDistribution()
    preds = torch.randn(8, 1)
    y = torch.randint(low=0, high=5, size=(8,), dtype=torch.float32)
    loss = dist.compute_loss(preds, y)
    _assert_finite(loss)


def test_inverse_gamma_distribution_loss():
    dist = InverseGammaDistribution()
    preds = torch.randn(8, 2)
    y = torch.rand(8) + 0.1
    loss = dist.compute_loss(preds, y)
    _assert_finite(loss)


def test_beta_distribution_loss():
    dist = BetaDistribution()
    preds = torch.randn(8, 2)
    y = torch.rand(8) * 0.9 + 0.05
    loss = dist.compute_loss(preds, y)
    _assert_finite(loss)


def test_dirichlet_distribution_loss():
    dist = DirichletDistribution()
    preds = torch.randn(8, 3)
    y = torch.rand(8, 3)
    y = y / y.sum(dim=1, keepdim=True)
    loss = dist.compute_loss(preds, y)
    _assert_finite(loss)


def test_gamma_distribution_loss():
    dist = GammaDistribution()
    preds = torch.randn(8, 2)
    y = torch.rand(8) + 0.1
    loss = dist.compute_loss(preds, y)
    _assert_finite(loss)


def test_student_t_distribution_loss():
    dist = StudentTDistribution()
    preds = torch.randn(8, 3)
    y = torch.randn(8)
    loss = dist.compute_loss(preds, y)
    _assert_finite(loss)


def test_negative_binomial_distribution_loss():
    dist = NegativeBinomialDistribution()
    preds = torch.randn(8, 2)
    y = torch.randint(low=0, high=5, size=(8,), dtype=torch.float32)
    loss = dist.compute_loss(preds, y)
    _assert_finite(loss)


def test_categorical_distribution_loss():
    dist = CategoricalDistribution()
    preds = torch.randn(8, 3)
    y = torch.randint(low=0, high=3, size=(8,))
    loss = dist.compute_loss(preds, y)
    _assert_finite(loss)


def test_quantile_distribution_loss():
    dist = Quantile(quantiles=[0.1, 0.5, 0.9])
    preds = torch.randn(8, 3)
    y = torch.randn(8)
    loss = dist.compute_loss(preds, y)
    _assert_finite(loss)


def test_robust_normal_distribution_loss():
    dist = RobustNormalDistribution(rob=0.1)
    preds = torch.randn(8, 2)
    y = torch.randn(8)
    loss = dist.compute_loss(preds, y)
    _assert_finite(loss)

    metrics = dist.evaluate_nll(y.numpy(), preds.detach().numpy())
    assert "NLL" in metrics
    assert "mse" in metrics
    assert "rmse" in metrics
    assert "mae" in metrics
