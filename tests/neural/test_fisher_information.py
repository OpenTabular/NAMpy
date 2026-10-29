"""Independent exact, score and signed-Hessian neural information contracts."""

import random

import numpy as np
import pandas as pd
import pytest
import torch
from numpy.testing import assert_allclose

from nampy.diagnostics.fisher import FisherDiagnostics
from nampy.neural.families.continuous import (
    BetaDistribution,
    GammaDistribution,
    NormalDistribution,
    WeibullDistribution,
)
from nampy.neural.fisher import estimate_fisher_information
from nampy.neural.fisher_likelihood import FisherLikelihood
from nampy.neural.objectives.core import (
    BinaryObjective,
    DistributionObjective,
    MulticlassObjective,
    RegressionObjective,
)


class Preprocessing:
    def __init__(self, limit=1024):
        self.limit, self.rows = limit, []

    def preprocess_tensors(self, frame):
        assert len(frame) <= self.limit
        self.rows.extend(frame.index.tolist())
        return {}, {"x": torch.tensor(frame.to_numpy(), dtype=torch.float64)}


class Model(torch.nn.Module):
    supports_fisher_hessian = True

    def __init__(self, objective, *, nonlinear=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.2, -0.1], dtype=torch.float64))
        self.bias = torch.nn.Parameter(torch.tensor(0.4, dtype=torch.float64))
        self.objective = objective
        self.nonlinear = nonlinear
        self.register_buffer("calls", torch.tensor(0))
        self.child = torch.nn.Identity()

    def forward(self, num_features, cat_features):
        self.calls.add_(1)
        value = num_features["x"] @ self.weight + self.bias
        return {"output": (value.sin() if self.nonlinear else value)[:, None]}


def estimate(model, X, **kwargs):
    options = {
        "group_by": "layer",
        "record_diagonal": "all",
        "regression_scale": 1.3,
        **kwargs,
    }
    config = FisherDiagnostics(**options)
    return estimate_fisher_information(
        model, Preprocessing(), X, config, random_state=17
    )


def test_exact_gaussian_and_hessian_equal_mean_design_information():
    X = pd.DataFrame([[1.0, 2.0], [-1.0, 0.5], [0.25, -0.5]])
    model = Model(RegressionObjective())
    expected = np.r_[np.mean(X.to_numpy() ** 2, axis=0), 1.0] / 1.3**2
    exact = estimate(model, X)
    hessian = estimate(model, X, estimator="negative_hessian", n_draws=4)
    assert_allclose(exact.diagonal, expected, atol=1e-14)
    assert_allclose(hessian.diagonal, expected, atol=1e-14)
    assert_allclose(hessian.estimator_variance, 0, atol=1e-30)
    assert exact.estimator_variance is None and exact.n_draws is None
    repeated = estimate(model, pd.concat([X, X], ignore_index=True))
    assert_allclose(exact.diagonal, repeated.diagonal, atol=1e-14)


def test_exact_bernoulli_includes_offset_and_probe_batches_preserve_rows():
    X = pd.DataFrame([[1.0, 2.0], [-1.0, 0.5], [0.25, -0.5]], index=[8, 3, 1])
    model = Model(BinaryObjective())
    config = FisherDiagnostics(group_by="tensor", record_diagonal="all")
    preprocessing = Preprocessing(limit=2)
    offset = np.array([0.2, 1.0, -0.3])
    result = estimate_fisher_information(
        model, preprocessing, X, config, offset=offset, batch_size=2
    )
    eta = X.to_numpy() @ model.weight.detach().numpy() + model.bias.item() + offset
    probability = 1 / (1 + np.exp(-eta))
    expected = np.r_[
        np.mean((probability * (1 - probability))[:, None] * X.to_numpy() ** 2, axis=0),
        np.mean(probability * (1 - probability)),
    ]
    assert_allclose(result.diagonal, expected, atol=2e-16)
    assert preprocessing.rows == [8, 3, 1]
    assert model.calls.item() == 0


def test_score_variances_match_gaussian_truth_and_groups_keep_covariance():
    X = pd.DataFrame([[1.0, 2.0]])
    model = Model(RegressionObjective())
    result = estimate(model, X, estimator="score_outer_product", n_draws=2000)
    truth = np.r_[X.iloc[0].to_numpy() ** 2, 1.0] / 1.3**2
    assert_allclose(result.diagonal, truth, rtol=0.08)
    assert_allclose(result.output_sampling_variance, 2 * truth**2, rtol=0.2)
    assert_allclose(result.estimator_variance, result.output_sampling_variance / 2000)
    # Weight coordinates share the same residual: variance of their sum is not
    # the sum of their marginal variances.
    group = next(g for g in result.groups if g.name == "weight")
    assert group.output_sampling_variance > result.output_sampling_variance[:2].sum()


def test_sampled_negative_hessian_is_signed_and_matches_nonlinear_information():
    class Nonlinear(torch.nn.Module):
        supports_fisher_hessian = True

        def __init__(self):
            super().__init__()
            self.theta = torch.nn.Parameter(torch.tensor(1.2, dtype=torch.float64))
            self.objective = RegressionObjective()

        def forward(self, num_features, cat_features):
            return {"output": self.theta.sin().expand(len(num_features["x"]), 1)}

    model = Nonlinear()
    X = pd.DataFrame([[1.0]])
    exact = estimate(model, X, regression_scale=1.0)
    hessian = estimate(
        model, X, estimator="negative_hessian", n_draws=4000, regression_scale=1.0
    )
    assert_allclose(exact.diagonal, [np.cos(1.2) ** 2], atol=1e-14)
    assert_allclose(hessian.diagonal, exact.diagonal, atol=0.04)
    assert_allclose(hessian.output_sampling_variance, [np.sin(1.2) ** 2], rtol=0.1)
    values = [
        estimate_fisher_information(
            model,
            Preprocessing(),
            X,
            FisherDiagnostics(
                estimator="negative_hessian",
                n_draws=2,
                regression_scale=1.0,
                group_by="tensor",
                record_diagonal="all",
            ),
            random_state=seed,
        ).diagonal[0]
        for seed in range(8)
    ]
    assert min(values) < 0


def test_modes_grad_flags_rng_and_buffers_restored_on_success_and_failure():
    X = pd.DataFrame([[1.0, 2.0], [-1.0, 0.5]])
    model = Model(RegressionObjective()).train()
    model.child.eval()
    model.weight.grad = torch.ones_like(model.weight)
    reference = model.weight.grad
    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)
    expected = random.random(), np.random.random(), torch.rand(1)
    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)
    estimate(model, X, estimator="score_outer_product", n_draws=4)
    actual = random.random(), np.random.random(), torch.rand(1)
    assert actual[:2] == expected[:2]
    torch.testing.assert_close(actual[2], expected[2])
    assert model.training and not model.child.training
    assert model.weight.grad is reference and torch.equal(
        reference, torch.ones_like(reference)
    )
    assert model.calls.item() == 0
    with pytest.raises(AssertionError):
        estimate_fisher_information(
            model,
            Preprocessing(limit=1),
            X,
            FisherDiagnostics(regression_scale=1.0, group_by="layer"),
            batch_size=2,
        )
    assert model.training and not model.child.training and model.calls.item() == 0


@pytest.mark.parametrize(
    "activation", [torch.nn.LeakyReLU, torch.nn.Softsign, torch.nn.Softshrink]
)
def test_known_nonsmooth_models_reject_hessian_but_allow_score(activation):
    model = Model(RegressionObjective())
    model.child = activation()
    X = pd.DataFrame([[1.0, 2.0]])
    with pytest.raises(NotImplementedError, match="smooth"):
        estimate(model, X, estimator="negative_hessian")
    estimate(model, X, estimator="score_outer_product", n_draws=2)


@pytest.mark.parametrize(
    "family,point",
    [
        (GammaDistribution(shape_transform="none", rate_transform="none"), [3.0, 2.0]),
        (BetaDistribution(shape_transform="none", scale_transform="none"), [2.0, 3.0]),
        (
            WeibullDistribution(scale_transform="none", shape_transform="none"),
            [2.0, 3.0],
        ),
    ],
)
def test_small_output_metric_matches_independent_expected_hessian(family, point):
    config = FisherDiagnostics()
    adapter = FisherLikelihood.from_objective(DistributionObjective(family), config)
    raw = torch.tensor([point], dtype=torch.float64, requires_grad=True)
    torch.manual_seed(25)
    target = adapter.distribution(raw.detach()).sample((40000,)).detach()
    expected = -torch.autograd.functional.hessian(
        lambda p: adapter.log_prob(p, target).sum() / len(target), raw
    )[0, :, 0, :]
    _, metric = adapter.coordinates_and_information(raw)
    assert_allclose(metric.detach(), expected.detach(), rtol=0.025, atol=0.01)


def test_categorical_exact_enumeration_handles_shared_logit_shift():
    class Categorical(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.shift = torch.nn.Parameter(torch.tensor(0.2, dtype=torch.float64))
            self.logits = torch.nn.Parameter(
                torch.tensor([0.1, 0.5, -0.3], dtype=torch.float64)
            )
            self.objective = MulticlassObjective(3)

        def forward(self, num_features, cat_features):
            return {"output": (self.logits + self.shift)[None, :]}

    model = Categorical()
    result = estimate(model, pd.DataFrame([[1.0]]))
    p = torch.softmax(model.logits.detach(), dim=0).numpy()
    assert_allclose(result.diagonal, np.r_[0.0, p * (1 - p)], atol=2e-16)


def test_parameterized_family_preserves_double_precision_derivatives():
    raw = torch.tensor([[0.2, -0.4]], dtype=torch.float64, requires_grad=True)
    family = NormalDistribution()
    assert family(raw).dtype == torch.float64
    assert family(torch.tensor([[1, 2]])).dtype == torch.float32
    assert torch.autograd.gradcheck(
        lambda p: family.compute_loss(p, torch.tensor([1.5], dtype=torch.float64)),
        (raw,),
    )


def test_custom_losses_and_family_subclasses_cannot_inherit_fisher_eligibility():
    class CustomNormal(NormalDistribution):
        pass

    for objective in (
        RegressionObjective(loss_fct=lambda p, y: (p - y).abs()),
        DistributionObjective(CustomNormal()),
    ):
        with pytest.raises(NotImplementedError, match="normalized"):
            FisherLikelihood.from_objective(
                objective, FisherDiagnostics(regression_scale=1.0)
            )


def test_frozen_native_coordinates_are_temporary_and_parameter_mutation_rejected():
    X = pd.DataFrame([[1.0, 2.0]])
    model = Model(RegressionObjective())
    for p in model.parameters():
        p.requires_grad_(False)
    result = estimate_fisher_information(
        model,
        Preprocessing(),
        X,
        FisherDiagnostics(
            regression_scale=1.3, group_by="tensor", record_diagonal="all"
        ),
        include_frozen=True,
    )
    assert_allclose(result.diagonal, np.array([1.0, 4.0, 1.0]) / 1.3**2)
    assert all(not p.requires_grad for p in model.parameters())

    class Mutating(Model):
        def forward(self, **kwargs):
            with torch.no_grad():
                self.bias.add_(1.0)
            return super().forward(**kwargs)

    mutating = Mutating(RegressionObjective())
    before = mutating.bias.detach().clone()
    with pytest.raises(RuntimeError, match="mutated"):
        estimate(mutating, X)
    torch.testing.assert_close(mutating.bias, before)
    assert mutating.calls.item() == 0


def test_failure_during_forward_restores_changed_buffers():
    class Failing(Model):
        def forward(self, **kwargs):
            super().forward(**kwargs)
            raise RuntimeError("forward failed")

    model = Failing(RegressionObjective()).train()
    model.child.eval()
    with pytest.raises(RuntimeError, match="forward failed"):
        estimate(model, pd.DataFrame([[1.0, 2.0]]))
    assert model.calls.item() == 0
    assert model.training and not model.child.training
