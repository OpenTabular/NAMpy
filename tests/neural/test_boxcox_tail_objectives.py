"""Normalized neural Box-Cox losses and analytical first derivatives."""

import pytest
import torch
from numpy.testing import assert_allclose

from nampy.distributions import BCPELaw, BCTLaw, DistributionPrediction
from nampy.neural.families.boxcox_tail import BCPEDistribution, BCTDistribution


@pytest.mark.parametrize(
    "objective,law", [(BCTDistribution, BCTLaw), (BCPEDistribution, BCPELaw)]
)
def test_loss_and_gradcheck_against_independent_probability(objective, law):
    raw = torch.tensor(
        [[0.2, -0.1, -1.2, 0.3], [-0.1, 0.3, 0.7, 1.2]],
        dtype=torch.float64,
        requires_grad=True,
    )
    target = torch.tensor([2.3, 0.7], dtype=torch.float64)
    family = objective()
    parameters = family(raw).detach().numpy()
    dist = DistributionPrediction(
        law(), dict(zip(law.spec.parameter_names, parameters.T, strict=True))
    )
    assert_allclose(
        family.compute_loss_per_sample(raw, target).detach(),
        -dist.log_prob(target.numpy()),
        atol=3e-12,
    )
    assert torch.autograd.gradcheck(
        lambda p: family.compute_loss_per_sample(p, target),
        (raw,),
        eps=1e-5,
        atol=1e-7,
        rtol=1e-6,
    )


@pytest.mark.parametrize("objective", [BCTDistribution, BCPEDistribution])
def test_log_response_limit_has_finite_first_derivatives(objective):
    raw = torch.tensor([[0.2, -0.1, 0, 1.2]], dtype=torch.float64, requires_grad=True)
    loss = objective().compute_loss_per_sample(
        raw, torch.tensor([2.3], dtype=torch.float64)
    )
    loss.sum().backward()
    assert torch.isfinite(loss).all() and torch.isfinite(raw.grad).all()


@pytest.mark.parametrize("name", ["bct", "bcpe"])
def test_public_neural_fit_and_distribution_roundtrip(name):
    import pandas as pd

    from nampy import GradientTrainingConfig, LinRegLSS, ObjectiveConfig

    frame = pd.DataFrame({"x": torch.linspace(-1, 1, 32).numpy()})
    y = (frame.x * 0.2 + 0.5).map(float).to_numpy()
    model = LinRegLSS(
        objective=ObjectiveConfig(family=name),
        training=GradientTrainingConfig(
            max_epochs=1,
            batch_size=32,
            trainer_kwargs={
                "logger": False,
                "enable_progress_bar": False,
                "enable_model_summary": False,
            },
        ),
    ).fit(frame, y)
    law = model.predict_distribution(frame)
    assert law.log_prob(y).shape == (len(y),)
    assert torch.isfinite(torch.as_tensor(law.log_prob(y))).all()
