"""Censor/entry likelihoods and exact fixed-smoothing mixed derivatives."""

from itertools import combinations_with_replacement

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose
from scipy.special import log_ndtr

from nampy.gam.families.gamlss._taylor import Taylor
from nampy.gam.families.gamlss.censored import CensoredLogNormal, CensoredWeibull
from nampy.neural.survival import CensoredParametricObjective
from nampy.survival import SurvivalTarget
from nampy.survival.likelihood import observed_log_likelihood, parametric_distribution


def reference_loglik(name, eta, target):
    time = torch.tensor(target.time.copy(), dtype=eta.dtype)
    entry = torch.tensor(target.entry.copy(), dtype=eta.dtype)
    event = torch.tensor(target.event.copy())
    if name == "weibull":
        scale, shape = eta[:, 0].exp(), eta[:, 1].exp()
        law = torch.distributions.Weibull(scale, shape)
        logsf = -((time / scale) ** shape)
        entry_logsf = -torch.where(
            entry > 0, (torch.where(entry > 0, entry, 1.0) / scale) ** shape, 0.0
        )
    else:
        loc, sd = eta[:, 0], eta[:, 1].exp()
        law = torch.distributions.LogNormal(loc, sd)
        logsf = torch.special.log_ndtr(-(time.log() - loc) / sd)
        za = (torch.where(entry > 0, entry, 1.0).log() - loc) / sd
        entry_logsf = torch.where(entry > 0, torch.special.log_ndtr(-za), 0.0)
    return torch.where(event, law.log_prob(time), logsf) - entry_logsf


@pytest.mark.parametrize(
    "name,family_class",
    [("weibull", CensoredWeibull), ("lognormal", CensoredLogNormal)],
)
@pytest.mark.parametrize("events", [[1, 0, 1, 0], [1, 1, 1, 1], [0, 0, 0, 0]])
def test_values_mixed_derivatives_and_weighted_coefficient_assembly(
    name, family_class, events
):
    target = SurvivalTarget([0.4, 1.2, 2.1, 4.0], events, [0.0, 0.2, 0.5, 1.0])
    family = family_class()
    family.bind_fit_context({"event": target.event, "entry": target.entry}, len(target))
    eta = np.array([[0.2, -0.3], [-0.1, 0.1], [0.4, -0.2], [0.7, 0.2]])
    variables = [Taylor(eta[:, k], 2, 2, k) for k in range(2)]
    value = family._log_density(target.time, *variables)
    tensor = torch.tensor(eta, dtype=torch.float64)
    neural = CensoredParametricObjective(name)
    targets = torch.tensor(np.column_stack([target.time, target.event, target.entry]))
    assert_allclose(neural.loss_values(tensor, targets), -value.value, atol=2e-12)
    expected = reference_loglik(name, tensor, target)
    assert_allclose(value.value, expected.numpy(), rtol=1e-12, atol=1e-12)
    params = family.distribution_parameters_from_eta(eta)
    assert_allclose(
        observed_log_likelihood(target, parametric_distribution(name, params)),
        expected,
        atol=2e-12,
    )
    for row in range(len(target)):
        single = target.take([row])

        def function(x, single=single):
            return reference_loglik(name, x[None, :], single)[0]

        def neural_function(x, row=row):
            return -neural.loss_values(x[None, :], targets[row : row + 1])[0]

        for degree in (1, 2):
            function = torch.func.jacfwd(function)
            neural_function = torch.func.jacfwd(neural_function)
            derivative = function(tensor[row]).numpy()
            assert_allclose(
                neural_function(tensor[row]), derivative, atol=2e-11, rtol=2e-11
            )
            packed = [
                derivative[index]
                for index in combinations_with_replacement(range(2), degree)
            ]
            assert_allclose(
                value.packed_derivatives(degree)[row], packed, atol=2e-11, rtol=2e-11
            )
    X = np.array(
        [[1.0, -0.3, 0.2], [1.0, 0.2, -0.2], [1.0, 0.7, 0.4], [1.0, 1.2, -0.1]]
    )
    jj = [np.array([0, 1]), np.array([0, 2])]
    coef = np.array([0.1, -0.2, 0.15])
    weights = np.array([1.0, 0.0, 2.0, 0.5])
    result = family.ll(target.time, X, jj, coef, weights, offset=list(eta.T), deriv=1)

    def objective(b):
        mat = torch.tensor(X)
        linear = torch.stack([mat[:, block] @ b[block] for block in jj], dim=1)
        return (
            reference_loglik(name, linear + tensor, target) * torch.tensor(weights)
        ).sum()

    beta = torch.tensor(coef)
    assert_allclose(result["lb"], torch.func.jacfwd(objective)(beta), atol=2e-11)
    assert_allclose(result["lbb"], torch.func.hessian(objective)(beta), atol=2e-10)
    assert family.max_derivative_order == 4
    assert family.supports_reml


@pytest.mark.parametrize("z", [-40.0, -10.0, 0.0, 10.0, 40.0])
def test_log_normal_survival_tail_values_and_derivatives(z):
    value = Taylor(np.array([z]), 1, 2, 0).log_normal_survival()

    def f(x):
        return torch.special.log_ndtr(-x)

    x = torch.tensor(z, dtype=torch.float64)
    assert_allclose(value.value, log_ndtr(-z), atol=1e-12)
    assert_allclose(
        value.packed_derivatives(1)[0, 0],
        torch.func.jacfwd(f)(x),
        rtol=1e-11,
        atol=1e-12,
    )
    assert_allclose(
        value.packed_derivatives(2)[0, 0],
        torch.func.hessian(f)(x),
        rtol=2e-9,
        atol=2e-10,
    )


def test_weibull_close_entry_increment_and_zero_entry_are_stable():
    time = np.array([2.0, 2.0])
    entry = np.array([np.nextafter(2.0, 0.0), 0.0])
    target = SurvivalTarget(time, [0, 0], entry)
    prediction = parametric_distribution("weibull", [[1.0, 2.0], [1.0, 2.0]])
    value = observed_log_likelihood(target, prediction)
    assert_allclose(
        value[0], -(time[0] - entry[0]) * (time[0] + entry[0]), rtol=1e-14, atol=0
    )
    assert_allclose(value[1], -4.0)
