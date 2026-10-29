"""Normalized test families exercise dimensions without adding public families."""

from itertools import combinations_with_replacement

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy.optimize._numdiff import approx_derivative

from nampy import GAMLSS
from nampy.gam.families.gamlss.distributional import _DistributionalFamily
from nampy.gam.fit.solvers.gamlss_utils import trind_generator


class _Student3(_DistributionalFamily):
    name = "test_student3"
    parameter_names = ("loc", "scale", "df")
    link_names = ("identity", "log", "log")
    parameter_bounds = {"scale": (0, None), "df": (0, None)}
    max_derivative_order = 4

    def _log_density(self, y, loc, log_scale, log_df):
        df = log_df.exp()
        z = (y - loc) * (-log_scale).exp()
        return (
            ((df + 1) / 2).lgamma()
            - (df / 2).lgamma()
            - (log_df + np.log(np.pi)) / 2
            - log_scale
            - (df + 1) / 2 * (1 + z * z / df).log()
        )

    def _initial_predictors(self, y):
        return np.mean(y), np.log(np.std(y)), np.log(5.0)


class _Mixture4(_DistributionalFamily):
    name = "test_mixture4"
    parameter_names = ("loc1", "loc2", "scale", "weight")
    link_names = ("identity", "identity", "log", "logit")
    parameter_bounds = {"scale": (0, None), "weight": (0, 1)}
    max_derivative_order = 4

    def _log_density(self, y, loc1, loc2, log_scale, logit_weight):
        weight = 1 / (1 + (-logit_weight).exp())
        z1, z2 = (y - loc1) * (-log_scale).exp(), (y - loc2) * (-log_scale).exp()
        density = weight * (-z1 * z1 / 2).exp() + (1 - weight) * (-z2 * z2 / 2).exp()
        return density.log() - log_scale - 0.5 * np.log(2 * np.pi)

    def _initial_predictors(self, y):
        return np.quantile(y, 0.2), np.quantile(y, 0.8), np.log(np.std(y) / 2), 0.0


@pytest.mark.parametrize(
    "family,eta",
    [
        (_Student3(), [0.2, -0.3, 1.4]),
        (_Mixture4(), [-0.5, 0.8, -0.2, 0.4]),
    ],
)
def test_general_dimension_mixed_tensors_and_coefficient_assembly(family, eta):
    torch = pytest.importorskip("torch")
    from nampy.gam.families.gamlss._taylor import Taylor

    k = len(eta)
    variables = [Taylor(np.array([eta[j]]), k, 4, j) for j in range(k)]
    value = family._log_density(np.array([0.3]), *variables)

    def likelihood(z):
        response = torch.tensor(0.3, dtype=torch.float64)
        if k == 3:
            return torch.distributions.StudentT(z[2].exp(), z[0], z[1].exp()).log_prob(
                response
            )
        components = torch.stack(
            [
                torch.nn.functional.logsigmoid(z[3])
                + torch.distributions.Normal(z[0], z[2].exp()).log_prob(response),
                torch.nn.functional.logsigmoid(-z[3])
                + torch.distributions.Normal(z[1], z[2].exp()).log_prob(response),
            ]
        )
        return torch.logsumexp(components, dim=0)

    tensor = torch.tensor(eta, dtype=torch.float64)
    assert_allclose(value.value, likelihood(tensor).detach().numpy(), atol=1e-13)
    function = likelihood
    for degree in range(1, 5):
        function = torch.func.jacfwd(function)
        derivatives = function(tensor).detach().numpy()
        packed = [
            derivatives[index]
            for index in combinations_with_replacement(range(k), degree)
        ]
        assert_allclose(value.packed_derivatives(degree)[0], packed, atol=2e-11)
    # Independent coefficient-space gradient/Hessian, with overlapping blocks,
    # row weights and offsets. No family/solver changes are needed for K>2.
    rng = np.random.default_rng(9)
    X = rng.normal(size=(5, k + 1))
    jj = [np.array([j, k]) for j in range(k)]
    beta = np.zeros(k + 1)
    offsets = [np.full(5, e) for e in eta]
    y, weights = np.linspace(-0.1, 0.6, 5), np.array([1.0, 0.0, 2.0, 0.5, 3.0])
    result = family.ll(y, X, jj, beta, weights, offset=offsets, deriv=1)

    def objective(b):
        return np.array([family.ll(y, X, jj, b, weights, offset=offsets)["l"]])

    numeric_gradient = approx_derivative(objective, beta).ravel()
    numeric_hessian = approx_derivative(
        lambda b: family.ll(y, X, jj, b, weights, offset=offsets, deriv=1)["lb"], beta
    )
    assert_allclose(result["lb"], numeric_gradient, atol=2e-7)
    assert_allclose(result["lbb"], numeric_hessian, atol=2e-6)
    assert family.nlp == family.n_linear_predictors == k
    assert_allclose(family.tri["i4"], trind_generator(k)["i4"])


@pytest.mark.parametrize("family", [_Student3(), _Mixture4()])
def test_multidimensional_fixed_fit_initialization_offsets_and_components(family):
    rng = np.random.default_rng(14)
    if isinstance(family, _Student3):
        y = 0.4 + rng.standard_t(4, size=500) * 0.7
    else:
        y = np.where(rng.uniform(size=500) < 0.4, -1.5, 1.0) + rng.normal(
            scale=0.3, size=500
        )
    frame = pd.DataFrame({"y": y})
    formula = {
        name: "y ~ 1" if j == 0 else "~ 1"
        for j, name in enumerate(family.parameter_names)
    }
    offsets = [np.full(len(y), 0.05 * (j + 1)) for j in range(family.nlp)]
    model = GAMLSS(family=family, formula=formula, optimize_smoothing=False).fit(
        frame, offset=offsets
    )
    from nampy.gam.model_state import _fit_result

    assert _fit_result(model.gam_).converged
    assert model.predict(frame).shape == (500, family.nlp)
    assert np.isfinite(model.standard_errors(frame)).all()
    model.predict_components(frame).validate_additive_reconstruction()


class _CoupledStudent3(_Student3):
    distribution_parameter_names = ("loc", "scale", "df", "scale_squared")
    parameter_bounds = {"scale": (0, None), "df": (0, None), "scale_squared": (0, None)}

    def _log_density(self, y, loc, log_scale, log_df):
        return super()._log_density(y, loc, log_scale + 0.2 * loc, log_df)

    def distribution_parameters_from_eta(self, eta):
        eta = np.asarray(eta)
        scale = np.exp(eta[:, 1] + 0.2 * eta[:, 0])
        return np.column_stack([eta[:, 0], scale, np.exp(eta[:, 2]), scale**2])

    def distribution_parameter_jacobian(self, eta):
        parameters = self.distribution_parameters_from_eta(eta)
        result = np.zeros((len(eta), 4, 3))
        result[:, 0, 0] = 1
        result[:, 1, 0] = 0.2 * parameters[:, 1]
        result[:, 1, 1] = parameters[:, 1]
        result[:, 2, 2] = parameters[:, 2]
        result[:, 3, 0] = 0.4 * parameters[:, 3]
        result[:, 3, 1] = 2 * parameters[:, 3]
        return result


def test_coupled_parameter_covariance_retains_cross_blocks_and_extra_parameters():
    family = _CoupledStudent3()
    x = np.linspace(-0.5, 0.5, 4)
    X = np.column_stack([np.ones(4), x, np.ones(4), x, np.ones(4), x])
    jj = [np.array([0, 1]), np.array([2, 3]), np.array([4, 5])]
    beta = np.array([0.4, 0.2, -0.3, 0.1, 1.5, 0.1])
    rng = np.random.default_rng(4)
    A = rng.normal(size=(6, 6))
    covariance = A @ A.T / 100
    covariance[0, 2] = covariance[2, 0] = -0.01
    assert np.linalg.eigvalsh(covariance).min() > 0
    parameters, se = family.predict_response(X, jj, beta, se=True, Vb=covariance)
    expected = []
    for row in range(len(x)):
        derivative = approx_derivative(
            lambda b, row=row: family.predict_fitted(X, jj, b)[row], beta
        )
        expected.append(np.sqrt(np.diag(derivative @ covariance @ derivative.T)))
    assert parameters.shape == se.shape == (4, 4)
    assert_allclose(se, expected, atol=1e-9)
    family._check_parameters(parameters)


def test_fixed_only_extension_does_not_inherit_automatic_smoothing_support():
    class FixedOnly(_Student3):
        max_derivative_order = 2

    family = FixedOnly()
    assert not family.supports_reml
    assert not family.supports_analytic_outer_hessian
    fixed_data = pd.DataFrame(
        {"y": 0.4 + 0.7 * np.random.default_rng(14).standard_t(4, 500)}
    )
    fixed = GAMLSS(
        family=family,
        formula={"loc": "y~1", "scale": "~1", "df": "~1"},
        optimize_smoothing=False,
    ).fit(fixed_data)
    from nampy.gam.model_state import _fit_result

    assert _fit_result(fixed.gam_).converged
    with pytest.raises(NotImplementedError, match="declares likelihood derivatives"):
        family.ll(
            np.array([0.3]),
            np.ones((1, 3)),
            [np.array([j]) for j in range(3)],
            np.array([0.1, 0.0, 1.5]),
            None,
            deriv=4,
        )
    frame = pd.DataFrame(
        {"x": np.linspace(-1, 1, 60), "y": np.random.default_rng(4).standard_t(4, 60)}
    )
    with pytest.raises(NotImplementedError, match="not supported"):
        GAMLSS(
            family=family, formula={"loc": "y~s(x,k=4)", "scale": "~1", "df": "~1"}
        ).fit(frame)


def test_extension_metadata_and_initialization_are_checked():
    class BadLinks(_Student3):
        link_names = ("identity", "log")

    with pytest.raises(ValueError, match="one link"):
        BadLinks()

    class BadBounds(_Student3):
        parameter_bounds = {"unknown": (0, None)}

    with pytest.raises(ValueError, match="bounds"):
        BadBounds()

    class BadStart(_Student3):
        def _initial_predictors(self, y):
            return [0.0]

    with pytest.raises(ValueError, match="one target"):
        BadStart().initialize(
            np.ones(5), np.ones((5, 3)), [np.array([j]) for j in range(3)]
        )


def test_unsupported_transform_and_inconsistent_automatic_declaration_fail_before_fitting():
    class ExtraParameters(_Student3):
        distribution_parameter_names = ("loc", "scale", "df", "derived")

    with pytest.raises(NotImplementedError, match="before fitting"):
        ExtraParameters()

    class DefaultProbit(_Student3):
        link_names = ("identity", "log", "probit")

    with pytest.raises(NotImplementedError, match="full parameter Jacobian"):
        DefaultProbit()

    class IncompleteAutomatic(_Student3):
        max_derivative_order = 2
        supports_reml = True

    with pytest.raises(ValueError, match="fourth-order"):
        IncompleteAutomatic()
