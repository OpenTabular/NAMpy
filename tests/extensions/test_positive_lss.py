"""Positive LSS probability, exact derivatives, fitting and route boundaries."""

from itertools import combinations_with_replacement
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy.optimize import brentq
from scipy.optimize._numdiff import approx_derivative
from sklearn.base import clone

from nampy import GAMLSS
from nampy.gam.families.gamlss._taylor import Taylor
from nampy.gam.families.registry import make_gam_family
from nampy.gam.fit.capabilities import coerce_general_family_smoothing_method
from nampy.gam.fit.selection import supports_smoothing_method
from nampy.gam.model_state import _fit_result
from nampy.gam.predict.distribution_adapter import distribution_prediction

FIXTURES = Path(__file__).parents[1] / "reference_fixtures" / "positive_lss"


def _data(name):
    return pd.read_csv(FIXTURES / f"{name}fit.csv")


def _model(name, *, smooth=False, **kwargs):
    first, second = make_gam_family(name + "_lss").parameter_names
    formula = {
        first: "y ~ s(x, bs='cr', k=5)" if smooth else "y ~ x",
        second: "~ s(x, bs='cr', k=4)" if smooth else "~ 1",
    }
    return GAMLSS(family=name + "_lss", formula=formula, **kwargs)


@pytest.mark.parametrize("name", ["lognormal", "weibull"])
def test_probability_and_unpenalized_fit_match_recorded_r_gamlss(name):
    family = make_gam_family(name + "_lss")
    reference = pd.read_csv(FIXTURES / f"{name}_probability.csv")
    parameters = reference[["first", "second"]].to_numpy()
    distribution = distribution_prediction(family, parameters)
    assert_allclose(
        family.logpdf_from_parameters(reference.y, parameters),
        reference.log_density,
        atol=2e-12,
    )
    assert_allclose(
        distribution.log_prob(reference.y), reference.log_density, atol=2e-12
    )
    assert_allclose(distribution.cdf(reference.y), reference.cdf, atol=2e-13)
    for j, probability in enumerate(reference.probability):
        assert_allclose(
            distribution.quantile(probability)[j, 0],
            reference["quantile"][j],
            rtol=2e-12,
        )
    frame = _data(name)
    model = _model(name, optimize_smoothing=False).fit(frame)
    assert _fit_result(model.gam_).converged
    assert_allclose(model.predict(frame), frame[["first", "second"]], atol=2e-5, rtol=0)
    assert_allclose(
        model.predict_distribution(frame).log_prob(frame.y),
        frame.log_density,
        atol=2e-5,
        rtol=0,
    )
    assert np.isfinite(
        model.predict_distribution(frame).quantile_residuals(frame.y).residuals
    ).all()


@pytest.mark.parametrize("name", ["lognormal", "weibull"])
@pytest.mark.parametrize("eta", [[-0.8, 0.2], [0.2, -0.4], [1.0, 0.7]])
def test_exact_fourth_order_likelihood_tensors_against_torch(name, eta):
    torch = pytest.importorskip("torch")
    family = make_gam_family(name + "_lss")
    variables = [Taylor(np.array([eta[k]]), 2, 4, k) for k in range(2)]
    value = family._log_density(np.array([1.2]), *variables)

    def likelihood(z):
        distribution = (
            torch.distributions.LogNormal(z[0], z[1].exp())
            if name == "lognormal"
            else torch.distributions.Weibull(z[0].exp(), z[1].exp())
        )
        return distribution.log_prob(torch.tensor(1.2, dtype=torch.float64))

    tensor = torch.tensor(eta, dtype=torch.float64)
    assert_allclose(value.value, likelihood(tensor).detach().numpy(), atol=1e-12)
    function = likelihood
    for degree in range(1, 5):
        function = torch.func.jacfwd(function)
        derivatives = function(tensor).detach().numpy()
        packed = [
            derivatives[idx] for idx in combinations_with_replacement(range(2), degree)
        ]
        assert_allclose(
            value.packed_derivatives(degree)[0], packed, atol=2e-9, rtol=2e-9
        )


def test_lognormal_weighted_mle_and_response_jacobian():
    frame = _data("lognormal")
    weights = np.tile([1.0, 2.0, 3.0], len(frame) // 3)
    model = _model("lognormal", optimize_smoothing=False).fit(
        frame, sample_weight=weights
    )
    X = np.column_stack([np.ones(len(frame)), frame.x])
    log_y = np.log(frame.y.to_numpy())
    beta = np.linalg.lstsq(
        X * np.sqrt(weights[:, None]), log_y * np.sqrt(weights), rcond=None
    )[0]
    loc = X @ beta
    scale = np.sqrt(np.average((log_y - loc) ** 2, weights=weights))
    assert_allclose(model.predict(frame)[:, 0], loc, atol=2e-7)
    assert_allclose(model.predict(frame)[:, 1], scale, atol=2e-7)
    density = (
        -0.5 * np.log(2 * np.pi)
        - np.log(scale)
        - 0.5 * ((log_y - loc) / scale) ** 2
        - log_y
    )
    assert_allclose(
        model.score(frame, frame.y, sample_weight=weights),
        np.average(density, weights=weights),
        atol=1e-12,
    )
    assert_allclose(model.predict_point(frame), np.exp(loc + scale**2 / 2), atol=3e-7)


def test_weibull_intercept_mle_matches_independent_profile_root():
    frame = _data("weibull")
    model = GAMLSS(
        family="weibull_lss",
        formula={"scale": "y~1", "shape": "~1"},
        optimize_smoothing=False,
    ).fit(frame)
    log_y = np.log(frame.y.to_numpy())

    def profile(k):
        powers = np.exp(k * (log_y - log_y.max()))
        return np.average(log_y, weights=powers) - log_y.mean() - 1 / k

    shape = brentq(profile, 0.01, 100.0)
    scale = np.exp(np.log(np.mean(np.exp(shape * log_y))) / shape)
    assert_allclose(
        model.predict(frame), np.tile([scale, shape], (len(frame), 1)), atol=2e-6
    )


@pytest.mark.parametrize("name", ["lognormal", "weibull"])
def test_fixed_penalty_weights_offsets_persistence_and_prediction_uncertainty(
    name, tmp_path
):
    frame = _data(name)
    weights = np.tile([1, 2, 3], len(frame) // 3)
    offsets = [np.linspace(-0.1, 0.1, len(frame)), np.linspace(0.05, -0.05, len(frame))]
    model = _model(
        name, smooth=True, optimize_smoothing=False, smoothing_params=[0.3, 0.7]
    ).fit(frame, sample_weight=weights, offset=offsets)
    repeated = _model(
        name, smooth=True, optimize_smoothing=False, smoothing_params=[0.3, 0.7]
    ).fit(
        frame.loc[frame.index.repeat(weights)].reset_index(drop=True),
        offset=[np.repeat(o, weights) for o in offsets],
    )
    assert_allclose(
        model.predict(frame, offset=offsets),
        repeated.predict(frame, offset=offsets),
        atol=2e-6,
    )
    components = model.predict_components(frame, offset=offsets)
    components.validate_additive_reconstruction()
    assert np.isfinite(model.standard_errors(frame)).all()
    native, native_se = model.gam_.predict(frame, return_se=True)
    assert_allclose(native, model.predict(frame))
    assert_allclose(native_se, model.standard_errors(frame), atol=1e-13)
    distribution = model.predict_distribution(frame)
    assert_allclose(model.predict_point(frame), distribution.mean(), rtol=1e-12)
    assert_allclose(
        model.gam_.family.variance_from_parameters(model.predict(frame)),
        distribution.variance(),
        rtol=1e-12,
    )
    path = model.save_model(tmp_path / f"{name}.pkl")
    restored = GAMLSS.load_model(path)
    assert_allclose(restored.predict_distribution(frame).mean(), distribution.mean())
    assert clone(model).get_params() == model.get_params()
    assert model.parameter_names_ == make_gam_family(name + "_lss").parameter_names


@pytest.mark.parametrize("name", ["lognormal", "weibull"])
def test_outer_reml_gradient_hessian_match_independent_differences(name):
    from nampy.gam.fit.selection.criteria import (
        criterion_gradient,
        criterion_hessian,
        criterion_value,
    )

    model = _model(
        name, smooth=True, optimize_smoothing=False, smoothing_params=[0.3, 0.7]
    ).fit(_data(name))
    gam = model.gam_
    rho = np.log(gam.smoothing_params)

    def value(z):
        return np.array([criterion_value(gam, gam.y_, z, method="reml")])

    gradient = criterion_gradient(gam, gam.y_, rho, method="reml")
    assert_allclose(
        gradient, approx_derivative(value, rho).ravel(), atol=2e-4, rtol=2e-4
    )
    hessian = criterion_hessian(gam, gam.y_, rho, method="reml")
    assert_allclose(
        hessian,
        approx_derivative(
            lambda z: criterion_gradient(gam, gam.y_, z, method="reml"), rho
        ),
        atol=2e-3,
        rtol=2e-3,
    )


@pytest.mark.parametrize("name", ["lognormal", "weibull"])
@pytest.mark.parametrize("optimizer", ["outer_newton", "bfgs", "efs"])
def test_automatic_general_reml_optimizers(name, optimizer):
    model = _model(name, smooth=True, smoothing_optimizer=optimizer).fit(_data(name))
    assert _fit_result(model.gam_).converged
    assert model.gam_._optim_method == "reml"
    assert np.all(np.isfinite(model.gam_.smoothing_params))
    assert np.all(np.asarray(model.gam_.smoothing_params) > 0)
    assert np.isfinite(model.score(_data(name), _data(name).y))


@pytest.mark.parametrize("name", ["lognormal", "weibull"])
def test_method_coercion_and_unsupported_optimizer_routes_are_explicit(name):
    family = make_gam_family(name + "_lss")
    model = SimpleNamespace(family=family, n_samples_=0)
    for method in ["ml", "laml", "gcv", "ubre", "ncv"]:
        assert coerce_general_family_smoothing_method(model, method) == "reml"
    assert not supports_smoothing_method(model, "ncv")
    assert not supports_smoothing_method(model, "gcv")
    for optimizer in ["magic", "nlm.fd", "optim"]:
        with pytest.raises(NotImplementedError):
            _model(name, smooth=True, smoothing_optimizer=optimizer).fit(_data(name))
    with pytest.raises(ValueError, match="Unsupported"):
        make_gam_family({"name": name + "_lss", "link": "identity"})


@pytest.mark.parametrize("name", ["lognormal", "weibull"])
@pytest.mark.parametrize("y", [[0.0, 1.0], [-1.0, 2.0], [1.0, np.inf], [1.0, np.nan]])
def test_positive_response_support_is_not_clipped(name, y):
    with pytest.raises(ValueError, match="strictly positive"):
        make_gam_family(name + "_lss").validate_y(y)
