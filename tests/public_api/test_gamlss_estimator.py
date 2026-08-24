"""Public contracts for the first-class GAM-backed GAMLSS estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.special import gammaln
from sklearn.base import clone, is_classifier, is_regressor

from nampy import GAMLSS
from nampy.gam import GAM
from nampy.gam.families import gammals, gaulss
from nampy.models import GAMClassifier, GAMRegressor


def _normal_data(n=100, seed=31):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "x0": rng.uniform(-1.0, 1.0, n),
            "x1": rng.uniform(-1.0, 1.0, n),
        }
    )
    mu = 0.4 + 1.1 * X["x0"].to_numpy()
    sigma = np.exp(-0.5 + 0.25 * X["x1"].to_numpy())
    y = rng.normal(mu, sigma)
    return X, y


def _fixed_normal(**overrides):
    params = {
        "family": "normal",
        "k": 5,
        "optimize_smoothing": False,
        "smoothing_method": "fixed",
        "smoothing_params": [1.0, 1.0, 1.0, 1.0],
    }
    params.update(overrides)
    return GAMLSS(**params)


def test_gamlss_is_exported_cloneable_and_has_distributional_tags():
    estimator = _fixed_normal()
    cloned = clone(estimator)

    assert type(cloned) is GAMLSS
    assert cloned.get_params() == estimator.get_params()
    assert not is_regressor(estimator)
    assert not is_classifier(estimator)


def test_array_mode_matches_raw_gam_link_predictions():
    X, y = _normal_data()
    estimator = _fixed_normal().fit(X, y)
    raw = GAM(
        family="gaulss",
        k=5,
        optimize_smoothing=False,
        smoothing_method="fixed",
        smoothing_params=[1.0, 1.0, 1.0, 1.0],
    ).fit(X, y)

    assert estimator.parameter_names_ == ("mu", "sigma")
    assert_array_equal(estimator.predict(raw=True), raw.predict(type="link"))
    assert_array_equal(estimator.predict(X, raw=True), raw.predict(X, type="link"))


def test_normal_predict_returns_natural_parameters_and_density_score():
    X, y = _normal_data()
    estimator = _fixed_normal().fit(X, y)

    eta = estimator.predict_link(X)
    parameters = estimator.predict(X)
    expected_mu = estimator.gam_.family.linfo[0].linkinv(eta[:, 0])
    expected_tau = estimator.gam_.family.linfo[1].linkinv(eta[:, 1])
    expected = np.column_stack([expected_mu, 1.0 / expected_tau])
    assert_allclose(parameters, expected, rtol=1e-13, atol=1e-13)
    assert_allclose(estimator.predict_point(X), parameters[:, 0])

    mu, sigma = parameters.T
    logpdf = (
        -0.5 * np.log(2.0 * np.pi)
        - np.log(sigma)
        - 0.5 * np.square((y - mu) / sigma)
    )
    assert estimator.score(X, y) == pytest.approx(float(np.mean(logpdf)))
    assert estimator.evaluate(X, y) == {
        "Negative Log-Likelihood": pytest.approx(float(-np.mean(logpdf)))
    }


def test_gamma_alias_returns_mean_and_standard_deviation_parameters():
    family = gammals()
    eta = np.array([[0.2, -0.4], [-0.3, 0.1]], dtype=np.float64)
    parameters = family.distribution_parameters_from_eta(eta)

    log_dispersion = family.linfo[1].linkinv(eta[:, 1])
    expected = np.column_stack(
        [np.exp(eta[:, 0]), np.sqrt(np.exp(log_dispersion))]
    )
    assert_allclose(parameters, expected)

    y = np.array([0.8, 1.4])
    mu, sigma = parameters.T
    dispersion = sigma**2
    shape = 1.0 / dispersion
    scale = mu * dispersion
    expected_logpdf = (
        (shape - 1.0) * np.log(y)
        - y / scale
        - gammaln(shape)
        - shape * np.log(scale)
    )
    assert_allclose(family.logpdf_from_parameters(y, parameters), expected_logpdf)


def test_named_formula_mapping_is_parameter_ordered():
    X, y = _normal_data()
    data = X.assign(y=y)
    estimator = GAMLSS(
        family="gaulss",
        formula={"sigma": "~ 1", "mu": "y ~ x0"},
        optimize_smoothing=False,
    ).fit(data)
    raw = GAM(
        family=gaulss(),
        formula=["y ~ x0", "~ 1"],
        optimize_smoothing=False,
    ).fit(data=data)

    assert_array_equal(estimator.predict_link(data), raw.predict(data, type="link"))
    assert estimator.predict(data).shape == (len(data), 2)


def test_components_are_zero_padded_and_reconstruct_every_predictor():
    X, y = _normal_data()
    estimator = _fixed_normal().fit(X, y)
    components = estimator.predict_components(X)

    assert set(components.terms) == {
        "mu:x0",
        "mu:x1",
        "sigma:x0",
        "sigma:x1",
    }
    assert components.intercept.shape == (2,)
    assert components.link.shape == components.response.shape == (len(X), 2)
    assert np.all(components.terms["mu:x0"][:, 1] == 0.0)
    assert np.all(components.terms["sigma:x0"][:, 0] == 0.0)
    components.validate_additive_reconstruction()
    assert estimator.standard_errors(X).shape == (len(X), 2)
    assert estimator.standard_errors(X, type="link").shape == (len(X), 2)


def test_role_guards_make_estimator_responsibilities_explicit():
    X, y = _normal_data(n=60)
    labels = (y > np.median(y)).astype(int)

    with pytest.raises(ValueError, match="Use .* GAMLSS"):
        GAMRegressor(family="gaulss").fit(X, y)
    with pytest.raises(ValueError, match="binary binomial"):
        GAMClassifier(family="gaussian").fit(X, labels)
    with pytest.raises(ValueError, match="Unknown GAMLSS family"):
        GAMLSS(family="poisson").fit(X, y)


@pytest.mark.parametrize(
    ("formula", "message"),
    [
        ({"mu": "y ~ x0"}, "exactly match"),
        ({"mu": "y ~ x0", "sigma": "y ~ x1"}, "one-sided"),
        (["y ~ x0"], "expects 2 formulas"),
        ("y ~ x0", "mapping or an ordered sequence"),
    ],
)
def test_formula_validation_is_early_and_actionable(formula, message):
    X, y = _normal_data(n=40)
    with pytest.raises((TypeError, ValueError), match=message):
        GAMLSS(formula=formula).fit(X.assign(y=y))


def test_offsets_require_one_explicit_entry_per_parameter():
    X, y = _normal_data(n=40)
    with pytest.raises(ValueError, match="one entry per"):
        _fixed_normal().fit(X, y, offset=np.zeros(len(X)))
    with pytest.raises(ValueError, match="expects 2 offsets"):
        _fixed_normal().fit(X, y, offset=[np.zeros(len(X))])


def test_persistence_round_trip(tmp_path):
    X, y = _normal_data(n=60)
    estimator = _fixed_normal().fit(X, y)
    path = estimator.save_model(tmp_path / "gamlss.nampy")
    restored = GAMLSS.load_model(path)

    assert restored.parameter_names_ == ("mu", "sigma")
    assert_array_equal(restored.predict(X), estimator.predict(X))


def test_shared_formula_component_is_exposed_for_each_target_parameter():
    X, y = _normal_data(n=70)
    data = X.assign(y=y)
    estimator = GAMLSS(
        formula=[
            "y ~ x0",
            "~ 1",
            '1 + 2 ~ s(x1, bs="cr", k=6, sp=0.8) - 1',
        ],
        optimize_smoothing=False,
        smoothing_method="fixed",
    ).fit(data)

    components = estimator.predict_components(data)
    shared_keys = [key for key in components.terms if "s(x1" in key]
    assert shared_keys == [
        'mu:s(x1, bs="cr", k=6, sp=0.8)',
        'sigma:s(x1, bs="cr", k=6, sp=0.8)',
    ]
    np.testing.assert_array_equal(components.terms[shared_keys[0]][:, 1], 0.0)
    np.testing.assert_array_equal(components.terms[shared_keys[1]][:, 0], 0.0)
    np.testing.assert_allclose(
        components.terms[shared_keys[0]][:, 0],
        components.terms[shared_keys[1]][:, 1],
        atol=0.0,
        rtol=0.0,
    )
    components.validate_additive_reconstruction()
