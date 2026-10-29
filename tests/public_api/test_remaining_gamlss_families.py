"""Public distribution contracts for the six newly registered general families.

Source fitting parity is owned by the family workflow suites. These checks use
their frozen numerical inputs to validate registration, adapters and normalized
distribution semantics. Multinomial natural-parameter SEs use the conventional
full coefficient delta covariance; native GAM prediction separately preserves
mgcv's clamp of negative cross-predictor covariances.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, softmax
from scipy.stats import genextreme, gumbel_r, norm, poisson

from nampy import GAMLSS
from nampy.gam import families as family_exports
from nampy.gam.families.gamlss import (
    GevlssFamily,
    GumblsFamily,
    MultinomFamily,
    ShashFamily,
    TwlssFamily,
    ZiplssFamily,
)
from nampy.gam.families.registry import make_gam_family
from nampy.gam.families.tweedie import ldTweedie
from nampy.models import GAMRegressor

ROOT = Path(__file__).parents[1] / "reference_fixtures"
FAMILIES = {
    "gumbls": GumblsFamily,
    "ziplss": ZiplssFamily,
    "multinom": MultinomFamily,
    "gevlss": GevlssFamily,
    "shash": ShashFamily,
    "twlss": TwlssFamily,
}


def _case(name, K=1):
    namespace = {"multinom": "mgcv_multinom_workflows", "gevlss": "mgcv_gevlss_workflows"}.get(name, f"mgcv_{name}")
    values = json.loads((ROOT / namespace / "inputs.json").read_text())
    if name in {"gumbls", "ziplss"}:
        values = values["data"]
        formulas = ["y~s(x,bs='cr',k=5)+offset(o1)", "~s(z,bs='cr',k=4)+offset(o2)"]
        sp = [.4, .7]
    elif name == "multinom":
        values = values[str(K)]
        formulas = ["y~s(x,bs='cr',k=6)"] + ["~s(x,bs='cr',k=6)"] * (K - 1)
        sp = [.7] * K
    elif name == "gevlss":
        formulas, sp = ["y~s(x,bs='cr',k=6)", "~1", "~1"], [.7]
    elif name == "shash":
        formulas, sp = ["y~s(x,bs='cr',k=6)", "~s(x,bs='cr',k=5)", "~1", "~1"], [.7, .9]
    else:
        formulas, sp = ["y~s(x,bs='cr',k=6)", "~1", "~s(x,bs='cr',k=5)"], [.7, .9]
    return pd.DataFrame(values), formulas, sp


def _spec(name, kind, options=None):
    options = {} if options is None else options
    if kind == "instance":
        return FAMILIES[name](**options)
    if kind == "name":
        assert not options
        return name
    if kind == "dict":
        return {"name": name, **options}
    return name, options


def _fit(name, spec, *, named=False, K=1):
    frame, formulas, sp = _case(name, K)
    if named:
        names = make_gam_family(spec).parameter_names
        # Deliberately reverse insertion order; parameters own formula order.
        formulas = dict(reversed(list(zip(names, formulas, strict=True))))
    model = GAMLSS(formula=formulas, family=spec, optimize_smoothing=False,
                   smoothing_method="fixed", smoothing_params=sp).fit(frame)
    return model, frame


@pytest.fixture(scope="module", params=tuple(FAMILIES))
def fitted_family(request):
    name = request.param
    model, frame = _fit(name, FAMILIES[name]())
    return name, model, frame


@pytest.mark.parametrize("kind", ["instance", "name", "dict", "tuple"])
@pytest.mark.parametrize("name", tuple(FAMILIES))
def test_remaining_general_family_registration(name, kind):
    family = make_gam_family(_spec(name, kind))
    assert isinstance(family, FAMILIES[name])
    assert family.name == name
    assert family.family_class == "general"
    assert isinstance(getattr(family_exports, name)(), FAMILIES[name])


@pytest.mark.parametrize("name,options,attribute,expected", [
    ("gumbls", {"b": -3.}, "b", -3.),
    ("ziplss", {"link": ["identity", "identity"]}, "link_names", ("identity", "identity")),
    ("multinom", {"K": 3}, "K", 3),
    ("gevlss", {"link": ["log", "identity", "identity"]}, "link_names", ("log", "identity", "identity")),
    ("shash", {"phiPen": .03}, "phi_pen", .03),
    ("twlss", {"a": 1.2, "b": 1.8}, "a", 1.2),
])
@pytest.mark.parametrize("kind", ["dict", "tuple"])
def test_general_family_registry_preserves_constructor_options(name, options, attribute, expected, kind):
    family = make_gam_family(_spec(name, kind, options))
    assert getattr(family, attribute) == expected


@pytest.mark.parametrize("kind", ["name", "dict", "tuple"])
def test_public_family_specs_and_named_formula_order_match_instances(fitted_family, kind):
    name, baseline, frame = fitted_family
    model, _ = _fit(name, _spec(name, kind), named=kind != "name")
    assert model.predictor_names_ == baseline.predictor_names_
    assert model.parameter_names_ == baseline.parameter_names_
    # These are independently fitted models. Equivalent family specifications
    # need numerical agreement; BLAS and nonlinear transforms can round their
    # results differently by a few float64 ulps across allocations/platforms.
    roundoff = 32 * np.finfo(np.float64).eps
    np.testing.assert_allclose(
        model.predict(frame), baseline.predict(frame), atol=roundoff, rtol=roundoff
    )
    np.testing.assert_allclose(
        model.predict_link(frame), baseline.predict_link(frame),
        atol=roundoff, rtol=roundoff,
    )


def _independent_distribution(name, eta, y, family):
    if name == "gumbls":
        params = np.column_stack((eta[:, 0], np.exp(eta[:, 1]) + np.exp(family.b)))
        point = params[:, 0] + np.euler_gamma * params[:, 1]
        density = gumbel_r.logpdf(y, loc=params[:, 0], scale=params[:, 1])
    elif name == "ziplss":
        lam, presence = np.exp(eta[:, 0]), -np.expm1(-np.exp(eta[:, 1]))
        params = np.column_stack((lam, presence))
        point = presence * lam / -np.expm1(-lam)
        density = np.log(presence) + poisson.logpmf(y, lam) - np.log(-np.expm1(-lam))
        density[y == 0] = np.log1p(-presence[y == 0])
    elif name == "multinom":
        params = softmax(np.column_stack((np.zeros(len(y)), eta)), axis=1)
        point = np.argmax(params, axis=1)
        density = np.log(params[np.arange(len(y)), np.rint(y).astype(int)])
    elif name == "gevlss":
        params = np.column_stack((eta[:, 0], np.exp(eta[:, 1]), 1.5 * expit(eta[:, 2]) - 1))
        point = genextreme.mean(c=-params[:, 2], loc=params[:, 0], scale=params[:, 1])
        density = genextreme.logpdf(y, c=-params[:, 2], loc=params[:, 0], scale=params[:, 1])
    elif name == "shash":
        params = np.column_stack((eta[:, 0], np.exp(eta[:, 1]) + family.b, eta[:, 2], np.exp(eta[:, 3])))
        mu, sigma, epsilon, delta = params.T
        point = mu + sigma * delta * np.sinh(epsilon / delta)
        z = (y - mu) / (sigma * delta)
        transformed = delta * np.arcsinh(z) - epsilon
        density = norm.logpdf(np.sinh(transformed)) + np.log(np.cosh(transformed)) - np.log(sigma) - .5 * np.log1p(z**2)
    else:
        params = np.column_stack((np.exp(eta[:, 0]), family.a + (family.b - family.a) * expit(eta[:, 1]), np.exp(eta[:, 2])))
        point = params[:, 0]
        density = ldTweedie(y, params[:, 0], p=params[:, 1], phi=params[:, 2], a=family.a, b=family.b)[:, 0]
    return params, point, density


def test_natural_parameters_points_scores_and_component_shapes(fitted_family):
    name, model, frame = fitted_family
    eta = model.predict_link(frame)
    family = model.gam_.family
    assert eta.shape == (len(frame), family.nlp)
    np.testing.assert_array_equal(model.predict(frame, raw=True), eta)
    params, point, density = _independent_distribution(name, eta, frame.y.to_numpy(), family)
    np.testing.assert_allclose(model.predict(frame), params, atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(model.predict_point(frame), point, atol=2e-12, rtol=2e-12)
    assert model.score(frame, frame.y) == pytest.approx(np.mean(density), abs=2e-12, rel=2e-12)
    weights = np.linspace(.5, 2, len(frame))
    assert model.score(frame, frame.y, sample_weight=weights) == pytest.approx(np.average(density, weights=weights), abs=2e-12, rel=2e-12)
    assert model.evaluate(frame, frame.y)["Negative Log-Likelihood"] == pytest.approx(-np.mean(density), abs=2e-12, rel=2e-12)
    components = model.predict_components(frame)
    components.validate_additive_reconstruction()
    np.testing.assert_array_equal(components.link, eta)
    assert components.link.shape == eta.shape
    assert model.standard_errors(frame).shape == params.shape
    assert model.standard_errors(frame, type="link").shape == eta.shape
    if name != "multinom":
        if name == "gumbls":
            jacobian = np.column_stack((np.ones(len(frame)), np.exp(eta[:, 1])))
        elif name == "ziplss":
            et = np.exp(eta[:, 1])
            jacobian = np.column_stack((np.exp(eta[:, 0]), et * np.exp(-et)))
        elif name == "gevlss":
            q = expit(eta[:, 2])
            jacobian = np.column_stack((np.ones(len(frame)), np.exp(eta[:, 1]), 1.5 * q * (1 - q)))
        elif name == "shash":
            jacobian = np.column_stack((np.ones(len(frame)), np.exp(eta[:, 1]), np.ones(len(frame)), np.exp(eta[:, 3])))
        else:
            q = expit(eta[:, 1])
            jacobian = np.column_stack((np.exp(eta[:, 0]), (family.b - family.a) * q * (1 - q), np.exp(eta[:, 2])))
        np.testing.assert_allclose(model.standard_errors(frame), jacobian * model.standard_errors(frame, type="link"), atol=2e-12, rtol=2e-12)
    for term, metadata in components.term_metadata.items():
        assert metadata.parameter in model.predictor_names_
        assert components.terms[term].shape == eta.shape


def test_registered_general_family_persistence_roundtrip(fitted_family, tmp_path):
    name, model, frame = fitted_family
    path = model.save_model(tmp_path / f"{name}.nampy")
    restored = GAMLSS.load_model(path)
    assert restored.predictor_names_ == model.predictor_names_
    assert restored.parameter_names_ == model.parameter_names_
    np.testing.assert_array_equal(
        restored.gam_.fit_result().coef_full, model.gam_.fit_result().coef_full
    )
    np.testing.assert_array_equal(restored.gam_.vcov(), model.gam_.vcov())
    # Pickle preserves fitted arrays exactly, but recomputing predictions can
    # change the final few bits through BLAS memory-layout/alignment choices.
    for method in ("predict", "predict_link", "predict_point", "standard_errors"):
        np.testing.assert_allclose(
            getattr(restored, method)(frame), getattr(model, method)(frame),
            atol=1e-14, rtol=1e-14, err_msg=f"{name} restored {method}",
        )
    restored.predict_components(frame).validate_additive_reconstruction()


@pytest.mark.parametrize("K", [1, 2, 3])
def test_multinom_baseline_components_and_full_coefficient_delta_covariance(K, tmp_path, monkeypatch):
    from nampy.gam.model_state import _predictor_full_indices

    model, frame = _fit("multinom", {"name": "multinom", "K": K}, named=True, K=K)
    new = frame.iloc[[0, 17, 39, 61, 89, len(frame) - 1]]
    eta, probability = model.predict_link(new), model.predict(new)
    assert eta.shape == (len(new), K)
    assert probability.shape == (len(new), K + 1)
    assert model.predictor_names_ == tuple(f"eta{k}" for k in range(1, K + 1))
    assert model.parameter_names_ == tuple(f"p{k}" for k in range(K + 1))
    np.testing.assert_allclose(probability.sum(axis=1), 1, atol=2e-15, rtol=0)
    np.testing.assert_allclose(probability, softmax(np.column_stack((np.zeros(len(new)), eta)), axis=1), atol=2e-15)
    components = model.predict_components(new)
    components.validate_additive_reconstruction()
    np.testing.assert_array_equal(components.link, eta)
    assert components.predictor_names == model.predictor_names_
    assert all(value.shape == eta.shape for value in components.terms.values())

    # Independent coefficient-space delta covariance. Keep every cross-LP
    # covariance, including negative ones: unlike native mgcv prediction,
    # this public natural-parameter uncertainty convention never clamps them.
    P, V = model.gam_.predict(new, type="lpmatrix"), model.gam_.vcov()
    jj = _predictor_full_indices(model.gam_)
    def coefficient_delta_errors(covariance):
        expected = np.zeros_like(probability)
        for row in range(len(new)):
            eta_gradient = np.zeros((K, P.shape[1]))
            for k, columns in enumerate(jj):
                eta_gradient[k, columns] = P[row, columns]
            covariance_eta = eta_gradient @ covariance @ eta_gradient.T
            jacobian = np.empty((K + 1, K))
            for category in range(K + 1):
                for k in range(K):
                    jacobian[category, k] = probability[row, category] * ((category == k + 1) - probability[row, k + 1])
            expected[row] = np.sqrt(np.maximum(np.diag(jacobian @ covariance_eta @ jacobian.T), 0))
        return expected

    np.testing.assert_allclose(model.standard_errors(new), coefficient_delta_errors(V), atol=2e-12, rtol=2e-12)
    if K > 1:
        # Supply an SPD coefficient covariance with a negative cross-predictor
        # intercept covariance. This makes the semantic distinction observable,
        # even when the fitted dataset's own cross-covariances are all positive.
        negative_cross_covariance = np.eye(P.shape[1]) * .03
        first, second = jj[0][0], jj[1][0]
        negative_cross_covariance[first, second] = negative_cross_covariance[second, first] = -.01
        assert np.linalg.eigvalsh(negative_cross_covariance).min() > 0
        monkeypatch.setattr("nampy.gam.predict.predictions.select_prediction_covariance_matrix",
                            lambda *args, **kwargs: negative_cross_covariance)
        public_se = model.standard_errors(new)
        np.testing.assert_allclose(public_se, coefficient_delta_errors(negative_cross_covariance), atol=2e-12, rtol=2e-12)
        _, source_se = model.gam_.family.predict(eta=eta, X=P, jj=jj, coef=np.zeros(P.shape[1]),
                                                se=True, Vb=negative_cross_covariance)
        assert np.max(np.abs(public_se - source_se)) > 1e-5
    np.testing.assert_array_equal(model.predict_point(new), np.argmax(probability, axis=1))
    path = model.save_model(tmp_path / f"multinom-{K}.nampy")
    restored = GAMLSS.load_model(path)
    np.testing.assert_allclose(
        restored.standard_errors(new), model.standard_errors(new),
        atol=1e-14, rtol=1e-14,
    )


def test_single_predictor_general_multinom_is_not_a_regression_family():
    frame, _, _ = _case("multinom")
    with pytest.raises(ValueError, match="GAMLSS"):
        GAMRegressor(family={"name": "multinom", "K": 1}, optimize_smoothing=False).fit(frame[["x"]], frame.y)


@pytest.mark.parametrize("na_action", ["omit", "exclude"])
def test_multinom_natural_standard_errors_align_missing_offset_rows(na_action):
    model, frame = _fit("multinom", {"name": "multinom", "K": 2}, K=2)
    new = frame.iloc[[0, 17, 39, 61, 89, len(frame) - 1]]
    offsets = [np.array([.1, np.nan, -.2, .3, .1, -.1]),
               np.array([-.1, .2, .1, -.2, np.nan, .3])]
    retained = np.isfinite(np.column_stack(offsets)).all(axis=1)
    expected = model.standard_errors(new.iloc[np.flatnonzero(retained)],
                                     offset=[value[retained] for value in offsets])
    actual = model.standard_errors(new, offset=offsets, na_action=na_action)
    # Like the underlying GAM prediction API, both omit and exclude return
    # only the retained rows for explicit newdata; pass reinserts missing rows.
    assert actual.shape == (int(retained.sum()), 3)
    np.testing.assert_array_equal(actual, expected)
