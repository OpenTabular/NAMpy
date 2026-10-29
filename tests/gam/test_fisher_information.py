"""Likelihood information in the actual statistical coefficient coordinates."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy import integrate

from nampy import GAMLSS
from nampy.diagnostics.fisher import FisherDiagnostics
from nampy.gam import GAM
from nampy.gam.families.binomial import (
    BinomialCloglogFamily,
    BinomialLogitFamily,
    BinomialProbitFamily,
)
from nampy.gam.families.gamlss.distributional import (
    BetaLocationPrecision,
    StudentLocationScale,
)
from nampy.gam.families.gamlss.gammals import GammalsFamily
from nampy.gam.families.gamlss.gaulss import GaulssFamily
from nampy.gam.families.gamlss.positive import (
    LogNormalLocationScale,
    WeibullLocationShape,
)
from nampy.gam.families.registry import make_gam_family
from nampy.gam.fisher import (
    _general_derivatives,
    _general_information,
    _glm_state,
    statistical_fisher_information,
    validate_fisher_family,
)


def config(**kw):
    return FisherDiagnostics(n_inputs=10000, random_state=17, **kw)


@pytest.mark.parametrize("family", ["gaussian", "poisson", "binomial", "gamma"])
def test_glm_mean_information_matches_expected_working_system(family):
    x = np.linspace(-0.8, 0.8, 40)
    X = x[:, None]
    mu = np.exp(0.2 + 0.3 * x)
    y = (
        0.4 + x + 0.2 * np.cos(np.arange(40))
        if family == "gaussian"
        else np.round(mu * 3)
        if family == "poisson"
        else np.resize([0.0, 1.0, 1.0, 0.0, 1.0], 40)
        if family == "binomial"
        else mu * (1 + 0.1 * np.cos(np.arange(40)))
    )
    model = GAM(family=family, k=5, optimize_smoothing=False).fit(X, y)
    result = statistical_fisher_information(model, config=config())
    design = model.lpmatrix(None)
    eta = model.predict(None, type="link")
    mu = model.family.inverse_link(eta)
    weight = model.family.mu_eta(eta) ** 2 / (
        model.family.variance(mu) * model.fit_result().scale
    )
    assert_allclose(
        result.diagonal, np.mean(design**2 * weight[:, None], axis=0), atol=1e-11
    )
    assert result.normalization == "mean"
    assert sum(group.trace for group in result.groups) == pytest.approx(
        result.diagonal.sum()
    )


def test_quasi_is_rejected_by_likelihood_capability_before_fit():
    for name in ["quasi", "quasipoisson", "quasibinomial"]:
        with pytest.raises(NotImplementedError, match="Quasi-likelihood"):
            validate_fisher_family(make_gam_family(name), config())


def test_public_offset_alignment_and_missing_required_offset():
    X = pd.DataFrame({"x": np.linspace(-1, 1, 30)})
    offset = np.linspace(-0.5, 0.5, 30)
    model = GAM(formula="y~x", family="poisson", optimize_smoothing=False).fit(
        data=X.assign(y=np.resize([0, 1, 2, 1, 3], 30)), offset=offset
    )
    model._fisher_offset_required_ = True
    with pytest.raises(ValueError, match="offset"):
        statistical_fisher_information(model, X, config=config())
    selected = np.array([5, 2, 13, 20])
    got = statistical_fisher_information(
        model, X.iloc[selected], offset=offset[selected], config=config()
    )
    design = model.lpmatrix(X.iloc[selected])
    mu = model.predict(X.iloc[selected], type="response", offset=offset[selected])
    assert_allclose(got.diagonal, np.mean(design**2 * mu[:, None], axis=0))
    with pytest.raises(ValueError, match="offset|length|shape|rows"):
        statistical_fisher_information(
            model, X.iloc[selected], offset=offset, config=config()
        )


def test_public_rows_and_internal_probe_budget():
    X = np.linspace(-0.8, 0.8, 32)[:, None]
    model = GAM(family="poisson", k=4, optimize_smoothing=False).fit(
        X, np.resize([1, 2, 1, 3], 32)
    )
    budget = FisherDiagnostics(n_inputs=4, random_state=17)
    public = statistical_fisher_information(model, X, config=budget)
    probe = statistical_fisher_information(model, X, config=budget, probe=True)
    assert public.n_inputs == len(X)
    assert probe.n_inputs == 4
    indices = np.sort(np.random.default_rng(17).choice(len(X), 4, replace=False))
    truth = statistical_fisher_information(model, X[indices], config=budget)
    assert_allclose(probe.diagonal, truth.diagonal)


def test_finite_population_reference_and_probe_validation():
    X = np.linspace(-0.8, 0.8, 24)[:, None]
    model = GAM(family="poisson", k=4, optimize_smoothing=False).fit(
        X, np.resize([1, 2, 1, 3], len(X))
    )
    census = statistical_fisher_information(
        model, X, config=config(input_sampling="without_replacement")
    )
    assert_allclose(census.input_estimator_variance, 0)
    sampled = config(input_sampling="without_replacement", population_size=100)
    subset = statistical_fisher_information(model, X, config=sampled)
    assert np.any(subset.input_estimator_variance > 0)
    with pytest.raises(ValueError, match="population_size"):
        statistical_fisher_information(model, config=sampled, probe=True)


def test_training_formula_and_separate_offsets_are_both_retained():
    X = pd.DataFrame({"x": np.linspace(-1, 1, 30), "z": np.linspace(0.1, 0.4, 30)})
    offset = np.linspace(-0.3, 0.2, 30)
    model = GAM(
        formula="y ~ x + offset(z)", family="poisson", optimize_smoothing=False
    ).fit(data=X.assign(y=np.resize([1, 2, 1, 3, 2], 30)), offset=offset)
    training = statistical_fisher_information(
        model, offset=offset.tolist(), config=config()
    )
    explicit = statistical_fisher_information(model, X, offset=offset, config=config())
    assert_allclose(training.diagonal, explicit.diagonal)
    prediction = model.predict(X, type="response", offset=offset)
    assert_allclose(
        training.diagonal,
        np.mean(model.lpmatrix(X) ** 2 * prediction[:, None], axis=0),
    )


@pytest.mark.parametrize(
    "family_type,point",
    [(BinomialLogitFamily, 31), (BinomialProbitFamily, 12), (BinomialCloglogFamily, 8)],
)
def test_solver_clipped_links_do_not_claim_likelihood_information(family_type, point):
    family = family_type()
    with pytest.raises(ValueError, match="solver-clipped"):
        _glm_state(family, point, 1.0)


def test_grouped_binomial_uses_one_normalized_binomial_likelihood_per_row():
    X = np.linspace(-0.8, 0.8, 24)[:, None]
    successes = np.resize([1, 3, 2, 4], len(X))
    trials = np.resize([5, 6, 7], len(X))
    model = GAM(family="binomial", k=4, optimize_smoothing=False).fit(
        X, np.column_stack([successes, trials - successes])
    )
    result = statistical_fisher_information(model, config=config())
    design = model.lpmatrix(None)
    mu = model.predict(None, type="response")
    assert_allclose(
        result.diagonal,
        np.mean(design**2 * (trials * mu * (1 - mu))[:, None], axis=0),
    )
    with pytest.raises(ValueError, match="trials"):
        statistical_fisher_information(model, X[:1], config=config())
    sample = statistical_fisher_information(
        model,
        X[:1],
        context={"trials": trials[:1]},
        config=config(estimator="score_outer_product", n_draws=3000),
    )
    truth = statistical_fisher_information(
        model, X[:1], context={"trials": trials[:1]}, config=config()
    )
    assert_allclose(sample.diagonal, truth.diagonal, rtol=0.1, atol=0.02)


@pytest.mark.parametrize(
    "family_type,point",
    [
        (GaulssFamily, [0.2, -0.3]),
        (GammalsFamily, [0.2, -0.3]),
        (BetaLocationPrecision, [0.2, 1.6]),
        (StudentLocationScale, [0.2, -0.3]),
        (LogNormalLocationScale, [0.2, -0.3]),
        (WeibullLocationShape, [0.2, 0.3]),
    ],
)
def test_general_exact_information_matches_integrated_analytic_hessian(
    family_type, point
):
    family = family_type()
    point = np.array(point)
    parameters = family.distribution_parameters_from_eta(point[None, :])
    from nampy.gam.predict.distribution_adapter import distribution_prediction

    prediction = distribution_prediction(family, parameters)
    expected = _general_information(family, point[None, :])[0]
    lower = -np.inf if family_type in {GaulssFamily, StudentLocationScale} else 0.0
    upper = 1.0 if family_type is BetaLocationPrecision else np.inf
    for i, j in np.ndindex(2, 2):

        def integrand(y, i=i, j=j):
            return -_general_derivatives(family, point, y)[1][i, j] * np.exp(
                prediction.log_prob([y])[0]
            )

        value, error = integrate.quad(integrand, lower, upper, epsabs=1e-8)
        assert error < 1e-6
        assert_allclose(value, expected[i, j], atol=1e-7, rtol=1e-7)


def test_normal_lss_expected_design_and_sampled_estimators():
    rng = np.random.default_rng(7)
    X = pd.DataFrame({"x": np.linspace(-1, 1, 60)})
    y = 0.3 + 0.2 * X.x.to_numpy() + rng.normal(size=60) * 0.5
    model = (
        GAMLSS(
            formula={"mu": "y~x", "sigma": "~1"},
            family="normal",
            optimize_smoothing=False,
        )
        .fit(X.assign(y=y))
        .gam_
    )
    exact = statistical_fisher_information(model, config=config())
    from nampy.gam.model_state import _predictor_full_indices

    design = model.lpmatrix(None)
    eta = model.predict(None, type="link")
    metric = _general_information(model.family, eta)
    expected = np.zeros(design.shape[1])
    for k, indices in enumerate(_predictor_full_indices(model)):
        expected[indices] += np.mean(
            design[:, indices] ** 2 * metric[:, k, k, None], axis=0
        )
    assert_allclose(exact.diagonal, expected)
    # One row is enough to independently exercise each output derivative.
    for estimator in ["score_outer_product", "negative_hessian"]:
        sample = statistical_fisher_information(
            model, X.iloc[:1], config=config(estimator=estimator, n_draws=3000)
        )
        truth = statistical_fisher_information(model, X.iloc[:1], config=config())
        assert_allclose(sample.diagonal, truth.diagonal, rtol=0.16, atol=0.1)
        assert sample.conditional_standard_error is not None


def test_shared_predictor_coefficient_uses_compiled_mapping_once():
    from nampy.gam.model_state import _predictor_full_indices

    rng = np.random.default_rng(32)
    X = pd.DataFrame({"x": np.linspace(-1, 1, 60)})
    data = X.assign(y=rng.normal(0.2 * X.x, 0.7, len(X)))
    model = GAM(
        family="gaulss",
        formula=["y~1", "~1", "1 + 2 ~ x - 1"],
        optimize_smoothing=False,
    ).fit(data=data)
    result = statistical_fisher_information(model, config=config())
    blocks = _predictor_full_indices(model)
    shared = np.intersect1d(*blocks)
    assert shared.size == 1
    design = model.lpmatrix(None)
    metric = _general_information(model.family, model.predict(None, type="link"))
    expected = np.zeros(design.shape[1])
    for row in range(len(X)):
        jacobian = np.zeros((2, design.shape[1]))
        for k, indices in enumerate(blocks):
            jacobian[k, indices] = design[row, indices]
        expected += np.diag(jacobian.T @ metric[row] @ jacobian) / len(X)
    assert_allclose(result.diagonal, expected)
    assert sum(group.n_parameters for group in result.groups) == design.shape[1]


@pytest.mark.parametrize("estimator", ["score_outer_product", "negative_hessian"])
def test_glm_sampled_estimators_agree_with_exact_and_preserve_rng(estimator):
    X = np.linspace(-0.8, 0.8, 32)[:, None]
    model = GAM(family="poisson", k=4, optimize_smoothing=False).fit(
        X, np.resize([1, 2, 1, 3], 32)
    )
    kwargs = {"estimator": estimator, "n_draws": 3000}
    sample = statistical_fisher_information(model, X[:1], config=config(**kwargs))
    exact = statistical_fisher_information(model, X[:1], config=config())
    assert_allclose(sample.diagonal, exact.diagonal, rtol=0.15, atol=0.02)
    if estimator == "negative_hessian":
        assert_allclose(sample.diagonal, exact.diagonal, rtol=1e-12, atol=1e-12)
        assert_allclose(sample.estimator_variance, 0, atol=1e-25)


def test_correlated_sequence_and_sparse_design_guards(monkeypatch):
    from scipy import sparse

    X = np.linspace(-0.8, 0.8, 32)[:, None]
    y = np.sin(X[:, 0]) + 0.1 * np.cos(np.arange(32))
    model = GAM(k=4, ar1_rho=0.3, optimize_smoothing=False).fit(X, y)
    exact = statistical_fisher_information(model, config=config())
    assert np.isfinite(exact.diagonal).all()
    with pytest.raises(NotImplementedError, match="entire fitted sequence"):
        statistical_fisher_information(model, X, config=config())
    plain = GAM(k=4, optimize_smoothing=False).fit(X, y)
    expected = statistical_fisher_information(plain, X, config=config())
    matrix = plain.lpmatrix(X)
    monkeypatch.setattr(plain, "lpmatrix", lambda X, *, block_size=None: sparse.csr_matrix(matrix))
    got = statistical_fisher_information(plain, X, config=config())
    assert_allclose(got.diagonal, expected.diagonal)
