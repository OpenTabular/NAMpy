"""Seven-stage contracts for family gaps identified in the 2026-10-28 history audit.

Earlier ports remain in test_gam_seven_stage_pipeline_contracts and
 test_remaining_family_pipeline_contracts. mgcv cases use frozen R captures;
new joint families use independent SciPy likelihood optimization, alongside
existing gamlss.dist density/MLE and torch fourth-derivative references.
"""

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize
from scipy.optimize._numdiff import approx_derivative
from scipy.special import expit
from scipy.stats import beta as beta_distribution
from scipy.stats import nbinom, t

from nampy import GAMLSS, GAMClassifier, GAMRegressor
from nampy.gam import GAM
from nampy.gam.families.registry import GENERAL_FAMILIES, make_gam_family
from nampy.gam.fit.state import FitCoreSolution
from tests.parity._registered_pipeline_reference import pipeline_reference
from tests.parity.test_remaining_family_pipeline_contracts import (
    test_remaining_stage_2_runtime_basis_ownership as check_runtime,
)
from tests.parity.test_remaining_family_pipeline_contracts import (
    test_remaining_stage_3_constructed_basis_penalty_pair as check_construction,
)
from tests.parity.test_remaining_family_pipeline_contracts import (
    test_remaining_stage_4_predictor_blocks_and_smoothing_maps as check_compilation,
)
from tests.parity.test_remaining_family_pipeline_contracts import (
    test_remaining_stage_5_identifiability_and_prediction_pairing as check_identification,
)

MGCV_FAMILIES = (
    "inverse_gaussian",
    "quasipoisson",
    "quasibinomial",
    "binomial",
    "poisson",
    "gamma",
    "nb",
    "negbin",
    "tw",
    "betar",
    "ocat",
    "gaulss",
    "gammals",
)
JOINT_FAMILIES = ("student_lss", "beta_lss", "nb_lss")
EARLIER_GENERAL = {"gumbls", "ziplss", "multinom", "gevlss", "shash", "twlss"}
pytestmark = [pytest.mark.surface_regression]


@dataclass
class FamilyCase:
    name: str
    data: pd.DataFrame
    newdata: pd.DataFrame
    formulas: list
    smoothing_params: list
    model: GAM
    estimator: object
    expected: dict | None


def family_inputs(name):
    rng = np.random.default_rng(281027)
    x = np.linspace(-1, 1, 140)
    mu = np.exp(0.3 + 0.4 * x)
    probability = expit(0.3 + 0.8 * x)
    if name in JOINT_FAMILIES:
        path = (
            Path(__file__).parents[1] / "reference_fixtures/extensions/neighbors.json"
        )
        source = json.loads(path.read_text())["inputs"]["families"]
        x, y = np.array(source["x"]), np.array(source[name])
    elif name == "inverse_gaussian":
        y = rng.wald(mu, 4)
    elif name in {"quasipoisson", "poisson"}:
        y = rng.poisson(mu)
    elif name in {"binomial", "quasibinomial"}:
        y = rng.binomial(1, probability)
    elif name in {"gamma", "gammals"}:
        y = rng.gamma(4, mu / 4)
    elif name in {"nb", "negbin"}:
        y = rng.negative_binomial(2, 2 / (2 + mu))
    elif name == "tw":
        y = rng.gamma(2, mu / 2)
        y[::5] = 0
    elif name == "betar":
        y = rng.beta(8 * probability, 8 * (1 - probability))
    elif name == "ocat":
        y = np.digitize(0.3 + x + rng.logistic(size=len(x)), [-0.7, 0.8]) + 1
    else:
        y = 0.3 + np.sin(2 * x) + rng.normal(0, np.exp(-0.4 + 0.1 * x))
    frame = pd.DataFrame({"x": x, "y": y, "off": 0.03 * x, "w": 0.8 + (x + 1) / 2})
    general = name in GENERAL_FAMILIES
    formulas = ["y ~ s(x,bs='cr',k=6) + offset(off)"]
    if general:
        formulas.append("~ s(x,bs='cr',k=5) + offset(off)")
    options = {
        "nb": {"name": "nb", "theta": 2.0},
        "negbin": {"name": "negbin", "theta": 2.0},
        "tw": {"name": "tw", "theta": 1.5},
        "betar": {"name": "betar", "theta": 8.0},
        "ocat": {"name": "ocat", "theta": [1.5]},
    }
    source = {
        "inverse_gaussian": "inverse.gaussian()",
        "nb": "nb(theta=2)",
        "negbin": "negbin(theta=2)",
        "tw": "tw(theta=1.5)",
        "betar": "betar(theta=8)",
        "ocat": "ocat(theta=1.5)",
        "gamma": "Gamma()",
    }.get(name, name + "()")
    newdata = pd.DataFrame(
        {"x": np.linspace(-0.93, 0.93, 13), "off": np.linspace(0.07, -0.02, 13)}
    )
    return frame, newdata, formulas, options.get(name, name), source


@pytest.fixture(scope="module", params=MGCV_FAMILIES + JOINT_FAMILIES)
def family_pipeline(request):
    name = request.param
    frame, newdata, formulas, family, source = family_inputs(name)
    sp = [0.4] * len(formulas)
    kwargs = {
        "family": family,
        "formula": formulas if len(formulas) > 1 else formulas[0],
        "smoothing_params": sp,
        "optimize_smoothing": False,
        "smoothing_method": "fixed",
        "max_irls_iter": 200,
        "irls_tol": 1e-10,
    }
    # Conditional fixed-scale Tweedie parity must use the same scale on both
    # sides: default fixed-SP Pearson scale differs from R's REML profile.
    if name == "tw":
        kwargs["scale"] = 1.0
    # Ordered-category probabilities are a core GAM interface; the regression
    # and binary classifier adapters deliberately reject this family.
    if name == "ocat":
        estimator = None
        model = GAM(**kwargs).fit(data=frame, sample_weight=frame.w.to_numpy())
    else:
        cls = (
            GAMLSS
            if name in GENERAL_FAMILIES
            else GAMClassifier
            if name in {"binomial", "quasibinomial"}
            else GAMRegressor
        )
        estimator = cls(**kwargs).fit(
            frame, frame.y.to_numpy(), sample_weight=frame.w.to_numpy()
        )
        model = estimator.gam_
    expected = (
        None
        if name in JOINT_FAMILIES
        else pipeline_reference(
            name, frame, newdata, formulas, source, scale=kwargs.get("scale")
        )
    )
    return FamilyCase(name, frame, newdata, formulas, sp, model, estimator, expected)


def test_general_registry_has_assigned_seven_stage_coverage():
    inventory = json.loads(Path(__file__).with_name("seven_stage_coverage.json").read_text())
    extensions = inventory["extension_family_test_modules"]
    assert set(GENERAL_FAMILIES) == EARLIER_GENERAL | {"gaulss", "gammals"} | set(
        JOINT_FAMILIES
    ) | set(extensions)
    assert set(extensions) == set(inventory["family_groups"]["gamlss_extensions"])
    assert all(
        modules and set(modules) <= set(inventory["test_modules"])
        for modules in extensions.values()
    )


def test_stage_1_family_formula_predictors_and_support(family_pipeline):
    case = family_pipeline
    assert case.model.formula_.nlp == len(case.formulas)
    assert len(case.model.predictor_specs) == len(case.formulas)
    assert case.model.formula_.response_name == "y"
    case.model.family.validate_y(case.data.y.to_numpy())
    for spec in case.model.predictor_specs:
        assert spec.offset_name == "off" and spec.has_intercept
        assert spec.terms[0].smooth_spec.bs == "cr"
    np.testing.assert_array_equal(case.model.prior_weights_, case.data.w)


def test_stage_2_family_runtime_basis(family_pipeline):
    check_runtime(family_pipeline)


def test_stage_3_family_constructed_penalty_pairs(family_pipeline):
    check_construction(family_pipeline)


def test_stage_4_family_predictor_and_smoothing_maps(family_pipeline):
    check_compilation(family_pipeline)


def test_stage_5_family_identified_prediction_maps(family_pipeline):
    check_identification(family_pipeline)


def _joint_reference(case):
    """Optimize a SciPy density, never calling NAMpy likelihood/derivative code."""
    model = case.model
    X = model.lpmatrix(case.data)
    indices = model.gam_result_.compiled_model.predictor_full_indices
    penalty = np.zeros((X.shape[1], X.shape[1]))
    compiled = model.gam_result_.compiled_model
    for block in compiled.compiled_penalties:
        jj = compiled.coef_reduced_to_full_idx[block.coef_slice]
        penalty[np.ix_(jj, jj)] += 0.4 * block.matrix
    y = case.data.y.to_numpy()

    def objective(beta):
        eta = np.column_stack([X[:, jj] @ beta[jj] + case.data.off for jj in indices])
        first, second = eta.T
        if case.name == "student_lss":
            log_density = t.logpdf(y, 5, loc=first, scale=np.exp(second))
        elif case.name == "beta_lss":
            mu, phi = expit(first), np.exp(second)
            log_density = beta_distribution.logpdf(y, mu * phi, (1 - mu) * phi)
        else:
            mu, alpha = np.exp(first), np.exp(second)
            log_density = nbinom.logpmf(y, 1 / alpha, 1 / (1 + alpha * mu))
        return -float(case.data.w @ log_density) + 0.5 * beta @ penalty @ beta

    def gradient(beta):
        return approx_derivative(
            lambda b: np.array([objective(b)]), beta, method="3-point"
        ).ravel()

    solution = minimize(
        objective,
        np.zeros(X.shape[1]),
        jac=gradient,
        method="BFGS",
        options={"gtol": 2e-7, "maxiter": 300},
    )
    assert np.linalg.norm(gradient(solution.x), ord=np.inf) < 1e-5, solution.message
    information = approx_derivative(
        gradient, solution.x, method="3-point", rel_step=1e-3
    )
    return (
        solution.x,
        np.linalg.inv((information + information.T) / 2),
        objective,
        penalty,
    )


def test_stage_6_family_fit_matches_independent_reference(family_pipeline):
    case = family_pipeline
    model = case.model
    solution = model.gam_result_.fit_core_solution
    assert isinstance(solution, FitCoreSolution)
    fit = solution.fit_result
    assert fit.converged
    assert np.isfinite(fit.coef_full).all() and np.isfinite(fit.cov_bayes).all()
    X = model.lpmatrix(case.data)
    indices = model.gam_result_.compiled_model.predictor_full_indices
    eta = np.column_stack(
        [X[:, jj] @ fit.coef_full[jj] + case.data.off for jj in indices]
    )
    np.testing.assert_allclose(
        fit.eta, eta if eta.shape[1] > 1 else eta[:, 0], atol=2e-9, rtol=2e-9
    )
    if case.expected is not None:
        from tests.parity._pipeline_invariants import assert_reference_invariants

        tolerance = 5e-6 if case.name in {"gaulss", "gammals", "ocat"} else 2e-7
        assert_reference_invariants(
            model, case.data, case.newdata, case.expected, tolerance=tolerance
        )
    else:
        beta, covariance, objective, penalty = _joint_reference(case)
        from tests.parity._pipeline_invariants import predictor_evaluation

        E = predictor_evaluation(model, case.data)
        np.testing.assert_allclose(
            objective(fit.coef_full), objective(beta), atol=1e-8, rtol=0
        )
        np.testing.assert_allclose(E @ fit.coef_full, E @ beta, atol=2e-5, rtol=2e-5)
        np.testing.assert_allclose(
            E @ fit.cov_bayes @ E.T, E @ covariance @ E.T, atol=3e-5, rtol=3e-5
        )

        expected_ll = -objective(beta) + 0.5 * beta @ penalty @ beta
        expected_edf = np.trace(np.eye(len(beta)) - penalty @ covariance)
        np.testing.assert_allclose(model.loglik(), expected_ll, atol=1e-5, rtol=1e-6)
        np.testing.assert_allclose(fit.edf, expected_edf, atol=3e-5, rtol=3e-5)
        np.testing.assert_allclose(
            model.aic(), -2 * expected_ll + 2 * expected_edf, atol=2e-4, rtol=2e-5
        )
        np.testing.assert_allclose(
            model.bic(),
            -2 * expected_ll + np.log(len(case.data)) * expected_edf,
            atol=2e-4,
            rtol=2e-5,
        )


def test_stage_7_family_outputs_domains_and_persistence(family_pipeline, tmp_path):
    case = family_pipeline
    link, se = case.model.predict(case.newdata, type="link", return_se=True)
    response = case.model.predict(case.newdata, type="response")
    assert np.isfinite(response).all() and np.isfinite(link).all()
    if case.expected is not None:
        tolerance = 5e-6 if case.name in {"gaulss", "gammals", "ocat"} else 2e-7
        np.testing.assert_allclose(
            link, case.expected["eta"], atol=tolerance, rtol=tolerance
        )
        np.testing.assert_allclose(
            se, case.expected["se"], atol=tolerance, rtol=tolerance
        )
        np.testing.assert_allclose(
            response, case.expected["response"], atol=tolerance, rtol=tolerance
        )
    X = case.model.lpmatrix(case.newdata)
    for i, jj in enumerate(
        case.model.gam_result_.compiled_model.predictor_full_indices
    ):
        variance = np.einsum(
            "ij,jk,ik->i", X[:, jj], case.model.vcov()[np.ix_(jj, jj)], X[:, jj]
        )
        np.testing.assert_allclose(
            np.asarray(se)[:, i] ** 2
            if len(case.formulas) > 1
            else np.asarray(se) ** 2,
            variance,
            atol=2e-9,
            rtol=2e-9,
        )
    if case.name in {"binomial", "quasibinomial", "ocat"}:
        assert ((response >= 0) & (response <= 1)).all()
        if case.name == "ocat":
            np.testing.assert_allclose(response.sum(axis=1), 1, atol=1e-12)
    if case.name in JOINT_FAMILIES:
        params = case.estimator.predict(case.newdata)
        assert (params[:, 1] > 0).all()
        if case.name == "beta_lss":
            assert ((params[:, 0] > 0) & (params[:, 0] < 1)).all()
        elif case.name == "nb_lss":
            assert (params[:, 0] > 0).all()
        make_gam_family(case.name).validate_y(case.data.y.to_numpy())
    if case.estimator is not None:
        case.estimator.predict_components(
            case.newdata
        ).validate_additive_reconstruction()
        path = case.estimator.save_model(tmp_path / (case.name + ".nampy"))
        restored = type(case.estimator).load_model(path)
        np.testing.assert_array_equal(
            restored.predict(case.newdata), case.estimator.predict(case.newdata)
        )
        restored.predict_components(case.newdata).validate_additive_reconstruction()
    else:
        restored = pickle.loads(pickle.dumps(case.model))
        np.testing.assert_array_equal(
            restored.predict(case.newdata, type="response"), response
        )


# Reuse the established seven-stage fixtures while extending their invariant
# output checks to likelihood criteria, covariance kernels and residuals.


@pytest.fixture(
    scope="module", params=["gaussian", "quasi", "scat", "zip", "cnorm", "clog"]
)
def established_single_case(request):
    from tests.parity.test_gam_seven_stage_pipeline_contracts import pipeline_case

    return pipeline_case.__wrapped__(request)


@pytest.fixture(
    scope="module",
    params=["cpois", "bcg", "gumbls", "ziplss", "multinom", "gevlss", "shash", "twlss"],
)
def established_general_case(request):
    from tests.parity.test_remaining_family_pipeline_contracts import family_pipeline

    return family_pipeline.__wrapped__(request)


PRIOR_SOURCE_FAMILIES = {
    "gaussian": "gaussian()",
    "quasi": "quasi(link='identity',variance='constant')",
    "scat": "scat(theta=c(5,.3))",
    "zip": "ziP(theta=c(-.5,log(.5)),b=.3)",
    "cnorm": "cnorm(theta=1)",
    "clog": "clog(theta=1)",
    "cpois": "cpois()",
    "bcg": "local({f<-bcg(theta=c(.7,.8));f$putTheta(c(.7,log(.8)));f})",
    "gumbls": "gumbls(b=-3)",
    "ziplss": "ziplss()",
    "multinom": "multinom(K=2)",
    "gevlss": "gevlss()",
    "shash": "shash()",
    "twlss": "twlss()",
}


def check_established_family(name, model, data, newdata, formulas):
    from tests.parity._pipeline_invariants import assert_reference_invariants

    frame = data.copy()
    frame["w"] = (
        np.ones(len(frame)) if model.prior_weights_ is None else model.prior_weights_
    )
    expected = pipeline_reference(
        name,
        frame,
        newdata,
        formulas,
        PRIOR_SOURCE_FAMILIES[name],
        sp=model.smoothing_params,
    )
    tolerance = {"gumbls": 3e-5, "ziplss": 3e-5, "multinom": 4e-6, "gevlss": 5e-6}.get(
        name, 2e-7
    )
    assert_reference_invariants(model, frame, newdata, expected, tolerance=tolerance)


def test_established_single_predictor_family_invariants(established_single_case):
    case = established_single_case
    check_established_family(
        case.gam.family.name, case.gam, case.data, case.newdata, [case.gam.formula]
    )


def test_established_multi_predictor_family_invariants(established_general_case):
    case = established_general_case
    check_established_family(
        case.name, case.model, case.data, case.newdata, case.formulas
    )


def test_history_inventory_accounts_for_registered_features():
    from nampy.gam.basis_registry import basis_descriptors

    inventory = json.loads(
        Path(__file__).with_name("seven_stage_coverage.json").read_text()
    )
    assert len(inventory["stages"]) == 7
    assert set(inventory["smooth_codes"]) == {d.name for d in basis_descriptors()}
    documented = [
        name for names in inventory["family_groups"].values() for name in names
    ]
    assert len(documented) == len(set(documented))
    assert set(documented) == set(inventory["extension_family_test_modules"]) | set(MGCV_FAMILIES + JOINT_FAMILIES) | set(
        PRIOR_SOURCE_FAMILIES
    )
    # Include ordinary/extended constructors too: checking only the general
    # registry lets a newly imported single-predictor family escape this gate.
    import inspect

    from nampy.gam.families import registry
    from nampy.gam.families.family_base import BaseFamily

    constructor_names = {
        family.name
        for family in vars(registry).values()
        if inspect.isclass(family)
        and family is not BaseFamily
        and issubclass(family, BaseFamily)
    }
    # nb and negbin share a constructor but have distinct theta policies.
    assert set(documented) == constructor_names | set(GENERAL_FAMILIES) | {"nb"}
    root = Path(__file__).resolve().parents[2]
    assert all((root / name).is_file() for name in inventory["test_modules"])
