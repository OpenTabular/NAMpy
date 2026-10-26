"""Quasi-likelihood workflow references from actual mgcv; normal tests need no R."""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from nampy.gam.model_state import _coef_full, _edf_total, _fit_scale
from tests.mgcv_parity_utils import _build_r_command
from tests.reference_fixtures import load_reference, reference_key, save_reference

pytestmark = pytest.mark.surface_regression


def _case(
    family="quasipoisson",
    method="fixed",
    link=None,
    variance=None,
    zero=False,
    multiple=False,
    rank_deficient=False,
):
    rng = np.random.default_rng(1729)
    x = np.linspace(-1, 1, 90)
    z = rng.uniform(-1, 1, len(x))
    off = 0.07 * np.cos(2 * x)
    mean = np.exp(0.4 + 0.7 * np.sin(3 * x) + 0.2 * z)
    if family == "quasibinomial" or variance == "mu(1-mu)":
        mean = 1 / (1 + np.exp(-0.6 * np.sin(3 * x) - 0.3 * z))
        y = rng.binomial(8, mean) / 8
    elif variance == "constant":
        y = 0.9 + 0.4 * np.sin(3 * x) + rng.normal(0, 0.2, len(x))
    elif variance in {"mu^2", "mu^3"}:
        y = rng.gamma(4, mean / 4)
    else:
        y = rng.negative_binomial(3, 3 / (3 + mean)).astype(float)
    w = 0.5 + (np.arange(len(x)) % 4) * 0.5
    if zero:
        w[[0, 17, 44]] = 0
    formula = "y~s(x,bs='cr',k=7)+offset(off)"
    if multiple:
        formula += "+s(z,bs='cr',k=6)"
    if rank_deficient:
        formula += "+z+I(2*z)"
    if link is None:
        link = (
            "logit"
            if family == "quasibinomial" or variance == "mu(1-mu)"
            else "identity"
            if variance == "constant"
            else "log"
        )
    return {
        "family": family,
        "method": method,
        "link": link,
        "variance": variance,
        "formula": formula,
        "sp": [0.4, 0.7] if multiple else [0.4],
        "data": {
            "x": x.tolist(),
            "z": z.tolist(),
            "off": off.tolist(),
            "y": y.tolist(),
            "w": w.tolist(),
        },
    }


_SCRIPT = r"""
suppressPackageStartupMessages(library(mgcv));library(jsonlite)
a<-commandArgs(TRUE);p<-fromJSON(a[1]);d<-as.data.frame(p$data)
fam<-if(p$family=='quasi') do.call(quasi,list(link=if(is.null(p$power)) p$link else 'identity',variance=p$variance)) else get(p$family)(link=p$link)
if(!is.null(p$power)) fam<-eval(substitute(quasi(link=power(P),variance=V),list(P=p$power,V=p$variance)))
args<-list(formula=as.formula(p$formula),data=d,family=fam,weights=d$w,
 method=if(p$method=='fixed') 'REML' else if(p$method %in% c('GCV','UBRE')) 'GCV.Cp' else p$method)
if(p$method=='fixed') args$sp<-p$sp
if(!is.null(p$scale)) args$scale<-p$scale
if(isTRUE(p$joint_trace) || isTRUE(p$pearson_trace)) trace('gam.fit3',where=asNamespace('mgcv'),print=FALSE,exit=quote({
 if(deriv==2 && scoreType %in% c('REML','ML','P-REML','P-ML') && !exists('.joint_snapshot',envir=.GlobalEnv,inherits=FALSE)) {
  assign('.joint_snapshot',list(log_sp=as.numeric(sp),log_phi=log(scale),value=as.numeric(REML),grad=as.numeric(REML1),hess=REML2),envir=.GlobalEnv)
 }
}))
b<-do.call(gam,args)
nd<-data.frame(x=seq(-.9,.9,length.out=12),z=seq(.8,-.8,length.out=12),off=.13)
pr<-predict(b,nd,se.fit=TRUE)
res<-list(coef=coef(b),edf=sum(b$edf),scale=b$sig2,dispersion=summary(b)$dispersion,
 Vp=b$Vp,Ve=b$Ve,sp=b$sp,fitted=fitted(b),prediction=pr$fit,se=pr$se.fit,
 score=as.numeric(b$gcv.ubre),deviance=deviance(b),aic=AIC(b),loglik=as.numeric(logLik(b)),
 y=b$y,weights=b$prior.weights,
 residuals=lapply(c('deviance','pearson','working','response'),function(t) residuals(b,type=t)))
if(isTRUE(p$joint_trace) || isTRUE(p$pearson_trace)) res$joint <- .joint_snapshot
if(isTRUE(p$working_score)) res$working_score <- b$working.weights * residuals(b,type='working')
if(isTRUE(p$inference)) {
 su<-summary(b); an<-anova(b)
 res$inference<-list(p_table=su$p.table,pterms=su$pTerms.table,smooth=su$s.table,
  anova_parametric=an$pTerms.table,anova_smooth=an$s.table,scale_estimated=su$scale.estimated)
}
write_json(res,a[2],digits=17,auto_unbox=TRUE,null='null',na='null')
"""


def _reference(case):
    key = reference_key("quasi-workflow-v1", case, normalize_floats=True)
    cached = load_reference("mgcv_quasi_workflows", key)
    if cached is not None:
        return cached
    with tempfile.TemporaryDirectory() as tmp:
        script, inp, out = [
            Path(tmp) / name for name in ["ref.R", "in.json", "out.json"]
        ]
        script.write_text(_SCRIPT)
        inp.write_text(json.dumps(case))
        run = subprocess.run(
            _build_r_command(script, str(inp), str(out)), capture_output=True, text=True
        )
        assert run.returncode == 0, run.stderr
        cached = json.loads(out.read_text())
    save_reference("mgcv_quasi_workflows", key, cached)
    return cached


def _fit(case, family_override=None):
    from nampy.gam.families.quasi import (
        QuasiBinomialFamily,
        QuasiFamily,
        QuasiPoissonFamily,
    )

    if case["family"] == "quasi":
        family = QuasiFamily(
            link=case["link"], variance=case["variance"], power=case.get("power")
        )
    else:
        constructor = (
            QuasiPoissonFamily
            if case["family"] == "quasipoisson"
            else QuasiBinomialFamily
        )
        family = constructor(link=case["link"])
    return GAM(
        formula=case["formula"],
        family=family if family_override is None else family_override,
        smoothing_method=case["method"],
        optimize_smoothing=case["method"] != "fixed",
        smoothing_params=case["sp"] if case["method"] == "fixed" else None,
        scale=case.get("scale", 0),
    ).fit(data=pd.DataFrame(case["data"]), sample_weight="w")


def _assert_fit(case, reference, model):
    tol = 2e-6 if case["method"] == "fixed" else 4e-5
    np.testing.assert_allclose(_coef_full(model), reference["coef"], atol=tol, rtol=tol)
    np.testing.assert_allclose(_edf_total(model), reference["edf"], atol=tol, rtol=tol)
    np.testing.assert_allclose(
        _fit_scale(model), reference["scale"], atol=tol, rtol=tol
    )
    np.testing.assert_allclose(model.vcov(), reference["Vp"], atol=tol, rtol=tol)
    np.testing.assert_allclose(
        model.vcov(freq=True), reference["Ve"], atol=tol, rtol=tol
    )
    np.testing.assert_allclose(model.y_, reference["y"], atol=1e-15)
    np.testing.assert_allclose(model.prior_weights_, reference["weights"], atol=1e-15)
    nd = pd.DataFrame(
        {"x": np.linspace(-0.9, 0.9, 12), "z": np.linspace(0.8, -0.8, 12), "off": 0.13}
    )
    prediction, se = model.predict(nd, type="link", return_se=True)
    np.testing.assert_allclose(prediction, reference["prediction"], atol=tol, rtol=tol)
    np.testing.assert_allclose(se, reference["se"], atol=tol, rtol=tol)
    if case["method"] != "fixed":
        reference_sp = np.atleast_1d(reference["sp"])
        actual_sp = np.atleast_1d(model.smoothing_params)
        if (
            case["family"] == "quasibinomial"
            and case["method"] == "ML"
            and case.get("scale") == 0.8
        ):
            # Both optimizers converge to the linear null-space limit (EDF=2).
            # R stops at SP=2.8264e7, NumPy at SP=7.6831e7; the finite
            # representatives depend on the score-gradient stop tolerance.
            np.testing.assert_allclose(
                [reference["edf"], _edf_total(model)], 2.0, atol=1e-5, rtol=0
            )
            np.testing.assert_allclose(
                model._optim_result.fun, reference["score"], atol=2e-6, rtol=0
            )
            assert np.max(np.abs(model._optim_result.jac)) < 1e-6 * max(
                abs(reference["score"]), reference["scale"]
            )
        else:
            np.testing.assert_allclose(actual_sp, reference_sp, atol=5e-5, rtol=5e-4)
    assert reference["aic"] is None and reference["loglik"] is None
    assert np.isnan(model.aic())
    assert np.isnan(model.loglik())
    assert np.isnan(model.bic())
    from nampy.gam.model_state import _fit_result, _fitted_mu

    assert np.isnan(_fit_result(model).loglik)
    assert np.isnan(model.family.loglik(model.y_, _fitted_mu(model)))
    assert np.isnan(model.family.aic(model.y_, _fitted_mu(model)))
    for name, expected in zip(
        ["deviance", "pearson", "working", "response"],
        reference["residuals"],
        strict=True,
    ):
        actual = model.residuals(type=name)
        expected = np.asarray(expected, dtype=float)
        # Identity/Poisson zero counts have zero observed curvature. The
        # Newton working response divides by roundoff alpha (~machine eps),
        # so its huge magnitude is not a meaningful precision comparison.
        singular = np.zeros(len(expected), dtype=bool)
        if name == "working":
            if case["link"] == "identity" and (
                case["family"] == "quasipoisson" or case["variance"] == "mu"
            ):
                singular = np.asarray(model.y_) == 0
            elif case["link"] == "log" and (
                case["family"] == "quasibinomial" or case["variance"] == "mu(1-mu)"
            ):
                singular = np.asarray(model.y_) == 1
            if np.any(singular):
                from nampy.gam.model_state import _fit_state, _fitted_eta, _fitted_mu

                mu = _fitted_mu(model)
                score = _fit_state(model).working_weights * actual
                expected_score = (
                    model.prior_weights_
                    * (model.y_ - mu)
                    * model.family.mu_eta(_fitted_eta(model))
                    / model.family.variance(mu)
                )
                np.testing.assert_allclose(
                    score[singular], expected_score[singular], atol=tol, rtol=tol
                )
        np.testing.assert_allclose(
            actual[~singular], expected[~singular], atol=tol, rtol=tol, equal_nan=True
        )


@pytest.mark.parametrize("family", ["quasipoisson", "quasibinomial"])
@pytest.mark.parametrize("method", ["fixed", "GCV", "REML", "ML", "P-REML", "P-ML"])
def test_named_quasi_fitting_selection_and_inference(family, method):
    case = _case(family, method)
    reference = _reference(case)
    _assert_fit(case, reference, _fit(case))


_GENERAL_COMBINATIONS = [
    ("constant", "identity"),
    ("constant", "log"),
    ("mu", "log"),
    ("mu", "identity"),
    ("mu", "sqrt"),
    ("mu(1-mu)", "logit"),
    ("mu(1-mu)", "probit"),
    ("mu(1-mu)", "cloglog"),
    ("mu(1-mu)", "cauchit"),
    ("mu(1-mu)", "identity"),
    ("mu^2", "log"),
    ("mu^2", "inverse"),
    ("mu^3", "log"),
    ("mu^3", "inverse"),
    ("mu^3", "1/mu^2"),
]


@pytest.mark.parametrize("variance,link", _GENERAL_COMBINATIONS)
@pytest.mark.parametrize("method", ["fixed", "REML"])
def test_general_quasi_variance_link_workflows(variance, link, method):
    case = _case("quasi", method, link, variance)
    reference = _reference(case)
    _assert_fit(case, reference, _fit(case))


@pytest.mark.parametrize(
    "family,link",
    [
        ("quasipoisson", "identity"),
        ("quasipoisson", "sqrt"),
        ("quasibinomial", "probit"),
        ("quasibinomial", "cloglog"),
        ("quasibinomial", "cauchit"),
        ("quasibinomial", "log"),
        ("quasibinomial", "identity"),
    ],
)
@pytest.mark.parametrize("method", ["fixed", "REML"])
def test_named_noncanonical_quasi_links(family, link, method):
    case = _case(family, method, link)
    reference = _reference(case)
    _assert_fit(case, reference, _fit(case))


@pytest.mark.parametrize("family", ["quasipoisson", "quasibinomial"])
@pytest.mark.parametrize("method", ["fixed", "GCV", "REML", "ML", "P-REML", "P-ML"])
def test_quasi_zero_weights_and_multiple_penalties(family, method):
    case = _case(family, method, zero=True, multiple=True)
    reference = _reference(case)
    _assert_fit(case, reference, _fit(case))


@pytest.mark.parametrize("family", ["quasipoisson", "quasibinomial"])
def test_quasi_rank_deficiency_and_serialization(family, tmp_path):
    case = _case(family, rank_deficient=True)
    reference = _reference(case)
    model = _fit(case)
    _assert_fit(case, reference, model)
    path = tmp_path / "quasi.pkl"
    model.save_model(path)
    restored = GAM.load_model(path)
    _assert_fit(case, reference, restored)


@pytest.mark.parametrize(
    "family",
    [
        {"name": "quasi", "variance": "mu^4"},
        {"name": "quasi", "link": "unknown"},
        {"name": "quasipoisson", "link": "unknown"},
        {"name": "quasibinomial", "link": "unknown"},
    ],
)
def test_quasi_rejects_unsupported_family_controls(family):
    with pytest.raises((ValueError, NotImplementedError)):
        GAM(family=family)


@pytest.mark.parametrize(
    "family,link,variance",
    [
        ("quasipoisson", "log", None),
        ("quasibinomial", "probit", None),
        ("quasi", "log", "mu^2"),
    ],
)
@pytest.mark.parametrize("method", ["REML", "ML"])
def test_exact_joint_scale_and_sp_gradient_hessian(family, link, variance, method):
    from nampy.gam.fit.selection.criteria.pirls.family_gamma import (
        criterion_gradient_ml_reml_pirls_gamma_joint,
        criterion_hessian_ml_reml_pirls_gamma_joint,
        criterion_ml_reml_pirls_gamma_joint,
    )

    case = _case(family, method, link, variance, zero=True, multiple=True)
    case["joint_trace"] = True
    reference = _reference(case)["joint"]
    model = _fit(case)
    log_sp = np.atleast_1d(reference["log_sp"])
    log_phi = reference["log_phi"]
    args = (model, model.y_, log_sp, log_phi, method)
    np.testing.assert_allclose(
        criterion_ml_reml_pirls_gamma_joint(*args),
        reference["value"],
        rtol=2e-7,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        criterion_gradient_ml_reml_pirls_gamma_joint(*args),
        reference["grad"],
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        criterion_hessian_ml_reml_pirls_gamma_joint(*args),
        reference["hess"],
        rtol=1e-5,
        atol=3e-6,
    )


@pytest.mark.parametrize("family", ["quasipoisson", "quasibinomial"])
@pytest.mark.parametrize("method", ["fixed", "UBRE", "REML", "ML", "P-REML"])
def test_quasi_explicit_positive_scale(family, method):
    case = _case(family, method)
    case["scale"] = 0.8
    reference = _reference(case)
    model = _fit(case)
    _assert_fit(case, reference, model)
    assert _fit_scale(model) == 0.8


@pytest.mark.parametrize("scale", [0, 0.8])
@pytest.mark.parametrize("family", ["quasipoisson", "quasibinomial"])
def test_summary_and_single_model_anova_dispersion_tests(family, scale):
    from nampy.gam.inference.summary import summary_gam

    case = _case(family, "REML")
    case["formula"] += "+z"
    case.update(scale=scale, inference=True)
    reference = _reference(case)["inference"]
    model = _fit(case)
    summary = summary_gam(model)
    np.testing.assert_allclose(
        summary.p_table.to_numpy(dtype=float),
        reference["p_table"],
        atol=4e-5,
        rtol=4e-5,
    )
    np.testing.assert_allclose(
        summary.pterms_table[["df", "wald_stat", "p_value"]].to_numpy(dtype=float),
        reference["pterms"],
        atol=4e-5,
        rtol=4e-5,
    )
    np.testing.assert_allclose(
        summary.s_table[["edf", "ref_df", "wald_stat", "p_value"]].to_numpy(
            dtype=float
        ),
        reference["smooth"],
        atol=4e-5,
        rtol=4e-5,
    )
    assert summary.scale_estimated == (scale == 0)
    anova = model.anova()
    assert set(anova.parametric_table["test"]) == ({"F"} if scale == 0 else {"ChiSq"})
    assert set(anova.smooth_table["test"]) == ({"F"} if scale == 0 else {"ChiSq"})
    np.testing.assert_allclose(
        anova.parametric_table[["df", "wald_stat", "p_value"]],
        reference["anova_parametric"],
        atol=4e-5,
        rtol=4e-5,
    )
    np.testing.assert_allclose(
        anova.smooth_table[["edf", "ref_df", "wald_stat", "p_value"]],
        reference["anova_smooth"],
        atol=4e-5,
        rtol=4e-5,
    )


@pytest.mark.parametrize("method", ["fixed", "GCV", "REML", "P-REML"])
def test_grouped_quasibinomial_trial_weights_and_no_integer_warnings(method):
    import warnings

    case = _case("quasibinomial", method, zero=True)
    case["data"]["success"] = (np.asarray(case["data"]["y"]) * 8).tolist()
    case["data"]["failure"] = (8 - np.asarray(case["data"]["success"])).tolist()
    case["data"]["success"][3] = case["data"]["failure"][3] = 0
    case["formula"] = case["formula"].replace("y~", "cbind(success,failure)~")
    reference = _reference(case)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = _fit(case)
    assert not any("non-integer" in str(item.message) for item in caught)
    _assert_fit(case, reference, model)
    trials = np.asarray(case["data"]["success"]) + case["data"]["failure"]
    np.testing.assert_array_equal(model.binomial_trials_, trials)


def test_quasibinomial_classifier_and_regressor_roles():
    from nampy.gam.families.quasi import QuasiBinomialFamily
    from nampy.models import GAMClassifier, GAMRegressor

    rng = np.random.default_rng(238)
    x = np.linspace(0, 1, 70)[:, None]
    y = rng.binomial(1, 0.3 + 0.3 * x[:, 0])
    classifier = GAMClassifier(
        family=QuasiBinomialFamily(), k=6, optimize_smoothing=False
    ).fit(x, y)
    assert set(classifier.predict(x)) <= {0, 1}
    np.testing.assert_allclose(classifier.predict_proba(x).sum(axis=1), 1)
    with pytest.raises(ValueError, match="binomial|classifier|classification"):
        GAMRegressor(family=QuasiBinomialFamily(), k=6).fit(x, y)


@pytest.mark.parametrize("family", ["quasipoisson", "quasi", "quasibinomial"])
def test_singular_working_score_against_actual_mgcv(family):
    from nampy.gam.model_state import _fit_state

    case = _case(family, link="identity", variance="mu" if family == "quasi" else None)
    case["working_score"] = True
    reference = _reference(case)
    model = _fit(case)
    np.testing.assert_allclose(
        _fit_state(model).working_weights * model.residuals("working"),
        reference["working_score"],
        atol=2e-6,
        rtol=2e-6,
    )


@pytest.mark.parametrize(
    "public,family,variance,link",
    [
        ("quasipoisson", "quasipoisson", None, "log"),
        (("quasipoisson", "sqrt"), "quasipoisson", None, "sqrt"),
        (
            {"name": "quasipoisson", "link": "identity"},
            "quasipoisson",
            None,
            "identity",
        ),
        ("quasibinomial", "quasibinomial", None, "logit"),
        (("quasibinomial", "probit"), "quasibinomial", None, "probit"),
        (
            {"name": "quasibinomial", "link": "cloglog"},
            "quasibinomial",
            None,
            "cloglog",
        ),
        ("quasi", "quasi", "constant", "identity"),
        (("quasi", "log"), "quasi", "constant", "log"),
        ({"name": "quasi", "variance": "mu^3", "link": "log"}, "quasi", "mu^3", "log"),
    ],
)
def test_public_quasi_family_specs_match_upstream(public, family, variance, link):
    case = _case(family, link=link, variance=variance)
    reference = _reference(case)
    model = _fit(case, family_override=public)
    _assert_fit(case, reference, model)
    with pytest.raises(NotImplementedError, match="distribution"):
        model.residuals(type="rquantile")


@pytest.mark.parametrize(
    "family",
    [
        {"name": "quasi", "unrecognized": 1},
        {"name": "quasipoisson", "variance": "mu^2"},
        {"name": "quasibinomial", "power": 0.5},
        {"name": "quasi", "variance": lambda mu: mu},
    ],
)
def test_public_quasi_rejects_ignored_or_custom_options(family):
    with pytest.raises((ValueError, TypeError, NotImplementedError)):
        GAM(family=family)


@pytest.mark.parametrize("power", [0.5, 1.7])
@pytest.mark.parametrize("method", ["fixed", "REML"])
def test_general_quasi_power_links_through_public_registry(power, method):
    case = _case("quasi", method, link="power", variance="mu^2")
    case["power"] = power
    reference = _reference(case)
    model = _fit(
        case, family_override={"name": "quasi", "variance": "mu^2", "power": power}
    )
    _assert_fit(case, reference, model)


@pytest.mark.parametrize("family", ["quasipoisson", "quasibinomial"])
@pytest.mark.parametrize("method", ["P-REML", "P-ML"])
def test_zero_weight_pearson_scale_objective_gradient_hessian(family, method):
    from nampy.gam.fit.selection.criteria.pirls.derivatives import (
        criterion_gradient_p_ml_reml_pirls_exact,
        criterion_hessian_p_ml_reml_pirls_exact,
    )
    from nampy.gam.fit.selection.criteria.pirls.value import criterion_p_ml_reml_pirls

    case = _case(family, method, zero=True, multiple=True)
    case["pearson_trace"] = True
    reference = _reference(case)["joint"]
    model = _fit(case)
    args = (model, model.y_, np.atleast_1d(reference["log_sp"]), method)
    np.testing.assert_allclose(
        criterion_p_ml_reml_pirls(*args), reference["value"], atol=2e-6, rtol=2e-7
    )
    np.testing.assert_allclose(
        criterion_gradient_p_ml_reml_pirls_exact(*args),
        reference["grad"],
        atol=2e-6,
        rtol=2e-6,
    )
    np.testing.assert_allclose(
        criterion_hessian_p_ml_reml_pirls_exact(*args),
        reference["hess"],
        atol=3e-6,
        rtol=1e-5,
    )
