"""Grouped binomial parity against static stats/mgcv output.

R stats::binomInitialize, binomial()$aic and mgcv::fix.family.ls own the
count-to-proportion, weight and likelihood conventions tested here.
"""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from nampy.gam.model_state import _coef_full, _edf_total, _fitted_mu
from tests.mgcv_parity_utils import _build_r_command
from tests.reference_fixtures import load_reference, reference_key, save_reference

pytestmark = pytest.mark.surface_regression


def _case(link="logit", method="fixed", weighted=False, zero=False):
    rng = np.random.default_rng(431)
    x = np.linspace(-1, 1, 65)
    n = rng.integers(2, 31, len(x))
    s = rng.binomial(n, 1 / (1 + np.exp(-0.8 * np.sin(3 * x) + 0.3 * x)))
    if zero:
        n[[0, 12, 38]] = 0
        s[[0, 12, 38]] = 0
    w = (0.5 + (np.arange(len(x)) % 4) * 0.5) if weighted else np.ones(len(x))
    if weighted:
        w[8] = 0
    data = pd.DataFrame({"x": x, "s": s, "f": n - s, "w": w, "off": 0.1 * x})
    return {"link": link, "method": method, "data": data.to_dict("list")}


_SCRIPT = r"""
suppressPackageStartupMessages(library(mgcv));library(jsonlite)
a <- commandArgs(TRUE);p <- fromJSON(a[1]);d <- as.data.frame(p$data)
f <- cbind(s,f) ~ s(x,bs='cr',k=7)+offset(off)
args <- list(formula=f,data=d,family=binomial(link=p$link),weights=d$w,
             method=if(p$method=='fixed') 'REML' else p$method)
if (isTRUE(p$proportion)) {
 n <- d$s+d$f; d$y <- ifelse(n==0,0,d$s/n)
 args$formula <- y ~ s(x,bs='cr',k=7)+offset(off)
 args$data <- d; args$weights <- d$w*n
}
if(p$method=='fixed') args$sp <- .4
b <- do.call(gam,args)
mu <- fitted(b);y <- b$y;weights <- b$prior.weights
init <- local({nobs<-nrow(d); y<-as.matrix(d[c('s','f')]);weights<-d$w;
 eval(binomial()$initialize);list(y=y,n=n,weights=weights,mustart=mustart)})
out <- list(coef=coef(b),fitted=mu,sp=b$sp,edf=sum(b$edf),aic=AIC(b),loglik=as.numeric(logLik(b)),
 deviance=deviance(b),y=y,weights=weights,initial=init,
 residuals=lapply(c('response','working','deviance','pearson'),function(t) residuals(b,type=t)),
 prediction=predict(b,newdata=data.frame(x=c(-.9,-.1,.4,.95),off=0),type='response'),
 se=predict(b,se.fit=TRUE,type='link')$se.fit)
write_json(out,a[2],digits=17,auto_unbox=TRUE,null='null',na='null')
"""


def _reference(case):
    key = reference_key("grouped-binomial-v1", case, normalize_floats=True)
    result = load_reference("mgcv_grouped_binomial", key)
    if result is not None:
        return result
    with tempfile.TemporaryDirectory() as tmp:
        script, inp, out = (
            Path(tmp) / p for p in ("reference.R", "case.json", "out.json")
        )
        script.write_text(_SCRIPT)
        inp.write_text(json.dumps(case))
        proc = subprocess.run(
            _build_r_command(script, str(inp), str(out)), capture_output=True, text=True
        )
        assert proc.returncode == 0, proc.stderr
        result = json.loads(out.read_text())
    save_reference("mgcv_grouped_binomial", key, result)
    return result


@pytest.mark.parametrize("link", ["logit", "probit", "cloglog", "cauchit", "log"])
@pytest.mark.parametrize("weighted,zero", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("method", ["fixed", "REML"])
def test_grouped_binomial_fit_and_likelihood_match_mgcv(link, weighted, zero, method):
    case = _case(link, method, weighted, zero)
    ref = _reference(case)
    data = pd.DataFrame(case["data"])
    model = GAM(
        formula="cbind(s,f) ~ s(x,bs='cr',k=7)+offset(off)",
        family={"name": "binomial", "link": link},
        optimize_smoothing=method != "fixed",
        smoothing_method=method,
        **({"smoothing_params": [0.4]} if method == "fixed" else {}),
    )
    model.fit(data=data, sample_weight="w")
    np.testing.assert_allclose(model.y_, ref["y"], atol=1e-15)
    np.testing.assert_allclose(model.binomial_trials_, ref["initial"]["n"], atol=1e-15)
    np.testing.assert_allclose(model.prior_weights_, ref["weights"], atol=1e-15)
    np.testing.assert_allclose(
        model.family.initialize_mu(model.y_, model.prior_weights_),
        ref["initial"]["mustart"],
        atol=1e-15,
    )
    tol = 3e-7 if method == "fixed" else 5e-6
    np.testing.assert_allclose(_coef_full(model), ref["coef"], atol=tol, rtol=tol)
    np.testing.assert_allclose(_fitted_mu(model), ref["fitted"], atol=tol, rtol=tol)
    np.testing.assert_allclose(_edf_total(model), ref["edf"], atol=tol, rtol=tol)
    _, se = model.predict(return_se=True, type="link")
    np.testing.assert_allclose(se, ref["se"], atol=tol, rtol=tol)
    np.testing.assert_allclose(
        model.family.deviance(model.y_, _fitted_mu(model), model.prior_weights_),
        ref["deviance"],
        atol=tol * 10,
        rtol=tol,
    )
    if method != "fixed":
        np.testing.assert_allclose(
            model.smoothing_params, ref["sp"], atol=1e-5, rtol=1e-4
        )
    np.testing.assert_allclose(model.aic(), ref["aic"], atol=tol * 10, rtol=tol)
    np.testing.assert_allclose(model.loglik(), ref["loglik"], atol=tol * 10, rtol=tol)
    for kind, values in zip(
        ("response", "working", "deviance", "pearson"), ref["residuals"], strict=True
    ):
        np.testing.assert_allclose(
            model.residuals(kind),
            np.asarray(values, dtype=float),
            atol=tol * 10,
            rtol=tol,
            equal_nan=True,
        )
    new = pd.DataFrame({"x": [-0.9, -0.1, 0.4, 0.95], "off": np.zeros(4)})
    np.testing.assert_allclose(
        model.predict(new), ref["prediction"], atol=tol, rtol=tol
    )


def _fixed_model(**kwargs):
    return GAM(
        family="binomial",
        optimize_smoothing=False,
        smoothing_method="fixed",
        smoothing_params=[0.4],
        **kwargs,
    )


def test_count_matrix_array_and_explicit_formula_y_match_cbind():
    data = pd.DataFrame(_case(weighted=True, zero=True)["data"])
    formula = "cbind(s,f) ~ s(x,bs='cr',k=7)"
    count = data[["s", "f"]].to_numpy()
    expected = _fixed_model(formula=formula).fit(data=data, sample_weight="w")
    explicit = _fixed_model(formula="y ~ s(x,bs='cr',k=7)").fit(
        data=data, y=count, sample_weight="w"
    )
    array = _fixed_model(k=7, basis="cr").fit(data[["x"]], count, sample_weight=data.w)
    for actual in (explicit, array):
        np.testing.assert_allclose(actual.predict(), expected.predict(), atol=2e-13)
        np.testing.assert_allclose(actual.aic(), expected.aic(), atol=2e-12)


def test_grouped_response_arithmetic_and_one_trial_counts():
    x = np.linspace(-1, 1, 45)
    successes = (np.arange(len(x)) % 3 == 0).astype(int)
    data = pd.DataFrame(
        {
            "x": x,
            "s": successes,
            "total": np.ones(len(x)),
            "w": 1 + (np.arange(len(x)) % 2),
        }
    )
    grouped = _fixed_model(formula="cbind(s,total-s) ~ s(x,bs='cr',k=7)").fit(
        data=data, sample_weight="w"
    )
    scalar = _fixed_model(formula="s ~ s(x,bs='cr',k=7)").fit(
        data=data, sample_weight="w"
    )
    np.testing.assert_allclose(grouped.predict(), scalar.predict(), atol=1e-10)
    np.testing.assert_allclose(grouped.aic(), scalar.aic(), atol=1e-9)
    # Grouped initialization depends on trials, never replication weights.
    np.testing.assert_allclose(
        grouped.family.initialize_mu(grouped.y_, grouped.prior_weights_),
        (successes + 0.5) / 2,
    )


@pytest.mark.parametrize(
    "bad",
    [
        np.ones((10, 3)),
        np.full((10, 2), -1.0),
        np.full((10, 2), np.nan),
        np.full((10, 2), np.inf),
    ],
)
def test_invalid_grouped_counts_fail_before_fitting(bad):
    with pytest.raises(ValueError, match="Binomial"):
        _fixed_model(k=5).fit(np.arange(10)[:, None], bad)


@pytest.mark.parametrize("response", ["cbind(s)", "cbind(s,f,s)", "cbind(s,f,other=1)"])
def test_invalid_cbind_arity_is_explicit(response):
    with pytest.raises(ValueError, match="exactly two"):
        _fixed_model(formula=f"{response} ~ x").fit(
            data=pd.DataFrame({"s": [1, 2], "f": [2, 3], "x": [0, 1]})
        )


def test_count_response_requires_binomial_and_classifier_rejects_counts():
    from nampy.models.gam import GAMClassifier, GAMRegressor

    data = pd.DataFrame(_case()["data"])
    with pytest.raises(ValueError, match="requires family='binomial'"):
        GAM(formula="cbind(s,f) ~ x").fit(data=data)
    with pytest.raises(ValueError, match="one binary class label"):
        GAMClassifier().fit(data[["x"]], data[["s", "f"]])
    with pytest.raises(ValueError, match="regression family"):
        GAMRegressor(family="binomial").fit(data[["x"]], data[["s", "f"]])


def test_fractional_counts_warn_and_refit_clears_trial_state():
    data = pd.DataFrame(_case()["data"])
    count = data[["s", "f"]].to_numpy(dtype=float) + 0.2
    model = _fixed_model(k=7, basis="cr")
    with pytest.warns(UserWarning, match="non-integer counts"):
        model.fit(data[["x"]], count)
    assert model.binomial_trials_ is not None
    model.fit(data[["x"]], (data.s > data.f).astype(float))
    assert model.binomial_trials_ is None
    assert model.grouped_response_ is None
    assert model.family._trial_counts is None


@pytest.mark.parametrize("link", ["logit", "probit"])
def test_grouped_binomial_ml_with_zero_trials(link):
    test_grouped_binomial_fit_and_likelihood_match_mgcv(link, True, True, "ML")


@pytest.mark.parametrize("link", ["logit", "probit"])
@pytest.mark.parametrize("column_matrix", [False, True])
def test_proportion_vector_or_one_column_keeps_binomial_residual_contract(
    link, column_matrix
):
    case = _case(link, "fixed", True, True)
    # stats::binomInitialize resets zero-weight vector responses, whereas the
    # two-column branch retains their proportions. Use the actual vector fit.
    case["proportion"] = True
    data = pd.DataFrame(case["data"])
    ref = _reference(case)
    trials = data.s.to_numpy() + data.f.to_numpy()
    response = np.divide(
        data.s.to_numpy(), trials, out=np.zeros(len(data)), where=trials != 0
    )
    model = GAM(
        formula="y ~ s(x,bs='cr',k=7)+offset(off)",
        family={"name": "binomial", "link": link},
        optimize_smoothing=False,
        smoothing_method="fixed",
        smoothing_params=[0.4],
    ).fit(
        data=data,
        y=response[:, None] if column_matrix else response,
        sample_weight=trials * data.w.to_numpy(),
    )
    assert model.binomial_trials_ is None
    np.testing.assert_allclose(model.predict(), ref["fitted"], atol=3e-7, rtol=3e-7)
    for kind, expected in zip(
        ("response", "working", "deviance", "pearson"), ref["residuals"], strict=True
    ):
        np.testing.assert_allclose(
            model.residuals(kind),
            np.asarray(expected, dtype=float),
            atol=3e-6,
            rtol=3e-7,
            equal_nan=True,
        )
