"""Frozen actual-R censored Box-Cox Gaussian fitting workflows."""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from nampy.gam.families.bcg import CensoredBoxCoxGaussianFamily
from nampy.gam.inference.null_deviance import null_deviance
from nampy.gam.model_state import _coef_full, _deviance, _edf_total
from tests.mgcv_parity_utils import _build_r_command
from tests.reference_fixtures import load_reference, reference_key, save_reference

ROOT = Path(__file__).parents[1] / "reference_fixtures" / "mgcv_bcg_workflows"
SCRIPT = r'''
suppressPackageStartupMessages(library(mgcv));library(jsonlite)
a<-commandArgs(TRUE);p<-fromJSON(a[1]);d<-as.data.frame(p$data);d$bound<-as.numeric(d$bound)
f<-do.call(bcg,list(theta=p$theta,link=p$link))
if(!is.null(p$theta_state)) f$putTheta(p$theta_state)
out<-tryCatch({
b<-gam(as.formula(p$formula),data=d,family=f,weights=d$w,sp=if(p$method=="fixed") .7 else NULL,method=if(p$method=="ml") "ML" else "REML",optimizer=if(is.null(p$optimizer)) c("outer","newton") else p$optimizer)
pr<-predict(b,type="response",se.fit=TRUE)
list(coef=coef(b),theta=b$family$getTheta(),sp=b$sp,full_sp=b$full.sp,Vp=b$Vp,Ve=b$Ve,eta=b$linear.predictors,mu=fitted(b),deviance=deviance(b),null=b$null.deviance,edf=b$edf,score=b$gcv.ubre,aic=AIC(b),bic=BIC(b),loglik=as.numeric(logLik(b)),pred=pr$fit,se=pr$se.fit,residuals=lapply(c("deviance","working","response"),function(t) residuals(b,type=t)),outer=b$outer.info,converged=b$converged)
},error=function(e)list(error=conditionMessage(e)))
if(isTRUE(p$uncertainty)&&is.null(out$error)) out$uncertainty<-list(Vc=b$Vc,edf2=b$edf2)
write_json(out,a[2],digits=17,auto_unbox=TRUE,na="null")
'''


def reference(payload):
    key = reference_key("bcg-workflows-v1", payload)
    value = load_reference("mgcv_bcg_workflows", key)
    if value is not None:
        return value
    with tempfile.TemporaryDirectory() as tmp:
        paths = [Path(tmp) / name for name in ("source.R", "input.json", "out.json")]
        paths[0].write_text(SCRIPT)
        paths[1].write_text(json.dumps(payload))
        result = subprocess.run(_build_r_command(paths[0], str(paths[1]), str(paths[2])), capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        value = json.loads(paths[2].read_text())
    save_reference("mgcv_bcg_workflows", key, value)
    return value


def case(response="vector", method="fixed", estimated=False, link="identity"):
    data = json.loads((ROOT / "inputs.json").read_text())
    formula = ("cbind(y,bound)" if response == "mixed" else "y") + "~s(x,bs='cr',k=6)+offset(off)"
    p = {"data": data, "formula": formula, "theta": [1., 0.] if estimated else [.7, 1.2], "link": link, "method": method}
    if not estimated:
        # Source constructor has a third theta entry and fails in gam.fit4.
        # Its documented family-owned setter accepts a coherent two-value state.
        p["theta_state"] = [.7, -0.35667494393873245]
    return p


def fit(p):
    data = pd.DataFrame(p["data"])
    data["bound"] = data["bound"].astype(float)
    family = CensoredBoxCoxGaussianFamily(p["theta"], p["link"])
    if "theta_state" in p:
        family.putTheta(p["theta_state"])
    return GAM(formula=p["formula"], family=family, smoothing_params=[.7] if p["method"] == "fixed" else None, optimize_smoothing=p["method"] != "fixed", smoothing_method=p["method"], smoothing_optimizer=p.get("optimizer", "outer_newton")).fit(data=data, sample_weight="w")


@pytest.mark.parametrize("response", ["vector", "mixed"])
@pytest.mark.parametrize("method", ["fixed", "reml", "ml"])
@pytest.mark.parametrize("estimated", [False, True])
def test_bcg_identity_fitting_against_source(response, method, estimated):
    p = case(response, method, estimated)
    expected = reference(p)
    assert "error" not in expected, expected
    model = fit(p)
    checks = {"coef": _coef_full(model), "theta": model.family.getTheta(), "Vp": model.vcov(), "Ve": model.vcov(freq=True), "eta": model.predict(type="link"), "mu": model.predict(), "deviance": _deviance(model), "null": null_deviance(model), "aic": model.aic(), "bic": model.bic(), "loglik": model.loglik()}
    for key, actual in checks.items():
        np.testing.assert_allclose(actual, expected[key], rtol=2e-6, atol=2e-6, err_msg=key)
    np.testing.assert_allclose(_edf_total(model), np.sum(expected["edf"]), rtol=2e-6, atol=2e-6)
    if method != "fixed":
        np.testing.assert_allclose(model.smoothing_params, np.atleast_1d(expected["sp"]), rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(model.smoothing_score_, expected["score"], rtol=2e-6, atol=2e-6)
    prediction, se = model.predict(return_se=True)
    np.testing.assert_allclose(prediction, expected["pred"], rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(se, expected["se"], rtol=2e-6, atol=2e-6)
    for i, kind in enumerate(("deviance", "working", "response")):
        np.testing.assert_allclose(model.residuals(type=kind), expected["residuals"][i], rtol=2e-6, atol=2e-6, err_msg=kind)


@pytest.mark.parametrize("link", ["log", "sqrt"])
@pytest.mark.parametrize("response", ["vector", "mixed"])
def test_bcg_nonidentity_fixed_fit_against_source(link, response):
    p = case(response, link=link)
    expected = reference(p)
    assert "error" not in expected, expected
    model = fit(p)
    for key, actual in (("coef", _coef_full(model)), ("mu", model.predict()), ("Vp", model.vcov())):
        np.testing.assert_allclose(actual, expected[key], rtol=2e-6, atol=2e-6)


def test_bcg_public_components_persistence(tmp_path):
    from nampy.models.gam import GAMRegressor
    p = case("mixed")
    data = pd.DataFrame(p["data"])
    data["bound"] = data["bound"].astype(float)
    family = CensoredBoxCoxGaussianFamily(p["theta"])
    family.putTheta(p["theta_state"])
    model = GAMRegressor(formula=p["formula"], family=family, optimize_smoothing=False, smoothing_params=[.7]).fit(data, sample_weight=data.w)
    model.predict_components(data).validate_additive_reconstruction()
    path = tmp_path / "bcg.pkl"
    model.save_model(path)
    restored = GAMRegressor.load_model(path)
    np.testing.assert_array_equal(restored.predict(data), model.predict(data))
    np.testing.assert_array_equal(restored.gam_.response_.bounds, model.gam_.response_.bounds)


def test_bcg_fixed_constructor_source_fit_rejection():
    p = case()
    del p["theta_state"]
    expected = reference(p)
    assert "subscript out of bounds" in expected["error"]
    with pytest.raises(ValueError, match="three working theta"):
        fit(p)


@pytest.mark.parametrize("link", ["log", "sqrt"])
@pytest.mark.parametrize("method", ["fixed", "reml", "ml"])
def test_bcg_estimated_nonidentity_fit_against_source(link, method):
    p = case(method=method, estimated=True, link=link)
    expected = reference(p)
    assert "error" not in expected, expected
    model = fit(p)
    for key, actual in (("coef", _coef_full(model)), ("theta", model.family.getTheta()), ("mu", model.predict()), ("Vp", model.vcov())):
        np.testing.assert_allclose(actual, expected[key], rtol=2e-6, atol=2e-6, err_msg=key)


@pytest.mark.parametrize("response", ["vector", "mixed"])
@pytest.mark.parametrize("estimated", [False, True])
def test_bcg_efs_selected_fit_against_source(response, estimated):
    p = case(response, "reml", estimated)
    p["optimizer"] = "efs"
    expected = reference(p)
    assert "error" not in expected, expected
    model = fit(p)
    for key, actual in (("coef", _coef_full(model)), ("theta", model.family.getTheta()), ("mu", model.predict()), ("Vp", model.vcov()), ("deviance", _deviance(model)), ("score", model.smoothing_score_)):
        np.testing.assert_allclose(actual, expected[key], rtol=2e-6, atol=2e-6, err_msg=key)
    np.testing.assert_allclose(model.smoothing_params, np.atleast_1d(expected["sp"]), rtol=2e-5, atol=2e-5)


def test_bcg_all_fixed_sp_estimated_efs_source_rejection():
    p = case(method="fixed", estimated=True)
    p["optimizer"] = "efs"
    expected = reference(p)
    assert "NA/NaN/Inf in foreign function call" in expected["error"]
    with pytest.raises(NotImplementedError, match="free smoothing parameter"):
        fit(p)


@pytest.mark.parametrize("method", ["ml", "reml"])
@pytest.mark.parametrize("estimated", [False, True])
def test_bcg_unconditional_covariance_against_source(method, estimated):
    from nampy.gam.model_state import _edf2

    p = case("mixed" if estimated else "vector", method, estimated)
    p["uncertainty"] = True
    expected = reference(p)
    assert "error" not in expected, expected
    assert expected["uncertainty"]["Vc"] is not None
    model = fit(p)
    np.testing.assert_allclose(model.vcov(unconditional=True), expected["uncertainty"]["Vc"], rtol=2e-7, atol=2e-7)
    np.testing.assert_allclose(_edf2(model), expected["uncertainty"]["edf2"], rtol=2e-7, atol=2e-7)


def test_bcg_estimated_efs_public_components_persistence(tmp_path):
    from nampy.models.gam import GAMRegressor

    p = case("mixed", "reml", True)
    data = pd.DataFrame(p["data"])
    data["bound"] = data["bound"].astype(float)
    model = GAMRegressor(formula=p["formula"], family=CensoredBoxCoxGaussianFamily(), smoothing_method="reml", smoothing_optimizer="efs").fit(data, sample_weight=data.w)
    components = model.predict_components(data)
    components.validate_additive_reconstruction()
    assert components.link.shape == (len(data), 1)
    np.testing.assert_allclose(components.link, model.predict_link(data))
    assert model.gam_.family.n_theta == 2
    path = tmp_path / "bcg-estimated-efs.pkl"
    model.save_model(path)
    restored = GAMRegressor.load_model(path)
    np.testing.assert_array_equal(restored.predict(data), model.predict(data))
    np.testing.assert_array_equal(restored.gam_.family.getTheta(), model.gam_.family.getTheta())
    np.testing.assert_array_equal(restored.gam_.response_.values, model.gam_.response_.values)
    np.testing.assert_array_equal(restored.gam_.response_.bounds, model.gam_.response_.bounds)
    assert not restored.gam_.response_.values.flags.writeable
    assert not restored.gam_.response_.bounds.flags.writeable
