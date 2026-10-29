"""Actual mgcv gumbls kernels, fits and public distribution contracts."""

import importlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from nampy.gam.families.gamlss.gumbls import GumblsFamily
from tests.mgcv_parity_utils import _build_r_command
from tests.reference_fixtures import load_reference, reference_key, save_reference

ROOT = Path(__file__).resolve().parents[1] / "reference_fixtures" / "mgcv_gumbls"
SCRIPT = r'''
suppressPackageStartupMessages(library(mgcv));library(jsonlite)
a<-commandArgs(TRUE);p<-fromJSON(a[1],simplifyVector=TRUE)
f<-gumbls(link=as.list(p$link),b=p$b)
if(p$mode=="kernel") {
 X<-as.matrix(p$X);attr(X,"lpi")<-list(1:2,3:4)
 original<-get("gamlss.gH",asNamespace("mgcv"))
 assignInNamespace("gamlss.gH",function(X,jj,l1,l2,i2,l3,i3,l4,i4,...) list(l1=l1,l2=l2,l3=l3,l4=l4),ns="mgcv")
 out<-f$ll(p$y,X,p$coef,p$w,f,offset=as.list(as.data.frame(p$offset)),deriv=4)
 assignInNamespace("gamlss.gH",original,ns="mgcv")
} else if(p$mode=="init") {
 x<-as.matrix(p$X);attr(x,"lpi")<-list(1:2,3:4);y<-p$y;E<-as.matrix(p$E);nobs<-length(y);start<-NULL;family<-f;offset<-as.list(as.data.frame(p$offset))
 pen.reg<-mgcv:::pen.reg;eval(f$initialize);out<-list(start=start)
} else {
 d<-as.data.frame(p$data)
 b<-gam(list(y~s(x,bs="cr",k=5)+offset(o1),~s(z,bs="cr",k=4)+offset(o2)),data=d,family=f,method="REML",sp=if(p$fixed) c(.4,.7) else NULL,in.out=if(is.null(p$initial_sp)) NULL else list(sp=unlist(p$initial_sp),scale=1))
 pr<-predict(b,type="response",se.fit=TRUE)
 out<-list(coef=coef(b),sp=b$sp,Vp=b$Vp,edf=b$edf,score=b$gcv.ubre,fit=b$fitted.values,pred=pr$fit,se=pr$se.fit,eta=predict(b,type="link"),l=b$family$ll(b$y,model.matrix(b),coef(b),b$prior.weights,b$family,offset=b$offset)$l,residuals=lapply(c("deviance","pearson","response"),function(t) residuals(b,type=t)))
}
write_json(out,a[2],digits=17,auto_unbox=TRUE,na="null")
'''


def reference(payload):
    key = reference_key("gumbls-source-v1", payload)
    value = load_reference("mgcv_gumbls", key)
    if value is not None:
        return value
    with tempfile.TemporaryDirectory() as tmp:
        paths = [Path(tmp) / name for name in ("source.R", "input.json", "out.json")]
        paths[0].write_text(SCRIPT)
        paths[1].write_text(json.dumps(payload))
        result = subprocess.run(_build_r_command(paths[0], str(paths[1]), str(paths[2])), capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        value = json.loads(paths[2].read_text())
    save_reference("mgcv_gumbls", key, value)
    return value


def inputs():
    return json.loads((ROOT / "inputs.json").read_text())


@pytest.mark.parametrize("link", ["identity", "log"])
def test_gumbls_derivatives_through_four_against_source(link, monkeypatch):
    module = importlib.import_module("nampy.gam.families.gamlss.gumbls")

    p = inputs()["kernel"]
    p.update(mode="kernel", link=["identity", link], b=-3.0)
    expected = reference(p)
    monkeypatch.setattr(module, "gamlss_gH", lambda X, jj, l1, l2, i2, **kw: {"l1": l1, "l2": l2, "l3": kw["l3"], "l4": kw["l4"]})
    actual = GumblsFamily(p["link"], b=p["b"]).ll(np.array(p["y"]), np.array(p["X"]), [np.arange(2), np.arange(2, 4)], np.array(p["coef"]), np.array(p["w"]), offset=np.array(p["offset"]).T.tolist(), deriv=4)
    for key in ("l", "l0", "l1", "l2", "l3", "l4"):
        np.testing.assert_allclose(actual[key], np.asarray(expected[key]).ravel() if key == "l0" else expected[key], rtol=2e-11, atol=2e-11, err_msg=key)


@pytest.mark.parametrize("link", ["identity", "log"])
@pytest.mark.parametrize("fixed", [True, False])
@pytest.mark.usefixtures("general_penalty_orientation")
def test_gumbls_fit_and_prediction_against_source(link, fixed):
    p = {"mode": "fit", "data": inputs()["data"], "link": ["identity", link], "b": -3.0, "fixed": fixed}
    if not fixed:
        # initial.spg combines a reparameterized design with original penalties;
        # its implicit start depends on arbitrary eigenvector orientations.
        p["initial_sp"] = [100.0, 100.0]
    expected = reference(p)
    model = GAM(formula=["y~s(x,bs='cr',k=5)+offset(o1)", "~s(z,bs='cr',k=4)+offset(o2)"], family=GumblsFamily(p["link"], b=p["b"]), optimize_smoothing=not fixed, smoothing_params=[.4, .7] if fixed else p["initial_sp"], smoothing_method="reml").fit(data=pd.DataFrame(p["data"]))
    for key, actual in (("coef", model.fit_result().coef_full), ("Vp", model.vcov()), ("eta", model.predict(type="link")), ("pred", model.predict()), ("se", model.predict(return_se=True)[1])):
        np.testing.assert_allclose(actual, expected[key], rtol=3e-5, atol=3e-6, err_msg=key)
    from nampy.gam.model_state import _fitted_mu
    np.testing.assert_allclose(_fitted_mu(model), expected["fit"], rtol=3e-5, atol=3e-6)
    if not fixed:
        np.testing.assert_allclose(model.smoothing_score_, expected["score"], rtol=3e-6, atol=3e-6)
        np.testing.assert_allclose(model.smoothing_params, expected["sp"], rtol=3e-4, atol=3e-5)
    for i, kind in enumerate(("deviance", "pearson", "response")):
        np.testing.assert_allclose(model.residuals(type=kind), expected["residuals"][i], rtol=3e-5, atol=3e-6)


def test_gumbls_distribution_and_validation():
    family = GumblsFamily(link=("identity", "identity"))
    eta = np.array([[1.2, -0.4], [-0.3, .2]])
    params = family.distribution_parameters_from_eta(eta)
    np.testing.assert_allclose(family.point_prediction_from_parameters(params), params[:, 0] + np.euler_gamma * params[:, 1])
    from scipy.stats import gumbel_r
    np.testing.assert_allclose(family.logpdf_from_parameters([.1, .5], params), gumbel_r.logpdf([.1, .5], loc=params[:, 0], scale=params[:, 1]))
    with pytest.raises(ValueError):
        GumblsFamily(link=("log", "identity"))


def test_gumbls_source_initialization():
    p = inputs()["kernel"]
    p.update(mode="init", link=["identity", "log"], b=-3.0)
    p["E"] = (np.eye(4) * .2).tolist()
    expected = reference(p)
    actual = GumblsFamily(b=-3).initialize(np.array(p["y"]), np.array(p["X"]), [np.arange(2), np.arange(2, 4)], offset=np.array(p["offset"]).T.tolist(), E=np.array(p["E"]))
    np.testing.assert_allclose(actual, np.asarray(expected["start"]).ravel(), rtol=3e-11, atol=3e-11)


def test_gumbls_public_components_persistence(tmp_path):
    from nampy.models.gamlss import GAMLSS

    data = pd.DataFrame(inputs()["data"])
    model = GAMLSS(formula=["y~s(x,bs='cr',k=5)+offset(o1)", "~s(z,bs='cr',k=4)+offset(o2)"], family=GumblsFamily(b=-3), optimize_smoothing=False, smoothing_params=[.4, .7]).fit(data)
    components = model.predict_components(data)
    components.validate_additive_reconstruction()
    np.testing.assert_allclose(components.link, model.predict_link(data))
    assert model.standard_errors(data).shape == (len(data), 2)
    assert np.isfinite(model.score(data, data.y))
    path = tmp_path / "gumbls.pkl"
    model.save_model(path)
    restored = GAMLSS.load_model(path)
    np.testing.assert_array_equal(restored.predict_parameters(data), model.predict_parameters(data))
    np.testing.assert_array_equal(restored.predict_point(data), model.predict_point(data))
