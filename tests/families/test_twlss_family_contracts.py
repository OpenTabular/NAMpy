"""Static mgcv twlss likelihood, initialization, fit and inference contracts."""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from nampy.gam.families.gamlss.twlss import TwlssFamily, _tw_null_fit
from tests.mgcv_parity_utils import _build_r_command
from tests.reference_fixtures import load_reference, reference_key, save_reference

ROOT = Path(__file__).parents[1] / "reference_fixtures/mgcv_twlss"
SOURCE = r"""
library(mgcv);library(jsonlite)
pen.reg<-mgcv:::pen.reg
a<-commandArgs(TRUE);p<-fromJSON(a[1]);d<-as.data.frame(p$data)
f<-twlss(link=list(p$link,'identity','identity'),a=p$a,b=p$b)
if(p$kind=='kernel') {
 X<-do.call(cbind,rep(list(cbind(1,d$x)),3));jj<-lapply(0:2,function(i)2*i+1:2);attr(X,'lpi')<-jj
 co<-as.numeric(p$coef);off<-lapply(1:3,function(k)rep((k-2)*.03,nrow(d)))
 eta<-sapply(1:3,function(k)drop(X[,jj[[k]]]%*%co[jj[[k]]])+off[[k]])
 gh<-f$ll(d$y,X,co,seq(.5,2,length.out=nrow(d)),f,offset=off,deriv=1)
 fit<-sapply(1:3,function(k)f$linfo[[k]]$linkinv(eta[,k]))
 obj<-list(y=d$y,fitted.values=fit,prior.weights=seq(.5,2,length.out=nrow(d)))
 init<-function(unscaled) {
  x<-X;y<-d$y;family<-f;start<-NULL;nobs<-length(y);E<-diag(ncol(X))*.1
  if(unscaled)attr(E,'use.unscaled')<-TRUE
  eval(f$initialize);start
 }
 ans<-list(gh=gh,fitted=fit,initial=list(init(FALSE),init(TRUE)),
  residuals=lapply(c('deviance','pearson','response'),function(t)f$residuals(obj,type=t)),
  natural=ldTweedie(d$y,mu=fit[,1],theta=fit[,2],rho=fit[,3],a=p$a,b=p$b,all.derivs=TRUE),
  available_derivs=f$available.derivs)
} else {
 form<-list(y~s(x,bs='cr',k=6),~1,~s(x,bs='cr',k=5))
 if(p$kind=='fixed')form<-list(y~s(x,bs='cr',k=6,sp=.7),~1,~s(x,bs='cr',k=5,sp=.9))
 args<-list(formula=form,data=d,family=f,method='REML',optimizer='efs',
  control=gam.control(epsilon=1e-10))
 fit<-do.call(gam,args);su<-summary(fit);nd<-data.frame(x=seq(-1.4,1.4,length.out=17))
 pr<-predict(fit,nd,type='response',se.fit=TRUE)
 ans<-list(coef=coef(fit),Vp=fit$Vp,Ve=fit$Ve,sp=fit$sp,full_sp=fit$full.sp,
  score=as.numeric(fit$gcv.ubre),edf=sum(fit$edf),prediction=pr$fit,se=pr$se.fit,
  fitted=fitted(fit),loglik=as.numeric(logLik(fit)),aic=AIC(fit),
  ptable=su$p.table,stable=su$s.table,dev_expl=su$dev.expl,
  null=mgcv:::tw.null.fit(d$y),null_deviance=fit$null.deviance,
  residuals=lapply(c('deviance','pearson','response'),function(t)residuals(fit,type=t)),
  converged=fit$converged)
}
write_json(ans,a[2],digits=17,auto_unbox=TRUE,null='null',na='null')
"""


def reference(payload):
    key = reference_key("twlss-v1", {"payload": payload, "source": SOURCE})
    result = load_reference("mgcv_twlss", key)
    if result is None:
        with tempfile.TemporaryDirectory() as directory:
            source, inputs, output = [Path(directory) / name for name in ("run.R", "in.json", "out.json")]
            source.write_text(SOURCE)
            inputs.write_text(json.dumps(payload))
            run = subprocess.run(_build_r_command(source, str(inputs), str(output)), capture_output=True, text=True, timeout=120)
            assert run.returncode == 0, run.stderr
            result = json.loads(output.read_text())
        save_reference("mgcv_twlss", key, result)
    return result


@pytest.mark.parametrize("link", ["log", "identity", "sqrt"])
@pytest.mark.parametrize("a,b", [(1.01, 1.99), (1.2, 1.8)])
def test_twlss_analytic_likelihood_links_initialization_and_residuals(link, a, b):
    payload = {"kind": "kernel", "link": link, "a": a, "b": b,
               "data": {"x": [-.8, -.4, -.1, 0, .3, .5, .9],
                        "y": [0, .4, 0, 1.3, .9, 1.5, 2.1]},
               "coef": [.8, .1, -.2, .05, -.5, .02]}
    ref = reference(payload)
    family = TwlssFamily(link=(link, "identity", "identity"), a=a, b=b)
    x, y = [np.asarray(payload["data"][key]) for key in ("x", "y")]
    X = np.tile(np.column_stack((np.ones(x.size), x)), (1, 3))
    jj = [np.arange(2 * k, 2 * k + 2) for k in range(3)]
    coef, weights = np.asarray(payload["coef"]), np.linspace(.5, 2, y.size)
    offsets = [np.full(y.size, (k - 1) * .03) for k in range(3)]
    result = family.ll(y, X, jj, coef, weights, offset=offsets, deriv=1)
    for key in ("l", "l0", "lb", "lbb"):
        np.testing.assert_allclose(result[key], ref["gh"][key], atol=2e-9, rtol=2e-9, err_msg=key)
    eta = family._stacked_eta(X, jj, coef, offset=offsets)
    fitted = family._predict_response_from_eta(eta)
    np.testing.assert_allclose(fitted, ref["fitted"], atol=2e-12, rtol=2e-12)
    params = family.distribution_parameters_from_eta(eta)
    np.testing.assert_allclose(family.logpdf_from_parameters(y, params), np.asarray(ref["natural"])[:, 0], atol=2e-9, rtol=2e-9)
    for index, kind in enumerate(("deviance", "pearson", "response")):
        np.testing.assert_allclose(family.residuals(y, fitted, kind, weights=weights), ref["residuals"][index], atol=2e-10, rtol=2e-10)
    class RootPenalty(np.ndarray):
        pass

    for index, unscaled in enumerate((False, True)):
        E = (.1 * np.eye(6)).view(RootPenalty)
        E.use_unscaled = unscaled
        np.testing.assert_allclose(family.initialize(y, X, jj, E=E), ref["initial"][index], atol=2e-9, rtol=2e-9)
    assert ref["available_derivs"] == family.available_derivs == 0
    assert not family.supports_analytic_outer_gradient
    assert not family.supports_analytic_outer_hessian


@pytest.mark.parametrize("kind", ["fixed", "efs"])
def test_twlss_fits_match_mgcv(kind):
    from nampy.gam.inference.null_deviance import compute_null_deviance
    from nampy.gam.model_state import _coef_full, _edf_total

    data = json.loads((ROOT / "inputs.json").read_text())
    ref = reference({"data": data, "kind": kind, "link": "log", "a": 1.01, "b": 1.99})
    model = GAM(formula=["y~s(x,bs='cr',k=6)", "~1", "~s(x,bs='cr',k=5)"],
                family=TwlssFamily(), smoothing_params=[.7, .9] if kind == "fixed" else None,
                smoothing_method="fixed" if kind == "fixed" else "REML", smoothing_optimizer="efs",
                optimize_smoothing=kind != "fixed", control={"epsilon": 1e-10}).fit(pd.DataFrame(data))
    nd = pd.DataFrame({"x": np.linspace(-1.4, 1.4, 17)})
    prediction, se = model.predict(nd, type="response", return_se=True)
    for actual, expected in ((_coef_full(model), ref["coef"]), (model.vcov(), ref["Vp"]),
                             (model.vcov(freq=True), ref["Ve"]),
                             (prediction, ref["prediction"]), (se, ref["se"]),
                             (_edf_total(model), ref["edf"]), (model.aic(), ref["aic"]),
                             (model.loglik(), ref["loglik"])):
        np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=2e-7)
    np.testing.assert_allclose(_tw_null_fit(np.asarray(data["y"])), ref["null"], atol=2e-7, rtol=2e-7)
    np.testing.assert_allclose(compute_null_deviance(model), ref["null_deviance"], atol=2e-7, rtol=2e-7)
    source_sp = ref["full_sp"] if kind == "fixed" else ref["sp"]
    np.testing.assert_allclose(model.smoothing_params, source_sp, atol=2e-7, rtol=2e-7)
    if kind != "fixed":
        np.testing.assert_allclose(model.smoothing_score_, ref["score"], atol=2e-7, rtol=2e-7)
    for index, rtype in enumerate(("deviance", "pearson", "response")):
        np.testing.assert_allclose(model.residuals(type=rtype), ref["residuals"][index], atol=2e-7, rtol=2e-7)
    summary = model.summary()
    np.testing.assert_allclose(summary.p_table, ref["ptable"], atol=2e-7, rtol=2e-7)
    np.testing.assert_allclose(summary.s_table[["edf", "ref_df", "wald_stat", "p_value"]], ref["stable"], atol=2e-7, rtol=2e-7)
    np.testing.assert_allclose(summary.dev_expl, ref["dev_expl"], atol=2e-7, rtol=2e-7)


def test_twlss_public_components_and_persistence(tmp_path):
    from nampy import GAMLSS

    data = pd.DataFrame(json.loads((ROOT / "inputs.json").read_text()))
    model = GAMLSS(formula=["y~s(x,bs='cr',k=6)", "~1", "~s(x,bs='cr',k=5)"],
                   family=TwlssFamily(), smoothing_params=[.7, .9], optimize_smoothing=False).fit(data)
    model.predict_components(data).validate_additive_reconstruction()
    assert model.parameter_names_ == ("mu", "p", "phi")
    parameters = model.predict(data)
    np.testing.assert_array_equal(model.predict_point(data), parameters[:, 0])
    assert np.all((parameters[:, 1] >= 1.01) & (parameters[:, 1] <= 1.99))
    assert np.isfinite(model.score(data, data["y"].to_numpy()))
    path = model.save_model(tmp_path / "twlss.nampy")
    restored = GAMLSS.load_model(path)
    np.testing.assert_array_equal(restored.predict(data), parameters)


@pytest.mark.parametrize("kwargs", [{"link": ("inverse", "identity", "identity")},
                                   {"link": ("log", "log", "identity")},
                                   {"link": ("log",)}, {"a": 1}, {"b": 2},
                                   {"a": 1.8, "b": 1.2}])
def test_twlss_rejects_invalid_links_and_bounds(kwargs):
    with pytest.raises(ValueError):
        TwlssFamily(**kwargs)


@pytest.mark.parametrize("y", [[-1, 0, 1], [0, np.nan], [np.inf]])
def test_twlss_rejects_invalid_response(y):
    with pytest.raises(ValueError, match="nonnegative"):
        TwlssFamily().validate_y(y)


def test_twlss_default_selection_uses_source_efs():
    from nampy.gam.model_state import _coef_full

    data = json.loads((ROOT / "inputs.json").read_text())
    ref = reference({"data": data, "kind": "efs", "link": "log", "a": 1.01, "b": 1.99})
    model = GAM(formula=["y~s(x,bs='cr',k=6)", "~1", "~s(x,bs='cr',k=5)"],
                family=TwlssFamily(), smoothing_method="REML",
                optimize_smoothing=True, control={"epsilon": 1e-10}).fit(pd.DataFrame(data))
    assert model._optim_result.outer_info["optimizer"] == "efs"
    np.testing.assert_allclose(_coef_full(model), ref["coef"], atol=2e-7, rtol=2e-7)
    np.testing.assert_allclose(model.smoothing_params, ref["sp"], atol=2e-7, rtol=2e-7)
