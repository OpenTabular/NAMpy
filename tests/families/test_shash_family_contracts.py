"""Static mgcv shash kernel, link, initialization and fit contracts."""

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from nampy.gam.families.gamlss.shash import ShashFamily
from tests.families._general_efs_invariants import (
    INITIAL_SP,
    assert_efs_function_space,
    capture_source,
)
from tests.mgcv_parity_utils import _build_r_command
from tests.reference_fixtures import load_reference, reference_key, save_reference

ROOT = Path(__file__).parents[1] / "reference_fixtures/mgcv_shash"
SOURCE = r"""
library(mgcv);library(jsonlite)
pen.reg<-mgcv:::pen.reg
a<-commandArgs(TRUE);p<-fromJSON(a[1]);d<-as.data.frame(p$data)
f<-shash(b=p$b,phiPen=p$phi_pen)
if(p$kind=='kernel') {
 X<-do.call(cbind,rep(list(cbind(1,d$x)),4));jj<-lapply(0:3,function(i)2*i+1:2);attr(X,'lpi')<-jj
 co<-as.numeric(p$coef);eta<-sapply(jj,function(j)drop(X[,j]%*%co[j]))
 # Expose the source eta derivatives before contractions with smoothing
 # derivatives. No family algebra or upstream solver is replaced.
 ff<-f$ll;body(ff)<-parse(text=sub('ret <- gamlss.gH','return(c(list(l=l,l0=l0),de));ret <- gamlss.gH',paste(deparse(body(ff)),collapse='\n'),fixed=TRUE))[[1]]
 de<-ff(d$y,X,co,rep(1,nrow(d)),f,deriv=4)
 gh<-f$ll(d$y,X,co,seq(.5,2,length.out=nrow(d)),f,deriv=1)
 fit<-sapply(1:4,function(k)f$linfo[[k]]$linkinv(eta[,k]))
 obj<-list(y=d$y,fitted=fit)
 init<-function(unscaled) {
  x<-X;y<-d$y;family<-f;start<-NULL;nobs<-length(y);E<-diag(ncol(X))*.1
  if(unscaled)attr(E,'use.unscaled')<-TRUE
  eval(f$initialize);start
 }
 ans<-list(de=de,lb=gh$lb,lbb=gh$lbb,fitted=fit,
  likelihood_positive_infinity=is.infinite(de$l)&&de$l>0,
  initial=list(init(FALSE),init(TRUE)),
  residuals=lapply(c('deviance','response'),function(t)f$residuals(obj,type=t)),
  quantile=f$qf(.3,fit),cdf=f$cdf(d$y,fit,logp=FALSE),logcdf=f$cdf(d$y,fit,logp=TRUE),
  links=lapply(1:4,function(k)list(inv=f$linfo[[k]]$linkinv(eta[,k]),
   mu_eta=f$linfo[[k]]$mu.eta(eta[,k]),d2=f$linfo[[k]]$d2link(fit[,k]),
   d3=f$linfo[[k]]$d3link(fit[,k]),d4=f$linfo[[k]]$d4link(fit[,k]))))
} else {
 form<-list(y~s(x,bs='cr',k=6),~s(x,bs='cr',k=5),~1,~1)
 args<-list(formula=form,data=d,family=f,method='REML',
   control=gam.control(epsilon=1e-10,newton=list(conv.tol=1e-10)))
 if(p$kind=='fixed')args$sp<-c(.7,.9)
 if(p$kind=='efs')args$optimizer<-'efs'
 fit<-do.call(gam,args);su<-summary(fit);nd<-data.frame(x=seq(-1.4,1.4,length.out=17))
 pr<-predict(fit,nd,type='response',se.fit=TRUE)
 ans<-list(coef=coef(fit),Vp=fit$Vp,Ve=fit$Ve,sp=fit$sp,full_sp=fit$full.sp,
  score=as.numeric(fit$gcv.ubre),edf=sum(fit$edf),prediction=pr$fit,se=pr$se.fit,
  fitted=fitted(fit),loglik=as.numeric(logLik(fit)),aic=AIC(fit),
  ptable=su$p.table,stable=su$s.table,dev_expl=su$dev.expl,
  residuals=lapply(c('deviance','response'),function(t)residuals(fit,type=t)),
  converged=fit$converged)
}
write_json(ans,a[2],digits=17,auto_unbox=TRUE,null='null',na='null')
"""


def reference(payload):
    source_text = capture_source(SOURCE, payload)
    key = reference_key("shash-v1", {"payload": payload, "source": source_text})
    result = load_reference("mgcv_shash", key)
    if result is None:
        with tempfile.TemporaryDirectory() as directory:
            source, inputs, output = [Path(directory) / name for name in ("run.R", "in.json", "out.json")]
            source.write_text(source_text)
            inputs.write_text(json.dumps(payload))
            run = subprocess.run(_build_r_command(source, str(inputs), str(output)), capture_output=True, text=True, timeout=120)
            assert run.returncode == 0, run.stderr
            result = json.loads(output.read_text())
        save_reference("mgcv_shash", key, result)
    return result


def kernel_payload(b=.01, phi_pen=.001):
    return {"kind": "kernel", "b": b, "phi_pen": phi_pen,
            "data": {"x": [-.8, -.4, -.1, 0, .3, .5, .9],
                     "y": [-1.3, -.4, .1, .31, .9, 1.5, 2.1]},
            "coef": [.3, .1, -.5, .05, .2, -.03, .1, .02]}


@pytest.mark.parametrize("b,phi_pen", [(.01, .001), (.2, .03), (.01, 0)])
def test_shash_analytic_derivatives_links_and_distribution_hooks(b, phi_pen):
    payload = kernel_payload(b, phi_pen)
    ref = reference(payload)
    family = ShashFamily(b=b, phi_pen=phi_pen)
    x, y = [np.asarray(payload["data"][key]) for key in ("x", "y")]
    X = np.tile(np.column_stack((np.ones(x.size), x)), (1, 4))
    jj = [np.arange(2 * k, 2 * k + 2) for k in range(4)]
    coef = np.asarray(payload["coef"])
    eta = family._stacked_eta(X, jj, coef)
    result = family.ll(y, X, jj, coef, np.linspace(.5, 2, y.size), deriv=4,
                       d1b=np.zeros((8, 1)), d2b=np.zeros((8, 1)),
                       fh=np.eye(8), D=np.ones(8), ncv=True)
    for key in ("l", "l0", "l1", "l2", "l3", "l4"):
        np.testing.assert_allclose(result[key], ref["de"][key], atol=2e-10, rtol=2e-10, err_msg=key)
    for key in ("lb", "lbb"):
        np.testing.assert_allclose(result[key], ref[key], atol=2e-10, rtol=2e-10)
    fit = np.column_stack([family.linfo[k].linkinv(eta[:, k]) for k in range(4)])
    for k, link in enumerate(family.linfo):
        for name, actual in (("inv", link.linkinv(eta[:, k])), ("mu_eta", link.mu_eta(eta[:, k])),
                             ("d2", link.d2link(fit[:, k])), ("d3", link.d3link(fit[:, k])), ("d4", link.d4link(fit[:, k]))):
            np.testing.assert_allclose(actual, ref["links"][k][name], atol=2e-12, rtol=2e-12)
    for k, kind in enumerate(("deviance", "response")):
        np.testing.assert_allclose(family.residuals(y, fit, kind), ref["residuals"][k], atol=2e-10, rtol=2e-10)
    np.testing.assert_allclose(family.qf(.3, fit), ref["quantile"], atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(family.cdf(y, fit), ref["cdf"], atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(family.cdf(y, fit, logp=True), ref["logcdf"], atol=2e-12, rtol=2e-12)
    params = family.distribution_parameters_from_eta(eta)
    np.testing.assert_allclose(family.logpdf_from_parameters(y, params) - phi_pen * fit[:, 3]**2,
                               ref["de"]["l0"], atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(family.point_prediction_from_parameters(params), family.qf(.5, fit), atol=2e-12, rtol=2e-12)
    class RootPenalty(np.ndarray):
        pass

    for index, unscaled in enumerate((False, True)):
        E = (.1 * np.eye(8)).view(RootPenalty)
        E.use_unscaled = unscaled
        np.testing.assert_allclose(family.initialize(y, X, jj, E=E), ref["initial"][index], atol=2e-10, rtol=2e-10)


@pytest.mark.parametrize("kind", ["fixed", "selected", "efs"])
def test_shash_fits_match_mgcv(kind):
    from nampy.gam.model_state import _coef_full, _edf_total

    data = json.loads((ROOT / "inputs.json").read_text())
    payload = {"data": data, "kind": kind, "b": .01, "phi_pen": .001}
    if kind == "efs":
        payload["initial_sp"] = INITIAL_SP
    ref = reference(payload)
    model = GAM(formula=["y~s(x,bs='cr',k=6)", "~s(x,bs='cr',k=5)", "~1", "~1"],
                family=ShashFamily(), smoothing_params=[.7, .9] if kind == "fixed" else INITIAL_SP if kind == "efs" else None,
                smoothing_method="fixed" if kind == "fixed" else "REML",
                smoothing_optimizer="efs" if kind == "efs" else "outer_newton",
                optimize_smoothing=kind != "fixed",
                control={"epsilon": 1e-10, "newton": {"conv.tol": 1e-10}}).fit(pd.DataFrame(data))
    nd = pd.DataFrame({"x": np.linspace(-1.4, 1.4, 17)})
    prediction, se = model.predict(nd, type="response", return_se=True)
    if kind == "efs":
        assert_efs_function_space(model, nd, ref)
    else:
        for actual, expected in ((_coef_full(model), ref["coef"]), (model.vcov(), ref["Vp"]),
                                 (model.vcov(freq=True), ref["Ve"])):
            np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=2e-7)
    for actual, expected in ((prediction, ref["prediction"]), (se, ref["se"]),
                             (_edf_total(model), ref["edf"]), (model.aic(), ref["aic"]),
                             (model.loglik(), ref["loglik"])):
        np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=2e-7)
    source_sp = ref["full_sp"] if kind == "fixed" else ref["sp"]
    np.testing.assert_allclose(model.smoothing_params, source_sp, atol=2e-7, rtol=2e-7)
    if kind != "fixed":
        np.testing.assert_allclose(model.smoothing_score_, ref["score"], atol=2e-7, rtol=2e-7)
    for index, kind in enumerate(("deviance", "response")):
        np.testing.assert_allclose(model.residuals(type=kind), ref["residuals"][index], atol=2e-7, rtol=2e-7)
    summary = model.summary()
    np.testing.assert_allclose(summary.p_table, ref["ptable"], atol=2e-7, rtol=2e-7)
    np.testing.assert_allclose(summary.s_table[["edf", "ref_df", "wald_stat", "p_value"]], ref["stable"], atol=2e-7, rtol=2e-7)
    assert summary.dev_expl is None
    # Public additive components and persistence are exercised through GAMLSS,
    # while this owner test checks the raw mgcv fitted parameterization.


def test_shash_input_manifest_hash():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    assert hashlib.sha256((ROOT / "inputs.json").read_bytes()).hexdigest() == manifest["inputs_sha256"]


def test_shash_exact_zero_source_likelihood_anomaly():
    payload = kernel_payload()
    payload["data"]["y"][3] = .3
    ref = reference(payload)
    assert ref["likelihood_positive_infinity"]
    family = ShashFamily()
    x, y = [np.asarray(payload["data"][key]) for key in ("x", "y")]
    X = np.tile(np.column_stack((np.ones(x.size), x)), (1, 4))
    jj = [np.arange(2 * k, 2 * k + 2) for k in range(4)]
    coef = np.asarray(payload["coef"])
    assert np.isposinf(family.ll(y, X, jj, coef, np.ones(y.size))["l"])
    # The source residual expression uses log1p(z^2), so it stays finite.
    fitted = family.predict(X=X, jj=jj, coef=coef)
    for index, kind in enumerate(("deviance", "response")):
        np.testing.assert_allclose(family.residuals(y, fitted, kind), ref["residuals"][index], atol=2e-10, rtol=2e-10)


def test_shash_public_components_and_persistence(tmp_path):
    from nampy import GAMLSS

    data = pd.DataFrame(json.loads((ROOT / "inputs.json").read_text()))
    model = GAMLSS(formula=["y~s(x,bs='cr',k=6)", "~s(x,bs='cr',k=5)", "~1", "~1"],
                   family=ShashFamily(), smoothing_params=[.7, .9], optimize_smoothing=False).fit(data)
    model.predict_components(data).validate_additive_reconstruction()
    assert model.parameter_names_ == ("mu", "sigma", "epsilon", "delta")
    np.testing.assert_allclose(model.predict_point(data), model.gam_.family.point_prediction_from_parameters(model.predict(data)))
    assert np.isfinite(model.score(data, data["y"].to_numpy()))
    path = model.save_model(tmp_path / "shash.nampy")
    restored = GAMLSS.load_model(path)
    np.testing.assert_array_equal(restored.predict(data), model.predict(data))


@pytest.mark.parametrize("kwargs", [{"link": ("log", "logeb", "identity", "identity")},
                                   {"link": ("identity",) * 4}, {"link": ("identity",)},
                                   {"b": 0}, {"b": -1}, {"phi_pen": -1}])
def test_shash_rejects_invalid_links_and_options(kwargs):
    with pytest.raises(ValueError):
        ShashFamily(**kwargs)


def test_shash_rejects_nonzero_offset():
    family = ShashFamily()
    X = np.ones((3, 4))
    jj = [np.array([k]) for k in range(4)]
    with pytest.raises(ValueError, match="offset not still available"):
        family.ll(np.array([-1., .3, 1.2]), X, jj, np.zeros(4), np.ones(3),
                  offset=[np.ones(3), None, None, None])
