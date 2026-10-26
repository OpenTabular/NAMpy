"""Static source kernels and constructor semantics for mgcv bcg."""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from nampy.gam.families.bcg import CensoredBoxCoxGaussianFamily
from nampy.gam.response import normalize_censored_response
from tests.mgcv_parity_utils import _build_r_command
from tests.reference_fixtures import load_reference, reference_key, save_reference

SCRIPT = r'''
suppressPackageStartupMessages(library(mgcv));library(jsonlite)
a<-commandArgs(TRUE);p<-fromJSON(a[1],simplifyVector=TRUE)
f<-do.call(bcg,list(theta=p$ctor,link=p$link))
y<-p$y
if(!is.null(p$bound)) attr(y,"censor")<-as.numeric(p$bound)
out<-list(theta=f$getTheta(),trans=f$getTheta(TRUE),n=f$n.theta,ini=f$ini.theta)
if(!is.null(p$mu)) {
 out$Dd<-f$Dd(y,p$mu,p$theta,p$w,level=p$level)
 out$dev<-unclass(f$dev.resids(y,p$mu,p$w,p$theta))
 out$aic<-f$aic(y,p$mu,p$theta,p$w,NULL)
 out$ls<-f$ls(y,p$w,p$theta,1)
}
if(!is.null(p$init_sign)) {
 e<-new.env();e$family<-f;e$y<-if(is.null(p$bound)) y else cbind(p$y,as.numeric(p$bound));eval(f$initialize,e);out$mustart<-e$mustart
 out$sign<-attr(f$dev.resids(y,p$mu,p$w,p$theta),"sign")
}
if(!is.null(p$efs)) {
 warnings<-character();out$efs<-withCallingHandlers(mgcv:::estimate.theta(f$getTheta(),f,y,p$mu,scale=p$scale,wt=p$w,tol=1e-7),warning=function(w){warnings<<-c(warnings,conditionMessage(w));invokeRestart("muffleWarning")});out$warnings<-warnings
}
write_json(out,a[2],digits=17,auto_unbox=TRUE,na="null")
'''


def reference(payload):
    key = reference_key("bcg-source-v1", payload)
    value = load_reference("mgcv_bcg_family", key)
    if value is not None:
        return value
    with tempfile.TemporaryDirectory() as tmp:
        paths = [Path(tmp) / name for name in ("source.R", "input.json", "out.json")]
        paths[0].write_text(SCRIPT)
        paths[1].write_text(json.dumps(payload))
        result = subprocess.run(_build_r_command(paths[0], str(paths[1]), str(paths[2])), capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        value = json.loads(paths[2].read_text())
    save_reference("mgcv_bcg_family", key, value)
    return value


@pytest.mark.parametrize("theta", [None, [1., 2.], [.3, .8], [.7, -.8], [1., 0.]])
def test_bcg_source_constructor(theta):
    p = {"ctor": theta, "link": "identity", "y": [1., 2.]}
    expected = reference(p)
    f = CensoredBoxCoxGaussianFamily(theta)
    np.testing.assert_array_equal(f.getTheta(), expected["theta"])
    np.testing.assert_allclose(f.getTheta(True), expected["trans"], rtol=1e-15)
    assert f.n_theta == expected["n"]


def kernel_payload(lam, kind, level):
    p = {"link": "identity", "ctor": None, "level": level,
         "theta": [lam, -.3], "y": [.3, .8, 1.2, 2.5, 4., 5., 1.6, .6],
         "mu": [-.7, .2, .6, -.1, 1.4, .8, .3, -.3], "w": [1., .5, 2., 1., 3., 1.5, .8, 2.]}
    if kind == "mixed":
        p["bound"] = [.3, 0., 1.9, 1.8, "Inf", 5., .7, 0.]
    return p


@pytest.mark.parametrize("lam", [0., 1e-8, .7, -.3])
@pytest.mark.parametrize("kind", ["exact", "mixed"])
@pytest.mark.parametrize("level", [0, 1, 2])
def test_bcg_derivatives_and_likelihood_against_source(lam, kind, level):
    p = kernel_payload(lam, kind, level)
    expected = reference(p)
    f = CensoredBoxCoxGaussianFamily()
    y = np.array(p["y"]) if "bound" not in p else normalize_censored_response(np.column_stack([p["y"], p["bound"]]))
    mu, w = np.array(p["mu"]), np.array(p["w"])
    actual = f.Dd(y, mu, np.array(p["theta"]), w, level)
    assert set(actual) == set(expected["Dd"])
    for key in actual:
        np.testing.assert_allclose(actual[key], np.asarray(expected["Dd"][key], dtype=float), atol=2e-9, rtol=2e-9, err_msg=key)
    np.testing.assert_allclose(f.deviance_obs(y, mu, w, p["theta"]), expected["dev"], atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(f.aic(y, mu, theta=p["theta"], wt=w), expected["aic"], atol=2e-12, rtol=2e-12)
    saturated = f.ls(y, w, p["theta"])
    for key in saturated:
        np.testing.assert_allclose(saturated[key], expected["ls"][key], atol=2e-12, rtol=2e-12)


def test_bcg_zero_endpoint_sign_branches_against_source():
    p = kernel_payload(.7, "mixed", 2)
    p["y"][2], p["bound"][2], p["mu"][2] = 1., 1.5, 0.
    expected = reference(p)
    f = CensoredBoxCoxGaussianFamily()
    y = normalize_censored_response(np.column_stack([p["y"], p["bound"]]))
    actual = f.Dd(y, np.array(p["mu"]), np.array(p["theta"]), np.array(p["w"]), 2)
    for key in actual:
        np.testing.assert_allclose(actual[key], np.asarray(expected["Dd"][key], dtype=float), atol=2e-9, rtol=2e-9, err_msg=key)


@pytest.mark.parametrize("initial", [None, [.7, -.8]])
@pytest.mark.parametrize("kind", ["exact", "mixed"])
@pytest.mark.parametrize("scale", [1., 2.])
def test_bcg_efs_vector_theta_update_against_source(initial, kind, scale):
    p = kernel_payload(.7, kind, 2)
    p.update(ctor=initial, efs=True, scale=scale)
    expected = reference(p)
    family = CensoredBoxCoxGaussianFamily(initial)
    y = np.array(p["y"]) if "bound" not in p else normalize_censored_response(np.column_stack([p["y"], p["bound"]]))
    family.update_theta_efs(y, np.array(p["mu"]), np.array(p["w"]), scale=scale)
    np.testing.assert_allclose(family.getTheta(), expected["efs"], rtol=2e-8, atol=2e-8)


@pytest.mark.parametrize("link", ["identity", "log", "sqrt"])
def test_bcg_response_initialization_and_deviance_sign(link):
    p = kernel_payload(.7, "mixed", 0)
    p.update(link=link, init_sign=True)
    expected = reference(p)
    family = CensoredBoxCoxGaussianFamily(link=link)
    family.putTheta(p["theta"])
    y = normalize_censored_response(np.column_stack([p["y"], p["bound"]]))
    np.testing.assert_array_equal(family.initialize_mu(y), expected["mustart"])
    np.testing.assert_array_equal(family.deviance_residual_sign(y, np.array(p["mu"])), expected["sign"])
