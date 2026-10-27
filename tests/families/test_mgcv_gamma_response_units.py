"""Gamma fitting and selection must respect valid positive response units."""

import hashlib
import json
import pickle
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from nampy.gam.families.registry import make_gam_family
from nampy.gam.fit.selection.criteria.dispatch import (
    criterion_gradient,
    criterion_hessian,
    criterion_value,
)
from tests.reference_fixtures import load_reference, reference_key, save_reference

UNITS = [1e-12, 1e-6, 1.0, 1e6, 1e12]
LINKS = ["log", "identity", "inverse"]
SCRIPT = r"""
a <- commandArgs(TRUE)
lib <- Sys.getenv('MGCV_LIB_PATH','')
if(nzchar(lib)) .libPaths(c(lib,.libPaths()))
suppressPackageStartupMessages(library(mgcv)); library(jsonlite)
# Restore the recorded library's local Newton-sign mutation in this process.
nr <- get('newton',asNamespace('mgcv'))
body(nr) <- parse(text=sub('Sstep <- grad/max','Sstep <- -grad/max',
    paste(deparse(body(nr)),collapse='\n'),fixed=TRUE))[[1]]
assignInNamespace('newton',nr,ns='mgcv')
p <- fromJSON(a[1]);family <- Gamma(link=p$link)
control <- gam.control(epsilon=1e-12,newton=list(conv.tol=1e-11))
if(p$operation=='kernel') {
    y <- p$y;mu <- p$mu;w <- p$weights
    family <- mgcv:::fix.family.var(mgcv:::fix.family.link(family))
    eta <- family$linkfun(mu)
    h <- if(p$link=='log') rep(.001,length(eta)) else abs(eta)*.001
    info <- function(e,fisher=FALSE) {
        m <- family$linkinv(e);d <- family$mu.eta(e);v <- family$variance(m)
        alpha <- if(fisher) 1 else 1+(y-m)*(family$dvar(m)/v+family$d2link(m)*d)
        alpha*d^2/v
    }
    fd <- function(fisher=FALSE) {
        fm2 <- info(eta-2*h,fisher);fm1 <- info(eta-h,fisher)
        f0 <- info(eta,fisher);fp1 <- info(eta+h,fisher);fp2 <- info(eta+2*h,fisher)
        list(first=(fm2-8*fm1+8*fp1-fp2)/(12*h),
             second=(-fp2+16*fp1-30*f0+16*fm1-fm2)/(12*h^2))
    }
    out <- list(deviance=family$dev.resids(y,mu,w),eta=eta,
                observed=fd(),fisher=fd(TRUE),
                scale=sum(w*(y-mu)^2/family$variance(mu))/(length(y)-1.4))
    if(!any(y<=0)) out$initial <- local({nobs<-length(y);weights<-w;
        eval(family$initialize);mustart})
} else {
    d <- as.data.frame(p$data)
    fit <- function(delta=0,method='REML')
        gam(y~s(x,bs='cr',k=6)+offset(o),data=d,weights=w,family=family,
            sp=p$sp*exp(delta),method=method,control=control)
    m <- fit();pr <- predict(m,type='response',se.fit=TRUE);ll <- logLik(m)
    h <- .01
    gcv <- sapply(c(-2,-1,0,1,2)*h,function(z) as.numeric(fit(z,'GCV.Cp')$gcv.ubre))
    types <- c('response','working','deviance','pearson','scaled.pearson')
    residual <- lapply(types,function(t) unname(residuals(m,type=t)));names(residual)<-types
    out <- list(mu=unname(pr$fit),se=unname(pr$se.fit),edf=sum(m$edf),
                scale=m$sig2,deviance=m$deviance,null_deviance=m$null.deviance,
                ll=as.numeric(ll),aic=AIC(m),bic=BIC(m),df=attr(ll,'df'),
                residuals=residual,gcv=gcv[3],
                gradient=(gcv[1]-8*gcv[2]+8*gcv[4]-gcv[5])/(12*h),
                hessian=(-gcv[5]+16*gcv[4]-30*gcv[3]+16*gcv[2]-gcv[1])/(12*h^2))
}
out$version <- as.character(packageVersion('mgcv'));out$r_version <- R.version.string
write_json(out,a[2],digits=17,auto_unbox=TRUE,na='null')
"""


SELECTED_SCRIPT = (
    SCRIPT.split("p <- fromJSON(a[1]);family <- Gamma(link=p$link)")[0]
    + r"""
p <- fromJSON(a[1]);d <- as.data.frame(p$data)
m <- gam(y~s(x,bs='cr',k=6)+offset(o),data=d,weights=w,
         family=Gamma(link=p$link),method='GCV.Cp',
         control=gam.control(epsilon=1e-12,newton=list(conv.tol=1e-11)))
ll <- logLik(m);pr <- predict(m,type='response',se.fit=TRUE)
write_json(list(version=as.character(packageVersion('mgcv')),r_version=R.version.string,
    mu=unname(pr$fit),se=unname(pr$se.fit),sp=unname(m$sp),score=unname(m$gcv.ubre),
    edf=sum(m$edf),scale=m$sig2,ll=as.numeric(ll),aic=AIC(m),bic=BIC(m)),
    a[2],digits=17,auto_unbox=TRUE,na='null')
"""
)


def reference(payload, script=SCRIPT):
    key = reference_key(
        "gamma-response-units-v1",
        {
            "payload": payload,
            "source_sha256": hashlib.sha256(script.encode()).hexdigest(),
        },
        normalize_floats=True,
    )
    result = load_reference("mgcv_gamma_response_units", key)
    if result is None:
        with tempfile.TemporaryDirectory() as directory:
            source, output = (
                Path(directory) / name for name in ("source.R", "out.json")
            )
            source.write_text(script)
            run = subprocess.run(
                ["Rscript", str(source), json.dumps(payload), str(output)],
                capture_output=True,
                text=True,
            )
            assert run.returncode == 0, run.stderr
            result = json.loads(output.read_text())
        save_reference("mgcv_gamma_response_units", key, result)
    assert result["version"] == "1.9.4"
    return result


@pytest.mark.parametrize("link", LINKS)
@pytest.mark.parametrize("unit", UNITS)
def test_gamma_kernels_and_working_weight_derivatives_across_units(link, unit):
    y = unit * np.array([0.3, 0.6, 1.3, 3])
    mu = unit * np.array([0.6, 0.8, 1, 2])
    w = np.array([0.5, 1, 2, 0])
    expected = reference(
        {
            "operation": "kernel",
            "link": link,
            "y": y.tolist(),
            "mu": mu.tolist(),
            "weights": w.tolist(),
        }
    )
    family = make_gam_family({"name": "gamma", "link": link})
    np.testing.assert_array_equal(family.initialize_mu(y), expected["initial"])
    np.testing.assert_allclose(
        family.deviance_obs(y, mu, w), expected["deviance"], atol=2e-13
    )
    np.testing.assert_allclose(
        family.deviance(y, mu, w), sum(expected["deviance"]), atol=2e-13
    )
    np.testing.assert_allclose(
        family.estimate_dispersion(y, mu, edf=1.4, weights=w),
        expected["scale"],
        rtol=2e-13,
    )
    eta = family.link(mu)
    for order, method in [
        (1, "working_weight_derivative_eta"),
        (2, "working_weight_second_derivative_eta"),
    ]:
        key = "first" if order == 1 else "second"
        np.testing.assert_allclose(
            getattr(family, method)(eta, y),
            expected["observed"][key],
            atol=0,
            rtol=3e-6,
        )
        if link != "log":
            np.testing.assert_allclose(
                getattr(family, method)(eta),
                expected["fisher"][key],
                atol=0,
                rtol=3e-6,
            )


def test_gamma_zero_response_deviance_retains_r_kernel_and_fit_rejects_zero():
    family = make_gam_family("gamma")
    y = np.array([0, 1e-12, 2e-12])
    mu = np.array([1e-12, 2e-12, 3e-12])
    w = np.array([1, 0.5, 2])
    expected = reference(
        {
            "operation": "kernel",
            "link": "inverse",
            "y": y.tolist(),
            "mu": mu.tolist(),
            "weights": w.tolist(),
        }
    )
    np.testing.assert_allclose(
        family.deviance_obs(y, mu, w), expected["deviance"], atol=1e-13
    )
    with pytest.raises(ValueError, match="strictly positive"):
        family.validate_y(y)


def data_frame(unit, link, zero_weights):
    rng = np.random.default_rng(40971)
    x = rng.uniform(0.1, 0.9, 70).round(12)
    mean = np.exp(0.4 + 0.5 * np.sin(4 * x))
    y = rng.gamma(12, mean / 12).round(12) * unit
    w = np.resize([0.5, 1, 2], len(x))
    if zero_weights:
        w[::11] = 0
    eta_unit = unit if link == "identity" else (1 / unit if link == "inverse" else 1)
    o = (0.07 * np.cos(3 * x)).round(12) * eta_unit
    sp = 0.7 * (
        1 / unit**2 if link == "identity" else (unit**2 if link == "inverse" else 1)
    )
    return pd.DataFrame({"x": x, "y": y, "w": w, "o": o}), sp, eta_unit


@pytest.mark.parametrize("link", LINKS)
@pytest.mark.parametrize("unit", UNITS)
@pytest.mark.parametrize("zero_weights", [False, True])
@pytest.mark.parametrize("offset_mode", ["formula", "api", "mixed"])
def test_gamma_fits_and_gcv_derivatives_across_response_units(
    link, unit, zero_weights, offset_mode
):
    data, sp, eta_unit = data_frame(unit, link, zero_weights)
    expected = reference(
        {
            "operation": "fit",
            "link": link,
            "data": data.to_dict("list"),
            "sp": sp,
        }
    )
    formula = "y~s(x,bs='cr',k=6)"
    fit_data = data.copy()
    offset = None
    if offset_mode == "formula":
        formula += "+offset(o)"
    elif offset_mode == "api":
        offset = data.o.to_numpy()
    else:
        formula += "+offset(o)"
        fit_data["o"] *= 0.5
        offset = data.o.to_numpy() * 0.5
    model = GAM(
        formula=formula,
        family={"name": "gamma", "link": link},
        smoothing_params=[sp],
        optimize_smoothing=False,
        smoothing_method="fixed",
        control={"epsilon": 1e-12},
    ).fit(data=fit_data, sample_weight="w", offset=offset)
    prediction, se = model.predict(return_se=True, offset=data.o.to_numpy())
    # Compare in response units: a fixed absolute tolerance would hide tiny fits.
    np.testing.assert_allclose(
        prediction / unit, np.asarray(expected["mu"]) / unit, atol=2e-9, rtol=2e-9
    )
    np.testing.assert_allclose(
        se / unit, np.asarray(expected["se"]) / unit, atol=2e-9, rtol=2e-9
    )
    for fitted in (model, pickle.loads(pickle.dumps(model))):
        for fun, key in [("loglik", "ll"), ("aic", "aic"), ("bic", "bic")]:
            np.testing.assert_allclose(
                getattr(fitted, fun)(), expected[key], atol=2e-7, rtol=2e-10
            )
        summary = fitted.summary()
        np.testing.assert_allclose(
            summary.dev_expl,
            (expected["null_deviance"] - expected["deviance"])
            / expected["null_deviance"],
            atol=2e-9,
        )
    result = model.fit_result()
    np.testing.assert_allclose(result.deviance, expected["deviance"], atol=2e-8)
    np.testing.assert_allclose(result.edf_total, expected["edf"], atol=2e-9)
    np.testing.assert_allclose(result.scale, expected["scale"], atol=2e-9)
    for kind, values in expected["residuals"].items():
        factor = unit if kind == "response" else (eta_unit if kind == "working" else 1)
        np.testing.assert_allclose(
            model.residuals(kind) / factor,
            np.asarray(values, float) / factor,
            atol=2e-8,
            rtol=2e-8,
            equal_nan=True,
            err_msg=kind,
        )
    rho = np.log([sp])
    np.testing.assert_allclose(
        criterion_value(model, model.y_, rho, "gcv"),
        expected["gcv"],
        atol=2e-10,
        rtol=2e-9,
    )
    np.testing.assert_allclose(
        criterion_gradient(model, model.y_, rho, "gcv"),
        [expected["gradient"]],
        atol=2e-8,
        rtol=3e-4,
    )
    np.testing.assert_allclose(
        criterion_hessian(model, model.y_, rho, "gcv"),
        [[expected["hessian"]]],
        atol=2e-8,
        rtol=3e-4,
    )


@pytest.mark.parametrize("link", LINKS)
@pytest.mark.parametrize("unit", [1e-12, 1e12])
def test_gamma_selected_gcv_across_response_units(link, unit):
    data, sp, _ = data_frame(unit, link, False)
    expected = reference(
        {"operation": "selected", "link": link, "data": data.to_dict("list")},
        script=SELECTED_SCRIPT,
    )
    # These are physical-unit smoothing parameters: identity and inverse links
    # need a search interval wide enough to contain their rescaled optima.
    model = GAM(
        formula="y~s(x,bs='cr',k=6)+offset(o)",
        family={"name": "gamma", "link": link},
        smoothing_params=[sp],
        optimize_smoothing=True,
        smoothing_method="GCV.Cp",
        sp_log_bounds=(-100, 100),
        control={"epsilon": 1e-12, "newton": {"conv_tol": 1e-11}},
    ).fit(data=data, sample_weight="w")
    assert model._optim_result.success
    prediction, se = model.predict(return_se=True)
    np.testing.assert_allclose(
        prediction / unit, np.asarray(expected["mu"]) / unit, atol=2e-8, rtol=2e-8
    )
    np.testing.assert_allclose(
        se / unit, np.asarray(expected["se"]) / unit, atol=2e-8, rtol=2e-8
    )
    np.testing.assert_allclose(
        np.log(model.smoothing_params), np.log(expected["sp"]), atol=1e-5, rtol=0
    )
    np.testing.assert_allclose(
        model.smoothing_score_, expected["score"], atol=2e-10, rtol=2e-9
    )
    np.testing.assert_allclose(model.fit_result().edf_total, expected["edf"], atol=2e-8)
    np.testing.assert_allclose(model.fit_result().scale, expected["scale"], atol=2e-9)
    for method, key in [("loglik", "ll"), ("aic", "aic"), ("bic", "bic")]:
        np.testing.assert_allclose(
            getattr(model, method)(), expected[key], atol=2e-7, rtol=2e-10
        )
