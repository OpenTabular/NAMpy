"""Gaussian AR precision, reporting and smoothing checked against mgcv::bam."""

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
from nampy.gam.fit.selection.criteria import (
    criterion_gradient,
    criterion_hessian,
    criterion_value,
)
from nampy.gam.fit.selection.criteria.gaussian_dyn import (
    criterion_gradient_ml_reml_gaussian_dynamic_joint,
    criterion_hessian_ml_reml_gaussian_dynamic_joint,
    criterion_ml_reml_gaussian_dynamic_joint,
)
from nampy.gam.model_state import _coef_column_offset, _penalty_blocks_seq
from tests.reference_fixtures import load_reference, reference_key, save_reference

SCRIPT = r"""
a <- commandArgs(TRUE)
lib <- Sys.getenv('MGCV_LIB_PATH','')
if (nzchar(lib)) .libPaths(c(lib,.libPaths()))
suppressPackageStartupMessages(library(mgcv)); library(jsonlite)
# The recorded library has a documented local Newton-sign mutation. Restore
# the CRAN 1.9-4 expression in this temporary R process for selected fits.
newton_ref <- get('newton',asNamespace('mgcv'))
body(newton_ref) <- parse(text=sub('Sstep <- grad/max','Sstep <- -grad/max',
    paste(deparse(body(newton_ref)),collapse='\n'),fixed=TRUE))[[1]]
assignInNamespace('newton',newton_ref,ns='mgcv')
p <- fromJSON(a[1]); d <- as.data.frame(p$data); nd <- d[seq(1,nrow(d),7),]
sp <- if(p$selected) NULL else .7
fit <- if(p$starts) {
    bam(y~s(x,bs='cr',k=6)+offset(o),data=d,weights=w,rho=p$rho,
        AR.start=start,sp=sp,method='REML')
} else {
    # Make the implicit first series explicit upstream: bam's final AIC
    # counts NULL AR.start as zero starts, unlike its own selection criterion.
    bam(y~s(x,bs='cr',k=6)+offset(o),data=d,weights=w,rho=p$rho,
        AR.start=c(TRUE,rep(FALSE,nrow(d)-1)),sp=sp,method='REML')
}
pr <- predict(fit,nd,se.fit=TRUE)
upr <- predict(fit,nd,se.fit=TRUE,unconditional=TRUE)
lik <- c(as.numeric(logLik(fit)),AIC(fit),BIC(fit))
if(p$selected) {
    # Vb.corr differentiates a Cholesky root, which changes under a change of
    # coefficient coordinates. Evaluate that upstream correction in the
    # native basis after checking both the design and penalty transport.
    Xr <- predict(fit,d,type='lpmatrix'); Xp <- as.matrix(p$design)
    T <- qr.solve(Xp,Xr); Ti <- solve(T)
    stopifnot(max(abs(Xp %*% T-Xr)) < 5e-8)
    Sr <- matrix(0,ncol(Xr),ncol(Xr)); ii <- fit$smooth[[1]]$first.para:fit$smooth[[1]]$last.para
    Sr[ii,ii] <- fit$smooth[[1]]$S[[1]]
    Sp <- t(Ti) %*% Sr %*% Ti
    stopifnot(max(abs(Sp-as.matrix(p$penalty))) < 5e-8)
    arstart <- if(p$starts) d$start else c(TRUE,rep(FALSE,nrow(d)-1))
    whiten <- function(X) apply(sqrt(d$w)*X,2,function(v) mgcv:::AR.resid(v,p$rho,arstart))
    Xrw <- whiten(Xr); Xpw <- whiten(Xp)
    correction <- function(X,S) fit$sig2 * mgcv:::Vb.corr(
        X,L=NULL,lsp0=0,S=list(S),off=1,dw=NULL,w=NULL,
        rho=c(log(fit$sp),log(fit$sig2)),Vr=fit$V.sp,scale.est=TRUE)
    Vc <- T %*% (fit$Vc-correction(Xrw,Sr)) %*% t(T) + correction(Xpw,Sp)
    Xnp <- predict(fit,nd,type='lpmatrix') %*% Ti
    upr$se.fit <- sqrt(rowSums((Xnp %*% Vc)*Xnp))
    edf2 <- min(sum(Vc*crossprod(Xpw))/fit$sig2,sum(fit$edf1))
    df <- min(edf2+1,ncol(Xp)+1)
    lik[2:3] <- -2*lik[1]+c(2,log(nrow(d)))*df
}
rt <- c('working','response','pearson','scaled.pearson','deviance')
res <- lapply(rt,function(t) unname(residuals(fit,type=t))); names(res) <- rt
write_json(list(version=as.character(packageVersion('mgcv')),sp=unname(fit$sp),
                fit=unname(pr$fit),se=unname(pr$se.fit),unconditional_se=unname(upr$se.fit),residuals=res,
                scale=fit$sig2,edf=sum(fit$edf),deviance=fit$deviance,
                standardized=unname(fit$std.rsd),
                loglik=lik[1],aic=lik[2],bic=lik[3]),
           a[2],digits=17,auto_unbox=TRUE,na='null')
"""


def data_frame(weight_mode, offset):
    rng = np.random.default_rng(17563)
    x = rng.uniform(-1, 1, 90)
    o = 0.2 * np.cos(2 * x)
    y = 0.5 + np.sin(3 * x) + o + rng.normal(0, 0.3, len(x))
    weights = rng.uniform(0.2, 3, len(x))
    if weight_mode == "unit":
        weights[:] = 1
    elif weight_mode == "zero":
        weights[::11] = 0
    starts = np.zeros(len(x), dtype=bool)
    starts[[0, 45]] = True
    return pd.DataFrame(
        {"x": x, "y": y, "o": o if offset else 0, "w": weights, "start": starts}
    ).round(12)


def reference(data, rho, starts, selected, model):
    payload = {
        "data": data.to_dict("list"),
        "rho": rho,
        "starts": starts,
        "selected": selected,
    }
    if selected:
        payload["design"] = model.lpmatrix(data).round(10).tolist()
        penalty = np.zeros((len(model.fit_result().coef_full),) * 2)
        offset = _coef_column_offset(model)
        for block in _penalty_blocks_seq(model):
            columns = slice(
                block.coef_slice.start + offset, block.coef_slice.stop + offset
            )
            penalty[columns, columns] += block.matrix
        payload["penalty"] = penalty.round(10).tolist()
    key = reference_key(
        "bam-ar1-weighted-v1",
        {
            "payload": payload,
            "source_sha256": hashlib.sha256(SCRIPT.encode()).hexdigest(),
        },
    )
    result = load_reference("mgcv_ar1_weighted", key)
    if result is None:
        with tempfile.TemporaryDirectory() as directory:
            source, output = [
                Path(directory) / name for name in ("source.R", "out.json")
            ]
            source.write_text(SCRIPT)
            run = subprocess.run(
                ["Rscript", str(source), json.dumps(payload), str(output)],
                capture_output=True,
                text=True,
            )
            assert run.returncode == 0, run.stderr
            result = json.loads(output.read_text())
        save_reference("mgcv_ar1_weighted", key, result)
    return result


def fitted(data, rho=0.4, starts=True, selected=False):
    return GAM(
        formula="y~s(x,bs='cr',k=6)+offset(o)",
        ar1_rho=rho,
        ar_start="start" if starts else None,
        smoothing_params=None if selected else [0.7],
        optimize_smoothing=selected,
        smoothing_method="reml",
    ).fit(data=data, sample_weight="w")


def assert_matches_bam(data, rho, starts, selected):
    model = fitted(data, rho, starts, selected)
    expected = reference(data, rho, starts, selected, model)
    assert expected["version"] == "1.9.4"
    fit, se = model.predict(data.iloc[::7], return_se=True)
    tolerance = 2e-6 if selected else 2e-10
    np.testing.assert_allclose(fit, expected["fit"], atol=tolerance, rtol=tolerance)
    np.testing.assert_allclose(se, expected["se"], atol=tolerance, rtol=tolerance)
    if selected:
        _, unconditional_se = model.predict(
            data.iloc[::7], return_se=True, unconditional=True
        )
    else:
        with pytest.warns(
            UserWarning,
            match="Smoothness uncertainty corrected covariance not available",
        ):
            _, unconditional_se = model.predict(
                data.iloc[::7], return_se=True, unconditional=True
            )
    np.testing.assert_allclose(
        unconditional_se, expected["unconditional_se"], atol=tolerance, rtol=tolerance
    )
    result = model.fit_result()
    for actual, target in [
        (result.scale, "scale"),
        (result.edf_total, "edf"),
        (result.deviance, "deviance"),
    ]:
        np.testing.assert_allclose(
            actual, expected[target], atol=tolerance, rtol=tolerance
        )
    for rtype, values in expected["residuals"].items():
        np.testing.assert_allclose(
            model.residuals(type=rtype),
            np.asarray(values, dtype=float),
            atol=tolerance,
            rtol=tolerance,
            equal_nan=True,
            err_msg=rtype,
        )
    np.testing.assert_allclose(
        model.ar1_standardized_residuals(),
        expected["standardized"],
        atol=tolerance,
        rtol=tolerance,
    )
    if selected:
        np.testing.assert_allclose(
            model.smoothing_params, np.atleast_1d(expected["sp"]), rtol=2e-5, atol=2e-5
        )
    likelihood = [model.loglik(), model.aic(), model.bic()]
    if np.all(data.w > 0):
        np.testing.assert_allclose(
            likelihood,
            [expected[name] for name in ("loglik", "aic", "bic")],
            atol=tolerance,
            rtol=tolerance,
        )
    else:
        # Retain the documented finite zero-weight likelihood convention.
        assert np.all(np.isfinite(likelihood))
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.predict(data.iloc[::7]), fit)
    np.testing.assert_array_equal(
        restored.ar1_standardized_residuals(), model.ar1_standardized_residuals()
    )


@pytest.mark.parametrize("rho", [-0.35, 0.4])
@pytest.mark.parametrize("weight_mode", ["unit", "varying", "zero"])
@pytest.mark.parametrize("starts", [False, True])
@pytest.mark.parametrize("offset", [False, True])
def test_fixed_ar1_gaussian_fit_matches_bam(rho, weight_mode, starts, offset):
    assert_matches_bam(data_frame(weight_mode, offset), rho, starts, False)


@pytest.mark.parametrize("weight_mode", ["unit", "varying", "zero"])
def test_ar1_selected_smoothing_matches_bam(weight_mode):
    assert_matches_bam(data_frame(weight_mode, True), 0.4, True, True)


@pytest.mark.parametrize("weight_mode", ["unit", "varying", "zero"])
@pytest.mark.parametrize("method", ["gcv", "ml", "reml"])
def test_ar1_criterion_derivatives_use_the_same_whitened_system(weight_mode, method):
    model = fitted(data_frame(weight_mode, True))
    log_sp, step = np.log([0.7]), 1e-4
    grad = criterion_gradient(model, model.y_, log_sp, method)
    hess = criterion_hessian(model, model.y_, log_sp, method)
    numeric_grad = (
        criterion_value(model, model.y_, log_sp + step, method)
        - criterion_value(model, model.y_, log_sp - step, method)
    ) / (2 * step)
    numeric_hess = (
        criterion_gradient(model, model.y_, log_sp + step, method)
        - criterion_gradient(model, model.y_, log_sp - step, method)
    ) / (2 * step)
    np.testing.assert_allclose(grad, [numeric_grad], rtol=2e-5, atol=2e-8)
    np.testing.assert_allclose(hess, numeric_hess.reshape(1, 1), rtol=2e-5, atol=2e-8)


@pytest.mark.parametrize("method", ["ml", "reml"])
def test_joint_ar1_scale_and_smoothing_derivatives_match_differences(method):
    model = fitted(data_frame("varying", True))
    x, step = np.array([np.log(0.7), np.log(0.2)]), 1e-4

    def value(v):
        return criterion_ml_reml_gaussian_dynamic_joint(
            model, model.y_, v[:1], v[1], method
        )

    def gradient(v):
        return criterion_gradient_ml_reml_gaussian_dynamic_joint(
            model, model.y_, v[:1], v[1], method
        )

    analytic_gradient = gradient(x)
    analytic_hessian = criterion_hessian_ml_reml_gaussian_dynamic_joint(
        model, model.y_, x[:1], x[1], method
    )
    directions = np.eye(2) * step
    numeric_gradient = [(value(x + d) - value(x - d)) / (2 * step) for d in directions]
    numeric_hessian = np.column_stack(
        [(gradient(x + d) - gradient(x - d)) / (2 * step) for d in directions]
    )
    np.testing.assert_allclose(
        analytic_gradient, numeric_gradient, rtol=2e-5, atol=2e-7
    )
    np.testing.assert_allclose(analytic_hessian, numeric_hessian, rtol=2e-5, atol=2e-7)
