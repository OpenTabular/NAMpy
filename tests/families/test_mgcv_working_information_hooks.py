"""Family working-information hooks against independent mgcv calculations."""

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from nampy.gam.families.registry import make_gam_family
from tests.reference_fixtures import load_reference, reference_key, save_reference

SCRIPT = r"""
a <- commandArgs(TRUE)
lib <- Sys.getenv('MGCV_LIB_PATH','')
if(nzchar(lib)) .libPaths(c(lib,.libPaths()))
suppressPackageStartupMessages(library(mgcv)); library(jsonlite)
p <- fromJSON(a[1])
if(p$operation=='density') {
    family <- mgcv:::fix.family.ls(gaussian())
    out <- list(density=dnorm(p$y,p$mu,sqrt(p$scale),log=TRUE),
                ls=family$ls(p$y,p$weights,length(p$y),p$scale))
} else {
    family <- mgcv:::fix.family.var(mgcv:::fix.family.link(eval(parse(text=p$family))))
    if(p$nb) {
        # nb exposes scaled g2g/g3g/g4g instead of ordinary link derivatives.
        # Obtain the ordinary fixed-theta link/variance hooks from negbin.
        ordinary <- mgcv:::fix.family.var(mgcv:::fix.family.link(
            negbin(theta=family$getTheta(TRUE),link=family$link)))
        for(nm in c('d2link','d3link','d4link','dvar','d2var','d3var'))
            family[[nm]] <- ordinary[[nm]]
        inverse_expression <- switch(family$link,log=quote(exp(z)),
                                     identity=quote(z),sqrt=quote(z^2))
        inverse_derivative <- function(order,e) {
            expression <- inverse_expression
            for(i in seq_len(order)) expression <- D(expression,'z')
            eval(expression,list(z=e))
        }
    }
    eta <- p$eta;y <- p$y;mu <- family$linkinv(eta);me <- family$mu.eta(eta)
    canonical <- isTRUE(family$link==family$canonical)
    info <- function(e,fisher=FALSE) {
        m <- family$linkinv(e);g <- family$mu.eta(e);v <- family$variance(m)
        if(fisher || canonical) return(g^2/v)
        if(p$nb) {
            dd <- family$Dd(y,m,family$getTheta(),rep(1,length(y)),level=0)
            b <- inverse_derivative(2,e)
            return((dd$Dmu2*g^2+dd$Dmu*b)/2)
        }
        alpha <- 1+(y-m)*(family$dvar(m)/v+family$d2link(m)*g)
        alpha*g^2/v
    }
    fd <- function(fisher) {
        h <- p$eta_scale*.001
        fm2<-info(eta-2*h,fisher);fm1<-info(eta-h,fisher)
        f0<-info(eta,fisher);fp1<-info(eta+h,fisher);fp2<-info(eta+2*h,fisher)
        list(first=(fm2-8*fm1+8*fp1-fp2)/(12*h),
             second=(-fp2+16*fp1-30*f0+16*fm1-fm2)/(12*h^2))
    }
    analytic <- function(fisher) {
        g1<-1/me;g2<-family$d2link(mu)/g1;g3<-family$d3link(mu)/g1
        V<-family$variance(mu);V1<-family$dvar(mu)/V;V2<-family$d2var(mu)/V
        wf<-me^2/V
        if(fisher || canonical) {
            d1 <- -wf*(V1+2*g2)/g1
            d2 <- d1*(d1/wf-g2/g1)-wf*(V2-V1^2+2*g3-2*g2^2)/g1^2
        } else if(p$nb) {
            dd<-family$Dd(y,mu,family$getTheta(),rep(1,length(y)),level=2)
            # Exact R-side differentiation of the actual inverse links avoids
            # cancelling large terms to obtain the zero sqrt third/fourth derivatives.
            b <- inverse_derivative(2,eta)
            c <- inverse_derivative(3,eta)
            d <- inverse_derivative(4,eta)
            d1 <- (dd$Dmu3*me^3+3*dd$Dmu2*me*b+dd$Dmu*c)/2
            d2 <- (dd$Dmu4*me^4+6*dd$Dmu3*me^2*b+3*dd$Dmu2*b^2+4*dd$Dmu2*me*c+dd$Dmu*d)/2
        } else {
            # gam.fit3's normalized link/variance derivatives, independently
            # of the native deviance-chain and closed-form family hooks.
            g4<-family$d4link(mu)/g1;V3<-family$d3var(mu)/V
            alpha<-1+(y-mu)*(V1+g2);alpha[alpha==0]<-.Machine$double.eps
            alpha1<-(-(V1+g2)+(y-mu)*(V2-V1^2+g3-g2^2))/alpha
            alpha2<-(-2*(V2-V1^2+g3-g2^2)+(y-mu)*(V3-3*V1*V2+2*V1^3+g4-3*g3*g2+2*g2^3))/alpha
            w<-wf*alpha
            d1<-w*(alpha1-V1-2*g2)/g1
            d2<-d1*(d1/w-g2/g1)-w*(alpha1^2-alpha2+V2-V1^2+2*g3-2*g2^2)/g1^2
        }
        list(first=d1,second=d2)
    }
    out <- list(mu=mu,fisher_information=info(eta,TRUE),
                fisher=analytic(TRUE),observed=analytic(FALSE),
                fd_fisher=fd(TRUE),fd_observed=fd(FALSE))
}
out$version<-as.character(packageVersion('mgcv'));out$r_version<-R.version.string
write_json(out,a[2],digits=17,auto_unbox=TRUE,na='string')
"""


def reference(payload):
    key = reference_key(
        "working-information-hooks-v1",
        {
            "payload": payload,
            "source_sha256": hashlib.sha256(SCRIPT.encode()).hexdigest(),
        },
    )
    result = load_reference("mgcv_working_information_hooks", key)
    if result is None:
        with tempfile.TemporaryDirectory() as directory:
            source, output = (
                Path(directory) / name for name in ("source.R", "out.json")
            )
            source.write_text(SCRIPT)
            run = subprocess.run(
                ["Rscript", str(source), json.dumps(payload), str(output)],
                capture_output=True,
                text=True,
            )
            assert run.returncode == 0, run.stderr
            result = json.loads(output.read_text())
        save_reference("mgcv_working_information_hooks", key, result)
    assert result["version"] == "1.9.4"
    return result


CASES = []


def case(name, link, eta, *, y=None, theta=None, fd=True, label="normal"):
    family = {"name": name, "link": link}
    if theta is not None:
        family["theta"] = theta
    eta = np.asarray(eta, float)
    native = make_gam_family(family)
    if y is None:
        y = native.inverse_link(eta) * np.array([0.2, 0.8, 1.6, 4.7])
    eta_scale = (
        np.ones(len(eta))
        if link in {"log", "logit", "probit", "cloglog", "cauchit"}
        else np.abs(eta)
    )
    r_name = {"gamma": "Gamma", "inverse_gaussian": "inverse.gaussian", "nb": "nb"}.get(
        name, name
    )
    r_family = (
        f"{r_name}(link='{link}'"
        + (f",theta={theta}" if theta is not None else "")
        + ")"
    )
    CASES.append(
        pytest.param(
            family,
            {
                "operation": "derivative",
                "family": r_family,
                "eta": eta.tolist(),
                "eta_scale": eta_scale.tolist(),
                "y": np.asarray(y, float).tolist(),
                "nb": name == "nb",
            },
            fd,
            id=f"{name}-{link}-{theta}-{label}",
        )
    )


case("gaussian", "identity", [-2, -0.4, 0.7, 2])
case("gaussian", "log", [-18, -2, 0.4, 10])
case("gaussian", "log", [-100, -50, -40, -36.1], fd=False, label="clipped-tail")
for factor in [1e-4, 1, 1e4]:
    case(
        "gaussian",
        "inverse",
        np.array([-3.1, -0.4, 0.3, 2.7]) * factor,
        label=str(factor),
    )
case("poisson", "log", [-10, -1, 0.4, 3], y=[0, 1, 3, 9])
case("poisson", "identity", [1e-8, 0.3, 1.4, 7], y=[0, 1, 3, 9])
case("poisson", "sqrt", [1e-5, 0.4, 1.3, 3], y=[0, 1, 3, 9])
case("binomial", "logit", [-2, -0.4, 0.7, 2.1], y=[0, 0.25, 0.8, 1])
case("binomial", "probit", [-2, -0.4, 0.7, 2.1], y=[0, 0.25, 0.8, 1])
case(
    "binomial",
    "probit",
    [-12, -9, 9, 12],
    y=[0, 0, 1, 1],
    fd=False,
    label="clipped-tails",
)
case("binomial", "cloglog", [-2, -0.5, 0.3, 1], y=[0, 0.25, 0.8, 1])
case("binomial", "cloglog", [-30, -20, -10, -4], y=[0, 0, 0, 0.25], label="lower-tail")
case("binomial", "cauchit", [-4, -0.3, 0.6, 3], y=[0, 0.25, 0.8, 1])
case("binomial", "log", [-3, -1.4, -0.7, -0.1], y=[0, 0.25, 0.8, 1])
case("gamma", "log", [-20, -0.6, 0.3, 2])
case("gamma", "identity", [0.02, 0.4, 1.3, 4])
case("gamma", "inverse", [0.1, 0.6, 1.4, 4])
for link in ["1/mu^2", "log", "identity", "inverse"]:
    eta = [-2, -0.6, 0.3, 2] if link == "log" else [0.1, 0.5, 1.5, 3]
    case("inverse_gaussian", link, eta)
case("quasipoisson", "sqrt", [0.1, 0.4, 1.3, 3], y=[0, 1, 3, 9])
case("quasibinomial", "probit", [-2, -0.4, 0.7, 2.1], y=[0, 0.25, 0.8, 1])
for theta in [0.4, 3.2, 50]:
    case("nb", "log", [-25, -2, 0.5, 8], y=[0, 1, 3, 9], theta=theta)
    case("nb", "identity", [1e-12, 0.3, 1.3, 9], y=[0, 1, 3, 9], theta=theta)
    case("nb", "sqrt", [1e-6, 0.4, 1.3, 4], y=[0, 1, 3, 9], theta=theta)


@pytest.mark.parametrize("family,payload,fd", CASES)
@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("observed", [False, True])
def test_family_information_hooks_match_mgcv(family, payload, fd, order, observed):
    native = make_gam_family(family)
    eta = np.array(payload["eta"])
    y = np.array(payload["y"])
    method = (
        "working_weight_derivative_eta"
        if order == 1
        else "working_weight_second_derivative_eta"
    )
    if family["name"] == "gamma" and family["link"] == "log" and not observed:
        with pytest.raises(ValueError, match="requires targets"):
            getattr(native, method)(eta)
        return
    expected = reference(payload)
    np.testing.assert_allclose(
        native.inverse_link(eta), expected["mu"], atol=0, rtol=2e-13
    )
    actual = getattr(native, method)(eta, y if observed else None)
    field = "observed" if observed else "fisher"
    key = "first" if order == 1 else "second"
    # Dimensionless comparisons expose errors even when raw derivatives are tiny.
    factor = np.array(payload["eta_scale"]) ** order / np.array(
        expected["fisher_information"]
    )
    np.testing.assert_allclose(
        actual * factor,
        np.array(expected[field][key], float) * factor,
        atol=2e-8,
        rtol=2e-8,
    )
    if fd:
        np.testing.assert_allclose(
            actual * factor,
            np.array(expected["fd_" + field][key], float) * factor,
            atol=3e-6,
            rtol=3e-6,
        )
    if family["name"] == "nb" and observed:
        np.testing.assert_allclose(actual, expected[field][key], atol=0, rtol=2e-8)


@pytest.mark.parametrize("scale", [-1, 0, 1e-15, 1e-9, 0.2, 1e8, 1e300])
@pytest.mark.parametrize("zero_weights", [False, True])
def test_gaussian_density_and_saturated_likelihood_preserve_dispersion(
    scale, zero_weights
):
    mu = np.array([-0.7, 1, 0.3, 2])
    y = (
        (mu + np.sqrt(scale) * np.array([-0.8, 0, 0.4, 1.5])).round(12)
        if scale > 0
        else mu + [-1, 0, 0.3, 2]
    )
    if scale == 1e300:
        y[-1] = 1e200  # Squaring before dividing by dispersion would overflow.
    weights = np.array([0.5, 0, 2, 1]) if zero_weights else np.ones(4)
    payload = {
        "operation": "density",
        "scale": scale,
        "mu": mu.tolist(),
        "y": y.tolist(),
        "weights": weights.tolist(),
    }
    expected = reference(payload)
    family = make_gam_family("gaussian")
    np.testing.assert_allclose(
        family.loglik_obs(y, mu, scale=scale),
        np.asarray(expected["density"], float),
        atol=2e-10,
        rtol=2e-12,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        family.ls(y, weights, scale=scale),
        np.asarray(expected["ls"], float),
        atol=2e-10,
        rtol=2e-12,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        family.saturated_loglik(y, weights=weights, scale=scale),
        float(expected["ls"][0]),
        atol=2e-10,
        rtol=2e-12,
        equal_nan=True,
    )
