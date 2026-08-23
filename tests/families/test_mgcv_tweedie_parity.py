"""Direct Tweedie kernel parity against vendored ``mgcv``."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from nampy.gam.families.registry import make_gam_family
from nampy.gam.fit.selection.criteria import (
    criterion_gradient,
    criterion_hessian,
    criterion_value,
)
from tests._paths import REPO_ROOT
from tests.mgcv_parity_utils import _build_r_command
from tests.reference_fixtures import load_reference, reference_key, save_reference


def _run_r_json(code: str, payload: dict) -> dict | list:
    key = reference_key(
        "tweedie_r_json",
        {"code": code, "payload": payload},
        normalize_floats=True,
    )
    cached = load_reference("mgcv", key)
    if cached is not None:
        return cached
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        script = root / "probe.R"
        output = root / "output.json"
        script.write_text(code, encoding="utf-8")
        subprocess.run(
            _build_r_command(script, json.dumps(payload), str(output)),
            check=True,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        result = json.loads(output.read_text(encoding="utf-8"))
        save_reference("mgcv", key, result)
        return result


def _r_ld(payload: dict, *, working: bool, all_derivs: bool = False):
    code = r'''
suppressPackageStartupMessages(library(mgcv))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
p <- fromJSON(args[[1]])
y <- as.numeric(p$y); mu <- as.numeric(p$mu)
if (isTRUE(p$working)) {
  ans <- mgcv:::ldTweedie(y, mu, rho=as.numeric(p$rho),
                          theta=as.numeric(p$theta), a=p$a, b=p$b,
                          all.derivs=isTRUE(p$all_derivs))
} else {
  ans <- mgcv:::ldTweedie(y, mu, p=as.numeric(p$p),
                          phi=as.numeric(p$phi), a=p$a, b=p$b,
                          all.derivs=isTRUE(p$all_derivs))
}
write_json(unname(ans), args[[2]], digits=17, auto_unbox=FALSE, na="null")
'''
    return np.asarray(
        _run_r_json(code, {**payload, "working": working, "all_derivs": all_derivs}),
        dtype=np.float64,
    )


def test_ld_tweedie_matches_mgcv_fixed_parameter_vector_path():
    payload = {
        "y": [0.0, 0.2, 1.3, 4.5, 0.7, 2.1],
        "mu": [0.4, 0.8, 1.1, 3.0, 1.0, 2.5],
        "p": [1.2, 1.45, 1.7, 1.85, 1.3, 1.6],
        "phi": [0.4, 0.7, 1.1, 0.9, 1.7, 0.55],
        "a": 1.01,
        "b": 1.99,
    }
    expected = _r_ld(payload, working=False)
    from nampy.gam.families.tweedie import ldTweedie

    actual = ldTweedie(**{k: payload[k] for k in ("y", "mu", "p", "phi")}, a=1.01, b=1.99)
    np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-12)


def test_ld_tweedie_matches_mgcv_working_parameter_vector_and_mu_derivatives():
    payload = {
        "y": [0.0, 0.2, 1.3, 4.5, 0.7, 2.1],
        "mu": [0.4, 0.8, 1.1, 3.0, 1.0, 2.5],
        "rho": [-0.9, -0.2, 0.1, 0.6, -0.4, 0.3],
        "theta": [-1.3, -0.4, 0.0, 0.7, 1.2, 2.1],
        "a": 1.01,
        "b": 1.99,
    }
    expected = _r_ld(payload, working=True, all_derivs=True)
    from nampy.gam.families.tweedie import ldTweedie

    actual = ldTweedie(
        payload["y"],
        payload["mu"],
        rho=payload["rho"],
        theta=payload["theta"],
        a=1.01,
        b=1.99,
        all_derivs=True,
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-12)


def test_tweedie_family_kernels_match_mgcv():
    payload = {
        "y": [0.0, 0.2, 1.3, 4.5, 0.7, 2.1],
        "mu": [0.4, 0.8, 1.1, 3.0, 1.0, 2.5],
        "wt": [0.0, 1.0, 2.5, 0.4, 1.0, 1.7],
        "eta": [-0.9, -0.2, 0.1, 0.6, -0.4, 0.3],
        "theta": -1.3,
        "scale": 0.73,
        "a": 1.01,
        "b": 1.99,
    }
    code = r'''
suppressPackageStartupMessages(library(mgcv))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
p <- fromJSON(args[[1]])
y <- as.numeric(p$y); mu <- as.numeric(p$mu); wt <- as.numeric(p$wt)
eta <- as.numeric(p$eta); th <- as.numeric(p$theta); scale <- as.numeric(p$scale)
fam <- mgcv::tw(theta=th, link="log", a=p$a, b=p$b)
dd <- fam$Dd(y, mu, th, wt, level=2)
dev <- sum(fam$dev.resids(y, mu, wt, theta=th))
ls <- fam$ls(y, wt, th, scale)
ans <- list(
  dev = unname(as.numeric(fam$dev.resids(y, mu, wt, theta=th))),
  variance = unname(as.numeric(fam$variance(mu))),
  linkinv = unname(as.numeric(fam$linkinv(eta))),
  linkfun = unname(as.numeric(fam$linkfun(mu))),
  mustart = unname(as.numeric({ ev <- new.env(); assign("y", y, ev); eval(fam$initialize, ev); get("mustart", ev) })),
  dd = lapply(dd, unname),
  ls = list(ls=unname(as.numeric(ls$ls)), lsth1=unname(as.numeric(ls$lsth1)),
            LSTH1=unname(as.matrix(ls$LSTH1)), lsth2=unname(as.matrix(ls$lsth2))),
  aic = fam$aic(y, mu, theta=th, wt=wt, dev=dev)
)
write_json(ans, args[[2]], digits=17, auto_unbox=TRUE, na="null")
'''
    expected = _run_r_json(code, payload)
    family = make_gam_family(
        {"name": "tw", "link": "log", "theta": payload["theta"], "a": 1.01, "b": 1.99}
    )
    y = np.asarray(payload["y"], dtype=np.float64)
    mu = np.asarray(payload["mu"], dtype=np.float64)
    wt = np.asarray(payload["wt"], dtype=np.float64)
    eta = np.asarray(payload["eta"], dtype=np.float64)

    np.testing.assert_allclose(
        family.deviance_obs(y, mu, wt, theta=payload["theta"]),
        expected["dev"],
        rtol=2e-10,
        atol=2e-12,
    )
    np.testing.assert_allclose(family.variance(mu), expected["variance"], rtol=2e-12, atol=2e-13)
    np.testing.assert_allclose(family.inverse_link(eta), expected["linkinv"], rtol=2e-12, atol=2e-13)
    np.testing.assert_allclose(family.link(mu), expected["linkfun"], rtol=2e-12, atol=2e-13)
    np.testing.assert_allclose(family.initialize_mu(y), expected["mustart"], rtol=0, atol=0)

    actual_dd = family.Dd(y, mu, theta=payload["theta"], wt=wt, level=2)
    for key, value in expected["dd"].items():
        np.testing.assert_allclose(actual_dd[key], value, rtol=2e-9, atol=2e-11, err_msg=key)

    actual_ls = family.ls(y, wt, theta=payload["theta"], scale=payload["scale"])
    for key in ("ls", "lsth1", "LSTH1", "lsth2"):
        np.testing.assert_allclose(actual_ls[key], expected["ls"][key], rtol=2e-9, atol=2e-11, err_msg=key)

    dev = float(np.sum(family.deviance_obs(y, mu, wt, theta=payload["theta"])))
    np.testing.assert_allclose(
        family.aic(y, mu, theta=payload["theta"], weights=wt, dev=dev),
        expected["aic"],
        rtol=2e-9,
        atol=2e-11,
    )


def test_ld_tweedie_matches_mgcv_poisson_and_gamma_endpoints():
    from nampy.gam.families.tweedie import ldTweedie

    cases = [
        {
            "y": [0.0, 0.5, 1.0, 2.5],
            "mu": [0.2, 0.7, 1.2, 2.0],
            "p": 1.0,
            "phi": 0.5,
            "a": 1.01,
            "b": 1.99,
        },
        {
            "y": [0.1, 0.5, 1.0, 2.5],
            "mu": [0.2, 0.7, 1.2, 2.0],
            "p": 2.0,
            "phi": 0.7,
            "a": 1.01,
            "b": 1.99,
        },
    ]
    for payload in cases:
        expected = _r_ld(payload, working=False)
        actual = ldTweedie(
            payload["y"],
            payload["mu"],
            p=payload["p"],
            phi=payload["phi"],
            a=payload["a"],
            b=payload["b"],
        )
        np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-12)


def test_tweedie_fixed_sp_gam_fit_matches_mgcv():
    x = np.linspace(0.02, 0.98, 72)
    y = np.exp(0.15 + 0.7 * np.sin(2.0 * np.pi * x) + 0.25 * x)
    y[::9] = 0.0
    sp = 0.6
    payload = {"x": x.tolist(), "y": y.tolist(), "sp": sp}
    code = r'''
suppressPackageStartupMessages(library(mgcv))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
p <- fromJSON(args[[1]])
dat <- data.frame(x=as.numeric(p$x), y=as.numeric(p$y))
fit <- gam(y ~ s(x, bs="cr", k=8), data=dat,
           family=tw(theta=1.5, a=1.01, b=1.99),
           method="REML", sp=as.numeric(p$sp))
ans <- list(
  fitted=unname(as.numeric(fitted(fit))),
  response=unname(as.numeric(predict(fit, type="response"))),
  deviance=unname(as.numeric(deviance(fit))),
  scale=unname(as.numeric(fit$scale)),
  pearson_scale=unname(as.numeric(sum((dat$y-fitted(fit))^2 /
                                      fitted(fit)^fit$family$getTheta(TRUE)) /
                                      (nrow(dat)-sum(fit$edf)))),
  edf=unname(as.numeric(sum(fit$edf))),
  theta=unname(as.numeric(fit$family$getTheta(TRUE))),
  sp=unname(as.numeric(fit$full.sp))
)
write_json(ans, args[[2]], digits=17, auto_unbox=TRUE)
'''
    expected = _run_r_json(code, payload)
    data = pd.DataFrame({"x": x, "y": y})
    gam = GAM(
        family={"name": "tw", "theta": 1.5, "a": 1.01, "b": 1.99},
        formula='y ~ s(x, bs="cr", k=8)',
        optimize_smoothing=False,
        smoothing_method="fixed",
        smoothing_params=np.asarray([sp]),
    ).fit(data=data)
    actual = np.asarray(gam.predict(data), dtype=np.float64)
    fit_result = gam.fit_result(include_covariances=False)
    np.testing.assert_allclose(actual, expected["response"], rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(fit_result.deviance, expected["deviance"], rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(fit_result.edf_total, expected["edf"], rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(fit_result.scale, expected["pearson_scale"], rtol=2e-6, atol=2e-7)
    # Fixed-sp PIRLS endpoint matches. ``tw`` REML scale profiling is a
    # separate outer-family-parameter path and is not claimed by this test.
    np.testing.assert_allclose(gam.family.p, expected["theta"], rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(fit_result.smoothing_params, expected["sp"], rtol=2e-10, atol=2e-10)


@pytest.mark.parametrize("method", ["REML", "ML"])
@pytest.mark.parametrize("optimizer", ["outer_newton", "bfgs", "optim"])
def test_tweedie_joint_outer_matches_mgcv(method, optimizer):
    x = np.linspace(0.02, 0.98, 72)
    y = np.exp(0.15 + 0.7 * np.sin(2.0 * np.pi * x) + 0.25 * x)
    y[::9] = 0.0
    payload = {"x": x.tolist(), "y": y.tolist(), "method": method}
    code = r'''
suppressPackageStartupMessages(library(mgcv))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
p <- fromJSON(args[[1]])
dat <- data.frame(x=as.numeric(p$x), y=as.numeric(p$y))
fit <- gam(y ~ s(x, bs="cr", k=8), data=dat,
           family=tw(theta=-1.3, a=1.01, b=1.99), method=as.character(p$method))
ans <- list(
  response=unname(as.numeric(predict(fit, type="response"))),
  deviance=unname(as.numeric(deviance(fit))),
  outer_scale=unname(as.numeric(fit$scale)),
  pearson_scale=unname(as.numeric(sum((dat$y-fitted(fit))^2 /
                                      fitted(fit)^fit$family$getTheta(TRUE)) /
                                      (nrow(dat)-sum(fit$edf)))),
  edf=unname(as.numeric(sum(fit$edf))),
  theta=unname(as.numeric(fit$family$getTheta(TRUE))),
  sp=unname(as.numeric(fit$full.sp)),
  score=unname(as.numeric(fit$gcv.ubre))
)
write_json(ans, args[[2]], digits=17, auto_unbox=TRUE)
'''
    expected = _run_r_json(code, payload)
    data = pd.DataFrame({"x": x, "y": y})
    gam = GAM(
        family={"name": "tw", "theta": -1.3, "a": 1.01, "b": 1.99},
        formula='y ~ s(x, bs="cr", k=8)',
        optimize_smoothing=True,
        smoothing_method=method,
        smoothing_optimizer=optimizer,
    ).fit(data=data)
    actual = np.asarray(gam.predict(data), dtype=np.float64)
    fit_result = gam.fit_result(include_covariances=False)
    np.testing.assert_allclose(actual, expected["response"], rtol=2e-4, atol=2e-5)
    np.testing.assert_allclose(fit_result.deviance, expected["deviance"], rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(fit_result.edf_total, expected["edf"], rtol=3e-4, atol=3e-5)
    np.testing.assert_allclose(gam.family.p, expected["theta"], rtol=3e-4, atol=3e-5)
    np.testing.assert_allclose(fit_result.smoothing_params, expected["sp"], rtol=2e-3, atol=2e-4)
    np.testing.assert_allclose(
        fit_result.scale, expected["pearson_scale"], rtol=2e-3, atol=2e-4
    )
    np.testing.assert_allclose(
        np.exp(float(gam._optim_result.joint_log_phi)),
        expected["outer_scale"],
        rtol=2e-3,
        atol=2e-4,
    )
    np.testing.assert_allclose(
        gam.smoothing_score_, expected["score"], rtol=2e-3, atol=2e-4
    )


def test_tweedie_joint_outer_derivatives_are_consistent():
    x = np.linspace(0.02, 0.98, 48)
    y = np.exp(0.1 + 0.5 * np.sin(2.0 * np.pi * x))
    y[::8] = 0.0
    data = pd.DataFrame({"x": x, "y": y})
    gam = GAM(
        family={"name": "tw", "theta": -1.3, "a": 1.01, "b": 1.99},
        formula='y ~ s(x, bs="cr", k=7)',
        optimize_smoothing=True,
        smoothing_method="REML",
        smoothing_optimizer="outer_newton",
    ).fit(data=data)
    x0 = np.concatenate(
        [
            np.array([gam.family.getTheta(False)]),
            np.log(np.asarray(gam.smoothing_params, dtype=np.float64)),
            np.array([float(gam._optim_result.joint_log_phi)]),
        ]
    )
    y_fit = np.asarray(gam.y_, dtype=np.float64)
    actual_grad = np.asarray(
        criterion_gradient(gam, y_fit, x0, method="reml"), dtype=np.float64
    )
    actual_hess = np.asarray(
        criterion_hessian(gam, y_fit, x0, method="reml"), dtype=np.float64
    )
    step = 2e-5
    fd_grad = np.empty_like(actual_grad)
    for i in range(x0.size):
        plus = x0.copy()
        plus[i] += step
        minus = x0.copy()
        minus[i] -= step
        fd_grad[i] = (
            criterion_value(gam, y_fit, plus, method="reml")
            - criterion_value(gam, y_fit, minus, method="reml")
        ) / (2.0 * step)
    np.testing.assert_allclose(actual_grad, fd_grad, rtol=3e-4, atol=3e-5)
    fd_hess = np.empty_like(actual_hess)
    hstep = 5e-4
    for i in range(x0.size):
        plus = x0.copy()
        plus[i] += hstep
        minus = x0.copy()
        minus[i] -= hstep
        fd_hess[:, i] = (
            criterion_gradient(gam, y_fit, plus, method="reml")
            - criterion_gradient(gam, y_fit, minus, method="reml")
        ) / (2.0 * hstep)
    np.testing.assert_allclose(actual_hess, fd_hess, rtol=3e-3, atol=3e-4)
