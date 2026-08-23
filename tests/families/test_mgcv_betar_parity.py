"""Direct and fitted beta-regression parity against vendored ``mgcv``."""

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
from nampy.gam.inference.null_deviance import null_deviance
from tests._paths import REPO_ROOT
from tests.mgcv_parity_utils import _build_r_command
from tests.reference_fixtures import load_reference, reference_key, save_reference


def _run_r_json(code: str, payload: dict) -> dict:
    key = reference_key(
        "betar_r_json",
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


@pytest.mark.parametrize("link", ["logit", "probit", "cloglog", "cauchit"])
def test_betar_family_kernels_match_mgcv(link):
    payload = {
        "link": link,
        "y": [0.03, 0.21, 0.47, 0.72, 0.96],
        "mu": [0.08, 0.28, 0.43, 0.68, 0.91],
        "eta": [-2.1, -0.7, 0.1, 0.9, 2.2],
        "wt": [0.0, 1.0, 2.5, 0.4, 1.7],
        "ltheta": 1.15,
        "eps": float(np.finfo(np.float64).eps * 100.0),
    }
    code = r'''
suppressPackageStartupMessages(library(mgcv))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
p <- fromJSON(args[[1]])
y <- as.numeric(p$y); mu <- as.numeric(p$mu); eta <- as.numeric(p$eta)
wt <- as.numeric(p$wt); th <- as.numeric(p$ltheta)
link_name <- as.character(p$link)
fam <- do.call(betar, list(theta=-exp(th), link=link_name, eps=p$eps))
dd <- fam$Dd(y, mu, th, wt, level=2)
pre <- fam$preinitialize(y, fam)$y
sat <- fam$saturated.ll(y, wt, fam$getTheta(TRUE))
ans <- list(
  variance=unname(as.numeric(fam$variance(mu))),
  linkinv=unname(as.numeric(fam$linkinv(eta))),
  linkfun=unname(as.numeric(fam$linkfun(mu))),
  mu_eta=unname(as.numeric(fam$mu.eta(eta))),
  preinitialize=unname(as.numeric(pre)),
  dev=unname(as.numeric(fam$dev.resids(y, mu, wt, theta=th))),
  dd=lapply(dd, unname),
  aic=unname(as.numeric(fam$aic(y, mu, th, wt, 0))),
  sat=list(f=unname(as.numeric(sat$f)), term=unname(as.numeric(sat$term)),
           mu=unname(as.numeric(sat$mu)))
)
write_json(ans, args[[2]], digits=17, auto_unbox=TRUE)
'''
    expected = _run_r_json(code, payload)
    family = make_gam_family(
        {
            "name": "betar",
            "theta": -np.exp(payload["ltheta"]),
            "link": link,
            "eps": payload["eps"],
        }
    )
    y = np.asarray(payload["y"], dtype=np.float64)
    mu = np.asarray(payload["mu"], dtype=np.float64)
    eta = np.asarray(payload["eta"], dtype=np.float64)
    wt = np.asarray(payload["wt"], dtype=np.float64)

    np.testing.assert_allclose(family.variance(mu), expected["variance"], rtol=2e-12, atol=2e-13)
    np.testing.assert_allclose(family.inverse_link(eta), expected["linkinv"], rtol=2e-12, atol=2e-13)
    np.testing.assert_allclose(family.link(mu), expected["linkfun"], rtol=2e-12, atol=2e-13)
    np.testing.assert_allclose(family.mu_eta(eta), expected["mu_eta"], rtol=2e-11, atol=2e-12)
    np.testing.assert_allclose(family.validate_y(y), expected["preinitialize"], rtol=0, atol=0)
    np.testing.assert_allclose(
        family.deviance_obs(y, mu, wt, theta=payload["ltheta"]),
        expected["dev"],
        rtol=2e-11,
        atol=2e-12,
    )
    actual_dd = family.Dd(y, mu, theta=payload["ltheta"], wt=wt, level=2)
    for key, value in expected["dd"].items():
        np.testing.assert_allclose(actual_dd[key], value, rtol=3e-10, atol=3e-12, err_msg=key)
    np.testing.assert_allclose(
        family.aic(y, mu, theta=payload["ltheta"], weights=wt),
        expected["aic"],
        rtol=3e-10,
        atol=3e-12,
    )
    sat = family.saturated_loglik(y, wt)
    np.testing.assert_allclose(sat, expected["sat"]["f"], rtol=2e-9, atol=2e-11)


def test_betar_fixed_theta_fit_matches_mgcv():
    x = np.linspace(0.02, 0.98, 64)
    eta = -0.25 + 1.1 * np.sin(2.0 * np.pi * x) + 0.35 * x
    y = 1.0 / (1.0 + np.exp(-eta))
    y = np.clip(y + 0.08 * np.sin(13.0 * x), 0.02, 0.98)
    sp = 0.7
    payload = {"x": x.tolist(), "y": y.tolist(), "sp": sp}
    code = r'''
suppressPackageStartupMessages(library(mgcv))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
p <- fromJSON(args[[1]])
dat <- data.frame(x=as.numeric(p$x), y=as.numeric(p$y))
fit <- gam(y ~ s(x, bs="cr", k=8), data=dat,
           family=betar(theta=12, link="logit"), method="REML",
           sp=as.numeric(p$sp))
ans <- list(
  response=unname(as.numeric(fitted(fit))),
  deviance=unname(as.numeric(deviance(fit))),
  edf=unname(as.numeric(sum(fit$edf))),
  scale=unname(as.numeric(fit$scale)),
  sp=unname(as.numeric(fit$full.sp)),
  theta=unname(as.numeric(fit$family$getTheta(TRUE))),
  score=unname(as.numeric(fit$gcv.ubre))
)
write_json(ans, args[[2]], digits=17, auto_unbox=TRUE)
'''
    expected = _run_r_json(code, payload)
    gam = GAM(
        family={"name": "betar", "theta": 12.0},
        formula='y ~ s(x, bs="cr", k=8)',
        optimize_smoothing=False,
        smoothing_method="fixed",
        smoothing_params=np.asarray([sp]),
    ).fit(data=pd.DataFrame(payload))
    actual = np.asarray(gam.predict(pd.DataFrame(payload)), dtype=np.float64)
    fit_result = gam.fit_result(include_covariances=False)
    np.testing.assert_allclose(actual, expected["response"], rtol=3e-6, atol=3e-7)
    np.testing.assert_allclose(fit_result.deviance, expected["deviance"], rtol=3e-6, atol=3e-7)
    np.testing.assert_allclose(fit_result.edf_total, expected["edf"], rtol=3e-6, atol=3e-7)
    np.testing.assert_allclose(fit_result.scale, expected["scale"], rtol=3e-6, atol=3e-7)
    # Both sides recover theta through exp(log(theta)); libm may round that
    # round-trip by one binary64 ULP.
    np.testing.assert_allclose(
        gam.family.getTheta(True), expected["theta"], rtol=2e-15, atol=0
    )
    np.testing.assert_allclose(fit_result.smoothing_params, expected["sp"], rtol=2e-10, atol=2e-10)


def test_betar_intercept_only_matches_mgcv():
    y = np.asarray([0.04, 0.11, 0.22, 0.31, 0.48, 0.57, 0.69, 0.83, 0.91, 0.97])
    payload = {"y": y.tolist()}
    code = r'''
suppressPackageStartupMessages(library(mgcv))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
p <- fromJSON(args[[1]])
dat <- data.frame(y=as.numeric(p$y))
fit <- gam(y ~ 1, data=dat, family=betar(theta=9, link="logit"), method="REML")
write_json(list(response=unname(as.numeric(fitted(fit))),
                coef=unname(as.numeric(coef(fit)))), args[[2]],
           digits=17, auto_unbox=TRUE)
'''
    expected = _run_r_json(code, payload)
    gam = GAM(
        family={"name": "betar", "theta": 9.0},
        formula="y ~ 1",
        optimize_smoothing=False,
        smoothing_method="fixed",
        smoothing_params=np.empty(0, dtype=np.float64),
    ).fit(data=pd.DataFrame(payload))
    actual = np.asarray(gam.predict(pd.DataFrame(payload)), dtype=np.float64)
    np.testing.assert_allclose(actual, expected["response"], rtol=3e-7, atol=3e-8)
    np.testing.assert_allclose(
        gam.gam_result_.fit_core_solution.fit_result.intercept,
        expected["coef"],
        rtol=3e-7,
        atol=3e-8,
    )


def test_betar_joint_theta_outer_matches_mgcv():
    x = np.linspace(0.03, 0.97, 48)
    eta = -0.35 + 1.0 * np.sin(2.0 * np.pi * x) + 0.25 * x
    y = np.clip(1.0 / (1.0 + np.exp(-eta)) + 0.09 * np.sin(11.0 * x), 0.02, 0.98)
    payload = {"x": x.tolist(), "y": y.tolist()}
    code = r'''
suppressPackageStartupMessages(library(mgcv))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
p <- fromJSON(args[[1]])
dat <- data.frame(x=as.numeric(p$x), y=as.numeric(p$y))
fit <- gam(y ~ s(x, bs="cr", k=7), data=dat,
           family=betar(theta=-9, link="logit"), method="REML")
ans <- list(
  response=unname(as.numeric(fitted(fit))),
  deviance=unname(as.numeric(deviance(fit))),
  edf=unname(as.numeric(sum(fit$edf))),
  scale=unname(as.numeric(fit$scale)),
  theta=unname(as.numeric(fit$family$getTheta(TRUE))),
  sp=unname(as.numeric(fit$full.sp)),
  score=unname(as.numeric(fit$gcv.ubre))
)
write_json(ans, args[[2]], digits=17, auto_unbox=TRUE)
'''
    expected = _run_r_json(code, payload)
    gam = GAM(
        family={"name": "betar", "theta": -9.0},
        formula='y ~ s(x, bs="cr", k=7)',
        optimize_smoothing=True,
        smoothing_method="REML",
        smoothing_optimizer="outer_newton",
    ).fit(data=pd.DataFrame(payload))
    actual = np.asarray(gam.predict(pd.DataFrame(payload)), dtype=np.float64)
    fit_result = gam.fit_result(include_covariances=False)
    np.testing.assert_allclose(actual, expected["response"], rtol=4e-4, atol=4e-5)
    np.testing.assert_allclose(fit_result.deviance, expected["deviance"], rtol=4e-4, atol=4e-5)
    np.testing.assert_allclose(fit_result.edf_total, expected["edf"], rtol=8e-4, atol=8e-5)
    np.testing.assert_allclose(gam.family.getTheta(True), expected["theta"], rtol=4e-4, atol=4e-5)
    np.testing.assert_allclose(fit_result.smoothing_params, expected["sp"], rtol=5e-3, atol=5e-4)
    np.testing.assert_allclose(gam.smoothing_score_, expected["score"], rtol=5e-3, atol=5e-4)


@pytest.mark.parametrize("link", ["logit", "probit", "cloglog", "cauchit"])
def test_betar_intercept_fit_all_links_matches_mgcv(link):
    y = np.asarray([0.03, 0.08, 0.17, 0.31, 0.44, 0.58, 0.71, 0.86, 0.94])
    payload = {"y": y.tolist(), "link": link}
    code = r'''
suppressPackageStartupMessages(library(mgcv))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
p <- fromJSON(args[[1]])
dat <- data.frame(y=as.numeric(p$y))
fit <- gam(y ~ 1, data=dat,
           family=do.call(betar, list(theta=8, link=as.character(p$link))),
           method="REML")
write_json(list(coef=unname(as.numeric(coef(fit))),
                fitted=unname(as.numeric(fitted(fit))),
                deviance=unname(as.numeric(deviance(fit))),
                null_deviance=unname(as.numeric(fit$null.deviance))),
           args[[2]], digits=17, auto_unbox=TRUE)
'''
    expected = _run_r_json(code, payload)
    gam = GAM(
        family={"name": "betar", "theta": 8.0, "link": link},
        formula="y ~ 1",
        optimize_smoothing=False,
        smoothing_method="fixed",
        smoothing_params=np.empty(0, dtype=np.float64),
    ).fit(data=pd.DataFrame(payload))
    result = gam.fit_result(include_covariances=False)
    np.testing.assert_allclose(result.intercept, expected["coef"], rtol=3e-6, atol=3e-7)
    np.testing.assert_allclose(
        gam.predict(pd.DataFrame(payload)), expected["fitted"], rtol=3e-6, atol=3e-7
    )
    np.testing.assert_allclose(result.deviance, expected["deviance"], rtol=3e-6, atol=3e-7)
    np.testing.assert_allclose(
        null_deviance(gam), expected["null_deviance"], rtol=3e-6, atol=3e-7
    )


def test_betar_residuals_weights_offsets_and_postprocessing_match_mgcv():
    x = np.linspace(0.05, 0.95, 18)
    off = 0.12 * np.cos(2.0 * np.pi * x)
    weights = np.linspace(0.4, 2.0, x.size)
    y = np.clip(0.5 + 0.2 * np.sin(4.0 * x) + 0.08 * np.cos(11.0 * x), 0.02, 0.98)
    data = pd.DataFrame({"x": x, "off": off, "y": y, "w": weights})
    payload = {key: value.tolist() for key, value in data.items()}
    code = r'''
suppressPackageStartupMessages(library(mgcv))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
p <- fromJSON(args[[1]])
dat <- data.frame(x=as.numeric(p$x), off=as.numeric(p$off),
                  y=as.numeric(p$y), w=as.numeric(p$w))
fit <- gam(y ~ s(x, bs="cr", k=6) + offset(off), data=dat,
           weights=w, family=betar(theta=7), method="REML", sp=0.8)
write_json(list(
  fitted=unname(as.numeric(fitted(fit))),
  deviance=unname(as.numeric(deviance(fit))),
  null_deviance=unname(as.numeric(fit$null.deviance)),
  response=unname(as.numeric(residuals(fit, "response"))),
  pearson=unname(as.numeric(residuals(fit, "pearson"))),
  devres=unname(as.numeric(residuals(fit, "deviance")))
), args[[2]], digits=17, auto_unbox=TRUE)
'''
    expected = _run_r_json(code, payload)
    gam = GAM(
        family={"name": "betar", "theta": 7.0},
        formula='y ~ s(x, bs="cr", k=6) + offset(off)',
        optimize_smoothing=False,
        smoothing_method="fixed",
        smoothing_params=np.asarray([0.8]),
    ).fit(data=data, sample_weight=weights)
    result = gam.fit_result(include_covariances=False)
    np.testing.assert_allclose(gam.predict(data), expected["fitted"], rtol=4e-5, atol=4e-6)
    np.testing.assert_allclose(result.deviance, expected["deviance"], rtol=4e-5, atol=4e-6)
    np.testing.assert_allclose(null_deviance(gam), expected["null_deviance"], rtol=4e-5, atol=4e-6)
    for rtype, key in [("response", "response"), ("pearson", "pearson"), ("deviance", "devres")]:
        np.testing.assert_allclose(gam.residuals(type=rtype), expected[key], rtol=5e-4, atol=5e-5)


def test_betar_mu_theta_derivatives_match_finite_differences():
    family = make_gam_family({"name": "betar", "theta": -7.5, "link": "logit"})
    y = np.asarray([0.07, 0.19, 0.42, 0.73, 0.94])
    mu = np.asarray([0.12, 0.27, 0.48, 0.69, 0.87])
    log_theta = family.getTheta(False)
    step = 1e-5
    base = family.Dd(y, mu, log_theta, level=2)
    for index in range(y.size):
        plus_mu = mu.copy()
        plus_mu[index] += step
        minus_mu = mu.copy()
        minus_mu[index] -= step
        plus = family.Dd(y, plus_mu, log_theta, level=1)
        minus = family.Dd(y, minus_mu, log_theta, level=1)
        np.testing.assert_allclose(
            (plus["Dmu"][index] - minus["Dmu"][index]) / (2.0 * step),
            base["Dmu2"][index], rtol=3e-5, atol=3e-7,
        )
    plus_theta = family.Dd(y, mu, log_theta + step, level=1)
    minus_theta = family.Dd(y, mu, log_theta - step, level=1)
    np.testing.assert_allclose(
        (plus_theta["Dmu"] - minus_theta["Dmu"]) / (2.0 * step),
        base["Dmuth"], rtol=3e-5, atol=3e-7,
    )


def test_betar_boundary_preinitialize_matches_mgcv():
    eps = float(np.finfo(np.float64).eps * 100.0)
    y = np.asarray([0.0, eps / 2.0, eps, 0.2, 0.8, 1.0 - eps, 1.0])
    payload = {"y": y.tolist(), "eps": eps}
    code = r'''
suppressPackageStartupMessages(library(mgcv))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
p <- fromJSON(args[[1]])
fam <- betar(theta=7, eps=p$eps)
pre <- fam$preinitialize(as.numeric(p$y), fam)$y
write_json(list(y=unname(as.numeric(pre))), args[[2]],
           digits=17, auto_unbox=TRUE)
'''
    expected = _run_r_json(code, payload)
    family = make_gam_family({"name": "betar", "theta": 7.0, "eps": eps})
    np.testing.assert_allclose(family.validate_y(y), expected["y"], rtol=0, atol=0)


def test_betar_response_prediction_and_standard_errors_match_mgcv():
    x = np.linspace(0.03, 0.97, 30)
    eta = -0.4 + 1.15 * np.sin(2.0 * np.pi * x)
    y = np.clip(1.0 / (1.0 + np.exp(-eta)) + 0.07 * np.cos(9.0 * x), 0.02, 0.98)
    payload = {"x": x.tolist(), "y": y.tolist()}
    code = r'''
suppressPackageStartupMessages(library(mgcv))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
p <- fromJSON(args[[1]])
dat <- data.frame(x=as.numeric(p$x), y=as.numeric(p$y))
fit <- gam(y ~ s(x, bs="cr", k=7), data=dat,
           family=betar(theta=9), method="REML", sp=0.5)
pred <- predict(fit, newdata=dat, type="response", se.fit=TRUE)
write_json(list(fitted=unname(as.numeric(pred$fit)),
                se=unname(as.numeric(pred$se.fit))), args[[2]],
           digits=17, auto_unbox=TRUE)
'''
    expected = _run_r_json(code, payload)
    gam = GAM(
        family={"name": "betar", "theta": 9.0},
        formula='y ~ s(x, bs="cr", k=7)',
        optimize_smoothing=False,
        smoothing_method="fixed",
        smoothing_params=np.asarray([0.5]),
    ).fit(data=pd.DataFrame(payload))
    actual, actual_se = gam.predict(pd.DataFrame(payload), return_se=True)
    np.testing.assert_allclose(actual, expected["fitted"], rtol=5e-5, atol=5e-6)
    np.testing.assert_allclose(actual_se, expected["se"], rtol=2e-3, atol=2e-4)


@pytest.mark.parametrize("optimizer", ["outer_newton", "bfgs"])
def test_betar_joint_outer_optimizer_matrix_matches_mgcv(optimizer):
    x = np.linspace(0.03, 0.97, 40)
    eta = -0.3 + 0.9 * np.sin(2.0 * np.pi * x) + 0.2 * x
    y = np.clip(1.0 / (1.0 + np.exp(-eta)) + 0.08 * np.cos(10.0 * x), 0.02, 0.98)
    payload = {"x": x.tolist(), "y": y.tolist(), "optimizer": optimizer}
    code = r'''
suppressPackageStartupMessages(library(mgcv))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
p <- fromJSON(args[[1]])
dat <- data.frame(x=as.numeric(p$x), y=as.numeric(p$y))
optimizer_name <- if (p$optimizer == "outer_newton") "newton" else as.character(p$optimizer)
fit <- gam(y ~ s(x, bs="cr", k=6), data=dat,
           family=betar(theta=-8), method="REML",
           optimizer=c("outer", optimizer_name))
write_json(list(response=unname(as.numeric(fitted(fit))),
                deviance=unname(as.numeric(deviance(fit))),
                theta=unname(as.numeric(fit$family$getTheta(TRUE))),
                sp=unname(as.numeric(fit$full.sp)),
                score=unname(as.numeric(fit$gcv.ubre))),
           args[[2]], digits=17, auto_unbox=TRUE)
'''
    expected = _run_r_json(code, payload)
    data = pd.DataFrame({"x": x, "y": y})
    gam = GAM(
        family={"name": "betar", "theta": -8.0},
        formula='y ~ s(x, bs="cr", k=6)',
        optimize_smoothing=True,
        smoothing_method="REML",
        smoothing_optimizer=optimizer,
    ).fit(data=data)
    result = gam.fit_result(include_covariances=False)
    np.testing.assert_allclose(gam.predict(data), expected["response"], rtol=7e-4, atol=7e-5)
    np.testing.assert_allclose(result.deviance, expected["deviance"], rtol=7e-4, atol=7e-5)
    np.testing.assert_allclose(gam.family.getTheta(True), expected["theta"], rtol=7e-3, atol=7e-4)
    np.testing.assert_allclose(result.smoothing_params, expected["sp"], rtol=2e-2, atol=2e-3)
    np.testing.assert_allclose(gam.smoothing_score_, expected["score"], rtol=2e-2, atol=2e-3)


@pytest.mark.parametrize("seed", [7, 19])
def test_betar_randomized_kernel_parity(seed):
    rng = np.random.default_rng(seed)
    y = rng.uniform(0.01, 0.99, size=12)
    mu = rng.uniform(0.02, 0.98, size=12)
    weights = rng.uniform(0.2, 2.0, size=12)
    log_theta = float(rng.uniform(0.4, 2.0))
    payload = {
        "y": y.tolist(),
        "mu": mu.tolist(),
        "wt": weights.tolist(),
        "ltheta": log_theta,
    }
    code = r'''
suppressPackageStartupMessages(library(mgcv))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
p <- fromJSON(args[[1]])
fam <- betar(theta=-exp(p$ltheta))
dd <- fam$Dd(as.numeric(p$y), as.numeric(p$mu), p$ltheta,
             as.numeric(p$wt), level=2)
write_json(dd, args[[2]], digits=17, auto_unbox=TRUE)
'''
    expected = _run_r_json(code, payload)
    family = make_gam_family({"name": "betar", "theta": -np.exp(log_theta)})
    actual = family.Dd(y, mu, log_theta, weights, level=2)
    for name, value in expected.items():
        if value is not None and name in actual:
            np.testing.assert_allclose(actual[name], value, rtol=4e-10, atol=4e-12)
