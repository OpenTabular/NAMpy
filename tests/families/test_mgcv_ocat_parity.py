"""Direct and fitted ordered-categorical parity against vendored ``mgcv``."""

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
from nampy.gam.fit.selection.optimize.objectives import _JointOcatPirlsRemlObjective
from nampy.gam.inference.null_deviance import null_deviance
from tests._paths import REPO_ROOT
from tests.mgcv_parity_utils import _build_r_command


def _run_r_json(code: str, payload: dict) -> dict:
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
        return json.loads(output.read_text(encoding="utf-8"))


def test_ocat_family_kernels_match_mgcv():
    payload = {
        "theta": [0.7, 1.3, 2.0],
        "y": [1, 2, 3, 4, 1, 3, 2, 4],
        "mu": [-1.4, -0.3, 0.2, 1.1, 0.7, -0.8, 0.0, 1.8],
        "wt": [1.0, 0.5, 1.2, 0.8, 1.0, 1.5, 0.7, 0.9],
    }
    code = r'''
suppressPackageStartupMessages(library(mgcv))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
p <- fromJSON(args[[1]])
fam <- ocat(theta=as.numeric(p$theta))
y <- as.numeric(p$y); mu <- as.numeric(p$mu); wt <- as.numeric(p$wt)
d0 <- fam$Dd(y, mu, fam$getTheta(), wt=wt, level=0)
d1 <- fam$Dd(y, mu, fam$getTheta(), wt=wt, level=1)
d2 <- fam$Dd(y, mu, fam$getTheta(), wt=wt, level=2)
pre <- fam$preinitialize(y, fam)
ans <- list(
  theta=fam$getTheta(FALSE), cutpoints=fam$getTheta(TRUE),
  dev=fam$dev.resids(y, mu, wt), aic=fam$aic(y, mu, wt=wt, dev=0),
  pre_theta=pre$Theta,
  d0=d0, d1=d1, d2=d2,
  ls=fam$ls(y, wt, fam$getTheta(), 1)
)
write_json(ans, args[[2]], digits=17, auto_unbox=TRUE)
'''
    expected = _run_r_json(code, payload)
    family = make_gam_family({"name": "ocat", "theta": payload["theta"]})
    y = np.asarray(payload["y"], dtype=np.int64)
    mu = np.asarray(payload["mu"], dtype=np.float64)
    wt = np.asarray(payload["wt"], dtype=np.float64)
    np.testing.assert_allclose(family.getTheta(False), expected["theta"])
    np.testing.assert_allclose(family.getTheta(True), expected["cutpoints"])
    np.testing.assert_allclose(family.deviance_obs(y, mu, wt), expected["dev"])
    np.testing.assert_allclose(family.aic(y, mu, wt=wt, dev=0), expected["aic"])
    for level, key in [(0, "d0"), (1, "d1"), (2, "d2")]:
        actual = family.Dd(y, mu, family.getTheta(False), wt, level=level)
        for name, value in expected[key].items():
            if value is None or name not in actual:
                continue
            np.testing.assert_allclose(actual[name], value)
    ls = family.ls(y, wt, family.getTheta(False), 1.0)
    np.testing.assert_allclose(ls["lsth1"], expected["ls"]["lsth1"])
    np.testing.assert_allclose(ls["LSTH1"], expected["ls"]["LSTH1"])


def test_ocat_fixed_theta_fit_matches_mgcv():
    x = np.linspace(-1.0, 1.0, 36)
    eta = 0.35 + 0.9 * x - 0.25 * x**2
    cut = np.asarray([-1.0, -0.1, 1.1])
    y = np.asarray(
        [1 + int(np.searchsorted(cut, value, side="left")) for value in eta]
    )
    payload = {"x": x.tolist(), "y": y.tolist(), "theta": [0.9, 1.0, 1.3]}
    code = r'''
suppressPackageStartupMessages(library(mgcv))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
p <- fromJSON(args[[1]])
dat <- data.frame(x=as.numeric(p$x), y=as.numeric(p$y))
fit <- gam(y ~ s(x, bs="cr", k=7), data=dat,
           family=ocat(theta=as.numeric(p$theta)), method="REML", sp=0.6)
pred <- predict(fit, newdata=dat, type="response", se.fit=TRUE)
ans <- list(
  eta=unname(as.numeric(fit$linear.predictors)),
  response=unname(predict(fit, newdata=dat, type="response")),
  response_se=unname(pred$se.fit),
  deviance=unname(as.numeric(deviance(fit))),
  edf=unname(as.numeric(sum(fit$edf))), scale=unname(as.numeric(fit$scale)),
  theta=unname(as.numeric(fit$family$getTheta(TRUE))),
  sp=unname(as.numeric(fit$full.sp)), score=unname(as.numeric(fit$gcv.ubre))
)
write_json(ans, args[[2]], digits=17, auto_unbox=TRUE)
'''
    expected = _run_r_json(code, payload)
    gam = GAM(
        family={"name": "ocat", "theta": payload["theta"]},
        formula="y ~ s(x, bs='cr', k=7)",
        optimize_smoothing=False,
        smoothing_method="fixed",
        smoothing_params=np.asarray([0.6]),
    ).fit(data=pd.DataFrame({"x": x, "y": y}))
    fit_result = gam.fit_result(include_covariances=False)
    np.testing.assert_allclose(
        fit_result.core.eta, expected["eta"], rtol=4e-4, atol=4e-5
    )
    np.testing.assert_allclose(
        fit_result.deviance, expected["deviance"], rtol=4e-4, atol=4e-5
    )
    np.testing.assert_allclose(fit_result.scale, expected["scale"], rtol=4e-4, atol=4e-5)
    np.testing.assert_allclose(gam.family.getTheta(True), expected["theta"])
    np.testing.assert_allclose(
        gam.predict(pd.DataFrame({"x": x}), type="response"),
        expected["response"],
        rtol=4e-4,
        atol=4e-5,
    )
    response, response_se = gam.predict(
        pd.DataFrame({"x": x}), type="response", return_se=True
    )
    np.testing.assert_allclose(response, expected["response"], rtol=4e-4, atol=4e-5)
    np.testing.assert_allclose(
        response_se, expected["response_se"], rtol=2e-3, atol=2e-4
    )


def test_ocat_joint_cutpoint_outer_matches_mgcv():
    x = np.linspace(-1.0, 1.0, 42)
    latent = 0.25 + 1.1 * x - 0.35 * x**2
    cuts = np.asarray([-1.0, -0.2, 0.9])
    rng = np.random.default_rng(1234)
    u = rng.uniform(size=x.size)
    latent = latent + np.log(u / (1.0 - u))
    y = np.asarray(
        [1 + int(np.searchsorted(cuts, value, side="left")) for value in latent]
    )
    payload = {"x": x.tolist(), "y": y.tolist()}
    code = r'''
suppressPackageStartupMessages(library(mgcv))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
p <- fromJSON(args[[1]])
dat <- data.frame(x=as.numeric(p$x), y=as.numeric(p$y))
fit <- gam(y ~ s(x, bs="cr", k=7), data=dat,
           family=ocat(R=4), method="REML")
ans <- list(
  eta=unname(as.numeric(fit$linear.predictors)),
  deviance=unname(as.numeric(deviance(fit))),
  edf=unname(as.numeric(sum(fit$edf))), scale=unname(as.numeric(fit$scale)),
  theta=unname(as.numeric(fit$family$getTheta(TRUE))),
  sp=unname(as.numeric(fit$full.sp)), score=unname(as.numeric(fit$gcv.ubre))
)
write_json(ans, args[[2]], digits=17, auto_unbox=TRUE)
'''
    expected = _run_r_json(code, payload)
    gam = GAM(
        family={"name": "ocat", "R": 4},
        formula="y ~ s(x, bs='cr', k=7)",
        optimize_smoothing=True,
        smoothing_method="REML",
        smoothing_optimizer="outer_newton",
    ).fit(data=pd.DataFrame({"x": x, "y": y}))
    fit_result = gam.fit_result(include_covariances=False)
    np.testing.assert_allclose(
        fit_result.core.eta, expected["eta"], rtol=6e-4, atol=6e-5
    )
    np.testing.assert_allclose(
        fit_result.deviance, expected["deviance"], rtol=6e-4, atol=6e-5
    )
    np.testing.assert_allclose(
        fit_result.edf_total, expected["edf"], rtol=1e-3, atol=1e-4
    )
    np.testing.assert_allclose(gam.family.getTheta(True), expected["theta"], rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(
        # This data produce a very flat high-smoothing optimum; compare the
        # selected scale on the log scale while keeping the fitted quantities
        # and objective value tightly checked above/below.
        np.log(fit_result.smoothing_params),
        np.log(np.asarray(expected["sp"], dtype=np.float64)),
        rtol=0.0,
        atol=0.15,
    )
    np.testing.assert_allclose(
        gam.smoothing_score_, expected["score"], rtol=8e-3, atol=8e-4
    )


def test_ocat_joint_outer_derivatives_match_objective_finite_difference():
    x = np.linspace(-1.0, 1.0, 42)
    latent = 0.25 + 1.1 * x - 0.35 * x**2
    cuts = np.asarray([-1.0, -0.2, 0.9])
    rng = np.random.default_rng(1234)
    u = rng.uniform(size=x.size)
    latent = latent + np.log(u / (1.0 - u))
    y = np.asarray(
        [1 + int(np.searchsorted(cuts, value, side="left")) for value in latent]
    )
    gam = GAM(
        family={"name": "ocat", "R": 4},
        formula="y ~ s(x, bs='cr', k=7)",
        optimize_smoothing=True,
        smoothing_method="REML",
        smoothing_optimizer="outer_newton",
    ).fit(data=pd.DataFrame({"x": x, "y": y}))
    objective = _JointOcatPirlsRemlObjective(
        gam, np.asarray(y, dtype=np.float64), "REML"
    )
    point = np.asarray(gam._optim_result.joint_x, dtype=np.float64)
    analytic = objective.jac(point)
    finite_difference = np.empty_like(analytic)
    step = 1e-5
    for index in range(point.size):
        delta = np.zeros_like(point)
        delta[index] = step
        finite_difference[index] = (
            objective.fun(point + delta) - objective.fun(point - delta)
        ) / (2.0 * step)
    np.testing.assert_allclose(analytic, finite_difference, rtol=2e-4, atol=2e-5)


def test_ocat_r5_kernels_and_data_initialization_match_mgcv():
    payload = {
        "theta": [0.35, 0.8, 1.6],
        "y": [1, 2, 3, 4, 5, 1, 3, 5, 2, 4],
        "mu": [-2.2, -1.1, -0.2, 0.4, 1.6, 2.4, -0.7, 0.9, 1.2, -1.8],
        "wt": [0.5, 1.0, 1.4, 0.7, 1.2, 1.8, 0.9, 0.6, 1.1, 1.3],
    }
    code = r'''
suppressPackageStartupMessages(library(mgcv))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
p <- fromJSON(args[[1]])
fam <- ocat(theta=as.numeric(p$theta))
fam_init <- ocat(R=5)
y <- as.numeric(p$y); mu <- as.numeric(p$mu); wt <- as.numeric(p$wt)
pre <- fam_init$preinitialize(y, fam_init)
write_json(list(
  theta=fam$getTheta(FALSE), cutpoints=fam$getTheta(TRUE),
  pre_theta=pre$Theta,
  dev=fam$dev.resids(y, mu, wt),
  d1=fam$Dd(y, mu, fam$getTheta(), wt=wt, level=1),
  d2=fam$Dd(y, mu, fam$getTheta(), wt=wt, level=2)
), args[[2]], digits=17, auto_unbox=TRUE)
'''
    expected = _run_r_json(code, payload)
    family = make_gam_family({"name": "ocat", "theta": payload["theta"]})
    initialized_family = make_gam_family({"name": "ocat", "R": 5})
    y = np.asarray(payload["y"], dtype=np.int64)
    mu = np.asarray(payload["mu"], dtype=np.float64)
    wt = np.asarray(payload["wt"], dtype=np.float64)
    np.testing.assert_allclose(family.getTheta(False), expected["theta"])
    np.testing.assert_allclose(family.getTheta(True), expected["cutpoints"])
    initialized_family.initialize_mu(y)
    np.testing.assert_allclose(initialized_family.getTheta(False), expected["pre_theta"])
    np.testing.assert_allclose(family.deviance_obs(y, mu, wt), expected["dev"])
    for level, key in [(1, "d1"), (2, "d2")]:
        actual = family.Dd(y, mu, family.getTheta(False), wt, level=level)
        for name, value in expected[key].items():
            if value is not None and name in actual:
                np.testing.assert_allclose(actual[name], value)


def test_ocat_validation_and_link_lifecycle():
    with pytest.raises(ValueError, match="R >= 3"):
        make_gam_family({"name": "ocat", "R": 2})
    with pytest.raises(ValueError, match="identity"):
        make_gam_family({"name": "ocat", "R": 4, "link": "logit"})
    family = make_gam_family({"name": "ocat", "R": 4})
    with pytest.raises(ValueError, match="integer labels"):
        family.validate_y([1, 2.5, 4])
    with pytest.raises(ValueError, match="1..4"):
        family.validate_y([1, 5])


def test_ocat_derivatives_match_finite_differences_for_each_cutpoint():
    family = make_gam_family({"name": "ocat", "theta": [0.35, 0.8, 1.6]})
    y = np.asarray([1, 2, 3, 4, 5, 2, 4, 5])
    mu = np.asarray([-2.2, -1.1, -0.2, 0.4, 1.6, 2.4, -0.7, 0.9])
    wt = np.asarray([0.5, 1.0, 1.4, 0.7, 1.2, 1.8, 0.9, 0.6])
    theta = family.getTheta(False)
    step = 1e-5
    base = family.Dd(y, mu, theta, wt, level=2)
    for index in range(y.size):
        plus_mu = mu.copy()
        plus_mu[index] += step
        minus_mu = mu.copy()
        minus_mu[index] -= step
        plus = family.Dd(y, plus_mu, theta, wt, level=1)
        minus = family.Dd(y, minus_mu, theta, wt, level=1)
        np.testing.assert_allclose(
            (plus["Dmu"][index] - minus["Dmu"][index]) / (2.0 * step),
            base["Dmu2"][index], rtol=3e-5, atol=3e-7,
        )
    for index in range(theta.size):
        plus_theta = theta.copy()
        plus_theta[index] += step
        minus_theta = theta.copy()
        minus_theta[index] -= step
        plus = family.Dd(y, mu, plus_theta, wt, level=1)
        minus = family.Dd(y, mu, minus_theta, wt, level=1)
        np.testing.assert_allclose(
            (plus["Dmu"] - minus["Dmu"]) / (2.0 * step),
            base["Dmuth"][:, index], rtol=4e-5, atol=4e-7,
        )
        np.testing.assert_allclose(
            (plus["Dth"][:, index] - minus["Dth"][:, index]) / (2.0 * step),
            base["Dth2"][:, [0, 3, 5][index]], rtol=5e-5, atol=5e-7,
        )


def test_ocat_residuals_weights_offsets_and_null_deviance_match_mgcv():
    x = np.linspace(-1.0, 1.0, 28)
    off = 0.22 * np.sin(2.0 * np.pi * x)
    weights = np.linspace(0.5, 1.8, x.size)
    latent = 0.25 + 1.0 * x - 0.3 * x**2 + off
    cuts = np.asarray([-1.0, -0.15, 0.95])
    y = np.asarray([1 + int(np.searchsorted(cuts, value, side="left")) for value in latent])
    data = pd.DataFrame({"x": x, "off": off, "y": y, "w": weights})
    payload = {key: value.tolist() for key, value in data.items()}
    code = r'''
suppressPackageStartupMessages(library(mgcv))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
p <- fromJSON(args[[1]])
dat <- data.frame(x=as.numeric(p$x), off=as.numeric(p$off),
                  y=as.numeric(p$y), w=as.numeric(p$w))
fit <- gam(y ~ s(x, bs="cr", k=7) + offset(off), data=dat,
           weights=w, family=ocat(theta=c(0.9, 1.0, 1.3)),
           method="REML", sp=0.7)
write_json(list(
  fitted=unname(as.numeric(fitted(fit))),
  deviance=unname(as.numeric(deviance(fit))),
  null_deviance=unname(as.numeric(fit$null.deviance)),
  response=unname(as.numeric(residuals(fit, "response"))),
  devres=unname(as.numeric(residuals(fit, "deviance")))
), args[[2]], digits=17, auto_unbox=TRUE)
'''
    expected = _run_r_json(code, payload)
    gam = GAM(
        family={"name": "ocat", "theta": [0.9, 1.0, 1.3]},
        formula="y ~ s(x, bs='cr', k=7) + offset(off)",
        optimize_smoothing=False,
        smoothing_method="fixed",
        smoothing_params=np.asarray([0.7]),
    ).fit(data=data, sample_weight=weights)
    result = gam.fit_result(include_covariances=False)
    np.testing.assert_allclose(
        gam.predict(data, type="link"), expected["fitted"], rtol=5e-5, atol=5e-6
    )
    np.testing.assert_allclose(result.deviance, expected["deviance"], rtol=5e-5, atol=5e-6)
    np.testing.assert_allclose(null_deviance(gam), expected["null_deviance"], rtol=5e-5, atol=5e-6)
    for rtype, key in [("response", "response"), ("deviance", "devres")]:
        np.testing.assert_allclose(gam.residuals(type=rtype), expected[key], rtol=7e-4, atol=7e-5)


def test_ocat_extreme_eta_probabilities_and_categories_match_mgcv():
    theta = np.asarray([0.35, 0.8, 1.6])
    eta = np.asarray([-1.0e4, -50.0, -4.0, -1.0, 0.0, 2.0, 50.0, 1.0e4])
    eta_se = np.linspace(0.1, 0.8, eta.size)
    payload = {"theta": theta.tolist(), "eta": eta.tolist(), "eta_se": eta_se.tolist()}
    code = r'''
suppressPackageStartupMessages(library(mgcv))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
p <- fromJSON(args[[1]])
fam <- ocat(theta=as.numeric(p$theta))
eta <- as.numeric(p$eta); eta_se <- as.numeric(p$eta_se)
cut <- fam$getTheta(TRUE)
cdf <- matrix(0, length(eta), length(cut)+2)
cdf[, ncol(cdf)] <- 1
dcdf <- matrix(0, length(eta), length(cut)+2)
for (i in seq_along(cut)) {
  cdf[, i+1] <- plogis(cut[i]-eta)
  dcdf[, i+1] <- cdf[, i+1] * (cdf[, i+1]-1)
}
write_json(list(
  prob=unname(cdf[,2:ncol(cdf)]-cdf[,1:(ncol(cdf)-1)]),
  se=unname(abs(dcdf[,2:ncol(dcdf)]-dcdf[,1:(ncol(dcdf)-1)]) * eta_se),
  category=unname(as.numeric(fam$predict(family=fam, eta=eta)[[1]]))
), args[[2]], digits=17, auto_unbox=TRUE)
'''
    expected = _run_r_json(code, payload)
    family = make_gam_family({"name": "ocat", "theta": theta})
    np.testing.assert_allclose(family.response_from_eta(eta), expected["prob"], rtol=2e-15, atol=2e-15)
    np.testing.assert_allclose(family.response_se_from_eta(eta, eta_se), expected["se"], rtol=2e-15, atol=2e-15)
    categories = np.searchsorted(family._residual_cutpoints(), eta, side="left")
    np.testing.assert_array_equal(categories, expected["category"])


@pytest.mark.parametrize("optimizer", ["outer_newton", "bfgs", "optim"])
def test_ocat_joint_outer_optimizer_matrix_matches_mgcv(optimizer):
    x = np.linspace(-1.0, 1.0, 40)
    latent = 0.2 + 0.95 * np.sin(2.0 * np.pi * x) - 0.25 * x
    rng = np.random.default_rng(42)
    uniform = rng.uniform(size=x.size)
    latent = latent + np.log(uniform / (1.0 - uniform))
    cuts = np.asarray([-1.0, -0.15, 0.95])
    y = np.asarray([1 + int(np.searchsorted(cuts, value, side="left")) for value in latent])
    payload = {"x": x.tolist(), "y": y.tolist(), "optimizer": optimizer}
    code = r'''
suppressPackageStartupMessages(library(mgcv))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
p <- fromJSON(args[[1]])
dat <- data.frame(x=as.numeric(p$x), y=as.numeric(p$y))
optimizer_name <- if (p$optimizer == "outer_newton") "newton" else as.character(p$optimizer)
fit <- gam(y ~ s(x, bs="cr", k=6), data=dat,
           family=ocat(R=4), method="REML",
           optimizer=c("outer", optimizer_name))
write_json(list(eta=unname(as.numeric(fit$linear.predictors)),
                deviance=unname(as.numeric(deviance(fit))),
                theta=unname(as.numeric(fit$family$getTheta(TRUE))),
                sp=unname(as.numeric(fit$full.sp)),
                score=unname(as.numeric(fit$gcv.ubre))),
           args[[2]], digits=17, auto_unbox=TRUE)
'''
    expected = _run_r_json(code, payload)
    data = pd.DataFrame({"x": x, "y": y})
    gam = GAM(
        family={"name": "ocat", "R": 4},
        formula="y ~ s(x, bs='cr', k=6)",
        optimize_smoothing=True,
        smoothing_method="REML",
        smoothing_optimizer=optimizer,
    ).fit(data=data)
    result = gam.fit_result(include_covariances=False)
    np.testing.assert_allclose(result.core.eta, expected["eta"], rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(result.deviance, expected["deviance"], rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(gam.family.getTheta(True), expected["theta"], rtol=1e-2, atol=1e-3)
    # This data set has an essentially flat high-smoothing optimum. Compare
    # the endpoint on the log scale while checking fitted quantities and the
    # criterion tightly above.
    np.testing.assert_allclose(
        np.log(result.smoothing_params),
        np.log(np.asarray(expected["sp"], dtype=np.float64)),
        rtol=0.0,
        atol=0.8,
    )
    np.testing.assert_allclose(gam.smoothing_score_, expected["score"], rtol=3e-2, atol=3e-3)


@pytest.mark.parametrize("seed", [11, 23])
def test_ocat_randomized_kernel_parity(seed):
    rng = np.random.default_rng(seed)
    theta = np.asarray([0.3, 0.75, 1.4])
    y = np.concatenate((np.arange(1, 6), rng.integers(1, 6, size=10)))
    mu = rng.normal(loc=0.0, scale=2.0, size=y.size)
    weights = rng.uniform(0.2, 2.0, size=y.size)
    payload = {
        "theta": theta.tolist(),
        "y": y.tolist(),
        "mu": mu.tolist(),
        "wt": weights.tolist(),
    }
    code = r'''
suppressPackageStartupMessages(library(mgcv))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
p <- fromJSON(args[[1]])
fam <- ocat(theta=as.numeric(p$theta))
write_json(fam$Dd(as.numeric(p$y), as.numeric(p$mu), fam$getTheta(),
                  wt=as.numeric(p$wt), level=2),
           args[[2]], digits=17, auto_unbox=TRUE)
'''
    expected = _run_r_json(code, payload)
    family = make_gam_family({"name": "ocat", "theta": theta})
    actual = family.Dd(y, mu, family.getTheta(False), weights, level=2)
    for name, value in expected.items():
        if value is not None and name in actual:
            np.testing.assert_allclose(actual[name], value, rtol=4e-10, atol=4e-12)
