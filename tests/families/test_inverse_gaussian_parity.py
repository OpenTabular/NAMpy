"""Static stats/mgcv inverse-Gaussian family and public GAM parity."""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from nampy.gam.families import InverseGaussianFamily, make_gam_family
from tests.reference_fixtures import (
    generation_enabled,
    load_reference,
    reference_key,
    save_reference,
)

LINKS = ["1/mu^2", "inverse", "log", "identity"]
NAMESPACE = "mgcv_inverse_gaussian"


def _reference(payload, code):
    key = reference_key("inverse_gaussian", {"payload": payload, "code": code})
    if generation_enabled():
        with tempfile.TemporaryDirectory() as directory:
            src, inp, out = [
                Path(directory) / name
                for name in ("reference.R", "input.json", "output.json")
            ]
            src.write_text(
                'args <- commandArgs(TRUE)\nlib <- Sys.getenv("MGCV_LIB_PATH")\nif(nzchar(lib)) .libPaths(c(lib,.libPaths()))\nlibrary(mgcv)\nlibrary(jsonlite)\na <- fromJSON(args[1])\n'
                + code
                + "\nwrite_json(ans,args[2],digits=17,auto_unbox=TRUE)\n"
            )
            inp.write_text(json.dumps(payload))
            proc = subprocess.run(
                ["Rscript", str(src), str(inp), str(out)],
                capture_output=True,
                text=True,
            )
            assert proc.returncode == 0, proc.stderr
            save_reference(NAMESPACE, key, json.loads(out.read_text()))
    return load_reference(NAMESPACE, key)


@pytest.mark.parametrize("link", LINKS)
def test_inverse_gaussian_family_kernels(link):
    y = np.array([0.2, 0.7, 1.1, 1.8, 3.4])
    mu = np.array([0.4, 0.9, 1.2, 1.7, 2.9])
    w = np.array([0.0, 0.5, 1.0, 2.0, 3.0])
    phi = 0.37
    ref = _reference(
        {"y": y.tolist(), "mu": mu.tolist(), "w": w.tolist(), "phi": phi, "link": link},
        r"""
f <- mgcv:::fix.family.ls(mgcv:::fix.family.var(mgcv:::fix.family.link(inverse.gaussian(a$link))))
eta <- f$linkfun(a$mu)
expr <- switch(a$link,"1/mu^2"=quote(eta^-.5),inverse=quote(1/eta),log=quote(exp(eta)),identity=quote(eta))
Dexpr <- substitute((y-MU)^2/(y*MU^2),list(MU=expr))
y <- a$y
wd1 <- D(D(D(Dexpr,"eta"),"eta"),"eta")
wd2 <- D(wd1,"eta")
ans <- list(eta=eta,mu=f$linkinv(eta),mu_eta=f$mu.eta(eta),d2=f$d2link(a$mu),d3=f$d3link(a$mu),d4=f$d4link(a$mu),v=f$variance(a$mu),v1=f$dvar(a$mu),v2=f$d2var(a$mu),v3=f$d3var(a$mu),dev=f$dev.resids(a$y,a$mu,a$w),aic=f$aic(a$y,rep(1,length(y)),a$mu,a$w,sum(f$dev.resids(a$y,a$mu,a$w))),ls=f$ls(a$y,a$w,length(y),a$phi),wd1=eval(wd1)/2,wd2=eval(wd2)/2)
""",
    )
    f = InverseGaussianFamily(link)
    eta = f.link(mu)
    actual = {
        "eta": eta,
        "mu": f.inverse_link(eta),
        "mu_eta": f.mu_eta(eta),
        "d2": f.d2link(mu),
        "d3": f.d3link(mu),
        "d4": f.d4link(mu),
        "v": f.variance(mu),
        "v1": f.dvar(mu),
        "v2": f.d2var(mu),
        "v3": f.d3var(mu),
        "dev": f.deviance_obs(y, mu, w),
        "aic": f.aic(y, mu, weights=w),
        "ls": f.ls(y, w, scale=phi),
        "wd1": f.working_weight_derivative_eta(eta, y),
        "wd2": f.working_weight_second_derivative_eta(eta, y),
    }
    for name, value in actual.items():
        np.testing.assert_allclose(
            value, ref[name], rtol=2e-12, atol=2e-12, err_msg=name
        )
    np.testing.assert_allclose(
        f.loglik_obs(y, mu, phi),
        -0.5 * np.log(2 * np.pi * phi * y**3) - f.deviance_obs(y, mu) / (2 * phi),
    )


FIT_CODE = r"""
dat <- data.frame(x=a$x,y=a$y,w=a$w)
args0 <- list(formula=y~s(x,bs="cr",k=7),data=dat,family=inverse.gaussian(a$link),method=if(a$method=="fixed") "REML" else a$method,weights=dat$w,optimizer=c("outer",a$optimizer))
if(a$method=="fixed") args0$sp <- .4
fit <- do.call(gam,args0)
p <- predict(fit,type="response",se.fit=TRUE)
ans <- list(version=as.character(packageVersion("mgcv")),eta=predict(fit,type="link"),mu=p$fit,se=p$se.fit,edf=sum(fit$edf),scale=fit$sig2,dev=deviance(fit),aic=AIC(fit),bic=BIC(fit),loglik=as.numeric(logLik(fit)),Vp=unname(fit$Vp),Vc=unname(vcov(fit,unconditional=TRUE)),resdev=residuals(fit,type="deviance"),respearson=residuals(fit,type="pearson"),resresponse=residuals(fit,type="response"),resworking=residuals(fit,type="working"),sp=fit$sp,score=fit$gcv.ubre)
"""


def _generate_data(weighted):
    x = np.linspace(-1, 1, 100)
    mu = np.exp(0.25 + 0.35 * np.sin(3 * x))
    y = np.random.default_rng(931).wald(mu, 7.0)
    w = np.resize([0.5, 1.0, 1.5, 2.0], len(x)) if weighted else np.ones(len(x))
    return pd.DataFrame({"x": x, "y": y, "w": w})


def _data(weighted):
    # Freeze the exact R input doubles: libm/SIMD ULP differences in exp/sin
    # propagate through the Wald sampler and otherwise change fixture hashes.
    path = (
        Path(__file__).parents[1]
        / "reference_fixtures/mgcv_inverse_gaussian/input_data.json"
    )
    inputs = json.loads(path.read_text())
    return pd.DataFrame(inputs["weighted" if weighted else "base"])


@pytest.mark.parametrize("link", LINKS)
@pytest.mark.parametrize("method", ["fixed", "REML", "ML"])
@pytest.mark.parametrize("weighted", [False, True])
def test_inverse_gaussian_fit_mgcv(link, method, weighted):
    _assert_fit(link, method, weighted)


def _assert_fit(
    link, method, weighted, optimizer="outer_newton", family="inverse_gaussian"
):
    dat = _data(weighted)
    payload = {
        **{k: dat[k].tolist() for k in dat},
        "link": link,
        "method": method,
        "optimizer": "newton" if optimizer == "outer_newton" else optimizer,
    }
    code = (
        FIT_CODE
        if family == "inverse_gaussian"
        else FIT_CODE.replace("inverse.gaussian(a$link)", "Gamma(a$link)")
    )
    ref = _reference(payload, code)
    gam = GAM(
        family={"name": family, "link": link},
        formula="y ~ s(x, bs='cr', k=7)",
        optimize_smoothing=method != "fixed",
        smoothing_method="gcv" if method == "GCV.Cp" else method,
        smoothing_optimizer=optimizer,
        smoothing_params=[0.4] if method == "fixed" else None,
    ).fit(data=dat, sample_weight=dat.w.to_numpy())
    result = gam.fit_result()
    mu, se = gam.predict(type="response", return_se=True)
    for name, value in {
        "eta": gam.predict(type="link"),
        "mu": mu,
        "se": se,
        "edf": result.edf_total,
        "scale": result.scale,
        "dev": result.deviance,
        "aic": gam.aic(),
        "bic": gam.bic(),
        "loglik": gam.loglik(),
        "Vp": gam.vcov(),
    }.items():
        np.testing.assert_allclose(
            value,
            ref[name],
            rtol=2e-5,
            atol=2e-6,
            err_msg=f"{link}/{method}/{weighted}/{name}",
        )
    for kind, key in [
        ("deviance", "resdev"),
        ("pearson", "respearson"),
        ("response", "resresponse"),
        ("working", "resworking"),
    ]:
        np.testing.assert_allclose(
            gam.residuals(type=kind), ref[key], rtol=2e-5, atol=2e-6, err_msg=key
        )
    if method in {"REML", "ML"}:
        np.testing.assert_allclose(
            gam.vcov(unconditional=True), ref["Vc"], rtol=3e-4, atol=2e-6
        )


@pytest.mark.parametrize(
    "spec",
    [
        "inverse_gaussian",
        "inverse.gaussian",
        "inversegaussian",
        ("inverse_gaussian", "log"),
        {"name": "inverse_gaussian", "link": "identity"},
    ],
)
def test_inverse_gaussian_registry(spec):
    family = make_gam_family(spec)
    assert isinstance(family, InverseGaussianFamily)
    if isinstance(spec, str):
        assert family.link_name == "1/mu^2"


@pytest.mark.parametrize("bad", [[0.0, 1.0], [-1.0, 1.0], [np.nan, 1.0], [np.inf, 1.0]])
def test_inverse_gaussian_rejects_invalid_targets(bad):
    with pytest.raises(ValueError):
        InverseGaussianFamily().validate_y(bad)


@pytest.mark.parametrize("link", LINKS)
def test_inverse_gaussian_joint_scale_analytic_derivatives(link):
    from nampy.gam.fit.selection.optimize.objectives import (
        _JointGammaPirlsRemlObjective,
    )

    dat = _data(True)
    model = GAM(
        family={"name": "inverse_gaussian", "link": link},
        formula="y ~ s(x, bs='cr', k=7)",
        optimize_smoothing=False,
        smoothing_params=[0.4],
    ).fit(data=dat, sample_weight=dat.w.to_numpy())
    objective = _JointGammaPirlsRemlObjective(model, dat.y.to_numpy(), "REML")
    rho = np.array([np.log(0.6), np.log(0.2)])
    step = 1e-4
    directions = np.eye(2) * step
    numerical_gradient = np.array(
        [
            (objective.fun(rho + d) - objective.fun(rho - d)) / (2 * step)
            for d in directions
        ]
    )
    numerical_hessian = np.column_stack(
        [
            (objective.jac(rho + d) - objective.jac(rho - d)) / (2 * step)
            for d in directions
        ]
    )
    np.testing.assert_allclose(
        objective.jac(rho), numerical_gradient, rtol=3e-5, atol=2e-5
    )
    np.testing.assert_allclose(
        objective.hess(rho), numerical_hessian, rtol=3e-5, atol=2e-5
    )


@pytest.mark.parametrize("link", LINKS)
@pytest.mark.parametrize(
    "method,optimizer", [("GCV.Cp", "outer_newton"), ("REML", "bfgs")]
)
def test_inverse_gaussian_additional_optimizers(link, method, optimizer):
    _assert_fit(link, method, True, optimizer)


def test_inverse_gaussian_regressor_adapter():
    from nampy.models import GAMRegressor

    dat = _data(False)
    model = GAMRegressor(
        family="inverse_gaussian",
        formula="y ~ s(x, bs='cr', k=7)",
        optimize_smoothing=False,
        smoothing_params=[0.4],
    )
    model.fit(dat[["x"]], dat.y.to_numpy())
    assert model.predict(dat[["x"]]).shape == (len(dat),)
    assert np.all(np.isfinite(model.predict(dat[["x"]])))


@pytest.mark.parametrize("link", LINKS)
def test_inverse_gaussian_two_smooth_gcv_derivatives(link):
    from nampy.gam.fit.selection.criteria import (
        criterion_gradient,
        criterion_hessian,
        criterion_value,
    )

    dat = _data(True)
    dat["z"] = np.random.default_rng(53).permutation(dat.x.to_numpy())
    model = GAM(
        family={"name": "inverse_gaussian", "link": link},
        formula="y ~ s(x, bs='cr', k=7) + s(z, bs='cr', k=6)",
        optimize_smoothing=False,
        smoothing_params=[0.6, 0.4],
    ).fit(data=dat, sample_weight=dat.w.to_numpy())
    y = dat.y.to_numpy()
    rho = np.log([0.6, 0.4])
    step = 1e-4
    directions = np.eye(2) * step
    numerical_gradient = np.array(
        [
            (
                criterion_value(model, y, rho + d, "gcv")
                - criterion_value(model, y, rho - d, "gcv")
            )
            / (2 * step)
            for d in directions
        ]
    )
    numerical_hessian = np.column_stack(
        [
            (
                criterion_gradient(model, y, rho + d, "gcv")
                - criterion_gradient(model, y, rho - d, "gcv")
            )
            / (2 * step)
            for d in directions
        ]
    )
    np.testing.assert_allclose(
        criterion_gradient(model, y, rho, "gcv"),
        numerical_gradient,
        rtol=3e-5,
        atol=2e-8,
    )
    np.testing.assert_allclose(
        criterion_hessian(model, y, rho, "gcv"), numerical_hessian, rtol=3e-5, atol=2e-8
    )


@pytest.mark.parametrize(
    "family,link", [("inverse_gaussian", "inverse"), ("gamma", "log")]
)
def test_noncanonical_two_smooth_gcv_mgcv(family, link):
    dat = _data(True)
    dat["z"] = np.random.default_rng(53).permutation(dat.x.to_numpy())
    path = (
        Path(__file__).parents[1]
        / "reference_fixtures/mgcv_inverse_gaussian/input_data.json"
    )
    inputs = json.loads(path.read_text())["two_smooth"]
    dat["z"], dat["y"] = inputs["z"], inputs["y"]
    payload = {
        **{k: dat[k].tolist() for k in dat},
        "link": link,
        "method": "GCV.Cp",
        "optimizer": "newton",
    }
    code = FIT_CODE.replace(
        "data.frame(x=a$x,y=a$y,w=a$w)", "data.frame(x=a$x,z=a$z,y=a$y,w=a$w)"
    ).replace('y~s(x,bs="cr",k=7)', 'y~s(x,bs="cr",k=7)+s(z,bs="cr",k=6)')
    if family == "gamma":
        code = code.replace("inverse.gaussian(a$link)", "Gamma(a$link)")
    ref = _reference(payload, code)
    model = GAM(
        family={"name": family, "link": link},
        formula="y ~ s(x, bs='cr', k=7) + s(z, bs='cr', k=6)",
        optimize_smoothing=True,
        smoothing_method="gcv",
        smoothing_optimizer="outer_newton",
    ).fit(data=dat, sample_weight=dat.w.to_numpy())
    result = model.fit_result()
    for name, value in {
        "eta": model.predict(type="link"),
        "mu": model.predict(type="response"),
        "edf": result.edf_total,
        "scale": result.scale,
        "Vp": model.vcov(),
        "aic": model.aic(),
        "loglik": model.loglik(),
    }.items():
        np.testing.assert_allclose(value, ref[name], rtol=2e-5, atol=2e-6, err_msg=name)


@pytest.mark.parametrize("link", ["inverse", "log"])
def test_weighted_gamma_bfgs_scale_initialization_regression(link):
    _assert_fit(link, "REML", True, "bfgs", family="gamma")
