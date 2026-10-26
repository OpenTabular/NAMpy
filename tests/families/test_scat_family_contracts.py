"""SCAT leaf contracts against actual mgcv 1.9-4 R snapshots."""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from tests.reference_fixtures import (
    generation_enabled,
    load_reference,
    reference_key,
    save_reference,
)

CODE = """
f <- do.call(scat,list(theta=a$theta,link=a$link,min.df=a$min_df))
y <- a$y; mu <- a$mu; w <- a$w
ans <- list(version=as.character(packageVersion('mgcv')),theta=f$getTheta(), transformed=f$getTheta(TRUE),n_theta=f$n.theta,variance=f$variance(mu),dev=f$dev.resids(y,mu,w),aic=f$aic(y,mu,wt=w,dev=0),ls=f$ls(y,w,f$getTheta(),1),dd0=f$Dd(y,mu,f$getTheta(),w,0),dd1=f$Dd(y,mu,f$getTheta(),w,1),dd2=f$Dd(y,mu,f$getTheta(),w,2),pre=f$preinitialize(y,f),link=f$linkfun(mu),inverse=f$linkinv(a$eta),mu_eta=f$mu.eta(a$eta))
e <- new.env(); e$y <- y; eval(f$initialize,e); ans$initial_mu <- e$mustart
"""


def reference(payload, code=CODE):
    key = reference_key("scat-leaf", {"payload": payload, "code": code})
    if generation_enabled():
        with tempfile.TemporaryDirectory() as directory:
            source, inp, out = [
                Path(directory) / name for name in ["ref.R", "input.json", "out.json"]
            ]
            source.write_text(
                'args<-commandArgs(TRUE)\nlib<-Sys.getenv("MGCV_LIB_PATH"); if(nzchar(lib)) .libPaths(c(lib,.libPaths()))\nlibrary(mgcv);library(jsonlite)\na<-fromJSON(args[1])\n'
                + code
                + "\nwrite_json(ans,args[2],digits=17,auto_unbox=TRUE)"
            )
            inp.write_text(json.dumps(payload))
            result = subprocess.run(
                ["Rscript", str(source), str(inp), str(out)],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, result.stderr
            save_reference("mgcv_scat_family", key, json.loads(out.read_text()))
    return load_reference("mgcv_scat_family", key)


@pytest.mark.parametrize("link", ["identity", "log", "inverse"])
@pytest.mark.parametrize(
    "theta", [None, [7.0, 1.2], [-5.0, -0.4], [5.0, -0.8], [0.0, 2.0], [2.0, 0.7]]
)
def test_scat_complete_kernel_contract(link, theta):
    payload = {
        "theta": theta,
        "link": link,
        "min_df": 3.0,
        "y": [-9.0, 0.0, 0.3, 2.0, 15.0],
        "mu": [0.2, 0.6, 1.0, 2.5, 4.0],
        "eta": [0.1, 0.4, 0.8, 1.2, 2.0],
        "w": [0.0, 0.5, 1.0, 2.0, 3.0],
    }
    ref = reference(payload)
    from nampy.gam.families.scat import ScaledTFamily

    if theta == [2.0, 0.7]:
        with pytest.warns(UserWarning, match="min.df reset"):
            family = ScaledTFamily(theta=theta, link=link)
    else:
        family = ScaledTFamily(theta=theta, link=link)
    assert ref["version"] == "1.9.4"
    assert family.n_theta == ref["n_theta"]
    np.testing.assert_allclose(family.getTheta(), ref["theta"], atol=2e-14)
    np.testing.assert_allclose(family.getTheta(True), ref["transformed"], atol=2e-14)
    y, mu, w = [np.array(payload[name]) for name in ["y", "mu", "w"]]
    for level in range(3):
        actual = family.Dd(y, mu, family.getTheta(), w, level)
        assert actual.keys() == ref[f"dd{level}"].keys()
        for key, value in actual.items():
            np.testing.assert_allclose(
                value, ref[f"dd{level}"][key], atol=3e-12, rtol=3e-12, err_msg=key
            )
    for key, value in family.ls(y, w, family.getTheta(), 1.0).items():
        np.testing.assert_allclose(
            value, ref["ls"][key], atol=3e-12, rtol=3e-12, err_msg=key
        )
    np.testing.assert_allclose(
        family.variance(mu), np.asarray(ref["variance"], dtype=float), atol=2e-14
    )
    np.testing.assert_allclose(family.deviance_obs(y, mu, w), ref["dev"], atol=2e-13)
    np.testing.assert_allclose(family.aic(y, mu, weights=w), ref["aic"], atol=2e-13)
    np.testing.assert_allclose(family.initialize_mu(y), ref["initial_mu"])
    np.testing.assert_allclose(family.link(mu), ref["link"])
    np.testing.assert_allclose(family.inverse_link(payload["eta"]), ref["inverse"])
    np.testing.assert_allclose(family.mu_eta(payload["eta"]), ref["mu_eta"])
    pre = family.preinitialize(y)
    if not ref["pre"]:
        assert pre is None
    else:
        np.testing.assert_allclose(pre["Theta"], ref["pre"]["Theta"], atol=2e-14)


@pytest.mark.parametrize("link", ["identity", "log", "inverse"])
@pytest.mark.parametrize("theta", [[3.01, 0.01], [40.0, 12.0], [1.2, 0.5]])
def test_scat_tails_signed_means_scalar_weights_and_mutation(link, theta):
    from nampy.gam.families.scat import ScaledTFamily

    payload = {
        "theta": theta,
        "link": link,
        "min_df": 0.5,
        "y": [-100.0, -1.0, 0.0, 3.0, 500.0],
        "mu": [-2.0, -0.3, 1.0, 3.0, 0.01]
        if link != "log"
        else [0.1, 0.5, 1.0, 3.0, 0.01],
        "eta": [-3.0, -0.2, 0.1, 1.0, 3.0]
        if link != "log"
        else [0.1, 0.2, 1.0, 2.0, 3.0],
        "w": 2.0,
    }
    ref = reference(payload)
    f = ScaledTFamily(theta=theta, link=link, min_df=0.5)
    y, mu = np.array(payload["y"]), np.array(payload["mu"])
    for key, value in f.Dd(y, mu, wt=2.0, level=2).items():
        np.testing.assert_allclose(
            value, ref["dd2"][key], rtol=3e-12, atol=3e-12, err_msg=key
        )
    for key, value in f.ls(y, 2.0).items():
        np.testing.assert_allclose(
            value, ref["ls"][key], rtol=3e-12, atol=3e-12, err_msg=key
        )
    np.testing.assert_allclose(f.inverse_link(payload["eta"]), ref["inverse"])
    np.testing.assert_allclose(f.mu_eta(payload["eta"]), ref["mu_eta"])
    np.testing.assert_allclose(
        -2 * np.sum(f.loglik_obs(y, mu) * 2), ref["aic"], rtol=3e-12, atol=3e-12
    )
    old = f.getTheta()
    value = f.getTheta()
    value[:] = 0
    np.testing.assert_array_equal(f.getTheta(), old)
    f.putTheta(old + [0.1, 0.2])
    np.testing.assert_allclose(
        f.getTheta(True), [np.exp(old[0] + 0.1) + 0.5, np.exp(old[1] + 0.2)]
    )


@pytest.mark.parametrize("theta", [None, [-6.0, -0.7], [7.0, 1.2]])
def test_scat_initialization_runs_only_once(theta):
    from nampy.gam.families.scat import ScaledTFamily

    f = ScaledTFamily(theta=theta)
    initial = f.getTheta()
    y = np.array([-4.0, 0.0, 1.0, 8.0])
    f.initialize_mu(y)
    np.testing.assert_array_equal(f.getTheta(), initial)
    f.preinitialize(y)
    expected = (
        initial if not f.estimate_theta else [1.5, np.log(0.8 * np.std(y, ddof=1))]
    )
    np.testing.assert_allclose(f.getTheta(), expected)
    assert f.preinitialize(100 * y) is None
    np.testing.assert_allclose(f.getTheta(), expected)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"theta": [1.0]},
        {"theta": [1.0, 2.0, 3.0]},
        {"theta": [np.nan, 1.0]},
        {"link": "sqrt"},
        {"min_df": np.inf},
    ],
)
def test_scat_invalid_constructor_inputs(kwargs):
    from nampy.gam.families.scat import ScaledTFamily

    with pytest.raises(ValueError):
        ScaledTFamily(**kwargs)


def test_scat_validity_and_fisher_contracts():
    from nampy.gam.families.scat import ScaledTFamily

    f = ScaledTFamily(theta=[5.0, 1.0])
    assert f.valid_mu([-4.0, 0.0, 7.0])
    assert not f.valid_mu([np.nan])
    assert f.valid_eta([np.inf])
    assert not ScaledTFamily(link="inverse").valid_eta([0.0])
    with pytest.raises(ValueError):
        f.initialize_mu([np.nan])
    y = np.array([0.0, 0.1, 3.0, 8.0])
    dd = f.Dd(y, np.zeros_like(y), wt=[0.0, 0.5, 1.0, 2.0], level=2)
    assert np.any(dd["Dmu2"] < 0)
    assert np.all(dd["EDmu2"] > 0)
    assert dd["Dmu2"][0] == 0 and dd["EDmu2"][0] > 0
    np.testing.assert_array_equal(dd["EDmu2"], f.Dd(y, y, wt=20)["EDmu2"])


def test_scat_simulation_ignores_weights_and_external_scale():
    from nampy.gam.families.scat import ScaledTFamily

    f = ScaledTFamily(theta=[6.0, 0.8])
    mu = np.linspace(-2.0, 2.0, 17)
    expected = np.random.default_rng(190).standard_t(6.0, size=mu.shape) * 0.8 + mu
    np.testing.assert_allclose(
        f.rd(mu, wt=np.arange(17), scale=99.0, rng=np.random.default_rng(190)), expected
    )
    lower, upper = f.quantile_residual_bounds(mu, mu, weights=np.arange(17), scale=20.0)
    np.testing.assert_array_equal(lower, upper)
    np.testing.assert_allclose(lower, 0.5)


def test_scat_state_roundtrip_and_explicit_theta_do_not_mutate():
    import pickle

    from nampy.gam.families.scat import ScaledTFamily

    f = ScaledTFamily(theta=[-6.0, -0.8], link="inverse")
    f.preinitialize([-2.0, 0.0, 1.0, 4.0])
    theta = f.getTheta()
    restored = pickle.loads(pickle.dumps(f))
    np.testing.assert_array_equal(restored.getTheta(), theta)
    assert restored.preinitialize([3.0, 5.0]) is None
    f.Dd([0.0, 2.0], [1.0, 3.0], theta=theta + [0.1, 0.2], level=2)
    f.ls([0.0, 2.0], 1.0, theta=theta + [0.1, 0.2])
    f.aic([0.0, 2.0], [1.0, 3.0], theta=theta + [0.1, 0.2])
    np.testing.assert_array_equal(f.getTheta(), theta)


@pytest.mark.parametrize("link", ["identity", "log", "inverse"])
def test_scat_source_valideta(link):
    from nampy.gam.families.scat import ScaledTFamily

    code = "f<-do.call(scat,list(link=a$link)); ans<-list(eta=vapply(list(c(-1,0,1),Inf,NaN,-Inf),f$valideta,logical(1)),mu=vapply(list(c(-1,0,1),Inf,NaN,-Inf),f$validmu,logical(1)))"
    ref = reference({"link": link}, code)
    family = ScaledTFamily(link=link)
    values = [[-1.0, 0.0, 1.0], [np.inf], [np.nan], [-np.inf]]
    assert [family.valid_eta(value) for value in values] == ref["eta"]
    assert [family.valid_mu(value) for value in values] == ref["mu"]


@pytest.mark.parametrize("scale", [None, 1.0, 2.0, -1.0, 0.0])
def test_scat_upstream_external_scale_inventory(scale):
    from tests.families.test_mgcv_scat_workflows import _case

    payload = _case()
    payload["scale"] = scale
    code = """
d<-as.data.frame(a$data)
fit_args<-list(formula=as.formula(a$formula),data=d,weights=d$w,family=scat(theta=c(5,.28)),sp=a$sp,method='REML')
if(!is.null(a$scale)) fit_args$scale<-a$scale
b<-do.call(gam,fit_args)
ans<-list(coef=coef(b),scale=b$sig2,score=as.numeric(b$gcv.ubre),Vp=b$Vp,theta=b$family$getTheta(TRUE),aic=AIC(b),scale_estimated=b$scale.estimated)
"""
    ref = reference(payload, code)
    baseline_payload = dict(payload, scale=None)
    baseline = reference(baseline_payload, code)
    expected_scale = scale if scale is not None and scale > 0 else 1.0
    assert ref["scale"] == expected_scale
    assert ref["scale_estimated"] is False
    np.testing.assert_array_equal(ref["coef"], baseline["coef"])
    np.testing.assert_allclose(
        ref["Vp"], np.asarray(baseline["Vp"]) * expected_scale, atol=2e-14, rtol=2e-14
    )
    np.testing.assert_allclose(ref["aic"], baseline["aic"], atol=2e-13)
    if expected_scale == 1:
        assert ref["score"] == baseline["score"]
    else:
        assert ref["score"] != baseline["score"]
