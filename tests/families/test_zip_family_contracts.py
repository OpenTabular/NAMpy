"""ziP leaf contracts against actual mgcv 1.9-4 snapshots."""

import json
import pickle
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
f<-ziP(theta=a$theta,b=a$b);y<-a$y;mu<-a$mu;w<-a$w
ans<-list(version=as.character(packageVersion('mgcv')),theta=f$getTheta(),transformed=f$getTheta(TRUE),n_theta=f$n.theta,dev=f$dev.resids(y,mu,w),aic=f$aic(y,mu,wt=w,dev=0),ls=f$ls(y,w,f$getTheta(),1),dd0=f$Dd(y,mu,f$getTheta(),w,0),dd1=f$Dd(y,mu,f$getTheta(),w,1),dd2=f$Dd(y,mu,f$getTheta(),w,2),pred=f$predict(f,eta=mu)[[1]],pred_se=f$predict(f,se=TRUE,X=diag(length(mu)),beta=mu,off=0,Vb=diag(a$se^2)))
e<-new.env();e$y<-y;eval(f$initialize,e);ans$initial_mu<-e$mustart
"""


def reference(payload, code=CODE):
    key = reference_key("zip-leaf", {"payload": payload, "code": code})
    if generation_enabled():
        with tempfile.TemporaryDirectory() as directory:
            source, inputs, output = [
                Path(directory) / n for n in ["ref.R", "input.json", "out.json"]
            ]
            source.write_text(
                'args<-commandArgs(TRUE)\nlib<-Sys.getenv("MGCV_LIB_PATH");if(nzchar(lib)) .libPaths(c(lib,.libPaths()))\nsuppressPackageStartupMessages(library(mgcv));library(jsonlite)\na<-fromJSON(args[1])\n'
                + code
                + "\nwrite_json(ans,args[2],digits=17,auto_unbox=TRUE)"
            )
            inputs.write_text(json.dumps(payload))
            result = subprocess.run(
                ["Rscript", str(source), str(inputs), str(output)],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, result.stderr
            save_reference("mgcv_zip_family", key, json.loads(output.read_text()))
    return load_reference("mgcv_zip_family", key)


@pytest.mark.parametrize("theta", [None, [0.0, 0.0], [-0.7, -0.4], [1.2, 0.2]])
@pytest.mark.parametrize("b", [0.0, 0.3, -0.2])
def test_zip_complete_leaf_contract(theta, b):
    payload = {
        "theta": theta,
        "b": b,
        "y": [0, 1, 2, 0, 7, 15],
        "mu": [-3.0, -1.0, 0.2, 1.0, 1.6, 2.5],
        "w": [0.0, 0.5, 1.0, 2.0, 3.0, 1.0],
        "se": [0.1, 0.3, 0.2, 0.1, 0.4, 0.5],
    }
    ref = reference(payload)
    from nampy.gam.families.zip import ZeroInflatedPoissonFamily

    f = ZeroInflatedPoissonFamily(theta=theta, b=b)
    assert ref["version"] == "1.9.4"
    assert f.n_theta == ref["n_theta"]
    np.testing.assert_allclose(f.getTheta(), ref["theta"], atol=1e-14)
    np.testing.assert_allclose(f.getTheta(True), ref["transformed"], atol=1e-14)
    y, mu, w = [np.asarray(payload[k]) for k in ["y", "mu", "w"]]
    for level in range(3):
        actual = f.Dd(y, mu, f.getTheta(), w, level)
        assert actual.keys() == ref[f"dd{level}"].keys()
        for key, value in actual.items():
            np.testing.assert_allclose(
                value, ref[f"dd{level}"][key], atol=3e-11, rtol=3e-11, err_msg=key
            )
    for key, value in f.ls(y, w, f.getTheta(), 1).items():
        np.testing.assert_array_equal(value, ref["ls"][key])
    np.testing.assert_allclose(
        f.deviance_obs(y, mu, w), ref["dev"], atol=1e-13, rtol=1e-13
    )
    np.testing.assert_allclose(
        f.aic(y, mu, weights=w), ref["aic"], atol=1e-13, rtol=1e-13
    )
    np.testing.assert_allclose(f.initialize_mu(y), ref["initial_mu"], atol=1e-14)
    np.testing.assert_allclose(f.response_from_eta(mu), ref["pred"], atol=1e-14)
    np.testing.assert_allclose(
        f.response_se_from_eta(mu, payload["se"]),
        ref["pred_se"]["se.fit"],
        atol=1e-13,
        rtol=1e-13,
    )


HELPER_CODE = """
x<-a$x;ans<-list(l1ee=mgcv:::l1ee(x),lee1=mgcv:::lee1(x),ldg=mgcv:::ldg(x,4),lde=mgcv:::lde(x,4),zipll=mgcv:::zipll(a$y,x,rev(x),4))
"""


@pytest.mark.parametrize("band", ["lower", "upper", "interior"])
def test_zip_stable_helpers_against_source(band):
    eps = np.finfo(float).eps
    xmax = np.finfo(float).max
    values = {
        "lower": [
            -750.0,
            -710.0,
            -50.0,
            np.log(eps) / 3 - 1e-7,
            np.log(eps) / 3,
            np.log(eps) / 3 + 1e-7,
            -8.0,
        ],
        "upper": [
            2.0,
            np.log(np.log(xmax)),
            np.log(np.log(xmax)) + 1,
            np.log(xmax) / 5 - 1e-7,
            np.log(xmax) / 5 + 1e-7,
            240.0,
            360.0,
        ],
        "interior": [-7.0, -3.0, -0.5, 0.0, 0.5, 1.0, 1.5],
    }[band]
    y = [0, 1, 2, 0, 5, 1, 0]
    ref = reference({"x": values, "y": y}, HELPER_CODE)
    from nampy.gam.families._zip_helpers import l1ee, lde, ldg, lee1, zipll

    with np.errstate(all="ignore"):
        actual = {
            "l1ee": l1ee(values),
            "lee1": lee1(values),
            "ldg": ldg(values, 4),
            "lde": lde(values, 4),
            "zipll": zipll(y, values, values[::-1], 4),
        }
    for key, value in actual.items():
        if isinstance(value, dict):
            for field, array in value.items():
                np.testing.assert_allclose(
                    array,
                    np.asarray(ref[key][field], dtype=float),
                    rtol=3e-11,
                    atol=3e-11,
                    err_msg=f"{key}.{field}",
                )
        else:
            np.testing.assert_allclose(
                value,
                np.asarray(ref[key], dtype=float),
                rtol=3e-11,
                atol=3e-11,
                err_msg=key,
            )


POST_CODE = """
f<-ziP(theta=a$theta,b=a$b); y<-a$y;mu<-a$mu;w<-a$w
sat<-f$saturated.ll(y,f,w)
post<-f$postproc(f,y,w,mu,mu,rep(.2,length(y)),FALSE)
ans<-list(sat=sat,post=post,deviance_residual=sign(y-f$predict(f,eta=mu)[[1]])*sqrt(pmax(0,f$dev.resids(y,mu,w)-sat)))
"""


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("theta", [[-0.7, -0.4], [0.0, 0.0]])
def test_zip_saturated_and_null_postprocessing(theta, weighted):
    payload = {
        "theta": theta,
        "b": 0.3,
        "y": [0, 1, 2, 0, 7, 15],
        "mu": [-3.0, -1.0, 0.2, 1.0, 1.6, 2.5],
        "w": [0.0, 0.5, 1.0, 2.0, 3.0, 1.0] if weighted else [1.0] * 6,
    }
    ref = reference(payload, POST_CODE)
    from nampy.gam.families.zip import ZeroInflatedPoissonFamily

    f = ZeroInflatedPoissonFamily(theta=theta, b=0.3)
    y, mu, w = [np.asarray(payload[k]) for k in ["y", "mu", "w"]]
    np.testing.assert_allclose(
        f.saturated_deviance_obs(y, w), ref["sat"], atol=2e-12, rtol=2e-12
    )
    np.testing.assert_allclose(
        f.postprocess_deviance(y, mu, w),
        ref["post"]["deviance"],
        atol=2e-12,
        rtol=2e-12,
    )
    np.testing.assert_allclose(
        f.null_deviance(y, mu, w, offset=np.full(6, 0.2), intercept=False),
        ref["post"]["null.deviance"],
        atol=2e-9,
        rtol=2e-10,
    )
    actual = np.sqrt(
        np.maximum(f.deviance_obs(y, mu, w) - f.saturated_deviance_obs(y, w), 0)
    ) * np.sign(y - f.response_from_eta(mu))
    np.testing.assert_allclose(actual, ref["deviance_residual"], atol=2e-12, rtol=2e-12)


VALIDATE_CODE = """
f<-ziP();e<-new.env();e$y<-a$y
ans<-tryCatch({eval(f$initialize,e);list(ok=TRUE,mu=e$mustart)},error=function(e)list(ok=FALSE,message=conditionMessage(e)))
"""


@pytest.mark.parametrize(
    "y",
    [
        [0, 1, 2],
        [-1, 0, 2],
        [0, 0.2, 2],
        [0, 1, 0],
        [0, 0, 0],
        [1, 1, 1],
        [0, 2 + 1e-9, 3],
        [0, 2 + 1e-5, 3],
    ],
)
def test_zip_response_initialization_source(y):
    ref = reference({"y": y}, VALIDATE_CODE)
    from nampy.gam.families.zip import ZeroInflatedPoissonFamily

    f = ZeroInflatedPoissonFamily()
    if ref["ok"]:
        np.testing.assert_allclose(f.initialize_mu(y), ref["mu"], atol=1e-14)
    else:
        with pytest.raises(ValueError):
            f.initialize_mu(y)


RANDOM_CODE = """
f<-ziP(theta=a$theta,b=a$b)
assign('runif',function(n,min=0,max=1) min+(max-min)*a$q,envir=environment(f$rd))
ans<-list(draw=f$rd(a$mu,rep(1,length(a$mu)),1))
"""


@pytest.mark.parametrize("q", [0.1, 0.4, 0.9])
def test_zip_simulation_controlled_uniforms(q):
    payload = {
        "theta": [-0.7, -0.4],
        "b": 0.3,
        "mu": [-30.0, -5.0, -1.0, 0.0, 1.0, 3.0, 5.0],
        "q": q,
    }
    ref = reference(payload, RANDOM_CODE)
    from nampy.gam.families.zip import ZeroInflatedPoissonFamily

    class FixedUniform:
        def uniform(self, low=0.0, high=1.0, size=None):
            if size is not None:
                return np.full(size, q)
            return np.asarray(low) + (np.asarray(high) - low) * q

    actual = ZeroInflatedPoissonFamily(theta=payload["theta"], b=0.3).rd(
        payload["mu"], rng=FixedUniform()
    )
    np.testing.assert_array_equal(actual, ref["draw"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"link": "log"},
        {"theta": [1.0]},
        {"theta": [0.0, np.inf]},
        {"b": np.nan},
        {"b": [0.3]},
    ],
)
def test_zip_public_input_guards(kwargs):
    from nampy.gam.families.zip import ZeroInflatedPoissonFamily

    with pytest.raises(ValueError):
        ZeroInflatedPoissonFamily(**kwargs)


def test_zip_theta_copy_serialization_and_link_contract():
    from nampy.gam.families.zip import ZeroInflatedPoissonFamily

    f = ZeroInflatedPoissonFamily(theta=[0.0, -2.0], b=0.2)
    got = f.getTheta()
    got[:] = 100
    np.testing.assert_array_equal(f.getTheta(), [0.0, -2.0])
    f.putTheta([-0.5, 0.1])
    clone = pickle.loads(pickle.dumps(f))
    np.testing.assert_array_equal(clone.getTheta(), f.getTheta())
    assert f.n_theta == 0 and not f.estimate_theta
    x = np.array([-3.0, 0.0, 4.0])
    np.testing.assert_array_equal(f.inverse_link(x), x)
    np.testing.assert_array_equal(f.mu_eta(x), np.ones(3))
    for order in [2, 3, 4]:
        np.testing.assert_array_equal(f.inverse_link_derivatives(x, order), np.zeros(3))
    assert f.valid_eta([np.nan]) and not f.valid_mu([np.nan])
    assert f.known_scale == 1 and f.no_r_sq


@pytest.mark.parametrize("weight", [0.0, 0.5, 2.0])
def test_zip_tail_and_scalar_weight_Dd(weight):
    payload = {
        "theta": [-0.3, -2.0],
        "b": 0.3,
        "y": [0, 1, 1, 2, 0, 100],
        "mu": [-40.0, -20.0, -12.0, -3.0, 0.0, 6.0],
        "w": weight,
        "se": [0.1] * 6,
    }
    ref = reference(payload)
    from nampy.gam.families.zip import ZeroInflatedPoissonFamily

    f = ZeroInflatedPoissonFamily(theta=payload["theta"], b=0.3)
    for level in range(3):
        for key, value in f.Dd(
            payload["y"], payload["mu"], wt=weight, level=level
        ).items():
            np.testing.assert_allclose(
                value, ref[f"dd{level}"][key], atol=3e-11, rtol=3e-11, err_msg=key
            )
    np.testing.assert_allclose(
        f.deviance_obs(payload["y"], payload["mu"], weight),
        ref["dev"],
        atol=3e-11,
        rtol=3e-11,
    )
    np.testing.assert_allclose(
        f.response_from_eta(payload["mu"]), ref["pred"], atol=1e-13, rtol=1e-13
    )
    np.testing.assert_allclose(
        f.response_se_from_eta(payload["mu"], payload["se"]),
        ref["pred_se"]["se.fit"],
        atol=1e-11,
        rtol=1e-11,
    )


def test_zip_residual_object_and_scalar_contract():
    from types import SimpleNamespace

    from nampy.gam.families.zip import ZeroInflatedPoissonFamily

    f = ZeroInflatedPoissonFamily(theta=[-0.7, -0.4], b=0.3)
    y = np.array([0.0, 1.0, 3.0])
    eta = np.array([-2.0, 0.0, 1.0])
    offset = np.array([0.1, -0.2, 0.3])
    state = SimpleNamespace(working_response=np.array([3.0, np.inf, -2.0]))
    core = SimpleNamespace(fit_state=state, fit_result=SimpleNamespace(eta=eta, mu=eta))
    model = SimpleNamespace(
        y_=y,
        prior_weights_=np.array([1.0, 0.0, 2.0]),
        offset_train_=offset,
        gam_result_=SimpleNamespace(fit_core_solution=core),
        predict=lambda type: f.response_from_eta(eta),
    )
    np.testing.assert_allclose(f.residuals(model, "working"), [5.1, np.nan, -2.7])
    np.testing.assert_allclose(
        f.residuals(model, "response"), y - f.response_from_eta(eta)
    )
    expected = np.sqrt(
        np.maximum(
            f.deviance_obs(y, eta) - f.saturated_deviance_obs(y, model.prior_weights_),
            0,
        )
    ) * np.sign(y - f.response_from_eta(eta))
    np.testing.assert_allclose(f.residuals(model, "deviance"), expected)
    for kind in ["pearson", "quantile", "unknown"]:
        with pytest.raises(NotImplementedError):
            f.residuals(model, kind)
    np.testing.assert_allclose(f.deviance_obs(2.0, 0.5), f.deviance_obs([2.0], [0.5]))
    np.testing.assert_allclose(f.response_from_eta(0.5), f.response_from_eta([0.5]))
    np.testing.assert_allclose(
        f.response_se_from_eta(0.5, 0.1), f.response_se_from_eta([0.5], [0.1])
    )
