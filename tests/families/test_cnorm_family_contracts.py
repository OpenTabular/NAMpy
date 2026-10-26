"""Censored-normal leaf references from installed mgcv, never R in normal tests."""

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

SCRIPT = r"""
capture <- function(expr) {
  warnings <- character()
  value <- tryCatch(withCallingHandlers(expr, warning=function(w) {
    warnings <<- c(warnings,conditionMessage(w)); invokeRestart("muffleWarning")
  }),error=function(e) structure(list(message=conditionMessage(e)),class="captured_error"))
  list(value=if(inherits(value,"captured_error")) NULL else value,
       error=if(inherits(value,"captured_error")) value$message else NULL,
       warnings=warnings)
}
f <- do.call(cnorm,list(theta=a$theta,link=a$link))
y <- as.numeric(a$y); mu<-as.numeric(a$mu); w<-as.numeric(a$w)
if(!is.null(a$bounds)) attr(y,"censor")<-as.numeric(a$bounds)
ans<-list(version=as.character(packageVersion("mgcv")),R=as.character(getRversion()),
 theta=f$getTheta(),physical=f$getTheta(TRUE),n_theta=f$n.theta,
 dev=capture(f$dev.resids(y,mu,w)),aic=capture(f$aic(y,mu,wt=w,dev=0)),
 ls=capture(f$ls(y,w,f$getTheta(),a$scale)),
 dd0=capture(f$Dd(y,mu,f$getTheta(),w,0)),
 dd1=capture(f$Dd(y,mu,f$getTheta(),w,1)),
 dd2=capture(f$Dd(y,mu,f$getTheta(),w,2)))
e<-new.env();e$y<-if(is.null(a$bounds)) y else cbind(a$y,as.numeric(a$bounds));e$family<-f
ans$initialize<-capture({eval(f$initialize,e);list(mu=e$mustart,y=as.numeric(e$y),bounds=attr(e$y,"censor"))})
ans$subset<-capture({z<-f$subsety(y,rev(seq_along(y)));list(y=as.numeric(z),bounds=attr(z,"censor"))})
ans$link<-capture(list(eta=f$linkfun(mu),inverse=f$linkinv(a$eta),derivative=f$mu.eta(a$eta),validmu=f$validmu(mu),valideta=f$valideta(a$eta)))
"""


def reference(payload, script=SCRIPT):
    key = reference_key("cnorm-leaf-v1", {"payload": payload, "script": script})
    if generation_enabled():
        with tempfile.TemporaryDirectory() as tmp:
            source, inputs, output = [
                Path(tmp) / s for s in ["run.R", "in.json", "out.json"]
            ]
            source.write_text(
                'args<-commandArgs(TRUE)\nlib<-Sys.getenv("MGCV_LIB_PATH");if(nzchar(lib)) .libPaths(c(lib,.libPaths()))\nsuppressPackageStartupMessages(library(mgcv));library(jsonlite)\na<-fromJSON(args[1])\n'
                + script
                + '\nwrite_json(ans,args[2],digits=17,auto_unbox=TRUE,na="string")'
            )
            inputs.write_text(json.dumps(payload))
            run = subprocess.run(
                ["Rscript", str(source), str(inputs), str(output)],
                capture_output=True,
                text=True,
            )
            assert run.returncode == 0, run.stderr
            save_reference("mgcv_cnorm_family", key, json.loads(output.read_text()))
    return load_reference("mgcv_cnorm_family", key)


def case(band="mixed", theta=1.3, link="identity"):
    y = [0.2, 1.2, 2.3, -0.5, 2.0, 1.0]
    bounds = [0.2, "-Inf", "Inf", 0.7, 1.1, 1.0]
    mu = [0.5, 0.9, 1.8, 0.1, 1.4, 0.8]
    w = [1.0] * 6
    if band == "precision":
        w = [0.25, 0.5, 2.0, 3.0, 1.5, 4.0]
    elif band == "zero":
        w = [0.0] * 6
    elif band == "uncensored":
        y = [-2.0, -0.1, 0.0, 0.5, 1.2, 3.0]
        bounds = None
    elif band == "tails":
        y = [-35.0, 35.0, -30.0, 30.0, 0.1, -0.1]
        bounds = ["-Inf", "Inf", -29.99, 30.01, "-Inf", "Inf"]
        mu = [0.0] * 6
    elif band == "endpoint_zero":
        y = [0.0, -1.0, -2.0, -0.4, 0.0, -2.0]
        bounds = [1.0, 0.0, 0.3, 0.6, 2.0, 0.0]
        mu = [0.0] * 6
    elif band == "symmetric":
        y = [-1.0, -2.0, -0.25, -3.0, -0.5, -0.75]
        bounds = [-v for v in y]
        mu = [0.0] * 6
    return {
        "y": y,
        "bounds": bounds,
        "mu": mu,
        "w": w,
        "theta": theta,
        "link": link,
        "eta": [-2.0, -0.2, 0.0, 0.2, 1.0, 3.0],
        "scale": 1.0,
    }


CONSTRUCTORS = [
    (theta, link)
    for theta in [None, 0.0, -0.7, 0.7, 2.0]
    for link in ["identity", "log", "sqrt"]
]
BANDS = [
    "mixed",
    "precision",
    "zero",
    "uncensored",
    "tails",
    "endpoint_zero",
    "symmetric",
]


@pytest.mark.parametrize("theta,link", CONSTRUCTORS)
def test_cnorm_source_constructor_inventory(theta, link):
    ref = reference(case(theta=theta, link=link))
    assert ref["version"] == "1.9.4"
    assert ref["n_theta"] == (0 if theta is not None and theta > 0 else 1)
    expected = 0.0 if theta is None else np.log(theta) if theta > 0 else theta
    np.testing.assert_allclose(ref["theta"], expected, atol=1e-15)


@pytest.mark.parametrize("band", BANDS)
def test_cnorm_source_encoding_inventory(band):
    ref = reference(case(band))
    assert ref["version"] == "1.9.4"
    for name in ["dev", "aic", "ls", "dd0", "dd1", "dd2", "initialize", "subset"]:
        if band == "zero" and name in ("dd1", "dd2"):
            assert (
                ref[name]["error"] == "NAs are not allowed in subscripted assignments"
            )
        else:
            assert not ref[name]["error"], (band, name, ref[name])
    assert len(ref["dd0"]["value"]["Dmu"]) == 6


def _numeric(value):
    """jsonlite retains NA/NaN/infinity explicitly in these references."""
    if isinstance(value, list):
        return np.asarray([_numeric(v) for v in value], dtype=float)
    if value in ("NA", "NaN"):
        return np.nan
    return float(value)


def _response(payload):
    values = np.asarray(payload["y"], dtype=float)
    if payload["bounds"] is None:
        return values
    return np.column_stack([values, _numeric(payload["bounds"])])


def _assert_kernel(payload):
    from nampy.gam.families.cnorm import CensoredNormalFamily

    ref = reference(payload)
    family = CensoredNormalFamily(theta=payload["theta"], link=payload["link"])
    y = _response(payload)
    mu, w = [np.asarray(payload[k]) for k in ["mu", "w"]]
    assert family.n_theta == ref["n_theta"]
    np.testing.assert_allclose(family.getTheta(), ref["theta"], atol=1e-15)
    np.testing.assert_allclose(family.getTheta(True), ref["physical"], atol=1e-15)
    with np.errstate(all="ignore"):
        for level in range(3):
            if ref[f"dd{level}"]["error"]:
                with pytest.raises((ValueError, FloatingPointError)):
                    family.Dd(y, mu, family.getTheta(), w, level)
                continue
            actual = family.Dd(y, mu, family.getTheta(), w, level)
            expected = ref[f"dd{level}"]["value"]
            assert actual.keys() == expected.keys()
            for field in actual:
                np.testing.assert_allclose(
                    actual[field],
                    _numeric(expected[field]),
                    rtol=3e-10,
                    atol=3e-10,
                    equal_nan=True,
                    err_msg=field,
                )
        np.testing.assert_allclose(
            family.deviance_obs(y, mu, w),
            _numeric(ref["dev"]["value"]),
            rtol=3e-12,
            atol=3e-12,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            family.aic(y, mu, weights=w),
            _numeric(ref["aic"]["value"]),
            rtol=3e-12,
            atol=3e-12,
            equal_nan=True,
        )
        actual_ls = family.ls(y, w, family.getTheta(), payload["scale"])
        for field, value in actual_ls.items():
            np.testing.assert_allclose(
                value,
                _numeric(ref["ls"]["value"][field]),
                rtol=3e-12,
                atol=3e-12,
                equal_nan=True,
            )
        np.testing.assert_allclose(
            family.initialize_mu(y, weights=w),
            _numeric(ref["initialize"]["value"]["mu"]),
            atol=1e-15,
        )


@pytest.mark.parametrize("theta,link", CONSTRUCTORS)
def test_cnorm_constructor_and_complete_kernel(theta, link):
    _assert_kernel(case(theta=theta, link=link))


@pytest.mark.parametrize("band", BANDS)
def test_cnorm_encoding_precision_tail_kernels(band):
    _assert_kernel(case(band))


@pytest.mark.parametrize("link", ["identity", "log", "sqrt"])
def test_cnorm_link_and_initialization_contract(link):
    from nampy.gam.families.cnorm import CensoredNormalFamily

    payload = case(link=link)
    ref = reference(payload)
    f = CensoredNormalFamily(link=link)
    expected = ref["link"]["value"]
    for actual, key in [
        (f.link(payload["mu"]), "eta"),
        (f.inverse_link(payload["eta"]), "inverse"),
        (f.mu_eta(payload["eta"]), "derivative"),
    ]:
        np.testing.assert_allclose(
            actual, _numeric(expected[key]), atol=1e-15, rtol=1e-15
        )
    assert f.valid_mu(payload["mu"]) == expected["validmu"]
    assert f.valid_eta(payload["eta"]) == expected["valideta"]


INVALID_SCRIPT = r"""
ans<-list(error=tryCatch({do.call(cnorm,list(theta=a$theta,link=a$link));NULL},error=function(e)conditionMessage(e)))
"""


@pytest.mark.parametrize(
    "theta,link", [([1, 2], "identity"), (1, "inverse"), (1, "probit")]
)
def test_cnorm_rejected_constructor_against_source(theta, link):
    from nampy.gam.families.cnorm import CensoredNormalFamily

    ref = reference({"theta": theta, "link": link}, INVALID_SCRIPT)
    assert ref["error"]
    with pytest.raises(ValueError):
        CensoredNormalFamily(theta=theta, link=link)


def test_cnorm_source_interval_endpoint_is_batch_dependent():
    """Preserve upstream ddnorm's zero-sign full-vector replacement evidence."""
    payload = case("endpoint_zero")
    mixed = reference(payload)
    one = {k: v[:1] if isinstance(v, list) else v for k, v in payload.items()}
    single = reference(one)
    assert mixed["dd0"]["warnings"]
    # The later zero endpoint inherits an earlier row of the full RHS.
    one = {k: v[4:5] if isinstance(v, list) else v for k, v in payload.items()}
    second = reference(one)
    assert abs(mixed["dd0"]["value"]["Dmu2"][4] - second["dd0"]["value"]["Dmu2"]) > 0.1
    assert single["dd0"]["value"]["Dmu2"] > 0


def test_cnorm_structured_response_subset_and_pickle():
    import pickle

    from nampy.gam.families.cnorm import CensoredNormalFamily
    from nampy.gam.response import CensoredResponse

    payload = case("precision")
    ref = reference(payload)
    response = CensoredResponse(payload["y"], _numeric(payload["bounds"]))
    reversed_response = pickle.loads(pickle.dumps(response)).subset(
        slice(None, None, -1)
    )
    expected = ref["subset"]["value"]
    np.testing.assert_array_equal(reversed_response.values, expected["y"])
    np.testing.assert_array_equal(
        reversed_response.bounds, _numeric(expected["bounds"])
    )
    family = CensoredNormalFamily(theta=payload["theta"])
    np.testing.assert_allclose(
        family.deviance_obs(response, payload["mu"], payload["w"]),
        ref["dev"]["value"],
        atol=3e-12,
    )
    family.putTheta([0.3])
    restored = pickle.loads(pickle.dumps(family))
    np.testing.assert_array_equal(restored.getTheta(), [0.3])
    np.testing.assert_allclose(restored.getTheta(True), [np.exp(0.3)], atol=1e-15)


def test_cnorm_external_scale_does_not_enter_leaf_ls():
    payload = case("precision")
    payload["scale"] = 2.0
    _assert_kernel(payload)
