"""Fresh static stats/mgcv quasi family kernels and boundary contracts."""

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

LINKS = [
    "log",
    "identity",
    "sqrt",
    "inverse",
    "1/mu^2",
    "logit",
    "probit",
    "cloglog",
    "cauchit",
]
VARIANCES = ["constant", "mu", "mu(1-mu)", "mu^2", "mu^3"]
CASES = [
    (family, variance, link, None)
    for family, variances in [
        ("quasipoisson", ["mu"]),
        ("quasibinomial", ["mu(1-mu)"]),
        ("quasi", VARIANCES),
    ]
    for variance in variances
    for link in LINKS
]
CASES += [
    ("quasi", variance, "power", power)
    for variance in VARIANCES
    for power in [0.5, 1.7]
]
CASES += [("quasi", "constant", "power", power) for power in [-1.0, 0.0, 1.0]]

CODE = r"""
link <- if(is.null(a$power)) a$link else power(a$power)
f <- if(a$family=="quasi") do.call(quasi,list(link=link,variance=a$variance)) else do.call(get(a$family),list(link=link))
f <- mgcv:::fix.family.ls(mgcv:::fix.family.var(mgcv:::fix.family.link(f)))
y <- a$y; weights <- a$weights; nobs <- length(y)
eval(f$initialize)
eta <- f$linkfun(a$mu)
V <- switch(a$variance,constant=quote(1),mu=quote(mu),"mu(1-mu)"=quote(mu*(1-mu)),"mu^2"=quote(mu^2),"mu^3"=quote(mu^3))
Dev <- switch(a$variance,constant=quote((y-mu)^2),mu=quote(2*(y*log(y/mu)-(y-mu))),"mu(1-mu)"=quote(2*(y*log(y/mu)+(1-y)*log((1-y)/(1-mu)))),"mu^2"=quote(-2*(log(y/mu)-(y-mu)/mu)),"mu^3"=quote((y-mu)^2/(y*mu^2)))
mu <- a$mu; y <- a$y
Dmu <- lapply(1:4,function(k) { ex <- Dev; for(j in seq_len(k)) ex <- D(ex,"mu"); eval(ex)+mu*0 })
linkname <- f$link
inv <- switch(linkname,identity=quote(eta),log=quote(exp(eta)),inverse=quote(1/eta),sqrt=quote(eta^2),"1/mu^2"=quote(eta^(-.5)),logit=quote(1/(1+exp(-eta))),probit=quote(pnorm(eta)),cloglog=quote(1-exp(-exp(eta))),cauchit=quote(atan(eta)/pi+.5))
if(is.null(inv)) inv <- substitute(eta^p,list(p=1/a$power))
DevEta <- do.call(substitute,list(Dev,list(mu=inv)))
W <- substitute(A^2/V,list(A=D(inv,"eta"),V=do.call(substitute,list(V,list(mu=inv)))))
obs1 <- D(D(D(DevEta,"eta"),"eta"),"eta")
obs2 <- D(obs1,"eta")
fisher1 <- D(W,"eta"); fisher2 <- D(fisher1,"eta")
ans <- list(link=f$link,canonical=f$canonical,eta=eta,inverse=f$linkinv(eta),mu_eta=f$mu.eta(eta),d2=f$d2link(mu),d3=f$d3link(mu),d4=f$d4link(mu),v=f$variance(mu),v1=f$dvar(mu),v2=f$d2var(mu),v3=f$d3var(mu),dev=f$dev.resids(y,mu,a$weights),start=mustart,ls=f$ls(y,a$weights,n,a$scale),aic=f$aic(y,n,mu,a$weights,sum(f$dev.resids(y,mu,a$weights))),dev_mu=Dmu,obs1=eval(obs1)/2+mu*0,obs2=eval(obs2)/2+mu*0,fisher1=eval(fisher1)+mu*0,fisher2=eval(fisher2)+mu*0)
"""


def _reference(payload, code=CODE):
    if payload.get("power") is not None:
        code = code.replace("list(link=link", "list(link=quote(link)")
    key = reference_key("quasi_family", {"payload": payload, "code": code})
    if generation_enabled():
        with tempfile.TemporaryDirectory() as directory:
            src, inp, out = [
                Path(directory) / n for n in ("ref.R", "in.json", "out.json")
            ]
            src.write_text(
                "args <- commandArgs(TRUE)\nlibrary(mgcv)\nlibrary(jsonlite)\na <- fromJSON(args[1])\n"
                + code
                + '\nwrite_json(ans,args[2],digits=17,auto_unbox=TRUE,na="null")\n'
            )
            inp.write_text(json.dumps(payload))
            result = subprocess.run(
                ["Rscript", str(src), str(inp), str(out)],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, result.stderr
            save_reference("mgcv_quasi_family", key, json.loads(out.read_text()))
    return load_reference("mgcv_quasi_family", key)


def _payload(case):
    family, variance, link, power = case
    return {
        "family": family,
        "variance": variance,
        "link": link,
        "power": power,
        "y": [0.12, 0.65, 0.3, 0.9],
        "mu": [0.2, 0.4, 0.6, 0.8],
        "weights": [0.0, 0.5, 1.0, 3.0],
        "scale": 1.7,
    }


def _family(case):
    from nampy.gam.families.quasi import (
        QuasiBinomialFamily,
        QuasiFamily,
        QuasiPoissonFamily,
    )

    family, variance, link, power = case
    if family == "quasipoisson":
        return QuasiPoissonFamily(link=link)
    if family == "quasibinomial":
        return QuasiBinomialFamily(link=link)
    return QuasiFamily(link=link, variance=variance, power=power)


@pytest.mark.parametrize("case", CASES)
def test_quasi_reference_kernels(case):
    a = _payload(case)
    ref = _reference(a)
    f = _family(case)
    y, mu, w = [np.asarray(a[key]) for key in ("y", "mu", "weights")]
    eta = f.link(mu)
    actual = {
        "eta": eta,
        "inverse": f.inverse_link(eta),
        "mu_eta": f.mu_eta(eta),
        "d2": f.d2link(mu),
        "d3": f.d3link(mu),
        "d4": f.d4link(mu),
        "v": f.variance(mu),
        "v1": f.dvar(mu),
        "v2": f.d2var(mu),
        "v3": f.d3var(mu),
        "dev": f.deviance_obs(y, mu, w),
        "start": f.initialize_mu(y, weights=w),
        "ls": f.ls(y, w, scale=a["scale"]),
        "dev_mu": [f.deviance_derivatives_mu(y, mu, k) for k in range(1, 5)],
        "obs1": f.working_weight_derivative_eta(eta, y),
        "obs2": f.working_weight_second_derivative_eta(eta, y),
        "fisher1": f.working_weight_derivative_eta(eta),
        "fisher2": f.working_weight_second_derivative_eta(eta),
    }
    for name, value in actual.items():
        np.testing.assert_allclose(
            value, ref[name], atol=2e-10, rtol=2e-11, err_msg=name
        )
    assert f.canonical_link == (ref["canonical"] == ref["link"])
    assert np.isnan(f.aic(y, mu, weights=w))
    assert np.isnan(f.loglik_obs(y, mu)).all()
    assert f.known_scale is None
    assert f.saturated_loglik(y, w, scale=a["scale"]) == f.ls(y, w, scale=a["scale"])[0]


TAIL_CODE = r"""
link <- if(is.null(a$power)) a$link else power(a$power)
f <- if(a$family=="quasi") do.call(quasi,list(link=link,variance=a$variance)) else do.call(get(a$family),list(link=link))
pack <- function(v) list(value=ifelse(is.finite(v),v,0),nan=is.na(v),posinf=is.infinite(v)&v>0,neginf=is.infinite(v)&v<0)
ans <- list(inverse=pack(f$linkinv(a$eta)),mu_eta=pack(f$mu.eta(a$eta)),valid=vapply(a$eta,f$valideta,TRUE))
"""


def _assert_packed(value, ref):
    value = np.asarray(value)
    np.testing.assert_array_equal(np.isnan(value), ref["nan"])
    np.testing.assert_array_equal(np.isposinf(value), ref["posinf"])
    np.testing.assert_array_equal(np.isneginf(value), ref["neginf"])
    np.testing.assert_allclose(
        np.where(np.isfinite(value), value, 0), ref["value"], rtol=2e-14, atol=1e-30
    )


@pytest.mark.parametrize("link", LINKS)
def test_quasi_link_extreme_eta(link):
    a = _payload(("quasi", "constant", link, None))
    a["eta"] = [
        -1000.0,
        -40.0,
        -31.0,
        -30.0,
        -20.0,
        -2.0,
        0.0,
        2.0,
        20.0,
        30.0,
        31.0,
        40.0,
        1000.0,
    ]
    ref = _reference(a, TAIL_CODE)
    f = _family(("quasi", "constant", link, None))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        _assert_packed(f.inverse_link(a["eta"]), ref["inverse"])
        _assert_packed(f.mu_eta(a["eta"]), ref["mu_eta"])
    assert [f.valid_eta([v]) for v in a["eta"]] == ref["valid"]


@pytest.mark.parametrize("power", [0.5, 1.7, 3.0])
def test_quasi_power_extreme_eta(power):
    case = ("quasi", "constant", "power", power)
    a = _payload(case)
    a["eta"] = [-1.0, 0.0, 1e-300, 1e-40, 0.2, 1.0, 1e40, 1e300]
    ref = _reference(a, TAIL_CODE)
    f = _family(case)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        _assert_packed(f.inverse_link(a["eta"]), ref["inverse"])
        _assert_packed(f.mu_eta(a["eta"]), ref["mu_eta"])
    assert [f.valid_eta([v]) for v in a["eta"]] == ref["valid"]


BOUNDARY_CASES = [
    ("quasipoisson", "mu", "log", None),
    ("quasibinomial", "mu(1-mu)", "logit", None),
] + [("quasi", v, "identity", None) for v in VARIANCES]
BOUNDARY_CODE = r"""
f <- if(a$family=="quasi") do.call(quasi,list(link=a$link,variance=a$variance)) else do.call(get(a$family),list(link=a$link))
y <- a$y; weights <- a$weights; nobs <- length(y)
eval(f$initialize)
pack <- function(v) list(value=ifelse(is.finite(v),v,0),nan=is.na(v),posinf=is.infinite(v)&v>0,neginf=is.infinite(v)&v<0)
ans <- list(start=mustart,dev=pack(f$dev.resids(a$y,a$mu,a$weights)))
"""


@pytest.mark.parametrize("case", BOUNDARY_CASES)
def test_quasi_response_boundaries_and_zero_weights(case):
    a = _payload(case)
    a.update(
        y=[0.0, 1.0, 0.5, 0.0],
        mu=[1e-15, 1 - 1e-15, 0.4, 0.2],
        weights=[0.0, 1e-12, 0.5, 1e9],
    )
    ref = _reference(a, BOUNDARY_CODE)
    f = _family(case)
    np.testing.assert_allclose(
        f.initialize_mu(a["y"], weights=a["weights"]),
        ref["start"],
        atol=1e-30,
        rtol=2e-14,
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        _assert_packed(f.deviance_obs(a["y"], a["mu"], a["weights"]), ref["dev"])


def test_quasibinomial_grouped_start():
    from nampy.gam.families.quasi import QuasiBinomialFamily

    a = {
        "success": [0.0, 1.0, 4.0, 250000.0],
        "failure": [0.0, 0.0, 3.0, 750000.0],
        "weights": [2.0, 0.25, 3.0, 0.0],
    }
    ref = _reference(
        a,
        r"""
f <- quasibinomial(); y <- cbind(a$success,a$failure); weights <- a$weights; nobs <- nrow(y)
eval(f$initialize)
ans <- list(y=y,n=n,start=mustart,weights=weights)
""",
    )
    f = QuasiBinomialFamily()
    f._trial_counts = np.array(ref["n"])
    np.testing.assert_allclose(
        f.initialize_mu(ref["y"], weights=ref["weights"]),
        ref["start"],
        rtol=2e-14,
        atol=1e-30,
    )


def test_quasi_cauchit_ultra_tail():
    case = ("quasi", "constant", "cauchit", None)
    a = _payload(case)
    a["eta"] = [-1e300, -1e30, -1e15, -1e8, 1e8, 1e15, 1e30, 1e300]
    ref = _reference(a, TAIL_CODE)
    f = _family(case)
    _assert_packed(f.inverse_link(a["eta"]), ref["inverse"])
    _assert_packed(f.mu_eta(a["eta"]), ref["mu_eta"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"variance": "not_a_variance"},
        {"variance": {}},
        {"link": "not_a_link"},
        {"link": "log", "power": 0.5},
        {"power": float("nan")},
        {"power": True},
        {"power": ".5"},
    ],
)
def test_quasi_invalid_specifications(kwargs):
    from nampy.gam.families.quasi import QuasiFamily

    with pytest.raises(ValueError):
        QuasiFamily(**kwargs)


@pytest.mark.parametrize(
    "family,y",
    [
        ("quasipoisson", [-0.1, 1]),
        ("quasibinomial", [-0.1, 0.5]),
        ("quasibinomial", [0.2, 1.1]),
        ("quasi", [float("nan"), 1]),
    ],
)
def test_quasi_invalid_targets(family, y):
    case = next(case for case in CASES if case[0] == family)
    with pytest.raises(ValueError):
        _family(case).validate_y(y)


@pytest.mark.parametrize("case", BOUNDARY_CASES)
def test_quasi_selection_scale_contract_and_no_distribution(case):
    f = _family(case)
    y = np.array([0.2, 0.4, 0.8])
    weights = np.array([0.0, 2.0, 0.5])
    scale = 1.7
    np.testing.assert_array_equal(f.ls(y, np.zeros(3), scale=scale), np.zeros(3))
    np.testing.assert_allclose(
        f.ls(y, weights, scale=scale), [-np.log(scale), -1 / scale, 1 / scale**2]
    )
    np.testing.assert_array_equal(
        f.ls(y, weights, n=np.array([5, 3, 8]), scale=scale),
        f.ls(y[::-1], weights, scale=scale),
    )
    assert f.supports_method("gcv") and f.supports_method("reml")
    with pytest.raises(NotImplementedError, match="distribution"):
        f.quantile_residual_bounds(y, y)


def test_quasi_general_initialization_and_variance_spelling():
    from nampy.gam.families.quasi import QuasiFamily

    f = QuasiFamily(variance="mu (1 - mu)")
    assert f.varfun == "mu(1-mu)"
    np.testing.assert_array_equal(f.initialize_mu([-3.0, 2.0]), [0.001, 0.999])
    np.testing.assert_array_equal(QuasiFamily().validate_y([-3.0, 2.0]), [-3.0, 2.0])
