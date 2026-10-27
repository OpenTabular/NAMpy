"""NCV selection controls and fixed-penalty criteria against actual mgcv."""

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from nampy.gam.fit.selection.criteria import criterion_gradient, criterion_value
from nampy.gam.fit.selection.criteria.ncv import _ncv_state, normalize_nei
from tests.reference_fixtures import load_reference, reference_key, save_reference

FAMILIES = {
    "gaussian": ({"name": "gaussian"}, "gaussian()", "gaussian"),
    "poisson": ({"name": "poisson"}, "poisson()", "count"),
    "gamma": ({"name": "gamma", "link": "log"}, "Gamma(link='log')", "positive"),
    "binomial": (
        {"name": "binomial", "link": "probit"},
        "binomial(link='probit')",
        "binary",
    ),
    "inverse_gaussian": (
        {"name": "inverse_gaussian", "link": "log"},
        "inverse.gaussian(link='log')",
        "positive",
    ),
}
SCRIPT = r"""
a <- commandArgs(TRUE)
lib <- Sys.getenv('MGCV_LIB_PATH','')
if(nzchar(lib)) .libPaths(c(lib,.libPaths()))
suppressPackageStartupMessages(library(mgcv)); library(jsonlite)
p <- fromJSON(a[1]); d <- as.data.frame(p$data); n <- nrow(d)
nei <- p$nei
normnei <- nei
if(is.null(normnei)) normnei <- list()
if(is.null(normnei$a) || is.null(normnei$ma)) {
    normnei$a <- normnei$ma <- normnei$d <- normnei$md <- seq_len(n)
}
if(is.null(normnei$d)) normnei$d <- normnei$md <- seq_len(n)
formula <- switch(p$geometry,
    linear=y~x,
    smooth=y~s(x,bs='cr',k=6),
    mixed=y~s(x,bs='cr',k=6)+s(z,bs='ps',k=5))
if(p$offset) formula <- update(formula,.~.+offset(o))

evaluate <- function(logsp,method) {
    sp <- if(length(logsp)) exp(logsp) else NULL
    m <- gam(formula,data=d,weights=w,family=eval(parse(text=p$family)),
             sp=sp,method=method,nei=nei,gamma=p$gamma,
             control=gam.control(epsilon=1e-12))
    fam <- mgcv:::fix.family.var(mgcv:::fix.family.link(m$family))
    X <- predict(m,d,type='lpmatrix'); eta <- as.numeric(m$linear.predictors)
    mu <- as.numeric(m$fitted.values); ig <- fam$mu.eta(eta); V <- fam$variance(mu)
    alpha <- if(fam$link==fam$canonical) rep(1,n) else
        1+(d$y-mu)*(fam$dvar(mu)/V+fam$d2link(mu)*ig)
    alpha[alpha==0] <- .Machine$double.eps
    W <- d$w*ig^2/V*alpha; ww <- d$w*(d$y-mu)*ig/V
    penalty <- matrix(0,ncol(X),ncol(X)); k <- 0L
    for(sm in m$smooth) for(S in sm$S) {
        k <- k+1L; ii <- sm$first.para:sm$last.para
        penalty[ii,ii] <- penalty[ii,ii]+sp[k]*S
    }
    A <- crossprod(X,W*X)+penalty
    cv <- numeric(length(normnei$d)); a0 <- d0 <- 0L
    for(j in seq_along(normnei$ma)) {
        ai <- normnei$a[(a0+1):normnei$ma[j]]
        di <- normnei$d[(d0+1):normnei$md[j]]
        Xa <- X[ai,,drop=FALSE]
        delta <- -solve(A-crossprod(Xa,W[ai]*Xa),crossprod(Xa,ww[ai]))
        cv[(d0+1):normnei$md[j]] <- eta[di]+X[di,,drop=FALSE]%*%delta
        a0 <- normnei$ma[j]; d0 <- normnei$md[j]
    }
    di <- normnei$d
    if(method=='QNCV') {
        diff <- cv-eta[di]
        value <- sum(fam$dev.resids(d$y[di],mu[di],d$w[di])) +
            p$gamma*sum(-2*ww[di]*diff+W[di]*diff^2)
    } else {
        cv <- p$gamma*cv-(p$gamma-1)*eta[di]
        value <- sum(fam$dev.resids(d$y[di],fam$linkinv(cv),d$w[di]))
    }
    # The direct RNcv path omits offsets and uses absolute weights for
    # negative-weight downdates. The independent signed equations above use
    # actual mgcv designs, penalties, fits and family derivatives. Check both
    # references agree whenever the direct upstream path has neither issue.
    raw <- as.numeric(m$gcv.ubre)
    if(!p$offset && all(W>=0)) {
        stopifnot(abs(value-raw) < 2e-7*(1+abs(raw)))
    }
    list(score=value,direct=raw,negative=sum(W<0),eta=cv,
         fit=unname(eta),version=as.character(packageVersion('mgcv')))
}
ans <- list()
for(method in c('NCV','QNCV')) {
    point <- evaluate(unlist(p$logsp),method)
    grad <- numeric(length(p$logsp)); h <- 1e-4
    for(j in seq_along(grad)) {
        plus <- minus <- unlist(p$logsp); plus[j] <- plus[j]+h; minus[j] <- minus[j]-h
        grad[j] <- (evaluate(plus,method)$score-evaluate(minus,method)$score)/(2*h)
    }
    point$gradient <- grad; ans[[method]] <- point
}
write_json(ans,a[2],digits=17,auto_unbox=TRUE,na='null')
"""

SELECT_SCRIPT = r"""
a <- commandArgs(TRUE)
lib <- Sys.getenv('MGCV_LIB_PATH','')
if(nzchar(lib)) .libPaths(c(lib,.libPaths()))
suppressPackageStartupMessages(library(mgcv)); library(jsonlite)
p <- fromJSON(a[1]); d <- as.data.frame(p$data)
m <- gam(y~s(x,bs='cr',k=6),data=d,weights=w,
         family=eval(parse(text=p$family)),method=p$method,gamma=p$gamma,
         nei=p$nei,optimizer=c('outer','bfgs'),control=gam.control(epsilon=1e-10))
pr <- predict(m,type='link',se.fit=TRUE)
write_json(list(version=as.character(packageVersion('mgcv')),sp=unname(m$sp),
                score=unname(m$gcv.ubre),fit=unname(pr$fit),se=unname(pr$se.fit),
                edf=sum(m$edf)),a[2],digits=17,auto_unbox=TRUE)
"""


def data_frame(family, zero_weights, offset):
    rng = np.random.default_rng(16893)
    x = rng.uniform(0.02, 0.98, 48)
    z = rng.uniform(0.02, 0.98, len(x))
    o = 0.2 * np.cos(3 * x) if offset else np.zeros(len(x))
    base = 0.4 + 0.6 * np.sin(4 * x) + o
    targets = {
        "gaussian": base + rng.normal(0, 0.3, len(x)),
        "positive": np.exp(base + rng.normal(0, 0.3, len(x))),
        "count": rng.poisson(np.exp(base)),
        "binary": rng.binomial(1, 1 / (1 + np.exp(-base))),
    }
    y = targets[FAMILIES[family][2]].copy()
    if family == "inverse_gaussian":
        y[5] *= 0.15  # A genuine negative Newton weight in an otherwise stable fit.
    w = rng.uniform(0.3, 2, len(x))
    if zero_weights:
        w[::11] = 0
    return pd.DataFrame({"x": x, "z": z, "o": o, "y": y, "w": w}).round(12)


def neighbourhood(name, n):
    if name == "default":
        return None
    if name == "implicit":
        return {"gamma": 2.3, "jackknife": False}
    out = {
        "a": np.arange(1, n + 1).tolist(),
        "ma": np.arange(3, n + 1, 3).tolist(),
        "gamma": 2.3,
        "jackknife": False,
    }
    if name == "block":
        out.update(d=out["a"].copy(), md=out["ma"].copy())
    else:
        out.update(
            d=np.arange(2, n + 1, 3).tolist(), md=np.arange(1, n // 3 + 1).tolist()
        )
        if name == "trailing":
            out["d"].append(n)
    return out


def reference(data, family, nei, gamma, geometry, offset):
    logsp = (
        []
        if geometry == "linear"
        else np.log([0.7, 1.1] if geometry == "mixed" else [0.7]).tolist()
    )
    payload = {
        "data": data.to_dict("list"),
        "family": FAMILIES[family][1],
        "nei": nei,
        "gamma": gamma,
        "geometry": geometry,
        "offset": offset,
        "logsp": logsp,
    }
    key = reference_key(
        "ncv-edge-selection-v1",
        {
            "payload": payload,
            "source_sha256": hashlib.sha256(SCRIPT.encode()).hexdigest(),
        },
        normalize_floats=True,
    )
    result = load_reference("mgcv_ncv_edges", key)
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
        save_reference("mgcv_ncv_edges", key, result)
    return result


def compare(
    family, nei_name, gamma, zero_weights, method, *, offset=False, layout="free"
):
    data = data_frame(family, zero_weights, offset)
    nei = neighbourhood(nei_name, len(data))
    geometry = (
        "mixed"
        if layout.startswith("mixed")
        else "linear"
        if layout == "linear"
        else "smooth"
    )
    formula = {
        "free": "y~s(x,bs='cr',k=6)",
        "fixed": "y~s(x,bs='cr',k=6,sp=.7)",
        "linear": "y~x",
        "mixed": "y~s(x,bs='cr',k=6,sp=.7)+s(z,bs='ps',k=5)",
        "mixed_free": "y~s(x,bs='cr',k=6)+s(z,bs='ps',k=5)",
        "mixed_fixed": "y~s(x,bs='cr',k=6,sp=.7)+s(z,bs='ps',k=5,sp=1.1)",
    }[layout]
    if offset:
        formula += "+offset(o)"
    initial = [0.7, 1.1] if geometry == "mixed" else [0.7]
    model = GAM(
        formula=formula,
        family=FAMILIES[family][0],
        smoothing_params=initial if layout in {"free", "mixed", "mixed_free"} else None,
        optimize_smoothing=False,
        nei=nei,
        score_gamma=gamma,
        control={"epsilon": 1e-12},
    ).fit(data=data, sample_weight="w")
    expected = reference(data, family, nei, gamma, geometry, offset)[method.upper()]
    assert expected["version"] == "1.9.4"
    full_logsp = np.log(
        [0.7, 1.1] if geometry == "mixed" else [] if geometry == "linear" else [0.7]
    )
    free = ~np.asarray(model.smoothing_fixed_mask_, dtype=bool)
    logsp = full_logsp[free]
    actual = criterion_value(model, model.y_, logsp, method)
    gradient = criterion_gradient(model, model.y_, logsp, method)
    np.testing.assert_allclose(
        model.predict(type="link"), expected["fit"], rtol=2e-9, atol=2e-9
    )
    np.testing.assert_allclose(actual, expected["score"], rtol=2e-9, atol=2e-9)
    ref_grad = np.asarray(expected["gradient"], dtype=float).reshape(-1)[free]
    np.testing.assert_allclose(gradient, ref_grad, rtol=2e-5, atol=3e-7)
    if not np.any(free):
        assert gradient.shape == (0,)
        assert actual > 0
    _, _, _, cv, _ = _ncv_state(model, model.y_, logsp, qapprox=method == "qncv")
    np.testing.assert_allclose(cv, expected["eta"], rtol=2e-9, atol=2e-9)
    if nei_name == "trailing":
        # Untouched rows begin at zero before the optional NCV inflation.
        expected_last = (
            0 if method == "qncv" else -(gamma - 1) * model.predict(type="link")[-1]
        )
        np.testing.assert_allclose(cv[-1], expected_last, atol=1e-12)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("nei_name", ["default", "block"])
@pytest.mark.parametrize("gamma", [1.0, 1.4])
@pytest.mark.parametrize("zero_weights", [False, True])
@pytest.mark.parametrize("method", ["ncv", "qncv"])
def test_ncv_criterion_and_gradient_match_mgcv(
    family, nei_name, gamma, zero_weights, method
):
    compare(family, nei_name, gamma, zero_weights, method)


@pytest.mark.parametrize("family", ["gaussian", "poisson", "inverse_gaussian"])
@pytest.mark.parametrize("nei_name", ["implicit", "subset", "trailing"])
@pytest.mark.parametrize("gamma", [1.0, 1.4])
@pytest.mark.parametrize("method", ["ncv", "qncv"])
def test_ncv_offsets_and_neighbourhood_edges(family, nei_name, gamma, method):
    compare(family, nei_name, gamma, True, method, offset=True)


@pytest.mark.parametrize("family", ["gaussian", "poisson", "binomial"])
@pytest.mark.parametrize(
    "layout", ["fixed", "linear", "mixed", "mixed_free", "mixed_fixed"]
)
@pytest.mark.parametrize("gamma", [1.0, 1.4])
@pytest.mark.parametrize("method", ["ncv", "qncv"])
def test_ncv_fixed_and_absent_smoothing_parameters(family, layout, gamma, method):
    compare(family, "default", gamma, False, method, layout=layout)


@pytest.mark.parametrize("index_base", [0, 1])
@pytest.mark.parametrize("partial", [{}, {"a": [1, 2]}, {"ma": [1, 2]}])
def test_default_neighbourhood_indices_preserve_other_options(index_base, partial):
    normal = normalize_nei(
        {**partial, "index_base": index_base, "gamma": 2.3, "jackknife": False}, 5
    )
    assert normal["gamma"] == 2.3
    assert normal["jackknife"] is False
    for key in ("a", "d"):
        np.testing.assert_array_equal(normal[key], [0, 1, 2, 3, 4])
    for key in ("ma", "md"):
        np.testing.assert_array_equal(normal[key], [1, 2, 3, 4, 5])


def test_null_neighbourhood_options_take_upstream_defaults():
    out = normalize_nei(
        {"a": [1, 2], "ma": [1, 2], "gamma": None, "jackknife": None}, 2
    )
    assert out["gamma"] == 1
    assert out["jackknife"] == -1


@pytest.mark.parametrize("family", ["gaussian", "poisson"])
@pytest.mark.parametrize("method,gamma", [("NCV", 1.0), ("QNCV", 1.4)])
def test_ncv_selection_uses_model_gamma_not_jackknife_gamma(family, method, gamma):
    data = data_frame(family, False, False)
    nei = neighbourhood("block", len(data))
    payload = {
        "data": data.to_dict("list"),
        "family": FAMILIES[family][1],
        "nei": nei,
        "gamma": gamma,
        "method": method,
    }
    key = reference_key(
        "ncv-selected-edge-v1",
        {
            "payload": payload,
            "source_sha256": hashlib.sha256(SELECT_SCRIPT.encode()).hexdigest(),
        },
        normalize_floats=True,
    )
    expected = load_reference("mgcv_ncv_edges", key)
    if expected is None:
        with tempfile.TemporaryDirectory() as directory:
            source, output = [
                Path(directory) / name for name in ("source.R", "out.json")
            ]
            source.write_text(SELECT_SCRIPT)
            run = subprocess.run(
                ["Rscript", str(source), json.dumps(payload), str(output)],
                capture_output=True,
                text=True,
            )
            assert run.returncode == 0, run.stderr
            expected = json.loads(output.read_text())
        save_reference("mgcv_ncv_edges", key, expected)
    assert expected["version"] == "1.9.4"
    model = GAM(
        formula="y~s(x,bs='cr',k=6)",
        family=FAMILIES[family][0],
        smoothing_method=method,
        smoothing_optimizer="bfgs",
        score_gamma=gamma,
        nei=nei,
        optimize_smoothing=True,
        control={"epsilon": 1e-10},
    ).fit(data=data, sample_weight="w")
    fit, se = model.predict(type="link", return_se=True)
    np.testing.assert_allclose(
        model.smoothing_params, np.atleast_1d(expected["sp"]), rtol=2e-5
    )
    np.testing.assert_allclose(model.smoothing_score_, expected["score"], rtol=2e-8)
    np.testing.assert_allclose(model.fit_result().edf_total, expected["edf"], rtol=2e-7)
    np.testing.assert_allclose(fit, expected["fit"], rtol=2e-7, atol=2e-7)
    np.testing.assert_allclose(se, expected["se"], rtol=2e-7, atol=2e-7)
