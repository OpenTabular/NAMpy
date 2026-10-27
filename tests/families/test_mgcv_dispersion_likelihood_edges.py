"""Known-dispersion Gaussian AIC and stable Gamma densities against mgcv."""

import hashlib
import json
import pickle
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from nampy.gam.families.registry import make_gam_family
from tests.reference_fixtures import load_reference, reference_key, save_reference

SCRIPT = r"""
a <- commandArgs(TRUE)
lib <- Sys.getenv('MGCV_LIB_PATH','')
if (nzchar(lib)) .libPaths(c(lib,.libPaths()))
suppressPackageStartupMessages(library(mgcv)); library(jsonlite)
p <- fromJSON(a[1])
if (p$operation == 'kernel') {
    out <- list(density=dgamma(p$y,shape=1/p$phi,scale=p$mu*p$phi,log=TRUE))
} else {
    d <- as.data.frame(p$data)
    f <- if(p$smooth) y~s(x,bs='cr',k=6)+offset(o) else y~x+offset(o)
    args <- list(formula=f,data=d,weights=d$w,scale=p$phi,
                 family=eval(parse(text=p$family)),method=p$method,
                 control=gam.control(epsilon=1e-12))
    if(p$smooth) args$sp <- .7
    m <- do.call(gam,args); ll <- logLik(m)
    out <- list(mu=unname(m$fitted.values),ll=as.numeric(ll),aic=AIC(m),bic=BIC(m),
                df=attr(ll,'df'),scale=m$sig2,edf=sum(m$edf))
    if(p$gamma) out$density <- dgamma(d$y,shape=1/p$phi,
                                      scale=m$fitted.values*p$phi,log=TRUE)
}
out$version <- as.character(packageVersion('mgcv'));out$r_version <- R.version.string
write_json(out,a[2],digits=17,auto_unbox=TRUE,na='string')
"""


def reference(payload):
    key = reference_key(
        "dispersion-likelihood-v1",
        {
            "payload": payload,
            "source_sha256": hashlib.sha256(SCRIPT.encode()).hexdigest(),
        },
    )
    result = load_reference("mgcv_dispersion_likelihood_edges", key)
    if result is None:
        with tempfile.TemporaryDirectory() as directory:
            source, output = (
                Path(directory) / name for name in ("source.R", "output.json")
            )
            source.write_text(SCRIPT)
            run = subprocess.run(
                ["Rscript", str(source), json.dumps(payload), str(output)],
                text=True,
                capture_output=True,
            )
            assert run.returncode == 0, run.stderr
            result = json.loads(output.read_text())
        save_reference("mgcv_dispersion_likelihood_edges", key, result)
    assert result["version"] == "1.9.4"
    return result


@pytest.mark.parametrize(
    "phi", [0.01, 0.1, 0.4, 1, 2, 10, 1e-8, 1e-12, 1e-15, 1e-300, 1e300]
)
def test_gamma_density_across_shapes_scales_and_boundaries(phi):
    mu = np.repeat([1e-30, 0.2, 1, 10, 1e30], 9)
    ratios = np.tile([0, 0.01, 0.1, 0.5, 1, 1.5, 3, 10, 100], 5)
    y = mu * ratios
    expected = reference(
        {
            "operation": "kernel",
            "phi": phi,
            "mu": mu.tolist(),
            "y": y.tolist(),
        }
    )
    family = make_gam_family("gamma")
    with np.errstate(over="ignore", under="ignore"):
        actual = family.loglik_obs(y, mu, scale=phi)
    np.testing.assert_allclose(
        actual,
        np.asarray(expected["density"], float),
        atol=2e-11,
        rtol=2e-12,
        equal_nan=True,
    )


FAMILIES = [("gaussian", "identity"), ("gamma", "log"), ("gamma", "inverse")]
FIT_CASES = [
    (name, link, phi, smooth, weights)
    for name, link in FAMILIES
    for phi in [1e-4, 1e-8, 1e-12, 1e-15]
    for smooth in [False, True]
    for weights in (
        ["unit", "varying"] if name == "gaussian" else ["unit", "varying", "zero"]
    )
]


def fit_data(name, link, phi, weights):
    rng = np.random.default_rng(98319)
    x = np.linspace(-1, 1, 64)
    o = 0.07 * np.cos(2 * x)
    eta = 0.8 + 0.2 * x + o
    mu = eta if link == "identity" else (np.exp(eta) if link == "log" else 1 / eta)
    y = mu + np.sqrt(phi) * rng.normal(size=len(x)) * (1 if name == "gaussian" else mu)
    w = np.ones(len(x)) if weights == "unit" else np.resize([0.5, 1, 2], len(x))
    if weights == "zero":
        w[::11] = 0
    return pd.DataFrame({"x": x, "y": y, "w": w, "o": o}).round(12)


@pytest.mark.parametrize("name,link,phi,smooth,weights", FIT_CASES)
@pytest.mark.parametrize("offset_mode", ["formula", "api", "mixed"])
def test_fitted_known_dispersion_likelihood_matches_mgcv(
    name, link, phi, smooth, weights, offset_mode
):
    data = fit_data(name, link, phi, weights)
    r_family = ("Gamma" if name == "gamma" else name) + f"(link='{link}')"
    expected = reference(
        {
            "operation": "fit",
            "family": r_family,
            "phi": phi,
            "gamma": name == "gamma",
            "smooth": smooth,
            "method": "REML",
            "data": data.to_dict("list"),
        }
    )
    formula = "y~s(x,bs='cr',k=6)" if smooth else "y~x"
    fit_frame = data.copy()
    if offset_mode == "formula":
        formula += "+offset(o)"
        offset = None
    elif offset_mode == "api":
        offset = data.o.to_numpy()
    else:
        formula += "+offset(o)"
        fit_frame["o"] *= 0.5
        offset = data.o.to_numpy() * 0.5
    model = GAM(
        formula=formula,
        family={"name": name, "link": link},
        scale=phi,
        optimize_smoothing=False,
        smoothing_method="fixed",
        smoothing_params=[0.7] if smooth else [],
        control={"epsilon": 1e-12},
    ).fit(data=fit_frame, sample_weight="w", offset=offset)
    np.testing.assert_allclose(
        model.predict(offset=data.o.to_numpy()),
        expected["mu"],
        atol=2e-12,
        rtol=2e-12,
    )
    for fitted in (model, pickle.loads(pickle.dumps(model))):
        for method, key in [("loglik", "ll"), ("aic", "aic"), ("bic", "bic")]:
            # At phi=1e-15, independent fitted means differ by a few ulps;
            # density sensitivity contributes at most a few micro-units.
            np.testing.assert_allclose(
                getattr(fitted, method)(),
                expected[key],
                atol=3e-6,
                rtol=2e-11,
                err_msg=method,
            )
        np.testing.assert_allclose(
            fitted._loglik_effective_df(), expected["df"], atol=2e-9
        )
    if name == "gamma":
        np.testing.assert_allclose(
            model.family.loglik_obs(data.y, expected["mu"], scale=phi),
            expected["density"],
            atol=2e-10,
            rtol=2e-12,
        )


@pytest.mark.parametrize("method", ["GCV.Cp", "REML", "ML"])
@pytest.mark.parametrize("optimize", [False, True])
def test_gaussian_fixed_dispersion_selection_routes_match_mgcv(method, optimize):
    # Pin SP in the formula so both routes retain the same conditional model.
    data = fit_data("gaussian", "identity", 1e-4, "varying")
    expected = reference(
        {
            "operation": "fit",
            "family": "gaussian()",
            "phi": 1e-4,
            "gamma": False,
            "smooth": True,
            "method": method,
            "data": data.to_dict("list"),
        }
    )
    model = GAM(
        formula="y~s(x,bs='cr',k=6,sp=.7)+offset(o)",
        family="gaussian",
        scale=1e-4,
        optimize_smoothing=optimize,
        smoothing_method=method,
    ).fit(data=data, sample_weight="w")
    for fun, key in [("loglik", "ll"), ("aic", "aic"), ("bic", "bic")]:
        np.testing.assert_allclose(
            getattr(model, fun)(), expected[key], atol=2e-8, rtol=2e-10
        )
