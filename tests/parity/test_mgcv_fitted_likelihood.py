"""Fitted likelihoods retain offsets, family parameter scales and weights."""

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
from tests.reference_fixtures import load_reference, reference_key, save_reference

CASES = [
    ("gaussian", {}, "gaussian()", "gaussian"),
    ("gaussian", {"link": "log"}, "gaussian(link='log')", "positive"),
    ("poisson", {}, "poisson()", "poisson"),
    ("poisson", {"link": "sqrt"}, "poisson(link='sqrt')", "poisson"),
    ("binomial", {}, "binomial()", "binomial"),
    ("binomial", {"link": "probit"}, "binomial(link='probit')", "binomial"),
    ("gamma", {"link": "log"}, "Gamma(link='log')", "positive"),
    ("gamma", {"link": "inverse"}, "Gamma(link='inverse')", "positive"),
    (
        "inverse_gaussian",
        {"link": "log"},
        "inverse.gaussian(link='log')",
        "positive",
    ),
    ("nb", {"theta": 2}, "nb(theta=2)", "poisson"),
    ("betar", {"theta": 12}, "betar(theta=12)", "beta"),
    ("betar", {"theta": 12}, "betar(theta=12)", "beta_boundary"),
    ("ocat", {"theta": [0.8, 1.3]}, "ocat(theta=c(.8,1.3))", "ocat"),
    ("scat", {"theta": [6, 1]}, "scat(theta=c(6,1))", "scat"),
    ("zip", {"theta": [-0.4, 0.2]}, "ziP(theta=c(-.4,.2))", "poisson"),
]
WEIGHTED_CASES = [
    (*case, zero_weights)
    for case in CASES
    for zero_weights in ([False] if case[0] in {"gaussian", "zip"} else [False, True])
]
SCRIPT = r"""
a <- commandArgs(TRUE)
lib <- Sys.getenv('MGCV_LIB_PATH', '')
if (nzchar(lib)) .libPaths(c(lib, .libPaths()))
suppressPackageStartupMessages(library(mgcv)); library(jsonlite)
# Restore the recorded library's local Newton-sign mutation inside this process.
newton_ref <- get('newton', asNamespace('mgcv'))
body(newton_ref) <- parse(text=sub('Sstep <- grad/max', 'Sstep <- -grad/max',
    paste(deparse(body(newton_ref)), collapse='\n'), fixed=TRUE))[[1]]
assignInNamespace('newton', newton_ref, ns='mgcv')
p <- fromJSON(a[1]); d <- as.data.frame(p$data)
m <- gam(y~s(x,bs='cr',k=6)+offset(o), data=d, weights=w,
         family=eval(parse(text=p$family)), sp=.7, method='REML',
         control=gam.control(epsilon=1e-12, newton=list(conv.tol=1e-12)))
ll <- logLik(m)
write_json(list(version=as.character(packageVersion('mgcv')),
                fit=unname(m$linear.predictors), ll=as.numeric(ll),
                df=attr(ll,'df'), aic=AIC(m), bic=BIC(m),
                raw_family_aic=if (inherits(m$family,'extended.family'))
                    m$family$aic(m$y,m$fitted.values,wt=m$prior.weights) else NULL),
           a[2], digits=17, auto_unbox=TRUE, na='null')
"""


def data_frame(target, name, zero_weights):
    rng = np.random.default_rng(28915)
    n = 84
    x = rng.uniform(0.02, 0.98, n)
    offset = 0.25 * np.cos(3 * x)
    eta = 0.3 + 0.6 * np.sin(4 * x) + offset
    mu = 1 / (1 + np.exp(-eta))
    targets = {
        "gaussian": eta + rng.normal(0, 0.3, n),
        "positive": np.exp(eta + rng.normal(0, 0.25, n)),
        "poisson": rng.poisson(np.exp(eta)),
        "binomial": rng.binomial(1, mu),
        "beta": rng.beta(10 * mu, 10 * (1 - mu)),
        "ocat": np.searchsorted([-1.0, -0.2, 1.1], eta + rng.logistic(size=n)) + 1,
        "scat": eta + rng.standard_t(6, n) * 0.4,
    }
    targets["beta_boundary"] = targets["beta"].copy()
    targets["beta_boundary"][[1, 7, 13, 20]] = [0, 1, 1e-15, 1 - 1e-15]
    weights = rng.uniform(0.4, 2, n)
    if target == "binomial":
        weights = np.resize([1.0, 2.0, 3.0], n)
    if name == "zip":
        weights = np.ones(n)  # Weighted ziP fitting is explicitly unsupported.
    if zero_weights:
        weights[::11] = 0
    return pd.DataFrame(
        {"x": x, "o": offset, "y": targets[target], "w": weights}
    ).round(12)


def reference(data, r_family):
    payload = {"data": data.to_dict("list"), "family": r_family}
    key = reference_key(
        "fitted-likelihood-v1",
        {
            "payload": payload,
            "source_sha256": hashlib.sha256(SCRIPT.encode()).hexdigest(),
        },
    )
    result = load_reference("mgcv_fitted_likelihood", key)
    if result is None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.R"
            output = Path(directory) / "output.json"
            source.write_text(SCRIPT)
            run = subprocess.run(
                ["Rscript", str(source), json.dumps(payload), str(output)],
                capture_output=True,
                text=True,
            )
            assert run.returncode == 0, run.stderr
            result = json.loads(output.read_text())
        save_reference("mgcv_fitted_likelihood", key, result)
    return result


@pytest.mark.parametrize("name,options,r_family,target,zero_weights", WEIGHTED_CASES)
@pytest.mark.parametrize("offset_mode", ["formula", "api", "mixed"])
def test_fitted_likelihood_matches_mgcv(
    name, options, r_family, target, zero_weights, offset_mode, monkeypatch
):
    data = data_frame(target, name, zero_weights)
    expected = reference(data, r_family)
    assert expected["version"] == "1.9.4"
    fit_data = data.copy()
    formula = "y~s(x,bs='cr',k=6)"
    offset = None
    if offset_mode == "formula":
        formula += "+offset(o)"
    elif offset_mode == "api":
        offset = data.o.to_numpy()
    else:
        formula += "+offset(o)"
        fit_data["o"] *= 0.5
        offset = data.o.to_numpy() * 0.5
    model = GAM(
        formula=formula,
        family={"name": name, **options},
        smoothing_params=[0.7],
        optimize_smoothing=False,
        smoothing_method="fixed",
        control={"epsilon": 1e-12},
    ).fit(data=fit_data, sample_weight="w", offset=offset)
    np.testing.assert_allclose(
        model.predict(type="link", offset=data.o.to_numpy()),
        expected["fit"],
        atol=2e-8,
        rtol=2e-8,
    )
    restored = pickle.loads(pickle.dumps(model))

    def no_fresh_predictions(*args, **kwargs):
        raise AssertionError("Likelihood must use the fitted state")

    monkeypatch.setattr(model, "predict", no_fresh_predictions)
    for fitted in (model, restored):
        np.testing.assert_allclose(
            fitted._loglik_effective_df(), expected["df"], atol=2e-8, rtol=2e-9
        )
        for method, key in [("loglik", "ll"), ("aic", "aic"), ("bic", "bic")]:
            np.testing.assert_allclose(
                getattr(fitted, method)(),
                expected[key],
                atol=2e-8,
                rtol=2e-9,
                err_msg=method,
            )
    if name == "ocat":
        latent = np.asarray(expected["fit"])
        # The common likelihood dispatcher passes the Python weight spelling.
        for keyword in ("wt", "weights"):
            np.testing.assert_allclose(
                model.family.aic(data.y, latent, **{keyword: data.w}),
                expected["raw_family_aic"],
                atol=2e-8,
                rtol=2e-9,
            )
        with pytest.raises(TypeError, match="only one"):
            model.family.aic(data.y, latent, wt=data.w, weights=data.w)
