"""Saved R references for large binomial denominators and boundary deviance."""

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from nampy.gam.families.registry import make_gam_family
from tests.reference_fixtures import load_reference, reference_key, save_reference

NAMESPACE = "mgcv_binomial_initialization_edges"
SCRIPT = r"""
a <- commandArgs(TRUE)
lib <- Sys.getenv('MGCV_LIB_PATH','')
if(nzchar(lib)) .libPaths(c(lib,.libPaths()))
suppressPackageStartupMessages(library(mgcv)); library(jsonlite)
p <- fromJSON(a[1])
f <- get(p$family)(link=p$link)
if(p$operation=='initialize') {
    y <- if(p$grouped) cbind(p$success,p$trials-p$success) else p$response
    weights <- p$weights
    nobs <- length(p$response)
    mustart <- etastart <- start <- NULL
    eval(f$initialize)
    out <- list(mu=mustart,response=y,weights=weights,trials=n)
} else {
    dev <- f$dev.resids(p$response,p$mu,p$weights)
    out <- list(dev=dev,total=sum(dev))
}
out$version <- as.character(packageVersion('mgcv'))
out$r_version <- R.version.string
write_json(out,a[2],digits=17,auto_unbox=TRUE,na='string')
"""


def reference(payload):
    key = reference_key(
        "binomial-initialization-edges-v1",
        {
            "payload": payload,
            "source_sha256": hashlib.sha256(SCRIPT.encode()).hexdigest(),
        },
    )
    result = load_reference(NAMESPACE, key)
    if result is None:
        with tempfile.TemporaryDirectory() as directory:
            source, output = (
                Path(directory) / name for name in ("source.R", "out.json")
            )
            source.write_text(SCRIPT)
            run = subprocess.run(
                ["Rscript", str(source), json.dumps(payload), str(output)],
                capture_output=True,
                text=True,
            )
            assert run.returncode == 0, run.stderr
            result = json.loads(output.read_text())
        save_reference(NAMESPACE, key, result)
    assert result["version"] == "1.9.4"
    return result


LINKS = ["logit", "probit", "cloglog", "cauchit", "log"]


@pytest.mark.parametrize("name", ["binomial", "quasibinomial"])
@pytest.mark.parametrize("link", LINKS)
@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("magnitude", [10, 10**9, 10**12, 10**15])
def test_large_binomial_denominators_initialize_like_r(name, link, grouped, magnitude):
    trials = np.full(6, magnitude, dtype=float)
    successes = np.array([0, magnitude, 1, magnitude - 1, 0, magnitude], float)
    y = successes / trials
    weights = np.array([1, 1, 0.5, 2, 0, 0.5])
    if not grouped:
        weights *= magnitude
    payload = {
        "operation": "initialize",
        "family": name,
        "link": link,
        "grouped": grouped,
        "trials": trials.tolist(),
        "success": successes.tolist(),
        "response": y.tolist(),
        "weights": weights.tolist(),
    }
    expected = reference(payload)
    family = make_gam_family({"name": name, "link": link})
    if grouped:
        family._trial_counts = trials
    actual = family.initialize_mu(y, weights=weights)
    np.testing.assert_allclose(actual, expected["mu"], rtol=2e-15, atol=0)
    # Relative comparisons of the failure probability also expose upper-tail clipping.
    np.testing.assert_allclose(
        1 - actual, 1 - np.array(expected["mu"]), rtol=2e-15, atol=0
    )


@pytest.mark.parametrize("name", ["binomial", "quasibinomial"])
@pytest.mark.parametrize("link", LINKS)
@pytest.mark.parametrize("weighted", [False, True])
def test_binomial_boundary_deviance_matches_r(name, link, weighted):
    pairs = [
        (y, mu) for y in [0, 0.25, 1] for mu in [0, 1e-30, 1e-15, 0.4, 1 - 1e-15, 1]
    ]
    y, mu = np.array(pairs).T
    weights = np.resize([0, 0.5, 1e15, 2], len(y)) if weighted else np.ones(len(y))
    expected = reference(
        {
            "operation": "deviance",
            "family": name,
            "link": link,
            "response": y.tolist(),
            "mu": mu.tolist(),
            "weights": weights.tolist(),
        }
    )
    family = make_gam_family({"name": name, "link": link})
    with np.errstate(divide="ignore", invalid="ignore"):
        actual = family.deviance_obs(y, mu, weights)
        total = family.deviance(y, mu, weights)
    np.testing.assert_allclose(
        actual, np.asarray(expected["dev"], float), rtol=2e-15, atol=0, equal_nan=True
    )
    np.testing.assert_allclose(
        total, float(expected["total"]), rtol=2e-15, atol=0, equal_nan=True
    )


@pytest.mark.parametrize("link", LINKS)
def test_bernoulli_density_preserves_exact_and_tiny_probabilities(link):
    from tests.families.test_mgcv_count_likelihood_edges import (
        reference as count_reference,
    )

    y = np.repeat([0.0, 1.0], 7)
    mu = np.tile([0, 1e-300, 1e-30, 1e-15, 0.4, 1 - 1e-15, 1], 2)
    expected = count_reference(
        {
            "operation": "kernel",
            "family": "binomial",
            "y": y.tolist(),
            "mu": mu.tolist(),
            "weights": np.ones(len(y)).tolist(),
            "trials": np.ones(len(y)).tolist(),
        }
    )
    family = make_gam_family({"name": "binomial", "link": link})
    np.testing.assert_allclose(
        family.loglik_obs(y, mu),
        np.array(expected["density"], float),
        rtol=2e-15,
        atol=0,
    )
    np.testing.assert_equal(family.loglik(y, mu), -np.inf)


@pytest.mark.parametrize(
    "magnitude,upper_tail,rare_multiplier",
    [(10**9, False, 1), (10**12, False, 1), (10**9, True, 1), (10**12, True, 100)],
)
def test_rare_grouped_binomial_fit_matches_mgcv(magnitude, upper_tail, rare_multiplier):
    import pickle

    import pandas as pd

    from nampy.gam import GAM
    from tests.families.test_mgcv_count_likelihood_edges import (
        reference as count_reference,
    )

    x = np.linspace(-1, 1, 60)
    rare = rare_multiplier * np.rint(3 + 2 * np.sin(3 * x))
    successes = magnitude - rare if upper_tail else rare
    data = pd.DataFrame(
        {
            "x": x,
            "success": successes,
            "failure": magnitude - successes,
            "w": np.resize([0.5, 1, 2], len(x)),
        }
    )
    expected = count_reference(
        {
            "operation": "fit",
            "family": "binomial",
            "data": data.to_dict("list"),
        }
    )
    model = GAM(
        formula="cbind(success,failure)~s(x,bs='cr',k=6)",
        family="binomial",
        smoothing_params=[0.7],
        optimize_smoothing=False,
        smoothing_method="fixed",
        control={"epsilon": 1e-12},
    ).fit(data=data, sample_weight="w")
    expected_mu = np.array(expected["mu"])
    actual = model.predict()
    if upper_tail:
        actual, expected_mu = 1 - actual, 1 - expected_mu
    np.testing.assert_allclose(
        actual, expected_mu, rtol=2e-6, atol=2e-16 if upper_tail else 1e-23
    )
    for fitted in [model, pickle.loads(pickle.dumps(model))]:
        np.testing.assert_allclose(
            fitted.loglik(), expected["ll"], rtol=2e-8, atol=2e-6
        )
        np.testing.assert_allclose(fitted.aic(), expected["aic"], rtol=2e-8, atol=4e-6)
        np.testing.assert_allclose(fitted.bic(), expected["bic"], rtol=2e-8, atol=4e-6)
        np.testing.assert_allclose(
            fitted._loglik_effective_df(), expected["df"], rtol=2e-7, atol=1e-8
        )
