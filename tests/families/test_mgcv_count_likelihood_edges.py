"""Large-count and support-boundary densities against stats/mgcv."""

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
from nampy.gam.families._r_poisson_math import log_dbinom
from nampy.gam.families.registry import make_gam_family
from tests.reference_fixtures import load_reference, reference_key, save_reference

SCRIPT = r"""
a <- commandArgs(TRUE)
lib <- Sys.getenv('MGCV_LIB_PATH', '')
if (nzchar(lib)) .libPaths(c(lib,.libPaths()))
suppressPackageStartupMessages(library(mgcv)); library(jsonlite)
p <- fromJSON(a[1])
if (p$operation == 'fit') {
    d <- as.data.frame(p$data)
    f <- if (p$family == 'poisson') y~s(x,bs='cr',k=6) else
        cbind(success,failure)~s(x,bs='cr',k=6)
    m <- gam(f,data=d,weights=w,family=eval(parse(text=p$family)),
             sp=.7,method='REML',control=gam.control(epsilon=1e-8))
    ll <- logLik(m)
    out <- list(mu=unname(m$fitted.values),ll=as.numeric(ll),df=attr(ll,'df'),
                aic=AIC(m),bic=BIC(m))
} else {
    y <- p$y; mu <- p$mu; w <- p$weights
    family <- eval(parse(text=paste0(p$family,'()')))
    if (p$family == 'poisson') {
        density <- dpois(y,mu,log=TRUE)
        aic <- family$aic(y,rep(1,length(y)),mu,w,0)
        saturated <- mgcv:::fix.family.ls(family)$ls(y,w,length(y),1)[1]
    } else {
        n <- p$trials
        density <- dbinom(round(n*y),round(n),mu,log=TRUE)
        aic <- family$aic(y,n,mu,w,0)
        saturated <- -family$aic(y,n,y,w,0)/2
    }
    out <- list(density=density,aic=aic,saturated=saturated)
}
# Strings retain Inf/-Inf/NaN explicitly in portable JSON captures.
out$version <- as.character(packageVersion('mgcv'))
out$r_version <- R.version.string
write_json(out,a[2],digits=17,auto_unbox=TRUE,na='string')
"""


def reference(payload):
    key = reference_key(
        "count-likelihood-edges-v1",
        {
            "payload": payload,
            "source_sha256": hashlib.sha256(SCRIPT.encode()).hexdigest(),
        },
    )
    result = load_reference("mgcv_count_likelihood_edges", key)
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
        save_reference("mgcv_count_likelihood_edges", key, result)
    assert result["version"] == "1.9.4"
    return result


def assert_values(actual, expected, *, atol=2e-10, rtol=2e-12):
    np.testing.assert_allclose(
        actual, np.asarray(expected, float), atol=atol, rtol=rtol, equal_nan=True
    )


@pytest.mark.parametrize("magnitude", [1, 100000, 10**10, 10**15])
@pytest.mark.parametrize("weighted", [False, True])
def test_poisson_density_and_saturated_likelihood_at_large_counts(magnitude, weighted):
    y = np.array([0, 1, 3, magnitude, magnitude + round(np.sqrt(magnitude))], float)
    mu = np.array([0.01, 1, 3.5, magnitude, magnitude], float)
    weights = np.array([0, 0.5, 1, 2, 1.3]) if weighted else np.ones(len(y))
    expected = reference(
        {
            "operation": "kernel",
            "family": "poisson",
            "y": y.tolist(),
            "mu": mu.tolist(),
            "weights": weights.tolist(),
        }
    )
    family = make_gam_family("poisson")
    assert_values(family.loglik_obs(y, mu), expected["density"])
    assert_values(family.aic(y, mu, weights=weights), expected["aic"])
    assert_values(family.saturated_loglik(y, weights=weights), expected["saturated"])


def test_poisson_enormous_counts_return_r_log_densities():
    y = np.full(3, 1e308)
    mu = np.array([1, 1e308, 1e-308])
    weights = np.array([1, 2, 1])
    expected = reference(
        {
            "operation": "kernel",
            "family": "poisson",
            "y": y.tolist(),
            "mu": mu.tolist(),
            "weights": weights.tolist(),
        }
    )
    family = make_gam_family("poisson")
    assert_values(family.loglik_obs(y, mu), expected["density"])
    assert_values(family.aic(y, mu, weights=weights), expected["aic"])
    assert_values(family.saturated_loglik(y, weights=weights), expected["saturated"])


@pytest.mark.parametrize("zero_weight_fractional", [False, True])
def test_poisson_noninteger_support_and_rounding_match_r(zero_weight_fractional):
    y = np.array([0, 0.25, 1 + 1e-8, 2.5, 3 - 1e-8])
    mu = np.array([0.2, 0.5, 1.2, 3, 2.8])
    weights = np.array([1, 0, 2, 0, 0.5]) if zero_weight_fractional else np.ones(5)
    expected = reference(
        {
            "operation": "kernel",
            "family": "poisson",
            "y": y.tolist(),
            "mu": mu.tolist(),
            "weights": weights.tolist(),
        }
    )
    family = make_gam_family("poisson")
    assert_values(family.loglik_obs(y, mu), expected["density"])
    with np.errstate(invalid="ignore"):
        assert_values(family.aic(y, mu, weights=weights), expected["aic"])
        assert_values(
            family.saturated_loglik(y, weights=weights), expected["saturated"]
        )


def test_poisson_zero_means_and_tiny_rates_match_r():
    y = np.array([0, 1, 3, 0, 1])
    mu = np.array([0, 0, 0, 1e-300, 1e-300])
    weights = np.ones(len(y))
    expected = reference(
        {
            "operation": "kernel",
            "family": "poisson",
            "y": y.tolist(),
            "mu": mu.tolist(),
            "weights": weights.tolist(),
        }
    )
    family = make_gam_family("poisson")
    assert_values(family.loglik_obs(y, mu), expected["density"])
    assert_values(family.aic(y, mu, weights=weights), expected["aic"])
    assert_values(family.saturated_loglik(y, weights=weights), expected["saturated"])


def test_poisson_rounding_inside_integer_tolerance_matches_r():
    y = np.array([0, 1 + 1e-10, 3 - 1e-10])
    mu = np.array([0.2, 1.2, 2.8])
    weights = np.array([1, 0.5, 2])
    expected = reference(
        {
            "operation": "kernel",
            "family": "poisson",
            "y": y.tolist(),
            "mu": mu.tolist(),
            "weights": weights.tolist(),
        }
    )
    family = make_gam_family("poisson")
    assert_values(family.loglik_obs(y, mu), expected["density"])
    assert_values(family.aic(y, mu, weights=weights), expected["aic"])
    assert_values(family.saturated_loglik(y, weights=weights), expected["saturated"])


@pytest.mark.parametrize("magnitude", [10, 100000, 10**10, 10**15])
@pytest.mark.parametrize("weighted", [False, True])
def test_grouped_binomial_large_trials_and_tail_densities(magnitude, weighted):
    probabilities = [1e-14, 0.01, 0.4, 0.5, 1 - 1e-14]
    pairs = [
        (p, x)
        for p in probabilities
        for x in [0, 1, round(magnitude * p), magnitude - 1, magnitude]
    ]
    mu = np.array([p for p, _ in pairs])
    successes = np.array([x for _, x in pairs], float)
    trials = np.full(len(mu), magnitude, dtype=float)
    y = successes / trials
    weights = np.resize([0, 0.5, 1, 2], len(y)) if weighted else np.ones(len(y))
    expected = reference(
        {
            "operation": "kernel",
            "family": "binomial",
            "y": y.tolist(),
            "mu": mu.tolist(),
            "weights": weights.tolist(),
            "trials": trials.tolist(),
        }
    )
    family = make_gam_family("binomial")
    # Match R's round(n*y) convention after proportion representation.
    assert_values(log_dbinom(np.rint(trials * y), trials, mu), expected["density"])
    assert_values(family.loglik_obs(y, mu, n=trials), expected["density"])
    assert_values(family.aic(y, mu, weights=weights, n=trials), expected["aic"])
    family._trial_counts = trials
    assert_values(family.saturated_loglik(y, weights=weights), expected["saturated"])


@pytest.mark.parametrize("magnitude", [10, 10**15])
def test_grouped_binomial_exact_probability_boundaries(magnitude):
    trials = np.array([0, magnitude, magnitude, magnitude], float)
    y = np.array([0, 0, 1, 1])
    mu = np.array([0, 0, 1, 1])
    weights = np.array([0, 1, 2, 0.5])
    expected = reference(
        {
            "operation": "kernel",
            "family": "binomial",
            "y": y.tolist(),
            "mu": mu.tolist(),
            "weights": weights.tolist(),
            "trials": trials.tolist(),
        }
    )
    family = make_gam_family("binomial")
    assert_values(
        family.loglik_obs(y, mu, n=trials), expected["density"], atol=0, rtol=0
    )
    assert_values(
        family.aic(y, mu, weights=weights, n=trials), expected["aic"], atol=0, rtol=0
    )
    family._trial_counts = trials
    assert_values(
        family.saturated_loglik(y, weights=weights),
        expected["saturated"],
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize("family", ["poisson", "binomial"])
@pytest.mark.parametrize("magnitude", [100000, 10**8, 10**10])
@pytest.mark.parametrize("weighted", [False, True])
def test_large_count_fitted_metrics_match_actual_mgcv(family, magnitude, weighted):
    rng = np.random.default_rng(9291)
    x = rng.uniform(-1, 1, 60).round(12)
    mean = magnitude * (np.exp(0.2 * np.sin(3 * x)) if family == "poisson" else 0.4)
    counts = np.rint(
        mean + np.sqrt(magnitude) * (0.6 * np.sin(3 * x) + rng.normal(size=len(x)))
    )
    weights = np.resize([0.5, 1, 2], len(x)) if weighted else np.ones(len(x))
    if weighted:
        weights[::13] = 0
    data = pd.DataFrame(
        {
            "x": x,
            "y": counts,
            "success": counts,
            "failure": magnitude - counts,
            "w": weights,
        }
    )
    expected = reference(
        {"operation": "fit", "family": family, "data": data.to_dict("list")}
    )
    formula = (
        "y" if family == "poisson" else "cbind(success,failure)"
    ) + "~s(x,bs='cr',k=6)"
    model = GAM(
        formula=formula,
        family=family,
        smoothing_params=[0.7],
        optimize_smoothing=False,
        smoothing_method="fixed",
        control={"epsilon": 1e-12},
    ).fit(data=data, sample_weight="w")
    assert_values(model.predict(), expected["mu"], atol=1e-8, rtol=2e-12)
    for fitted in (model, pickle.loads(pickle.dumps(model))):
        assert_values(fitted.loglik(), expected["ll"], atol=1e-5, rtol=1e-11)
        assert_values(fitted.aic(), expected["aic"], atol=2e-5, rtol=1e-11)
        assert_values(fitted.bic(), expected["bic"], atol=2e-5, rtol=1e-11)
        assert_values(fitted._loglik_effective_df(), expected["df"])
