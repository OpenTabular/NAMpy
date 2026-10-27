"""Gaussian magic scoring discards zero-weight rows without discarding predictions."""

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
from nampy.gam.fit.selection.criteria import (
    criterion_gradient,
    criterion_hessian,
    criterion_value,
)
from nampy.gam.inference.null_deviance import null_deviance
from tests.reference_fixtures import load_reference, reference_key, save_reference

FORMULAS = {
    "two_smooths": "y~s(x,bs='cr',k=6)+s(z,bs='ps',k=6)+offset(o)",
    "linked": "y~s(x,bs='cr',k=6,id=1)+s(z,bs='cr',k=6,id=1)+offset(o)",
    "tensor": "y~te(x,z,bs=c('cr','ps'),k=c(4,4))+offset(o)",
    "no_intercept": "y~0+s(x,bs='cr',k=6)+z+offset(o)",
}
SCRIPT = r"""
a <- commandArgs(TRUE)
lib <- Sys.getenv("MGCV_LIB_PATH", "")
if (nzchar(lib)) .libPaths(c(lib, .libPaths()))
suppressPackageStartupMessages(library(mgcv)); library(jsonlite)
p <- fromJSON(a[1]); d <- as.data.frame(p$data)
m <- gam(as.formula(p$formula), data=d, weights=w, method="GCV.Cp", scale=p$scale)
pr <- predict(m, d, type="link", se.fit=TRUE)
out <- list(version=as.character(packageVersion("mgcv")),sp=m$sp,
            fit=pr$fit,se=pr$se.fit,score=unname(m$gcv.ubre),scale=m$sig2,
            edf=sum(m$edf),null_deviance=m$null.deviance,
            working=unname(residuals(m,type="working")))
write_json(out,a[2],digits=17,auto_unbox=TRUE,na="null")
"""


def data_frame(zero_weights=True):
    rng = np.random.default_rng(801934)
    x, z = rng.uniform(-1, 1, (2, 125))
    offset = 0.12 * np.sin(3 * z)
    weights = rng.lognormal(0, 0.7, len(x))
    if zero_weights:
        weights[::13] = 0
    y = 0.1 + 0.65 * np.sin(3 * x) + 0.3 * z + offset + rng.normal(0, 0.3, len(x))
    return pd.DataFrame({"x": x, "z": z, "o": offset, "w": weights, "y": y})


def reference(data, formula, scale):
    payload = {"data": data.to_dict(orient="list"), "formula": formula, "scale": scale}
    key = reference_key(
        "gaussian-magic-zero-weights-v1",
        {
            "payload": payload,
            "source_sha256": hashlib.sha256(SCRIPT.encode()).hexdigest(),
        },
    )
    expected = load_reference("mgcv_gaussian_gcv_zero_weights", key)
    if expected is None:
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
            expected = json.loads(output.read_text())
        save_reference("mgcv_gaussian_gcv_zero_weights", key, expected)
    return expected


@pytest.mark.parametrize("formula_name", FORMULAS)
@pytest.mark.parametrize("zero_weights", [True, False])
@pytest.mark.parametrize("method,scale", [("gcv", 0.0), ("ubre", 0.2)])
def test_gaussian_magic_weighted_fit_matches_mgcv(
    formula_name, zero_weights, method, scale
):
    data = data_frame(zero_weights)
    formula = FORMULAS[formula_name]
    expected = reference(data, formula, scale)
    assert expected["version"] == "1.9.4"
    model = GAM(
        formula=formula,
        optimize_smoothing=True,
        smoothing_method=method,
        scale=scale,
    ).fit(data=data, sample_weight=data.w.to_numpy())
    fit, se = model.predict(data, type="link", return_se=True)
    np.testing.assert_allclose(fit, expected["fit"], rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(se, expected["se"], rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(model.fit_result().edf_total, expected["edf"], rtol=3e-5)
    np.testing.assert_allclose(model.fit_result().scale, expected["scale"], rtol=3e-5)
    np.testing.assert_allclose(model.smoothing_score_, expected["score"], rtol=3e-5)
    np.testing.assert_allclose(
        null_deviance(model), expected["null_deviance"], rtol=1e-10
    )
    np.testing.assert_allclose(
        model.residuals(type="working"),
        np.asarray(expected["working"], dtype=float),
        atol=3e-6,
        rtol=3e-5,
        equal_nan=True,
    )
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.predict(data), fit)
    np.testing.assert_allclose(
        null_deviance(restored), expected["null_deviance"], rtol=1e-10
    )


def test_magic_score_derivatives_keep_the_same_effective_row_count():
    data = data_frame()
    model = GAM(
        formula=FORMULAS["two_smooths"], optimize_smoothing=True, smoothing_method="gcv"
    ).fit(data=data, sample_weight=data.w.to_numpy())
    y = data.y.to_numpy()
    log_sp = np.log([0.8, 1.2])
    step = 1e-4
    grad = criterion_gradient(model, y, log_sp, method="gcv")
    hess = criterion_hessian(model, y, log_sp, method="gcv")
    numeric_gradient, numeric_hessian = [], []
    for direction in np.eye(2) * step:
        numeric_gradient.append(
            (
                criterion_value(model, y, log_sp + direction, "gcv")
                - criterion_value(model, y, log_sp - direction, "gcv")
            )
            / (2 * step)
        )
        numeric_hessian.append(
            (
                criterion_gradient(model, y, log_sp + direction, "gcv")
                - criterion_gradient(model, y, log_sp - direction, "gcv")
            )
            / (2 * step)
        )
    np.testing.assert_allclose(grad, numeric_gradient, rtol=2e-5, atol=2e-8)
    np.testing.assert_allclose(
        hess, np.asarray(numeric_hessian).T, rtol=2e-5, atol=2e-8
    )


def test_refit_from_magic_to_reml_resets_the_effective_row_policy():
    data = data_frame()
    kwargs = {"formula": FORMULAS["two_smooths"], "optimize_smoothing": True}
    model = GAM(**kwargs, smoothing_method="gcv").fit(
        data=data, sample_weight=data.w.to_numpy()
    )
    model.fit(data=data, sample_weight=data.w.to_numpy(), smoothing_method="reml")
    fresh = GAM(**kwargs, smoothing_method="reml").fit(
        data=data, sample_weight=data.w.to_numpy()
    )
    np.testing.assert_allclose(model.predict(data), fresh.predict(data), atol=1e-12)
    np.testing.assert_allclose(model.vcov(), fresh.vcov(), atol=1e-12)


@pytest.mark.parametrize("method", ["fixed", "gcv", "reml"])
def test_zero_weight_gaussian_likelihood_retains_positive_precision_rows(method):
    """Keep finite likelihood semantics when upstream's log(0) is undefined."""
    data = data_frame()
    model = GAM(
        formula=FORMULAS["two_smooths"],
        optimize_smoothing=method != "fixed",
        smoothing_method=method,
        smoothing_params=[0.7, 0.9] if method == "fixed" else None,
    ).fit(data=data, sample_weight=data.w.to_numpy())
    weights = data.w.to_numpy()
    positive = weights > 0
    nobs = np.count_nonzero(positive)
    rss = np.sum(weights * (data.y.to_numpy() - model.predict(data)) ** 2)
    loglik = -0.5 * (
        nobs * (np.log(2 * np.pi * rss / nobs) + 1) - np.sum(np.log(weights[positive]))
    )
    np.testing.assert_allclose(model.loglik(), loglik, atol=1e-10)
    assert np.isfinite(model.aic())
    assert np.isfinite(model.bic())
