"""Constant numeric by constraints preserve mgcv prediction covariance."""

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

SCRIPT = r"""
a <- commandArgs(TRUE)
lib <- Sys.getenv("MGCV_LIB_PATH", "")
if (nzchar(lib)) .libPaths(c(lib, .libPaths()))
suppressPackageStartupMessages(library(mgcv)); library(jsonlite)
p <- fromJSON(a[1]); d <- as.data.frame(p$data); nd <- as.data.frame(p$newdata)
m <- gam(as.formula(p$formula), data=d, sp=.7, method="REML")
pr <- predict(m, nd, se.fit=TRUE); X <- predict(m, nd, type="lpmatrix")
write_json(list(version=as.character(packageVersion("mgcv")),
                fit=unname(pr$fit), se=unname(pr$se.fit),
                covariance=unname(X %*% m$Vp %*% t(X)),
                ncoef=length(coef(m)), edf=sum(m$edf), scale=m$sig2),
           a[2], digits=17, auto_unbox=TRUE)
"""


def reference(data, newdata, formula):
    payload = {
        "data": data.to_dict(orient="list"),
        "newdata": newdata.to_dict(orient="list"),
        "formula": formula,
    }
    key = reference_key(
        "constant-numeric-by-v1",
        {
            "payload": payload,
            "source_sha256": hashlib.sha256(SCRIPT.encode()).hexdigest(),
        },
    )
    result = load_reference("mgcv_constant_numeric_by", key)
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
        save_reference("mgcv_constant_numeric_by", key, result)
    return result


@pytest.mark.parametrize("basis", ["cr", "cs", "ps", "tp", "ts", "gp"])
@pytest.mark.parametrize(
    "by_value,intercept",
    [(0.0, True), (1.0, True), (1.0, False), (-1.0, True), (-1.0, False)],
)
def test_constant_numeric_by_prediction_covariance_matches_mgcv(
    basis, by_value, intercept
):
    rng = np.random.default_rng(73)
    x = rng.uniform(size=90)
    data = pd.DataFrame(
        {
            "x": x,
            "z": rng.normal(size=len(x)),
            "b": by_value,
            "y": np.sin(4 * x) + rng.normal(0, 0.2, len(x)),
        }
    )
    newdata = data.iloc[::6].copy()
    # A negative constant retains the smooth's constant null space in mgcv.
    # With an intercept its allocation is unidentified; compare predictions on
    # estimable rows, rather than an arbitrary dropped-coordinate convention.
    if by_value >= 0 or not intercept:
        newdata["b"] = np.resize([0.0, 1.0, -1.0], len(newdata))
    formula = f"y~{'1' if intercept else '0'}+z+s(x,by=b,bs='{basis}',k=6)"
    expected = reference(data, newdata, formula)
    assert expected["version"] == "1.9.4"
    model = GAM(formula=formula, smoothing_params=[0.7], optimize_smoothing=False).fit(
        data=data
    )
    fit, se = model.predict(newdata, return_se=True)
    result = model.fit_result()
    design = model.lpmatrix(newdata)
    assert design.shape[1] == expected["ncoef"]
    np.testing.assert_allclose(fit, expected["fit"], atol=2e-9, rtol=2e-9)
    np.testing.assert_allclose(se, expected["se"], atol=2e-9, rtol=2e-9)
    np.testing.assert_allclose(
        design @ result.cov_bayes @ design.T,
        expected["covariance"],
        atol=2e-9,
        rtol=2e-9,
    )
    np.testing.assert_allclose(result.edf_total, expected["edf"], rtol=2e-9)
    np.testing.assert_allclose(result.scale, expected["scale"], rtol=2e-9)
    restored = pickle.loads(pickle.dumps(model))
    restored_fit, restored_se = restored.predict(newdata, return_se=True)
    np.testing.assert_array_equal(restored_fit, fit)
    np.testing.assert_array_equal(restored_se, se)
