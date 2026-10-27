"""Stored working residuals use the offset-free predictor for every family."""

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
    ("gaussian", "identity", "gaussian", "gaussian"),
    ("gaussian", "log", "gaussian", "positive"),
    ("poisson", "log", "poisson", "poisson"),
    ("poisson", "sqrt", "poisson", "poisson"),
    ("gamma", "log", "Gamma", "positive"),
    ("gamma", "inverse", "Gamma", "positive"),
    ("inverse_gaussian", "log", "inverse.gaussian", "positive"),
    ("negbin", "log", "nb", "poisson"),
    ("tw", "log", "tw", "positive"),
    ("binomial", "probit", "binomial", "binomial"),
    ("quasipoisson", "log", "quasipoisson", "poisson"),
]
RTYPES = ["working", "response", "deviance", "pearson", "scaled.pearson"]
SCRIPT = r"""
a <- commandArgs(TRUE)
lib <- Sys.getenv("MGCV_LIB_PATH", "")
if (nzchar(lib)) .libPaths(c(lib, .libPaths()))
suppressPackageStartupMessages(library(mgcv)); library(jsonlite)
p <- fromJSON(a[1]); d <- as.data.frame(p$data)
m <- gam(y~s(x,bs='cr',k=6)+offset(o),data=d,weights=w,
         family=eval(parse(text=p$family)),sp=.7,method='REML',
         control=gam.control(epsilon=1e-12))
r <- lapply(p$rtypes, function(t) unname(residuals(m,type=t)))
names(r) <- p$rtypes
write_json(list(version=as.character(packageVersion('mgcv')),
                fit=unname(predict(m,type='link')),residuals=r),
           a[2],digits=17,auto_unbox=TRUE,na='null')
"""


def reference(data, r_family, rtypes):
    payload = {"data": data.to_dict("list"), "family": r_family, "rtypes": rtypes}
    key = reference_key(
        "working-residual-offsets-v1",
        {
            "payload": payload,
            "source_sha256": hashlib.sha256(SCRIPT.encode()).hexdigest(),
        },
    )
    result = load_reference("mgcv_working_residual_offsets", key)
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
        save_reference("mgcv_working_residual_offsets", key, result)
    return result


def data_frame(target, zero_weights):
    rng = np.random.default_rng(92761)
    x = rng.uniform(-1, 1, 85)
    offset = 0.3 * np.cos(2 * x)
    base = 0.3 + 0.5 * np.sin(2 * x) + offset
    targets = {
        "gaussian": base + rng.normal(0, 0.4, len(x)),
        "positive": np.exp(base + rng.normal(0, 0.3, len(x))),
        "poisson": rng.poisson(np.exp(base)),
        "binomial": rng.binomial(1, 1 / (1 + np.exp(-base))),
    }
    weights = rng.uniform(0.3, 2, len(x))
    if zero_weights:
        weights[::11] = 0
    # Keep fixture identities independent of platform libm rounding.
    return pd.DataFrame(
        {"x": x, "o": offset, "y": targets[target], "w": weights}
    ).round(12)


@pytest.mark.parametrize("name,link,r_name,target", CASES)
@pytest.mark.parametrize("zero_weights", [False, True])
@pytest.mark.parametrize("offset_mode", ["formula", "api", "mixed"])
def test_working_residuals_with_offsets_and_zero_weights_match_mgcv(
    name, link, r_name, target, zero_weights, offset_mode
):
    data = data_frame(target, zero_weights)
    family = {"name": name, "link": link}
    r_family = f"{r_name}(link='{link}')"
    if name == "negbin":
        family["theta"] = 2.0
        r_family = "nb(theta=2,link='log')"
    elif name == "tw":
        family["theta"] = 1.5
        r_family = "tw(theta=1.5,link='log')"
    # Fixed-inner tw retains its documented Pearson dispersion fallback;
    # upstream's scaled Pearson uses a separately profiled outer dispersion.
    # The scale-independent residuals still have an exact upstream target.
    rtypes = RTYPES[:-1] if name == "tw" else RTYPES
    expected = reference(data, r_family, rtypes)
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
        family=family,
        smoothing_params=[0.7],
        optimize_smoothing=False,
        control={"epsilon": 1e-12},
    ).fit(data=fit_data, offset=offset, sample_weight="w")
    np.testing.assert_allclose(
        model.predict(type="link", offset=data.o.to_numpy()),
        expected["fit"],
        atol=2e-9,
        rtol=2e-9,
    )
    for rtype in rtypes:
        residuals = model.residuals(type=rtype)
        np.testing.assert_allclose(
            residuals,
            np.asarray(expected["residuals"][rtype], dtype=float),
            atol=2e-8,
            rtol=2e-8,
            equal_nan=True,
            err_msg=rtype,
        )
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(
        restored.residuals(type="working"), model.residuals(type="working")
    )
