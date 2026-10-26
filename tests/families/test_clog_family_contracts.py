"""Every returned clog kernel field compared with actual mgcv captures."""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from nampy.gam.families.clog import CensoredLogisticFamily
from tests.families.test_cnorm_family_contracts import SCRIPT, _numeric, _response, case
from tests.reference_fixtures import (
    generation_enabled,
    load_reference,
    reference_key,
    save_reference,
)

SOURCE = SCRIPT.replace("cnorm", "clog")


def reference(payload):
    key = reference_key("clog-leaf-v1", {"payload": payload, "script": SOURCE})
    if generation_enabled():
        with tempfile.TemporaryDirectory() as tmp:
            source, inputs, output = [Path(tmp) / name for name in ("run.R", "in.json", "out.json")]
            source.write_text(
                'args<-commandArgs(TRUE)\nlib<-Sys.getenv("MGCV_LIB_PATH");if(nzchar(lib)) .libPaths(c(lib,.libPaths()))\nsuppressPackageStartupMessages(library(mgcv));library(jsonlite)\na<-fromJSON(args[1])\n'
                + SOURCE + '\nwrite_json(ans,args[2],digits=17,auto_unbox=TRUE,na="string")'
            )
            inputs.write_text(json.dumps(payload))
            run = subprocess.run(["Rscript", str(source), str(inputs), str(output)], capture_output=True, text=True, timeout=60)
            assert run.returncode == 0, run.stderr
            save_reference("mgcv_clog_family", key, json.loads(output.read_text()))
    return load_reference("mgcv_clog_family", key)


@pytest.mark.parametrize("theta", [None, 0, -0.7, 0.7, 2])
@pytest.mark.parametrize("link", ["identity", "log", "sqrt"])
def test_clog_constructor_and_all_kernel_fields(theta, link):
    assert_kernel(case(theta=theta, link=link))


@pytest.mark.parametrize("band", ["mixed", "precision", "zero", "uncensored", "tails", "endpoint_zero", "symmetric"])
def test_clog_response_precision_and_tail_kernels(band):
    assert_kernel(case(band))


def assert_kernel(payload):
    ref = reference(payload)
    assert ref["version"] == "1.9.4"
    family = CensoredLogisticFamily(theta=payload["theta"], link=payload["link"])
    y, mu, w = _response(payload), np.asarray(payload["mu"]), np.asarray(payload["w"])
    assert family.n_theta == ref["n_theta"]
    np.testing.assert_allclose(family.getTheta(), ref["theta"], atol=1e-15)
    np.testing.assert_allclose(family.getTheta(True), ref["physical"], atol=1e-15)
    with np.errstate(all="ignore"):
        for level in range(3):
            expected = ref[f"dd{level}"]
            assert not expected["error"], expected
            actual = family.Dd(y, mu, family.getTheta(), w, level)
            assert actual.keys() == expected["value"].keys()
            for field, value in actual.items():
                np.testing.assert_allclose(value, _numeric(expected["value"][field]), rtol=3e-12, atol=3e-12, equal_nan=True, err_msg=field)
        for name, value in {"dev": family.deviance_obs(y, mu, w), "aic": family.aic(y, mu, weights=w)}.items():
            np.testing.assert_allclose(value, _numeric(ref[name]["value"]), rtol=3e-12, atol=3e-12, equal_nan=True, err_msg=name)
        for field, value in family.ls(y, w, family.getTheta()).items():
            np.testing.assert_allclose(np.asarray(value).ravel(), np.asarray(_numeric(ref["ls"]["value"][field])).ravel(), rtol=3e-12, atol=3e-12, equal_nan=True, err_msg=field)
        np.testing.assert_allclose(family.initialize_mu(y), _numeric(ref["initialize"]["value"]["mu"]), atol=1e-15)


@pytest.mark.parametrize("kwargs", [{"theta": [1, 2]}, {"theta": np.nan}, {"link": "inverse"}])
def test_clog_invalid_constructor(kwargs):
    with pytest.raises(ValueError):
        CensoredLogisticFamily(**kwargs)
