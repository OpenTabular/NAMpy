"""Alternative SCAT outer optimizers against fresh actual-R endpoints."""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nampy.gam import GAM
from tests.families.test_mgcv_scat_workflows import _case
from tests.mgcv_parity_utils import _build_r_command
from tests.reference_fixtures import load_reference, reference_key, save_reference

_SCRIPT = """
suppressPackageStartupMessages(library(mgcv));library(jsonlite)
a<-commandArgs(TRUE);p<-fromJSON(a[1]);d<-as.data.frame(p$data)
b<-gam(as.formula(p$formula),data=d,weights=d$w,family=scat(theta=p$theta),method='REML',optimizer=c('outer',p$optimizer))
pr<-predict(b,type='response',se.fit=TRUE)
write_json(list(version=as.character(packageVersion('mgcv')),theta=b$family$getTheta(),sp=b$sp,coef=coef(b),fit=pr$fit,se=pr$se.fit,Vp=b$Vp,edf=sum(b$edf),criterion=as.numeric(b$gcv.ubre),loglik=as.numeric(logLik(b)),aic=AIC(b)),a[2],digits=17,auto_unbox=TRUE)
"""


def _reference(case):
    script = _SCRIPT
    if "control" in case:
        script = script.replace(
            "method='REML',", "method='REML',control=do.call(gam.control,p$control),"
        )
    key = reference_key("scat-optimizer", {"case": case, "script": script})
    result = load_reference("mgcv_scat_optimizers", key)
    if result is None:
        with tempfile.TemporaryDirectory() as directory:
            source, inputs, output = [
                Path(directory) / name
                for name in ("reference.R", "input.json", "output.json")
            ]
            source.write_text(script)
            inputs.write_text(json.dumps(case))
            result_run = subprocess.run(
                _build_r_command(source, str(inputs), str(output)),
                capture_output=True,
                text=True,
            )
            assert result_run.returncode == 0, result_run.stderr
            result = json.loads(output.read_text())
        save_reference("mgcv_scat_optimizers", key, result)
    return result


@pytest.mark.parametrize("optimizer", ["bfgs", "optim", "nlm"])
@pytest.mark.parametrize("estimated", [False, True])
def test_scat_alternative_optimizer_matches_mgcv(optimizer, estimated):
    case = _case(method="REML", estimated=estimated)
    case["optimizer"] = optimizer
    if optimizer == "optim":
        # Match tighter source controls to resolve a flat theta endpoint.
        case["control"] = {"epsilon": 1e-10, "optim": {"factr": 10.0}}
    ref = _reference(case)
    model = GAM(
        formula=case["formula"],
        family={"name": "scat", "theta": case["theta"]},
        smoothing_method="REML",
        optimize_smoothing=True,
        smoothing_optimizer=optimizer,
        control=case.get("control"),
    ).fit(data=pd.DataFrame(case["data"]), sample_weight="w")
    assert ref["version"] == "1.9.4"
    prediction, se = model.predict(return_se=True)
    np.testing.assert_allclose(
        model.smoothing_params, np.atleast_1d(ref["sp"]), atol=5e-5, rtol=5e-4
    )
    from nampy.gam.model_state import _coef_full, _edf_total

    for name, actual in {
        "theta": model.family.getTheta(),
        "coef": _coef_full(model),
        "fit": prediction,
        "se": se,
        "Vp": model.vcov(),
        "edf": _edf_total(model),
        "criterion": model.smoothing_score_,
        "loglik": model.loglik(),
        "aic": model.aic(),
    }.items():
        np.testing.assert_allclose(
            actual, ref[name], atol=5e-5, rtol=5e-5, err_msg=name
        )
