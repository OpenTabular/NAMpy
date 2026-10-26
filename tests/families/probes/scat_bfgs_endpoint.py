"""Opt-in actual-R audit of SCAT BFGS path and uncertainty state."""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from nampy.gam import GAM
from nampy.gam.fit.selection.optimize import driver
from tests.families.test_mgcv_scat_optimizers import _SCRIPT
from tests.families.test_mgcv_scat_workflows import _case
from tests.mgcv_parity_utils import _build_r_command

case = _case(method="REML", estimated=True)
case["optimizer"] = "bfgs"
script = _SCRIPT.replace(
    "b<-gam(",
    "trace('bfgs',where=asNamespace('mgcv'),tracer=quote(assign('initial_lsp',lsp,envir=.GlobalEnv)),print=FALSE);b<-gam(",
).replace(
    "aic=AIC(b)",
    "aic=AIC(b),initial_lsp=initial_lsp,outer=b$outer.info,edf2=b$edf2,Vc=b$Vc",
)
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    source, inputs, output = [
        root / x for x in ("probe.R", "input.json", "output.json")
    ]
    source.write_text(script)
    inputs.write_text(json.dumps(case))
    subprocess.run(
        _build_r_command(source, str(inputs), str(output)),
        check=True,
        capture_output=True,
    )
    reference = json.loads(output.read_text())

original_bfgs = driver._optimize_outer_bfgs_strict
initial = {}


def record_start(**kwargs):
    initial["x0"] = np.array(kwargs["x0"]).copy()
    return original_bfgs(**kwargs)


driver._optimize_outer_bfgs_strict = record_start
model = GAM(
    formula=case["formula"],
    family={"name": "scat", "theta": case["theta"]},
    smoothing_method="REML",
    optimize_smoothing=True,
    smoothing_optimizer="bfgs",
).fit(data=pd.DataFrame(case["data"]), sample_weight="w")


def default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


print(
    json.dumps(
        {
            "R": reference,
            "Python": {
                "initial": initial,
                "theta": model.family.getTheta(),
                "sp": model.smoothing_params,
                "outer": model._optim_result.outer_info,
                "trace": getattr(model._optim_result, "optim_trace", None),
                "aic": model.aic(),
            },
        },
        default=default,
        indent=2,
    )
)
