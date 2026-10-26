"""Opt-in actual-R ziP BFGS evaluation and accepted-iterate audit."""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from nampy.gam import GAM
from nampy.gam.fit.selection.criteria.pirls import family_vector_theta
from nampy.gam.fit.selection.optimize import bfgs_strict, driver
from tests.families.test_mgcv_zip_workflows import SCRIPT, _case
from tests.mgcv_parity_utils import _build_r_command

case = _case(method="REML", estimated=True)
trace = r"""
.zip_trace<-list()
trace('bfgs',where=asNamespace('mgcv'),print=FALSE,tracer=quote(assign('.zip_initial',lsp,envir=.GlobalEnv)))
trace('gam.fit3',where=asNamespace('mgcv'),print=FALSE,
tracer=quote({.zip_sp<-sp;.zip_start<-start;.zip_mu<-mustart}),exit=quote({v<-returnValue(); .zip_trace[[length(.zip_trace)+1]]<<-list(x=.zip_sp,start=.zip_start,mustart=.zip_mu,deriv=deriv,score=v$REML,grad=v$REML1,coef=v$coefficients,iter=v$iter)}))
args$optimizer<-c('outer','bfgs')
"""
script = SCRIPT.replace("b<-do.call(gam,args)", trace + "b<-do.call(gam,args)").replace(
    "joint=if(exists", "evaluations=.zip_trace,initial=.zip_initial,joint=if(exists"
)
with tempfile.TemporaryDirectory() as directory:
    paths = [Path(directory) / x for x in ["probe.R", "input.json", "output.json"]]
    paths[0].write_text(script)
    paths[1].write_text(json.dumps(case))
    subprocess.run(
        _build_r_command(paths[0], str(paths[1]), str(paths[2])),
        check=True,
        capture_output=True,
    )
    reference = json.loads(paths[2].read_text())

joint_calls = []
old_joint = family_vector_theta._joint


def joint(*args, **kwargs):
    result = old_joint(*args, **kwargs)
    if len(joint_calls) < 2:
        joint_calls.append(
            {
                "coef": result[0]["coef_full"],
                "Dp1": result[1].Dp1,
                "K1": result[1].K1_full,
            }
        )
    return result


family_vector_theta._joint = joint
original = driver._optimize_outer_bfgs_strict
initial = {}
evaluations = []


def start(**kwargs):
    initial["x"] = kwargs["x0"].copy()
    from nampy.gam.model_state import _fit_workspace

    initial["starts"] = {
        k: v
        for k, v in vars(_fit_workspace(kwargs["objective"].model)).items()
        if "start" in k
    }
    return original(**kwargs)


driver._optimize_outer_bfgs_strict = start
orig_eval = bfgs_strict._eval_objective_at


def evaluate(objective, x, **kwargs):
    result = orig_eval(objective, x, **kwargs)
    from nampy.gam.model_state import _fit_workspace

    inner = _fit_workspace(objective.model).get("pirls_last_inner_trace", None)
    evaluations.append(
        {
            "x": np.array(x),
            "score": result[0],
            "grad": result[1],
            "coef": result[4],
            "need_grad": kwargs.get("need_grad", False),
            "inner": inner,
        }
    )
    return result


bfgs_strict._eval_objective_at = evaluate
model = GAM(
    formula=case["formula"],
    family={"name": "zip", "theta": case["theta"], "b": case["b"]},
    smoothing_method="REML",
    smoothing_optimizer="bfgs",
    optimize_smoothing=True,
).fit(data=pd.DataFrame(case["data"]), sample_weight="w")


def serialize(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return str(x)


print(
    json.dumps(
        {
            "R": reference,
            "Python": {
                "joint_calls": joint_calls,
                "initial": initial,
                "evaluations": evaluations,
                "outer": model._optim_result.outer_info,
                "accepted": model._optim_result.optim_trace,
                "theta": model.family.getTheta(),
                "sp": model.smoothing_params,
            },
        },
        default=serialize,
        indent=2,
    )
)
