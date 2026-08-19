"""Compare direct and default side-condition paths for an aliased PIRLS fit."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nampy.gam import GAM
from nampy.gam.model_state import _fit_workspace


def _fit(*, apply_side_conditions: bool, smoothing_params=None):
    rng = np.random.default_rng(44)
    n = 110
    x0 = rng.uniform(size=n)
    x1 = rng.uniform(size=n)
    eta = 0.15 + 0.55 * np.sin(2.0 * np.pi * x0) + 0.4 * x1
    data = pd.DataFrame({"x0": x0, "x1": x1, "z": x1})
    data["y"] = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta)))
    gam = GAM(
        formula='y ~ x1 + z + s(x0, bs="cr", k=8)',
        family="binomial",
        optimize_smoothing=smoothing_params is None,
        smoothing_method="REML" if smoothing_params is None else "fixed",
        smoothing_optimizer="outer_newton",
        apply_side_conditions=apply_side_conditions,
        smoothing_params=smoothing_params,
    )
    gam.fit(data=data)
    return gam


direct = _fit(apply_side_conditions=False)
default = _fit(apply_side_conditions=True)
fixed_direct = _fit(
    apply_side_conditions=False,
    smoothing_params=np.asarray(direct.smoothing_params, dtype=np.float64),
)
fixed_default = _fit(
    apply_side_conditions=True,
    smoothing_params=np.asarray(direct.smoothing_params, dtype=np.float64),
)

for name, lhs, rhs in (
    ("design", direct.gam_result_.compiled_model.design_matrix, default.gam_result_.compiled_model.design_matrix),
    (
        "prediction_map",
        direct.gam_result_.compiled_model.fit_to_prediction_parameterization_map,
        default.gam_result_.compiled_model.fit_to_prediction_parameterization_map,
    ),
    ("smoothing_params", direct.smoothing_params, default.smoothing_params),
    (
        "setup_U1",
        _fit_workspace(direct).penalty_subspace_cache["estimate_setup"].U1,
        _fit_workspace(default).penalty_subspace_cache["estimate_setup"].U1,
    ),
    (
        "setup_Eb",
        _fit_workspace(direct).penalty_subspace_cache["estimate_setup"].Eb,
        _fit_workspace(default).penalty_subspace_cache["estimate_setup"].Eb,
    ),
    (
        "fit_coef",
        direct.gam_result_.fit_core_solution.fit_result.coef_full,
        default.gam_result_.fit_core_solution.fit_result.coef_full,
    ),
    (
        "fit_vp_diag",
        np.diag(direct.gam_result_.fit_core_solution.fit_result.cov_bayes),
        np.diag(default.gam_result_.fit_core_solution.fit_result.cov_bayes),
    ),
    (
        "state_drop",
        direct.gam_result_.fit_core_solution.fit_state.dropped_column_indices,
        default.gam_result_.fit_core_solution.fit_state.dropped_column_indices,
    ),
):
    lhs = None if lhs is None else np.asarray(lhs)
    rhs = None if rhs is None else np.asarray(rhs)
    if lhs is None or rhs is None:
        print(name, lhs, rhs)
    else:
        print(
            name,
            "equal=", np.array_equal(lhs, rhs),
            "max_abs=", float(np.max(np.abs(lhs - rhs))) if lhs.size else 0.0,
            "direct=", lhs if lhs.size <= 12 else lhs.shape,
            "default=", rhs if rhs.size <= 12 else rhs.shape,
        )

print(
    "fixed_coef",
    np.asarray(fixed_direct.gam_result_.fit_core_solution.fit_result.coef_full),
    np.asarray(fixed_default.gam_result_.fit_core_solution.fit_result.coef_full),
)
print(
    "fixed_state_drop",
    fixed_direct.gam_result_.fit_core_solution.fit_state.dropped_column_indices,
    fixed_default.gam_result_.fit_core_solution.fit_state.dropped_column_indices,
)


def _r_fixed(sp: float):
    rng = np.random.default_rng(44)
    n = 110
    x0 = rng.uniform(size=n)
    x1 = rng.uniform(size=n)
    eta = 0.15 + 0.55 * np.sin(2.0 * np.pi * x0) + 0.4 * x1
    data = pd.DataFrame({"x0": x0, "x1": x1, "z": x1})
    data["y"] = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta)))
    r_code = """
suppressPackageStartupMessages(library(mgcv))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(TRUE)
d <- read.csv(args[[1]])
b <- gam(y ~ x1 + z + s(x0, bs="cr", k=8), data=d, family=binomial(),
         method="REML", sp=as.numeric(args[[2]]))
cat(toJSON(list(coef=as.numeric(coef(b)), vp=as.numeric(diag(b$Vp))),
           auto_unbox=TRUE, digits=17))
"""
    with tempfile.TemporaryDirectory() as tmp:
        csv = Path(tmp) / "data.csv"
        script = Path(tmp) / "probe.R"
        data.to_csv(csv, index=False)
        script.write_text(r_code, encoding="utf-8")
        proc = subprocess.run(
            ["Rscript", str(script), str(csv), repr(float(sp))],
            check=True,
            capture_output=True,
            text=True,
        )
    return json.loads(proc.stdout)


print("r_direct_sp", _r_fixed(float(direct.smoothing_params[0])))
print("r_default_sp", _r_fixed(float(default.smoothing_params[0])))
