"""Decompose the gammals_select_true_cr optimized-endpoint mismatch.

Prints, for both NAMpy and mgcv: sum(edf), sum(edf1), the raw
rowSums(Vc * crossprod(R)) sum before the cap, and whether the
`sum(edf2) > sum(edf1) -> edf2 <- edf1` replacement (gam.fit4.r:1715) fired.

It also compares optimized new-data predictions against mgcv on the original
and mirrored (``x -> -x``) representations of the same model.  The mirrored
newdata is transformed with the training data so both fits are evaluated at
the same physical covariate locations.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/home/ad32/projects/package/NAMpy")

import numpy as np

import nampy.gam.fit.solvers.general_family.newton as gnewton
from nampy.gam.fit.solvers.general_family.fixed_smoothing import (
    build_general_family_setup_state,
)
from nampy.gam.model.api import GAM
from nampy.gam.smoothing_selection.optimize.basics import (
    _initial_smoothing_params_from_design,
)
from nampy.gam.smoothing_selection.reparam import build_estimate_gam_setup_state
from tests.families.test_general_family_mgcv_parity import (
    _gammals_data,
    _general_newdata,
)
from tests.mgcv_parity_utils import _run_mgcv_predict_on_newdata

FORMULA = ['y ~ s(x, bs="cr", k=6)', "~ 1"]


def _run_mgcv_initial_spg(frame):
    with tempfile.TemporaryDirectory() as tmp:
        csv = Path(tmp) / "d.csv"
        output = Path(tmp) / "initial.json"
        frame.to_csv(csv, index=False)
        res = subprocess.run(
            [
                "Rscript",
                "tests/parity/mgcv_initial_spg.R",
                str(csv),
                str(output),
                str(FORMULA),
                "gammals",
                "ML",
                "true",
            ],
            capture_output=True,
            text=True,
        )
        if res.returncode != 0:
            raise RuntimeError(res.stderr[-1000:])
        return json.loads(output.read_text(encoding="utf-8"))

data = _gammals_data()
captured = {}

compiled = GAM(
    formula=FORMULA,
    family="gammals",
    select=True,
    optimize_smoothing=False,
    smoothing_method="ML",
)
compiled.fit(data=data)
compiled_setup = build_estimate_gam_setup_state(compiled)
compiled_fit5_setup = build_general_family_setup_state(
    compiled,
    np.ones(np.asarray(compiled.smoothing_params).size, dtype=np.float64),
    score_type="ML",
)
nampy_initial_sp = _initial_smoothing_params_from_design(compiled, compiled.y_)
mgcv_initial = _run_mgcv_initial_spg(data)
mirrored_initial_data = data.copy()
mirrored_initial_data["x"] = -mirrored_initial_data["x"]
mgcv_mirrored_initial = _run_mgcv_initial_spg(mirrored_initial_data)

print("initial.spg:")
print(f"  model sp count : {np.asarray(compiled.smoothing_params).size}")
print(f"  penalty count  : {len(compiled_setup.S)}")
print(f"  ranks / offsets: {compiled_setup.rank} / {compiled_setup.off}")
print(f"  L / lsp0      : {compiled_setup.L} / {compiled_setup.lsp0}")
for label, expected in (
    ("mgcv original", mgcv_initial),
    ("mgcv mirrored", mgcv_mirrored_initial),
):
    x_diff = np.max(
        np.abs(
            np.asarray(compiled_fit5_setup.X_initial, dtype=np.float64)
            - np.asarray(expected["X_initial"], dtype=np.float64)
        )
    )
    s_diffs = [
        np.max(np.abs(np.asarray(actual) - np.asarray(reference)))
        for actual, reference in zip(compiled_setup.S, expected["S"], strict=True)
    ]
    print(f"  setup vs {label}: X={x_diff:.3e}, S={s_diffs}")
print(f"  NAMpy          : {np.asarray(nampy_initial_sp, dtype=float)}")
print(f"  mgcv original  : {np.asarray(mgcv_initial['initial_sp'], dtype=float)}")
print(
    "  mgcv mirrored  : "
    f"{np.asarray(mgcv_mirrored_initial['initial_sp'], dtype=float)}"
)


def _fit_from_initial_sp(initial_sp):
    model = GAM(
        formula=FORMULA,
        family="gammals",
        select=True,
        optimize_smoothing=True,
        smoothing_params=np.asarray(initial_sp, dtype=np.float64),
        smoothing_method="ML",
        smoothing_optimizer="outer_newton",
    )
    model.fit(data=data)
    return model


gam_from_mgcv_initial = _fit_from_initial_sp(mgcv_initial["initial_sp"])
print(
    "  endpoint from ordinary mgcv start: "
    f"{np.asarray(gam_from_mgcv_initial.smoothing_params, dtype=float)}"
)

_orig = gnewton.postprocess_general_newton_fit


def _wrapped(fit, **kw):
    out = _orig(fit, **kw)
    R = out["R"]
    RTR = R.T @ R
    captured["edf_sum"] = float(np.sum(out["edf"]))
    captured["edf1_sum"] = float(np.sum(out["edf1"]))
    captured["edf2_sum"] = float(np.sum(out["edf2"]))
    captured["raw_edf2_sum"] = float(np.sum(out["Vc"] * RTR))
    captured["cap_fired"] = bool(
        captured["raw_edf2_sum"] > captured["edf1_sum"] + 1e-12
    )
    return out


gnewton.postprocess_general_newton_fit = _wrapped
try:
    gam = GAM(
        formula=FORMULA,
        family="gammals",
        select=True,
        optimize_smoothing=True,
        smoothing_method="ML",
        smoothing_optimizer="outer_newton",
    )
    gam.fit(data=data)
finally:
    gnewton.postprocess_general_newton_fit = _orig

print("NAMpy:")
print(f"  sp             : {np.asarray(gam.smoothing_params, dtype=float)}")
for k, v in captured.items():
    print(f"  {k:14s}: {v}")

r_code = """
library(mgcv)
d <- read.csv(commandArgs(TRUE)[1])
b <- gam(list(y ~ s(x, bs="cr", k=6), ~ 1), data=d, family=gammals(),
         method="ML", select=TRUE)
RtR <- crossprod(b$R)
raw <- sum(rowSums(b$Vc * RtR))
out <- list(sp = as.numeric(b$sp),
            edf_sum = sum(b$edf), edf1_sum = sum(b$edf1),
            edf2_sum = sum(b$edf2), raw_edf2_sum = raw,
            cap_fired = raw > sum(b$edf1))
cat(jsonlite::toJSON(out, digits=I(15), auto_unbox=TRUE))
"""
with tempfile.TemporaryDirectory() as tmp:
    csv = Path(tmp) / "d.csv"
    data.to_csv(csv, index=False)
    rfile = Path(tmp) / "probe.R"
    rfile.write_text(r_code)
    res = subprocess.run(
        ["Rscript", str(rfile), str(csv)], capture_output=True, text=True
    )
    if res.returncode != 0:
        print("R error:", res.stderr[-800:])
    else:
        out = json.loads(res.stdout)
        print("mgcv:")
        print(f"  sp             : {np.asarray(out['sp'], dtype=float)}")
        for k in ("edf_sum", "edf1_sum", "edf2_sum", "raw_edf2_sum", "cap_fired"):
            print(f"  {k:14s}: {out[k]}")

# Flatness + orientation checks (same method as gaulss_select probe).
from nampy.gam.smoothing_selection.criteria.dispatch import (  # noqa: E402
    criterion_gradient,
    criterion_value,
)

nam_end = np.log(np.asarray(gam.smoothing_params, dtype=np.float64))
mg_end = np.log(np.asarray(out["sp"], dtype=np.float64))
for name, pt in (("nampy", nam_end), ("mgcv", mg_end)):
    val = criterion_value(gam, gam.y_, pt, method="reml")
    grad = criterion_gradient(gam, gam.y_, pt, method="reml")
    print(
        f"criterion at {name} end: {val:.10f} |grad| {np.max(np.abs(grad)):.2e}"
    )

r_code2 = """
library(mgcv)
d <- read.csv(commandArgs(TRUE)[1])
d$x <- -d$x
b <- gam(list(y ~ s(x, bs="cr", k=6), ~ 1), data=d, family=gammals(),
         method="ML", select=TRUE)
cat(jsonlite::toJSON(list(sp = as.numeric(b$sp), edf1_sum = sum(b$edf1),
                          edf2_sum = sum(b$edf2)),
                     digits=I(15), auto_unbox=TRUE))
"""
with tempfile.TemporaryDirectory() as tmp:
    csv = Path(tmp) / "d.csv"
    data.to_csv(csv, index=False)
    rfile = Path(tmp) / "probe2.R"
    rfile.write_text(r_code2)
    res = subprocess.run(
        ["Rscript", str(rfile), str(csv)], capture_output=True, text=True
    )
    if res.returncode != 0:
        print("R error:", res.stderr[-500:])
    else:
        out2 = json.loads(res.stdout)
        print("mgcv mirrored basis (x -> -x):")
        print(f"  sp      : {np.asarray(out2['sp'], dtype=float)}")
        print(f"  edf1_sum: {out2['edf1_sum']}")
        print(f"  edf2_sum: {out2['edf2_sum']}")


def _max_abs_difference(actual, expected):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    if actual.shape != expected.shape and actual.size == expected.size:
        expected = expected.reshape(actual.shape, order="F")
    return float(np.max(np.abs(actual - expected)))


newdata = _general_newdata(data)
mirrored_data = data.copy()
mirrored_data["x"] = -mirrored_data["x"]
mirrored_newdata = newdata.copy()
mirrored_newdata["x"] = -mirrored_newdata["x"]

print("optimized new-data prediction max-absolute differences:")
for pred_type in ("link", "response", "terms"):
    actual_pred, actual_se = gam.predict(newdata, type=pred_type, return_se=True)
    original = _run_mgcv_predict_on_newdata(
        data,
        newdata,
        FORMULA,
        family="gammals",
        method="ML",
        type=pred_type,
        return_se=True,
        select=True,
    )
    mirrored = _run_mgcv_predict_on_newdata(
        mirrored_data,
        mirrored_newdata,
        FORMULA,
        family="gammals",
        method="ML",
        type=pred_type,
        return_se=True,
        select=True,
    )
    print(
        f"  {pred_type:8s} pred original={_max_abs_difference(actual_pred, original['pred']):.3e} "
        f"mirrored={_max_abs_difference(actual_pred, mirrored['pred']):.3e}"
    )
    print(
        f"  {pred_type:8s} se   original={_max_abs_difference(actual_se, original['se']):.3e} "
        f"mirrored={_max_abs_difference(actual_se, mirrored['se']):.3e}"
    )
