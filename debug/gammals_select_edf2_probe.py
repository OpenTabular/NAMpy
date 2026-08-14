"""Decompose the gammals_select_true_cr edf2_total mismatch.

Prints, for both NAMpy and mgcv: sum(edf), sum(edf1), the raw
rowSums(Vc * crossprod(R)) sum before the cap, and whether the
`sum(edf2) > sum(edf1) -> edf2 <- edf1` replacement (gam.fit4.r:1715) fired.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/home/ad32/projects/package/NAMpy")

import numpy as np

import nampy.gam.fit.solvers.general_family.newton as gnewton
from nampy.gam.model.api import GAM
from tests.families.test_general_family_mgcv_parity import _gammals_data

data = _gammals_data()
captured = {}

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
        formula=['y ~ s(x, bs="cr", k=6)', "~ 1"],
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
