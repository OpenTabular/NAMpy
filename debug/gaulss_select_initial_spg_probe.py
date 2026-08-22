"""Classify the gaulss_select_true_cr endpoint as eigen-sign-sensitive.

Evidence chain:
1. The NAMpy-vs-mgcv trace divergence must already be present in the
   iteration-0 log smoothing parameters (i.e., it originates in initial.spg,
   not in the Newton updates).
2. Both final endpoints must be equivalent optima of the same criterion
   surface: NAMpy's REML/LAML objective evaluated at each endpoint agrees, and
   the criterion gradient is ~0 at both.
3. The initial.spg output must be orientation-sensitive inside R itself: a
   pure row permutation or mirrored basis (mathematically equivalent fits)
   changes mgcv's own initial/optimized sp. Base R leaves the real signs of
   the DSYEVR eigenvectors used by Sl.setup unspecified, so the optimized
   endpoint must be checked by its flat-tail invariant rather than by one
   build's raw smoothing parameter.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/home/ad32/projects/package/NAMpy")

import numpy as np

from nampy.gam import GAM
from nampy.gam.fit.selection.criteria.dispatch import (
    criterion_gradient,
    criterion_value,
)
from tests.families.test_general_family_mgcv_parity import (
    GAULSS_FORMULA,
    _gaulss_data,
)
from tests.mgcv_parity_utils import _run_mgcv_snapshot
from tests.optimization.test_mgcv_outer_optimization_parity import (
    _run_mgcv_outer_trace,
)
from tests.optimization.test_mgcv_postprocessing_final_fit_parity import (
    _serialize_actual_final_fit,
    _serialize_expected_final_fit,
)

data = _gaulss_data()

gam = GAM(
    formula=GAULSS_FORMULA,
    family="gaulss",
    select=True,
    optimize_smoothing=True,
    smoothing_method="ML",
    smoothing_optimizer="outer_newton",
)
gam.fit(data=data)
nam_rows = list(getattr(gam, "_optim_trace", []) or [])

mgcv_trace = _run_mgcv_outer_trace(
    data, str(GAULSS_FORMULA), "gaulss", "ML", "newton", select=True
)
mgcv_rows = mgcv_trace["trace"]

nam0 = np.asarray(nam_rows[0]["log_sp"], dtype=np.float64)
mg0 = np.asarray(mgcv_rows[0]["log_sp"], dtype=np.float64)
print("1) iteration-0 lsp")
print("   nampy :", np.array2string(nam0, precision=8))
print("   mgcv  :", np.array2string(mg0, precision=8))
print("   max|diff| at iter 0:", float(np.max(np.abs(nam0 - mg0))))

nam_end = np.log(np.asarray(gam.smoothing_params, dtype=np.float64))
mg_end = np.asarray(mgcv_rows[-1]["log_sp"], dtype=np.float64)
print("2) endpoints")
print("   nampy end:", np.array2string(nam_end, precision=8))
print("   mgcv end :", np.array2string(mg_end, precision=8))
val_nam = criterion_value(gam, gam.y_, nam_end, method="reml")
val_mg = criterion_value(gam, gam.y_, mg_end, method="reml")
grad_nam = criterion_gradient(gam, gam.y_, nam_end, method="reml")
grad_mg = criterion_gradient(gam, gam.y_, mg_end, method="reml")
print(f"   criterion at nampy end: {val_nam:.10f} |grad| {np.max(np.abs(grad_nam)):.2e}")
print(f"   criterion at mgcv end : {val_mg:.10f} |grad| {np.max(np.abs(grad_mg)):.2e}")
print(f"   criterion difference  : {abs(val_nam - val_mg):.3e}")

actual_vc = np.asarray(gam.vcov(unconditional=True), dtype=np.float64)
expected_snapshot = _run_mgcv_snapshot(
    data,
    GAULSS_FORMULA,
    "gaulss",
    "ML",
    select=True,
)
expected_vc = np.asarray(expected_snapshot["fit"]["cov_unconditional"], dtype=np.float64)
lpmatrix = np.asarray(gam.predict(data, type="lpmatrix"), dtype=np.float64)
actual_var = np.sum((lpmatrix @ actual_vc) * lpmatrix, axis=1)
expected_var = np.sum((lpmatrix @ expected_vc) * lpmatrix, axis=1)
print("   max|Vc diagonal diff|:", float(np.max(np.abs(np.diag(actual_vc) - np.diag(expected_vc)))))
print("   max|fitted variance diff|:", float(np.max(np.abs(actual_var - expected_var))))
print("   max relative fitted variance diff:", float(np.max(np.abs(actual_var - expected_var) / np.maximum(np.abs(expected_var), 1e-15))))

actual_final = _serialize_actual_final_fit(
    gam,
    [],
    allow_synthetic_outer_info=False,
)
expected_final = _serialize_expected_final_fit(expected_snapshot)
for key in ("Vp", "Ve"):
    print(
        f"   max|{key} diagonal diff|:",
        float(
            np.max(
                np.abs(
                    np.diag(np.asarray(actual_final[key], dtype=np.float64))
                    - np.diag(np.asarray(expected_final[key], dtype=np.float64))
                )
            )
        ),
    )
for key in ("edf_by_term", "edf_total", "edf2_total", "trace_H", "scale", "aic"):
    actual_value = np.asarray(actual_final[key], dtype=np.float64)
    expected_value = np.asarray(expected_final[key], dtype=np.float64)
    print(f"   max|{key} diff|:", float(np.max(np.abs(actual_value - expected_value))))
actual_outer = actual_final["outer_info"]
expected_outer = expected_final["outer_info"]
for key in ("grad", "hess"):
    print(
        f"   max|outer_info {key} diff|:",
        float(
            np.max(
                np.abs(
                    np.asarray(actual_outer[key], dtype=np.float64)
                    - np.asarray(expected_outer[key], dtype=np.float64)
                )
            )
        ),
    )

r_code = """
library(mgcv)
args <- commandArgs(TRUE)
d <- read.csv(args[1])
fml <- list(y ~ s(x, bs="cr", k=6), ~ 1)
f1 <- gam(fml, data=d, family=gaulss(), method="ML", select=TRUE)
set.seed(3)
perm <- sample(nrow(d))
d2 <- d[perm, , drop=FALSE]
f2 <- gam(fml, data=d2, family=gaulss(), method="ML", select=TRUE)
d3 <- d
d3$x <- -d3$x
f3 <- gam(fml, data=d3, family=gaulss(), method="ML", select=TRUE)
out <- list(sp1 = as.numeric(f1$sp), sp2 = as.numeric(f2$sp),
            sp3 = as.numeric(f3$sp),
            score1 = as.numeric(f1$gcv.ubre), score2 = as.numeric(f2$gcv.ubre),
            score3 = as.numeric(f3$gcv.ubre))
cat(jsonlite::toJSON(out, digits=I(15)))
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
        print("R error:", res.stderr[-500:])
    else:
        out = json.loads(res.stdout)
        sp1 = np.log(np.asarray(out["sp1"], dtype=np.float64))
        sp2 = np.log(np.asarray(out["sp2"], dtype=np.float64))
        print("3) mgcv row-permutation sensitivity (same model, permuted rows)")
        print("   log sp original:", np.array2string(sp1, precision=8))
        print("   log sp permuted:", np.array2string(sp2, precision=8))
        print("   max|log sp diff| within R:", float(np.max(np.abs(sp1 - sp2))))
        print(
            "   score orig/perm:",
            f"{out['score1'][0]:.10f}",
            f"{out['score2'][0]:.10f}",
        )
        sp3 = np.log(np.asarray(out["sp3"], dtype=np.float64))
        print("4) mgcv x -> -x sensitivity (equivalent model, mirrored basis)")
        print("   log sp mirrored:", np.array2string(sp3, precision=8))
        print("   max|log sp diff| vs original:", float(np.max(np.abs(sp1 - sp3))))
        print("   score mirrored:", f"{out['score3'][0]:.10f}")
        print(
            "   nampy-vs-mgcv endpoint max|log sp diff|:",
            float(np.max(np.abs(nam_end - mg_end))),
        )
