"""Probe mgcv's rank-deficient Gaussian representative vs NAMpy.

Exactly aliased parametric columns (z == x1) force a genuine column drop.
Records mgcv's coefficient gauge (zeros at dropped canonical coordinates per
mgcv/src/gdi.c:2253-2292), Vp diagonal, rank, and predictions, then compares
NAMpy's current output.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/home/ad32/projects/package/NAMpy")

import numpy as np
import pandas as pd

from nampy.gam.model.api import GAM

rng = np.random.default_rng(42)
n = 90
x0 = rng.uniform(size=n)
x1 = rng.uniform(size=n)
data = pd.DataFrame({"x0": x0, "x1": x1, "z": x1})
data["y"] = np.sin(2.0 * np.pi * x0) + 0.5 * x1 + 0.1 * rng.standard_normal(n)

r_code = """
library(mgcv)
d <- read.csv(commandArgs(TRUE)[1])
b <- gam(y ~ x1 + z + s(x0, bs="cr", k=8), data=d, family=gaussian(),
         method="REML")
out <- list(coef = as.numeric(coef(b)), names = names(coef(b)),
            vp_diag = as.numeric(diag(b$Vp)), rank = as.integer(b$rank),
            np = length(coef(b)), sp = as.numeric(b$sp),
            fitted_head = as.numeric(head(fitted(b), 8)),
            edf_total = sum(b$edf))
cat(jsonlite::toJSON(out, digits=I(15)))
"""
with tempfile.TemporaryDirectory() as tmp:
    csv = Path(tmp) / "d.csv"
    data.to_csv(csv, index=False)
    rf = Path(tmp) / "p.R"
    rf.write_text(r_code)
    res = subprocess.run(["Rscript", str(rf), str(csv)], capture_output=True, text=True)
    if res.returncode != 0:
        print("R error:", res.stderr[-800:])
        sys.exit(1)
    out = json.loads(res.stdout)

print("mgcv:")
print("  names   :", out["names"])
print("  coef    :", np.round(np.asarray(out["coef"], float), 6))
print("  vp_diag :", np.round(np.asarray(out["vp_diag"], float), 8))
print("  rank/np :", out["rank"], "/", out["np"])
print("  sp      :", out["sp"])
print("  edf     :", out["edf_total"])
print("  fitted  :", np.round(np.asarray(out["fitted_head"], float), 6))

gam = GAM(
    formula='y ~ x1 + z + s(x0, bs="cr", k=8)',
    family="gaussian",
    optimize_smoothing=True,
    smoothing_method="REML",
    smoothing_optimizer="outer_newton",
)
try:
    gam.fit(data=data)
    fr = gam.fit_result()
    coef = np.asarray(fr.coef_full, float)
    vp = np.asarray(fr.cov_bayes, float)
    print("NAMpy:")
    print("  coef    :", np.round(coef, 6))
    print("  vp_diag :", np.round(np.diag(vp), 8))
    print("  sp      :", np.asarray(gam.smoothing_params, float))
    print("  edf     :", fr.edf_total)
    print("  fitted  :", np.round(gam.predict(data)[:8], 6))
except Exception as e:
    print("NAMpy FAILED:", type(e).__name__, e)
