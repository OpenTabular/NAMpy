"""Gaussian noncanonical-link ML/REML probe.

Checks that the profiled-scale criterion value/gradient/Hessian now routed
through dispatch agree with the joint (log sp, log phi) path at its optimum,
and that the EFS branch produces the same REML score/sp as mgcv's efs.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/home/ad32/projects/package/NAMpy")

import numpy as np
import pandas as pd

from nampy.gam import GAM
from nampy.gam.fit.selection.criteria.dispatch import (
    criterion_gradient,
    criterion_hessian,
    criterion_value,
)

rng = np.random.default_rng(7)
n = 150
x0 = rng.uniform(size=n)
mu = np.exp(0.4 + np.sin(2.0 * np.pi * x0))
y = mu + 0.15 * mu * rng.standard_normal(n)
y = np.maximum(y, 1e-3)
data = pd.DataFrame({"y": y, "x0": x0})

for method in ("REML", "ML"):
    gam = GAM(
        formula='y ~ s(x0, bs="cr", k=8)',
        family={"name": "gaussian", "link": "log"},
        optimize_smoothing=True,
        smoothing_method=method,
        smoothing_optimizer="outer_newton",
    )
    gam.fit(data=data)
    log_sp = np.log(np.asarray(gam.smoothing_params, dtype=np.float64))
    val = criterion_value(gam, gam.y_, log_sp, method=method.lower())
    grad = criterion_gradient(gam, gam.y_, log_sp, method=method.lower())
    hess = criterion_hessian(gam, gam.y_, log_sp, method=method.lower())
    print(
        f"{method}: newton score={gam.smoothing_score_:.8f} "
        f"profiled value={val:.8f} |grad|={np.max(np.abs(grad)):.2e} "
        f"hess_finite={np.all(np.isfinite(hess))}"
    )

efs = GAM(
    formula='y ~ s(x0, bs="cr", k=8)',
    family={"name": "gaussian", "link": "log"},
    optimize_smoothing=True,
    smoothing_method="REML",
    smoothing_optimizer="efs",
)
efs.fit(data=data)
print(
    f"EFS: sp={float(efs.smoothing_params[0]):.7f} "
    f"score={efs.smoothing_score_:.8f}"
)

r_code = """
library(mgcv)
d <- read.csv(commandArgs(TRUE)[1])
f1 <- gam(y ~ s(x0, bs="cr", k=8), data=d, family=gaussian(link="log"),
          method="REML", optimizer="efs")
f2 <- gam(y ~ s(x0, bs="cr", k=8), data=d, family=gaussian(link="log"),
          method="REML")
f3 <- gam(y ~ s(x0, bs="cr", k=8), data=d, family=gaussian(link="log"),
          method="ML")
cat(sprintf("mgcv efs: sp=%.7f score=%.8f\\n", f1$sp, f1$gcv.ubre))
cat(sprintf("mgcv newton REML: sp=%.7f score=%.8f scale=%.8f\\n",
            f2$sp, f2$gcv.ubre, f2$scale))
cat(sprintf("mgcv newton ML: sp=%.7f score=%.8f\\n", f3$sp, f3$gcv.ubre))
"""
with tempfile.TemporaryDirectory() as tmp:
    csv = Path(tmp) / "d.csv"
    data.to_csv(csv, index=False)
    rfile = Path(tmp) / "probe.R"
    rfile.write_text(r_code)
    out = subprocess.run(
        ["Rscript", str(rfile), str(csv)], capture_output=True, text=True
    )
    print(out.stdout.strip())
    if out.returncode != 0:
        print(out.stderr[-500:])
