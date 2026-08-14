"""Compare summary_gam numbers against R summary.gam for matched fits."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/home/ad32/projects/package/NAMpy")

import numpy as np
import pandas as pd

from nampy.gam.model.api import GAM

rng = np.random.default_rng(11)
n = 120
x0 = rng.uniform(size=n)
x1 = rng.uniform(size=n)
fac = rng.choice(["a", "b", "c"], size=n)
data = pd.DataFrame({"x0": x0, "x1": x1, "fac": fac})
data["y"] = (
    np.sin(2.0 * np.pi * x0)
    + 0.4 * x1
    + np.where(fac == "b", 0.5, 0.0)
    + 0.15 * rng.standard_normal(n)
)

r_code = """
library(mgcv)
d <- read.csv(commandArgs(TRUE)[1], stringsAsFactors = TRUE)
b <- gam(y ~ fac + x1 + s(x0, bs="cr", k=8), data=d, method="REML")
s <- summary(b)
out <- list(p_table = unname(as.matrix(s$p.table)),
            p_names = rownames(s$p.table),
            s_table = unname(as.matrix(s$s.table)),
            r_sq = s$r.sq, dev_expl = s$dev.expl,
            residual_df = s$residual.df, scale = s$scale,
            n = s$n, np = s$np, method = as.character(s$method),
            sp_criterion = as.numeric(s$sp.criterion),
            null_deviance = b$null.deviance)
cat(jsonlite::toJSON(out, digits=I(15)))
"""
with tempfile.TemporaryDirectory() as tmp:
    csv = Path(tmp) / "d.csv"
    data.to_csv(csv, index=False)
    rf = Path(tmp) / "p.R"
    rf.write_text(r_code)
    res = subprocess.run(["Rscript", str(rf), str(csv)], capture_output=True, text=True)
    if res.returncode != 0:
        print("R error:", res.stderr[-500:])
        sys.exit(1)
    out = json.loads(res.stdout)

gam = GAM(
    formula='y ~ fac + x1 + s(x0, bs="cr", k=8)',
    family="gaussian",
    optimize_smoothing=True,
    smoothing_method="REML",
)
gam.fit(data=data)
from nampy.gam.inference.summary import summary_gam  # noqa: E402

s = summary_gam(gam)

r_pt = np.asarray(out["p_table"], dtype=np.float64)
py_pt = s.p_table.to_numpy(dtype=np.float64)
print("p_table max|diff|      :", np.max(np.abs(py_pt - r_pt)))
print("p_names R              :", out["p_names"])
print("p_names nampy          :", list(s.p_table.index))
r_st = np.asarray(out["s_table"], dtype=np.float64)
py_st = s.s_table[["edf", "ref_df", "wald_stat", "p_value"]].to_numpy(np.float64)
print("s_table max|diff|      :", np.max(np.abs(py_st - r_st)))
for name, mine, theirs in [
    ("r_sq", s.r_sq, out["r_sq"][0]),
    ("dev_expl", s.dev_expl, out["dev_expl"][0]),
    ("residual_df", s.residual_df, out["residual_df"][0]),
    ("scale", s.scale, out["scale"][0]),
    ("null_dev", s.null_deviance, out["null_deviance"][0]),
    ("sp_criterion", s.sp_criterion, out["sp_criterion"][0]),
]:
    print(f"{name:14s} nampy={mine!r} mgcv={theirs!r} "
          f"rel={abs(mine - theirs) / max(abs(theirs), 1e-12):.2e}")
print("n", s.n, out["n"][0], " np", s.np, out["np"][0],
      " method", s.method, out["method"][0])
