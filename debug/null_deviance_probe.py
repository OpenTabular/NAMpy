"""Verify nampy.gam.inference.null_deviance against mgcv fit$null.deviance."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/home/ad32/projects/package/NAMpy")

import numpy as np
import pandas as pd

from nampy.gam import GAM
from nampy.gam.inference.null_deviance import null_deviance

rng = np.random.default_rng(3)
n = 120
x0 = rng.uniform(size=n)
off = rng.uniform(-0.2, 0.2, size=n)
w = rng.uniform(0.5, 1.5, size=n)

base = pd.DataFrame({"x0": x0, "off": off, "w": w})

cases = []

d = base.copy()
d["y"] = np.sin(2 * np.pi * x0) + 0.1 * rng.standard_normal(n)
cases.append(("gaussian_plain", d, 'y ~ s(x0, bs="cr", k=8)', "gaussian",
              'gam(y ~ s(x0, bs="cr", k=8), data=d, method="REML")', {}))

d = base.copy()
d["y"] = rng.poisson(np.exp(0.5 + np.sin(2 * np.pi * x0) + off))
cases.append(("poisson_offset", d, 'y ~ offset(off) + s(x0, bs="cr", k=8)',
              "poisson",
              'gam(y ~ offset(off) + s(x0, bs="cr", k=8), data=d, '
              'family=poisson(), method="REML")', {}))

d = base.copy()
mu = np.exp(0.4 + np.sin(2 * np.pi * x0))
theta = 1.2
d["y"] = rng.negative_binomial(theta, theta / (theta + mu))
cases.append(("negbin_est", d, 'y ~ s(x0, bs="cr", k=8)',
              {"name": "negbin", "theta": 1.5, "estimate_theta": True},
              'gam(y ~ s(x0, bs="cr", k=8), data=d, family=nb(), '
              'method="REML")', {}))

d = base.copy()
d["y"] = np.sin(2 * np.pi * x0) + 0.2 * rng.standard_normal(n)
cases.append(("gaulss", d, ['y ~ s(x0, bs="cr", k=6)', "~ 1"], "gaulss",
              'gam(list(y ~ s(x0, bs="cr", k=6), ~ 1), data=d, '
              'family=gaulss(), method="REML")', {}))

d = base.copy()
d["y"] = rng.gamma(shape=3.0, scale=np.exp(0.4 + 0.3 * x0) / 3.0)
cases.append(("gammals", d, ['y ~ s(x0, bs="cr", k=6)', "~ 1"], "gammals",
              'gam(list(y ~ s(x0, bs="cr", k=6), ~ 1), data=d, '
              'family=gammals(), method="REML")', {}))

d = base.copy()
d["y"] = rng.poisson(np.exp(0.5 + np.sin(2 * np.pi * x0)))
cases.append(("poisson_weights", d, 'y ~ s(x0, bs="cr", k=8)', "poisson",
              'gam(y ~ s(x0, bs="cr", k=8), data=d, family=poisson(), '
              'weights=w, method="REML")', {"sample_weight": "w"}))

for name, d, formula, family, r_fit, fit_kwargs in cases:
    r_code = f"""
library(mgcv)
d <- read.csv(commandArgs(TRUE)[1])
b <- {r_fit}
cat(jsonlite::toJSON(list(nd = as.numeric(b$null.deviance)), digits=I(15)))
"""
    with tempfile.TemporaryDirectory() as tmp:
        csv = Path(tmp) / "d.csv"
        d.to_csv(csv, index=False)
        rf = Path(tmp) / "p.R"
        rf.write_text(r_code)
        res = subprocess.run(
            ["Rscript", str(rf), str(csv)], capture_output=True, text=True
        )
        if res.returncode != 0:
            print(f"{name}: R error {res.stderr[-300:]}")
            continue
        r_nd = float(json.loads(res.stdout)["nd"][0])

    gam = GAM(
        formula=formula,
        family=family,
        optimize_smoothing=True,
        smoothing_method="REML",
    )
    kw = {}
    if "sample_weight" in fit_kwargs:
        kw["sample_weight"] = np.asarray(d[fit_kwargs["sample_weight"]], float)
    try:
        gam.fit(data=d, **kw)
        nd = null_deviance(gam)
        rel = abs(nd - r_nd) / max(abs(r_nd), 1.0)
        print(f"{name:16s} nampy={nd:.10f} mgcv={r_nd:.10f} rel={rel:.2e}")
    except Exception as e:
        print(f"{name:16s} NAMpy FAILED: {type(e).__name__}: {e}")
