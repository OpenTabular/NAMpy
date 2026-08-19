"""Probe: mgcv reTest p-value in the saturated gaussian_fs_select_reml case.

Case: y ~ s(f, x, bs="fs", k=6), gaussian, REML, select=TRUE on the noiseless
18-row _make_fs_data() frame (tests/parity/
test_mgcv_prediction_inference_diagnostics_parity.py). The smooth interpolates:
edf ~ 17, sig2 ~ 8e-11, residual df ~ 3e-5, reTest statistic ~ 1e11.

mgcv computes the p-value with psum.chisq (mgcv/R/mgcv.r:3466-3498), i.e. the
Davies (1980) qfc C routine at tol=2e-5, on the mixture
    P(sum_i ev_i*chisq_1 - (stat/k)*chisq_k > 0),   k = max(1, round(rdf)) = 1
with ev_i ~ 1 and stat ~ 1.07e11. The mathematically correct tail is
    ~ sqrt(2/(pi*stat)) * E[sqrt(chisq_17)] ~ 9.9e-6,
which is exactly what NAMpy's port returns. R's C routine instead returns 0.5
with ifault=0, and feeding it the same inputs truncated to 11 significant
digits flips the answer to 0.0 — the value is a numerical artifact, unstable
under last-bit input changes, and not a reproducible parity target.

The anova parity test therefore skips strict p-value comparison for rows whose
statistic exceeds Davies' resolution (|stat| >= 1e8) and only requires the
NAMpy p-value to be small.

Run this file to re-derive both sides (requires R + mgcv).
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from nampy.gam.inference.chi_square_mixtures import psum_chisq

R_SCRIPT = r"""
args <- commandArgs(trailingOnly = TRUE)
library(mgcv)
d <- read.csv(args[1], stringsAsFactors = TRUE)
b <- gam(y ~ s(f, x, bs = "fs", k = 6), data = d, method = "REML",
         select = TRUE)
res <- mgcv:::reTest(b, 1)
cat("mgcv stat:", sprintf("%.10e", res$stat), "\n")
cat("mgcv rank:", res$rank, "\n")
cat("mgcv pval:", sprintf("%.10e", res$pval), "\n")
cat("sig2:", sprintf("%.6e", b$sig2), " df.residual:",
    sprintf("%.6e", b$df.residual), "\n")
"""


def main() -> None:
    levels = ["a", "b", "c"]
    n = 18
    f = np.array([levels[i % len(levels)] for i in range(n)], dtype=object)
    x = np.linspace(0.1, 1.7, n)
    offsets = {"a": 1.0, "b": -0.5, "c": 0.9}
    y = 0.4 * np.sin(2 * x) + np.array([offsets[v] for v in f])
    data = pd.DataFrame({"y": y, "f": f, "x": x})

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "d.csv"
        r_path = Path(tmp) / "probe.R"
        data.to_csv(csv_path, index=False)
        r_path.write_text(R_SCRIPT)
        out = subprocess.run(
            ["Rscript", str(r_path), str(csv_path)],
            check=True,
            capture_output=True,
            text=True,
        )
    print(out.stdout)

    # NAMpy's Davies port on representative inputs from the R fit: 17 unit
    # weights against one astronomically negative weight.
    ev = np.ones(17, dtype=np.float64)
    stat = 1.0674102103e11
    lb = np.concatenate([ev, [-stat]])
    df = np.concatenate([np.ones(17, dtype=np.int64), [1]])
    p = float(psum_chisq(0.0, lb, df=df))
    print("nampy psum_chisq on the same mixture:", p)
    # Closed-form sanity check of the tail for stat >> sum(ev).
    approx = np.sqrt(2.0 / (np.pi * stat)) * np.sqrt(16.5)
    print("analytic approximation sqrt(2/(pi*stat))*E[sqrt(chisq_17)]:", approx)


if __name__ == "__main__":
    main()
