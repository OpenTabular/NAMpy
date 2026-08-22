"""Probe: is the cs fixed-sp GCV criterion gap fully explained by the penalty?

tests/families/test_fixed_sp_input_matrix_mgcv_parity.py::
test_fixed_sp_family_matrix_derivatives_match_mgcv[gaussian_cs_single] shows a
0.67%-relative GCV gap at mgcv's optimized sp for ``s(x0, bs="cs", k=8)``.
The cs shrinkage penalty is chaotic in the eigensolver's resolution of the cr
penalty's two near-zero eigenvectors (see cs_shrinkage_null_space_probe.py).

This probe reruns NAMpy's criterion with R's *exact* unscaled shrunk penalty
monkeypatched into the cs construction. If the criterion then matches mgcv to
tight tolerance, the entire gap is the platform-indeterminate null-space
orientation; any remainder would be a genuine criterion-path bug.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

cr_term_module = importlib.import_module("nampy.gam.smooths.univariate.cr")

from nampy.gam import GAM  # noqa: E402
from nampy.gam.fit.selection.criteria.dispatch import (  # noqa: E402
    criterion_gradient,
    criterion_value,
)

R_SCRIPT = r"""
args <- commandArgs(trailingOnly = TRUE)
library(mgcv)
d <- read.csv(args[1])
fam <- if (args[3] == "poisson") poisson() else gaussian()
b <- gam(y ~ s(x0, bs = "cs", k = 8), data = d, family = fam,
         method = "GCV.Cp")
sm <- smoothCon(s(x0, bs = "cs", k = 8), data = d, knots = NULL,
                absorb.cons = FALSE, scale.penalty = FALSE)[[1]]
out <- list(sp = b$sp, gcv = b$gcv.ubre, S_unscaled = sm$S[[1]])
writeLines(jsonlite::toJSON(out, digits = 17), args[2])
"""


def run_case(family: str, data: pd.DataFrame) -> None:
    print(f"--- family={family}")
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "d.csv"
        json_path = Path(tmp) / "out.json"
        r_path = Path(tmp) / "probe.R"
        data.to_csv(csv_path, index=False)
        r_path.write_text(R_SCRIPT)
        subprocess.run(
            ["Rscript", str(r_path), str(csv_path), str(json_path), family],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(json_path.read_text())

    sp = np.atleast_1d(np.asarray(payload["sp"], dtype=np.float64))
    gcv_r = float(np.atleast_1d(payload["gcv"])[0])
    S_r_unscaled = np.asarray(payload["S_unscaled"], dtype=np.float64)
    log_sp = np.log(sp)
    print("mgcv sp:", sp, " mgcv gcv:", gcv_r)

    def criterion_with_current_construction() -> tuple[float, np.ndarray]:
        gam = GAM(
            family=family,
            formula='y ~ s(x0, bs="cs", k=8)',
            optimize_smoothing=False,
            smoothing_method="fixed",
            smoothing_params=sp,
        ).fit(data=data)
        value = float(criterion_value(gam, gam.y_, log_sp, method="gcv.cp"))
        grad = np.asarray(
            criterion_gradient(gam, gam.y_, log_sp, method="gcv.cp"),
            dtype=np.float64,
        )
        return value, grad

    own_value, own_grad = criterion_with_current_construction()
    print("nampy criterion (own cs penalty):", own_value, " grad:", own_grad)

    original = cr_term_module.add_full_rank_shrinkage

    def r_exact_shrinkage(S, shrink=0.1, **kwargs):
        S = np.asarray(S, dtype=np.float64)
        if S.shape == S_r_unscaled.shape:
            base = 0.5 * (S + S.T)
            # Only substitute when this is the raw cr penalty of our term
            # (same matrix up to the shrink addition).
            if np.max(np.abs(base - 0.5 * (S_r_unscaled + S_r_unscaled.T))) < np.max(
                np.abs(base)
            ):
                return S_r_unscaled.copy()
        return original(S, shrink=shrink, **kwargs)

    cr_term_module.add_full_rank_shrinkage = r_exact_shrinkage
    try:
        crit_injected, grad_injected = criterion_with_current_construction()
    finally:
        cr_term_module.add_full_rank_shrinkage = original

    print(
        "nampy criterion (R-exact cs penalty):",
        crit_injected,
        " grad:",
        grad_injected,
    )
    print("gap vs mgcv with R-exact penalty:", abs(crit_injected - gcv_r))


def main() -> None:
    rng = np.random.default_rng(2024)
    n = 220
    x0 = rng.uniform(-2.0, 2.0, size=n)
    x1 = rng.uniform(-1.5, 1.5, size=n)
    y = np.sin(1.2 * x0) + 0.4 * x1**2 + rng.normal(scale=0.15, size=n)
    run_case("gaussian", pd.DataFrame({"y": y, "x0": x0, "x1": x1}))

    rng = np.random.default_rng(2039)
    n = 200
    x0 = rng.normal(size=n)
    x1 = rng.normal(size=n)
    mu = np.exp(0.2 + 0.7 * np.sin(x0) - 0.25 * x1)
    y = rng.poisson(mu)
    run_case("poisson", pd.DataFrame({"y": y, "x0": x0, "x1": x1}))


if __name__ == "__main__":
    main()
