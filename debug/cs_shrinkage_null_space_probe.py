"""Probe: is the mgcv `cs` shrinkage penalty platform-determined?

mgcv/R/smooth.r::smooth.construct.cr.smooth.spec() builds the cs penalty by
eigen-decomposing the rank-(k-2) cr penalty and assigning *different* shrunk
eigenvalues to the two null-space directions:

    es$values[nk-1] <- es$values[nk-2]*shrink      # 0.1 * lambda_min_pos
    es$values[nk]   <- es$values[nk-1]*shrink      # 0.01 * lambda_min_pos

The two "null" eigenvalues of the raw penalty are numerically ~1e-15 noise, so
which orthonormal basis of the 2-d null space the eigensolver returns (and how
it orders the two vectors) is LAPACK-implementation dependent. Because the two
directions receive different penalties, the reconstructed S — and therefore
fitted coefficients/predictions at fixed sp — legitimately differ between R's
eigen() and scipy's eigh() (and between BLAS builds) at O(1e-5).

This probe builds the exact transformed_cs case from
tests/parity/test_mgcv_output_parity.py, compares NAMpy's shrunk penalty with
R smoothCon()'s, and decomposes the difference into range-space vs null-space
components.

Findings (2026-08-18):
- Raw cr penalties agree with R to machine epsilon; the smoothCon rescale
  factor for cs is correctly computed from the *shrunk* penalty on both sides.
- The shrunk penalty itself is chaotic in the last bits of its input: knots
  from np.quantile(..., method="linear") vs NAMpy's _r_quantile_type7_sorted
  differ by 1 ulp, and that alone moves the shrunk penalty by ~4e-5 relative
  (max abs 0.023 on a norm-531 matrix) because the eigensolver's resolution of
  the degenerate near-zero pair (eigenvalues ~1e-14 and ~-1e-16) rotates
  freely. R's own eigen() is equally sensitive, so mgcv's cs penalty — and
  fitted values at fixed sp (~2e-5 shifts) — are platform/BLAS-dependent.
- Consequently the transformed_cs prediction tolerance in
  tests/parity/test_mgcv_output_parity.py is ~1e-4-level, not 1e-10; the basis
  (lpmatrix) remains exact.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from nampy.gam.splines.basis.cr import cr_spl
from nampy.gam.splines.univariate.cr import add_full_rank_shrinkage

R_SCRIPT = r"""
args <- commandArgs(trailingOnly = TRUE)
library(mgcv)
d <- read.csv(args[1])
sm <- smoothCon(s(xt, bs = "cs", k = 8), data = d, knots = NULL,
                absorb.cons = FALSE)[[1]]
smr <- smoothCon(s(xt, bs = "cr", k = 8), data = d, knots = NULL,
                 absorb.cons = FALSE)[[1]]
out <- list(S_cs = sm$S[[1]], S_cr = smr$S[[1]], knots = sm$xp,
            S_scale_cs = sm$S.scale, S_scale_cr = smr$S.scale)
writeLines(jsonlite::toJSON(out, digits = 17), args[2])
"""


def main() -> None:
    rng = np.random.default_rng(551)
    n = 150
    x0 = rng.uniform(-2.0, 2.0, size=n)
    x1 = rng.uniform(-1.5, 1.5, size=n)
    y = np.sin(1.2 * x0) + 0.4 * x1**2 + rng.normal(scale=0.15, size=n)
    x = x0
    xt = x + 0.15 * x**2

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "d.csv"
        json_path = Path(tmp) / "out.json"
        r_path = Path(tmp) / "probe.R"
        pd.DataFrame({"y": y, "xt": xt}).to_csv(csv_path, index=False)
        r_path.write_text(R_SCRIPT)
        subprocess.run(
            ["Rscript", str(r_path), str(csv_path), str(json_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(json_path.read_text())

    S_cs_r = np.asarray(payload["S_cs"], dtype=np.float64)
    S_cr_r = np.asarray(payload["S_cr"], dtype=np.float64)
    knots_r = np.asarray(payload["knots"], dtype=np.float64)

    knots = np.quantile(np.unique(xt), np.linspace(0.0, 1.0, 8), method="linear")
    print("knot agreement (max abs):", np.max(np.abs(knots - knots_r)))

    _, S_cr, _, _ = cr_spl(xt, 8, knots=knots)
    S_cr = 0.5 * (S_cr + S_cr.T)
    # smoothCon applies scale.penalty; recover the scalar from the cr penalties
    # (cr scaling happens before shrinkage per upstream ordering).
    mask = np.abs(S_cr_r) > 1e-8
    scale = np.median(S_cr[mask] / S_cr_r[mask])
    print("smoothCon penalty scale factor:", scale)
    print(
        "raw cr penalty agreement after rescale (max abs):",
        np.max(np.abs(S_cr / scale - S_cr_r)),
    )

    S_cs = add_full_rank_shrinkage(S_cr / scale, shrink=0.1)
    diff = S_cs - S_cs_r
    print("shrunk cs penalty diff (max abs):", np.max(np.abs(diff)))

    # Decompose the disagreement: project onto the raw penalty's null space.
    w, V = np.linalg.eigh(S_cr_r)
    null = V[:, :2]  # 2-d null space of the rank-(k-2) cr penalty
    rng_sp = V[:, 2:]
    print(
        "diff restricted to null space (max abs):",
        np.max(np.abs(null.T @ diff @ null)),
    )
    print(
        "diff restricted to range space (max abs):",
        np.max(np.abs(rng_sp.T @ diff @ rng_sp)),
    )

    # The invariant part of the cs construction: total shrinkage penalty mass on
    # the null space is NOT rotation-invariant (0.1λ vs 0.01λ split), but the
    # trace over the null space is basis-invariant only if both eigensolvers
    # split the same way. Report the null-space blocks explicitly.
    print("R cs penalty, null-space block:\n", null.T @ S_cs_r @ null)
    print("NAMpy cs penalty, null-space block:\n", null.T @ S_cs @ null)

    # Both constructions only ever modify eigenvalues, so S_cs - S_cr must be
    # (a) the null-space shrink addition plus (b) any perturbation of the range
    # eigenpairs the round-trip reconstruction introduces. Compare the two
    # addition matrices directly.
    base = S_cr / scale
    add_r = S_cs_r - base
    add_n = S_cs - base
    print("R addition matrix norm:", np.linalg.norm(add_r))
    print("NAMpy addition matrix norm:", np.linalg.norm(add_n))
    print("addition matrices diff (max abs):", np.max(np.abs(add_r - add_n)))
    wr = np.linalg.eigvalsh(add_r)
    wn = np.linalg.eigvalsh(add_n)
    print("eigvals of R addition:", wr)
    print("eigvals of NAMpy addition:", wn)
    # How much of each addition lies outside the null space?
    print(
        "R addition range-space leakage (max abs):",
        np.max(np.abs(rng_sp.T @ add_r @ rng_sp)),
    )
    print(
        "NAMpy addition range-space leakage (max abs):",
        np.max(np.abs(rng_sp.T @ add_n @ rng_sp)),
    )

    # Production-order comparison: both sides shrink the UNSCALED penalty, then
    # rescale by norm-based factors computed from their own shrunk penalty.
    scale_cs_r = float(np.atleast_1d(payload["S_scale_cs"])[0])
    scale_cr_r = float(np.atleast_1d(payload["S_scale_cr"])[0])
    print("R S.scale (cs, cr):", scale_cs_r, scale_cr_r)
    S_cs_unscaled_r = S_cs_r * scale_cs_r
    S_cs_unscaled_n = add_full_rank_shrinkage(S_cr, shrink=0.1)
    d_unscaled = S_cs_unscaled_n - S_cs_unscaled_r
    lam_max = float(np.max(np.abs(np.linalg.eigvalsh(S_cr))))
    print("unscaled shrunk penalty diff (max abs):", np.max(np.abs(d_unscaled)))
    print("relative to lambda_max:", np.max(np.abs(d_unscaled)) / lam_max)
    print(
        "unscaled diff null-block:\n",
        null.T @ d_unscaled @ null,
    )
    print(
        "unscaled diff range-block (max abs):",
        np.max(np.abs(rng_sp.T @ d_unscaled @ rng_sp)),
    )

    # Replicate CubicSplineTerm._main_penalty(raw=True) for cs and compare with
    # R smoothCon's scaled S.
    from nampy.gam.penalties.algebra import penalty_rescale_factor, scale_penalty

    X_raw, S_raw_unscaled, _, _ = cr_spl(xt, 8)
    S_prod = scale_penalty(X_raw, add_full_rank_shrinkage(S_raw_unscaled, shrink=0.1))
    print(
        "NAMpy production scaled cs penalty vs R sm$S (max abs):",
        np.max(np.abs(S_prod - S_cs_r)),
    )
    print(
        "NAMpy rescale factor (cs production):",
        penalty_rescale_factor(
            X_raw, add_full_rank_shrinkage(S_raw_unscaled, shrink=0.1)
        ),
    )


def norms_check() -> None:
    rng = np.random.default_rng(551)
    n = 150
    x0 = rng.uniform(-2.0, 2.0, size=n)
    rng.uniform(-1.5, 1.5, size=n)
    x = x0
    xt = x + 0.15 * x**2
    X_raw, S_raw_unscaled, _, _ = cr_spl(xt, 8)
    S_shrunk = add_full_rank_shrinkage(S_raw_unscaled, shrink=0.1)
    from nampy.gam.linalg import r_matrix_norm_one

    print("np one-norm S_shrunk:", np.linalg.norm(S_shrunk, 1))
    print("np inf-norm S_shrunk:", np.linalg.norm(S_shrunk, np.inf))
    print("np one-norm S_raw:", np.linalg.norm(S_raw_unscaled, 1))
    print(
        "sym one-norm S_raw:",
        np.linalg.norm(0.5 * (S_raw_unscaled + S_raw_unscaled.T), 1),
    )
    print("r_matrix_norm_one(X)^2:", r_matrix_norm_one(X_raw) ** 2)
    print("np inf-norm X^2:", np.linalg.norm(X_raw, np.inf) ** 2)
    print(
        "ratio shrunk-one / Xinf2:",
        np.linalg.norm(S_shrunk, 1) / np.linalg.norm(X_raw, np.inf) ** 2,
    )


if __name__ == "__main__":
    main()
    norms_check()
