"""Localize the fs null-space penalty ordering swap vs mgcv.

Case: s(f, x, bs="fs", xt=list(bs="ps", m=2, k=7)) on the lifecycle data.
Upstream contract (mgcv/R/smooth.r:2067-2075): null penalty i penalizes
nat.param(type=1) output column rank+i; the column order is R
eigen(RSR, symmetric=TRUE) descending. Compare the two null columns and the
raw RSR null eigenvalues between NAMpy and R.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/home/ad32/projects/package/NAMpy")

import numpy as np

from nampy.gam.smooths.categorical.fs import _penalty_rank_from_base_term
from nampy.splines.basis.natparam import (
    _r_linpack_qr,
    _r_symmetric_eigh_descending,
    _r_triangular_solve,
    nat_param_type1,
)
from tests._optimization_lifecycle_registry import _coverage_make_fs_lifecycle_data
from tests.smooths.test_mgcv_raw_constructor_parity import _build_runtime_term

data = _coverage_make_fs_lifecycle_data()
term, _X, _ = _build_runtime_term(
    data, 'y ~ s(f, x, bs="fs", xt=list(bs="ps", m=2, k=7))'
)
B0, S0, _unused = term._base_constructor_fit_matrices()
B0 = __import__("numpy").asarray(B0, dtype=float)
S0 = __import__("numpy").asarray(S0, dtype=float)
rank = int(_penalty_rank_from_base_term(term._base_term, B0, S0))
print("base X shape", B0.shape, "rank", rank)
rp = nat_param_type1(B0, S0, rank=rank, unit_fnorm=True)
Xn = np.asarray(rp["X"], dtype=np.float64)

# Raw RSR null eigenvalues on the NAMpy path
Q, R = _r_linpack_qr(B0, np.finfo(float).eps ** 0.8)
tmp = _r_triangular_solve(R.T, S0.T, lower=True)
RSR = _r_triangular_solve(R.T, tmp.T, lower=True)
evals, U = _r_symmetric_eigh_descending(RSR)
print("nampy eigenvalues (descending):", evals)
print("nampy null evals:", evals[rank:])

r_code = """
library(mgcv)
d <- commandArgs(TRUE)[1]
B0 <- as.matrix(read.csv(file.path(d, "B0.csv"), header=FALSE))
S0 <- as.matrix(read.csv(file.path(d, "S0.csv"), header=FALSE))
rank <- as.integer(commandArgs(TRUE)[2])
qrx <- qr(B0, tol=.Machine$double.eps^.8)
R <- qr.R(qrx)
RSR <- forwardsolve(t(R), t(forwardsolve(t(R), t(S0))))
er <- eigen(RSR, symmetric=TRUE)
rp <- mgcv:::nat.param(B0, S0, rank=rank, type=1, unit.fnorm=TRUE)
write.table(rp$X, file.path(d, "Xn.csv"), row.names=FALSE, col.names=FALSE, sep=",")
cat(jsonlite::toJSON(list(evals = er$values), digits=I(18)))
"""
with tempfile.TemporaryDirectory() as tmpd:
    tp = Path(tmpd)
    np.savetxt(tp / "B0.csv", B0, delimiter=",", fmt="%.17g")
    np.savetxt(tp / "S0.csv", S0, delimiter=",", fmt="%.17g")
    rf = tp / "p.R"
    rf.write_text(r_code)
    res = subprocess.run(
        ["Rscript", str(rf), str(tp), str(rank)], capture_output=True, text=True
    )
    if res.returncode != 0:
        print("R error:", res.stderr[-500:])
        sys.exit(1)
    out = json.loads(res.stdout)
    r_evals = np.asarray(out["evals"], dtype=np.float64)
    r_Xn = np.loadtxt(tp / "Xn.csv", delimiter=",", ndmin=2)

print("mgcv eigenvalues (descending):", r_evals)
print("mgcv null evals:", r_evals[rank:])

n_null = Xn.shape[1] - rank
print("\nnull-column cross correlations (nampy cols x mgcv cols):")
A = Xn[:, rank:]
B = r_Xn[:, rank:]
corr = (A.T @ B) / (
    np.linalg.norm(A, axis=0)[:, None] * np.linalg.norm(B, axis=0)[None, :]
)
print(np.round(corr, 6))
print("\npenalized-column max |diff| after sign align:")
Ap = Xn[:, :rank]
Bp = r_Xn[:, :rank]
signs = np.sign(np.sum(Ap * Bp, axis=0))
signs[signs == 0] = 1.0
print(float(np.max(np.abs(Ap * signs - Bp))))
