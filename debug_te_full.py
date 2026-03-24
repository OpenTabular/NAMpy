"""
Step-by-step comparison of Python te construction vs mgcv R.
"""
import numpy as np
import subprocess
import tempfile
import json
from pathlib import Path
import sys
sys.path.insert(0, 'tests')
from mgcv_parity_utils import _make_gaussian_data
from nampy.gam.smooths.univariate.cubic_regression import SplineTerm1D
from nampy.gam.basis.tensor import (
    rowwise_kronecker, tensor_product_penalties,
    normalize_tensor_marginal_penalty, rescale_tensor_penalties_for_fit,
)
from nampy.gam.constraints.absorption import full_term_sum_to_zero_constraint

data = _make_gaussian_data(seed=7, n=80)
X = data[['x0', 'x1']].to_numpy(dtype=np.float64)
feature_names = ['x0', 'x1']

# Python: Build marginals
m0 = SplineTerm1D(feature='x0', k=5, basis='cr')
m1 = SplineTerm1D(feature='x1', k=5, basis='cr')
m0.fit(X, feature_names)
m1.fit(X, feature_names)

s0 = m0._spline
s1 = m1._spline

# Python step-by-step
B0 = np.asarray(s0.raw_basis, dtype=np.float64)
S0_unscaled = np.asarray(s0.raw_penalty_unscaled, dtype=np.float64)
XP0 = s0._np_transform

B1 = np.asarray(s1.raw_basis, dtype=np.float64)
S1_unscaled = np.asarray(s1.raw_penalty_unscaled, dtype=np.float64)
XP1 = s1._np_transform

print(f"=== Python intermediate values ===")
print(f"B0[0,0]: {B0[0,0]:.10f}")
print(f"S0_unscaled[0,0]: {S0_unscaled[0,0]:.10f}")

if XP0 is not None:
    B0_np = B0 @ XP0
    S0_np = XP0.T @ S0_unscaled @ XP0
else:
    B0_np = B0
    S0_np = S0_unscaled

if XP1 is not None:
    B1_np = B1 @ XP1
    S1_np = XP1.T @ S1_unscaled @ XP1
else:
    B1_np = B1
    S1_np = S1_unscaled

print(f"B0_np[0,0]: {B0_np[0,0]:.10f}")
print(f"S0_np[0,0]: {S0_np[0,0]:.10f}")

ev0 = float(np.max(np.linalg.eigvalsh(0.5*(S0_np+S0_np.T))))
ev1 = float(np.max(np.linalg.eigvalsh(0.5*(S1_np+S1_np.T))))
print(f"max eig S0_np: {ev0:.6f}")
print(f"max eig S1_np: {ev1:.6f}")

S0_norm = normalize_tensor_marginal_penalty(S0_np)
S1_norm = normalize_tensor_marginal_penalty(S1_np)

B_raw = rowwise_kronecker([B0_np, B1_np])
S_list = tensor_product_penalties([S0_norm, S1_norm], basis_dims=[B0_np.shape[1], B1_np.shape[1]])

print(f"B_raw[0,0]: {B_raw[0,0]:.10f}")
print(f"S_list[0][0,0] before scale: {S_list[0][0,0]:.10f}")

S_scaled = rescale_tensor_penalties_for_fit(B_raw, S_list)
print(f"S_scaled[0][0,0]: {S_scaled[0][0,0]:.10f}")
print(f"S_scaled[1][0,0]: {S_scaled[1][0,0]:.10f}")

# Constraint
mean_row = B_raw.mean(axis=0)
print(f"mean_row[0:3]: {mean_row[:3]}")

B_te, S_te, C_te = full_term_sum_to_zero_constraint(B_raw, S_scaled)
print(f"B_te[0,0]: {B_te[0,0]:.10f}")
print(f"S_te[0][0,0]: {S_te[0][0,0]:.10f}")
print(f"S_te[1][0,0]: {S_te[1][0,0]:.10f}")

# Now run R with the same data
r_code = """
suppressPackageStartupMessages(library(mgcv))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
d <- read.csv(args[[1]], stringsAsFactors = FALSE)
out <- args[[2]]

x0 <- d$x0; x1 <- d$x1

# Step 1: Build marginals with smooth.construct
spec0 <- s(x0, k=5, bs="cr")
spec1 <- s(x1, k=5, bs="cr")
sm0 <- smooth.construct(spec0, list(x0=x0), list(x0=NULL))
sm1 <- smooth.construct(spec1, list(x1=x1), list(x1=NULL))

X0_raw <- sm0$X; S0_raw <- sm0$S[[1]]
X1_raw <- sm1$X; S1_raw <- sm1$S[[1]]

# Step 2: np reparam
np0 <- ncol(X0_raw)
knt0 <- seq(min(x0), max(x0), length=np0)
pd0 <- list(x0=knt0); names(pd0) <- sm0$term
sv0 <- svd(Predict.matrix(sm0, pd0))
XP0 <- sv0$v %*% (t(sv0$u) / sv0$d)
X0_np <- X0_raw %*% XP0
S0_np <- t(XP0) %*% S0_raw %*% XP0

np1 <- ncol(X1_raw)
knt1 <- seq(min(x1), max(x1), length=np1)
pd1 <- list(x1=knt1); names(pd1) <- sm1$term
sv1 <- svd(Predict.matrix(sm1, pd1))
XP1 <- sv1$v %*% (t(sv1$u) / sv1$d)
X1_np <- X1_raw %*% XP1
S1_np <- t(XP1) %*% S1_raw %*% XP1

# Step 3: Eigenvalue normalize
ev0 <- eigen(S0_np, symmetric=TRUE, only.values=TRUE)$values[1]
ev1 <- eigen(S1_np, symmetric=TRUE, only.values=TRUE)$values[1]
S0_norm <- S0_np / ev0
S1_norm <- S1_np / ev1

# Step 4: Tensor product
X_te <- tensor.prod.model.matrix(list(X0_np, X1_np))
S_te_list <- tensor.prod.penalties(list(S0_norm, S1_norm))

# Step 5: scale.penalty
maXX <- norm(X_te, type="I")^2
S_scaled <- lapply(S_te_list, function(S) S / (norm(S) / maXX))

# Step 6: constraint
C <- matrix(colMeans(X_te), 1, ncol(X_te))
k <- ncol(X_te); j <- nrow(C)
qrc <- qr(t(C))
S_final <- lapply(S_scaled, function(S) {
  ZSZ <- qr.qty(qrc, S)[(j+1):k,]
  t(qr.qty(qrc, t(ZSZ))[(j+1):k,])
})
X_final <- t(qr.qty(qrc, t(X_te))[(j+1):k,])

# smoothCon for comparison
sm_te <- smoothCon(te(x0, x1, bs=c("cr","cr"), k=c(5,5)), d, absorb.cons=TRUE, scale.penalty=TRUE)[[1]]

write_json(list(
  B0_raw_00 = X0_raw[1,1],
  S0_raw_00 = S0_raw[1,1],
  B0_np_00 = X0_np[1,1],
  S0_np_00 = S0_np[1,1],
  ev0 = ev0,
  ev1 = ev1,
  B_te_00 = X_te[1,1],
  S_te_pre_scale_00 = S_te_list[[1]][1,1],
  maXX = maXX,
  S_te_scaled_00 = S_scaled[[1]][1,1],
  S_te_scaled_11 = S_scaled[[2]][1,1],
  C_00 = C[1,1],
  S_final_00 = S_final[[1]][1,1],
  S_final_11 = S_final[[2]][1,1],
  X_final_00 = X_final[1,1],
  smoothCon_S0_00 = sm_te$S[[1]][1,1],
  smoothCon_S1_00 = sm_te$S[[2]][1,1],
  smoothCon_X_00 = sm_te$X[1,1]
), out, auto_unbox = TRUE, digits = 17)
"""

with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir_path = Path(tmpdir)
    csv_path = tmpdir_path / "data.csv"
    json_path = tmpdir_path / "result.json"
    script_path = tmpdir_path / "debug.R"
    data.to_csv(csv_path, index=False)
    script_path.write_text(r_code, encoding='utf-8')
    result = subprocess.run(
        ['Rscript', str(script_path), str(csv_path), str(json_path)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("R error:", result.stderr)
    else:
        r_vals = json.loads(json_path.read_text())
        print(f"\n=== R intermediate values ===")
        for k, v in r_vals.items():
            print(f"  {k}: {v:.10f}" if isinstance(v, float) else f"  {k}: {v}")
