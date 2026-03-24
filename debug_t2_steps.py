"""Debug t2 step-by-step."""
import numpy as np
import sys
sys.path.insert(0, 'tests')
import pandas as pd
from nampy.gam.smooths.univariate.cubic_regression import SplineTerm1D

rng = np.random.default_rng(31)
data = pd.DataFrame({"x": rng.uniform(-2.0, 2.0, size=120)})
term = SplineTerm1D(feature="x", k=5, basis="cr")
term.fit(data[["x"]].to_numpy(dtype=np.float64), ["x"])

X = np.asarray(term._spline.raw_basis, dtype=np.float64)
S = np.asarray(term._spline.raw_penalty, dtype=np.float64)
p = X.shape[1]

# Step 1: eigendecompose S in descending order
evals, U = np.linalg.eigh(0.5 * (S + S.T))
idx = np.argsort(evals)[::-1]
evals, U = evals[idx], U[:, idx]
print(f"evals: {evals}")
print(f"rank = {int(np.sum(evals > float(np.finfo(np.float64).eps)**0.8 * max(1.0, float(np.max(evals)))))}")

tol_eff = float(np.finfo(np.float64).eps)**0.8 * max(1.0, float(np.max(evals)))
rank = int(np.sum(evals > tol_eff))
null_exists = rank < p

E = np.ones(p, dtype=np.float64)
E[:rank] = np.sqrt(np.maximum(evals[:rank], 0.0))
print(f"E = {E}")

Xp = X @ U
col_norm = np.sum(Xp**2, axis=0) / E**2
av_norm = float(np.mean(col_norm[:rank]))
print(f"col_norm = {col_norm}")
print(f"av_norm = {av_norm}")

for i in range(rank, p):
    if av_norm > 0.0 and col_norm[i] > 0.0:
        E[i] = np.sqrt(col_norm[i] / av_norm)
print(f"E after null update = {E}")

P = U / E[np.newaxis, :]
Xp = Xp / E[np.newaxis, :]
print(f"Xp[0,:] before rotation = {Xp[0,:]}")

# Type 3 rotation
ind = list(range(rank, p))
rind = list(range(p-1, rank-1, -1))
print(f"ind={ind}, rind={rind}")
Xn = Xp[:, ind].copy()
n = Xn.shape[0]
one = np.ones(n, dtype=np.float64)
Xn -= (one[:, None] * (one[None, :] @ Xn)) / n
um_evals, um_vecs = np.linalg.eigh(Xn.T @ Xn)
desc = np.argsort(um_evals)[::-1]
um_vecs = um_vecs[:, desc]
print(f"um_evals (desc) = {um_evals[desc]}")
print(f"um_vecs[:,0] = {um_vecs[:,0]}")

Xp_before = Xp.copy()
Xp[:, rind] = Xp[:, ind] @ um_vecs
P[:, rind] = P[:, ind] @ um_vecs
print(f"Xp[0,:] after rotation = {Xp[0,:]}")

# Unit.fnorm
pen_idx = list(range(rank))
scale = 1.0 / np.sqrt(float(np.mean(Xp[:, pen_idx]**2)))
print(f"scale (penalized) = {scale}")
Xp[:, pen_idx] *= scale
P[pen_idx, :] *= scale

null_idx = list(range(rank, p))
scale_f = 1.0 / np.sqrt(float(np.mean(Xp[:, null_idx]**2)))
print(f"scale_f (null) = {scale_f}")
Xp[:, null_idx] *= scale_f
P[null_idx, :] *= scale_f

print(f"Python Xp[0,:] = {Xp[0,:]}")
print(f"Expected from R = [-0.26093879  0.21833587  0.95715008 -1.06556495 -1.20119191]")
