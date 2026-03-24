import numpy as np
import sys
sys.path.insert(0, 'tests')
from mgcv_parity_utils import _run_mgcv_smoothcon_penalties, _make_gaussian_data
from nampy.gam.formula.parser import parse_gam_formula
from nampy.gam.formula.compiler import compile_predictor_specs_from_formula
from nampy.gam.design.compiler import compile_predictor_designs
from nampy.gam.smooths.univariate.cubic_regression import SplineTerm1D
from nampy.gam.basis.tensor import (
    rowwise_kronecker, tensor_product_penalties,
    normalize_tensor_marginal_penalty, rescale_tensor_penalties_for_fit
)
from nampy.gam.constraints.absorption import full_term_sum_to_zero_constraint

data = _make_gaussian_data(seed=7, n=80)
X = data[['x0', 'x1']].to_numpy(dtype=np.float64)
feature_names = ['x0', 'x1']

# Build marginals manually
m0 = SplineTerm1D(feature='x0', k=5, basis='cr')
m1 = SplineTerm1D(feature='x1', k=5, basis='cr')
m0.fit(X, feature_names)
m1.fit(X, feature_names)

s0 = m0._spline
s1 = m1._spline

print(f'np_transform0 not None: {s0._np_transform is not None}')
print(f'np_transform1 not None: {s1._np_transform is not None}')

# Reconstruct what te.fit does
B0 = np.asarray(s0.raw_basis, dtype=np.float64)
S0_unscaled = np.asarray(s0.raw_penalty_unscaled, dtype=np.float64)
XP0 = s0._np_transform

print(f'B0 shape: {B0.shape}, S0_unscaled[0,0]: {S0_unscaled[0,0]:.6f}')
print(f'S0_scaled[0,0]: {s0.raw_penalty[0,0]:.6f}')

if XP0 is not None:
    B0_np = B0 @ XP0
    S0_np = XP0.T @ S0_unscaled @ XP0
else:
    B0_np = B0
    S0_np = S0_unscaled

evals0 = np.linalg.eigvalsh(0.5*(S0_np + S0_np.T))
print(f'max eig S0_np: {np.max(evals0):.6f}')
S0_norm = normalize_tensor_marginal_penalty(S0_np)

B1 = np.asarray(s1.raw_basis, dtype=np.float64)
S1_unscaled = np.asarray(s1.raw_penalty_unscaled, dtype=np.float64)
XP1 = s1._np_transform

if XP1 is not None:
    B1_np = B1 @ XP1
    S1_np = XP1.T @ S1_unscaled @ XP1
else:
    B1_np = B1
    S1_np = S1_unscaled

S1_norm = normalize_tensor_marginal_penalty(S1_np)

B_raw = rowwise_kronecker([B0_np, B1_np])
S_list = tensor_product_penalties([S0_norm, S1_norm], basis_dims=[B0_np.shape[1], B1_np.shape[1]])
S_scaled = rescale_tensor_penalties_for_fit(B_raw, S_list)

print(f'B_raw shape: {B_raw.shape}')
print(f'S_scaled[0] shape: {S_scaled[0].shape}')
print(f'S_scaled[0][0,0]: {S_scaled[0][0,0]:.6f}')
print(f'S_scaled[1][0,0]: {S_scaled[1][0,0]:.6f}')

B_te, S_te, C_te = full_term_sum_to_zero_constraint(B_raw, S_scaled)
print(f'B_te shape after constraint: {B_te.shape}')
print(f'S_te[0][0,0]: {S_te[0][0,0]:.6f}')
print(f'S_te[1][0,0]: {S_te[1][0,0]:.6f}')

# Compare with mgcv
smooth_expr_r = 'te(x0, x1, bs=c("cr", "cr"), k=c(5, 5), sp=c(0.7, 1.3))'
expected = _run_mgcv_smoothcon_penalties(data, smooth_expr_r, absorb_cons=True, scale_penalty=True)
target = [np.asarray(S, dtype=np.float64) for S in expected['S']]
print(f'\nmgcv S[0][0,0]: {target[0][0,0]:.6f}')
print(f'mgcv S[1][0,0]: {target[1][0,0]:.6f}')
print(f'max diff S[0]: {np.max(np.abs(S_te[0] - target[0])):.3e}')
print(f'max diff S[1]: {np.max(np.abs(S_te[1] - target[1])):.3e}')
