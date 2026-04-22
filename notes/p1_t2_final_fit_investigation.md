# P1 `t2` General-Family Final-Fit Investigation

Date: 2026-04-21

## Scope

This note captures investigation state for P1 in [xfails.md](/home/ad32/projects/package/NAMpy/xfails.md): general-family `t2` final-fit parity.

Tracked cases:

- `gaulss_t2_full_false`
- `gaulss_t2_full_true`
- `gammals_t2_full_false`
- `gammals_t2_full_true`
- `gevlss_t2_full_false`
- `gevlss_t2_full_true`
- `shashlss_t2_full_false`
- `shashlss_t2_full_true`
- `ziplss_t2_full_false`
- `ziplss_t2_full_true`

Investigation concentrated first on `gaulss_t2_full_false`, because it is the smallest clear reproducer for the remaining gap.

## High-Level Status

Main result:

- constructor/setup parity for `t2(tp, cr)` was repaired
- general-family preoptimization parity now holds for audited `gaulss_t2_full_false`
- fixed-smoothing inner-fit parity now holds for audited `gaulss_t2_full_false`
- remaining failure moved downstream into final high-level post-fit parameterization, specifically `Vp`

This is progress. It means current blocker is no longer low-level tensor basis assembly or general-family PIRLS/outer derivative core for this case.

## Upstream `mgcv` References Used

Primary behavioral references:

- [mgcv/R/smooth.r](/home/ad32/projects/package/NAMpy/mgcv/R/smooth.r)

Relevant upstream functions:

- `t2.model.matrix`
- `smooth.construct.t2.smooth.spec`
- `Predict.matrix.t2.smooth`
- `nat.param(..., type = 3)`

Secondary parity target:

- high-level `mgcv::gam(...)` fit output behavior for `coef(fit)`, `fit$Vp`, and unconditional post-fit surfaces

## Code Changes Already Made During Investigation

### 1. `t2` marginal orientation

File:

- [nampy/gam/smooths/algebra.py](/home/ad32/projects/package/NAMpy/nampy/gam/smooths/algebra.py)

Change:

- removed `tp/ts` static `t2` sign hack
- changed `cr/cs` static `t2` sign handling from flipping last null column to flipping first null column

Reason:

- align marginal null-space orientation with actual `mgcv` tensor-marginal reparameterization used by `nat.param(..., type = 3)`

### 2. Runtime absorption of `t2` null-block constraint

File:

- [nampy/gam/smooths/tensor/t2.py](/home/ad32/projects/package/NAMpy/nampy/gam/smooths/tensor/t2.py)

Changes:

- removed `tp/ts` covariance-based sign flip in `_orient_t2_marginal_like_mgcv`
- for constant/no-`by` `t2`, absorbed null-block constraint inside runtime basis assembly with:
  - `remove_constant_from_null_block=True`
- stopped wrapper-side explicit `fit_constraint_matrix` absorption when runtime already absorbed null constraint
- removed now-unused `null_space_basis_from_constraint_matrix` import

Reason:

- `mgcv::smoothCon(..., absorb.cons = TRUE)` for `t2` applies null-block constraint directly on assembled `t2` basis
- generic wrapper QR absorption rotated columns differently and broke exact tensor parity

## What Improved

### 1. Tensor marginal parity

On actual tensor-marginal path used by `TensorANOVASplineTerm.fit`:

- `tp` marginal matched `mgcv:::nat.param(..., type = 3)` to about `1e-14`
- `cr` marginal matched `mgcv:::nat.param(..., type = 3)` to about `4.4e-15`

### 2. Runtime `t2` training basis parity

`TensorANOVASplineTerm._basis_train` matched `mgcv::smoothCon(..., absorb.cons = TRUE)` to about:

- `2.353672812205332e-14`

### 3. General-family preoptimization setup parity

For `gaulss_t2_full_false`:

- `build_general_family_setup_state(...).X_full` matched to about `2.353672812205332e-14`
- `X_initial` matched to about `4.6629367034256575e-15`

This removed earlier space-only drift before optimization.

### 4. Fixed-smoothing inner-fit parity

At snapshot-optimized smoothing parameters

```text
[4.87925152e+02, 7.83566626e-02, 2.83260173e-04]
```

the audited `gaulss_t2_full_false` fixed-sp path matched `mgcv_fixed_sp_fit5.R` essentially exactly:

- coefficient block max abs about `2.44e-14`
- fit coefficient block max abs about `1.07e-12`
- `db_drho` max abs about `2.75e-15`
- fit-space `db_drho` max abs about `9.24e-14`

Interpretation:

- inner general-family solve is no longer the active bug for this case

## Targeted Test Commands Run

Passing slice:

```bash
pytest tests/families/test_general_family_mgcv_parity.py -k 'gaulss_t2_full_false and (outer_fit_matches or fixed_sp_outer_derivatives)' -v
```

Observed result:

- `2 passed`

Still failing slice:

```bash
pytest tests/optimization/test_mgcv_postprocessing_final_fit_parity.py -k 'gaulss_t2_full_false' -v --runxfail
```

Current failure moved to:

- `gaulss_t2_full_false: Vp diagonal mismatch`

Earlier `Vc`-level failure is no longer the first blocker on this case.

## Crucial Diagnostic Finding

Remaining mismatch is not in:

- tensor constructor
- preoptimization setup assembly
- fixed-sp inner `gam.fit5`-style solve
- `Sl.initial.repara(..., inverse = TRUE, cov = TRUE)` equivalent
- `compute_preconditioned_inverse`
- `db_drho`

Why this conclusion is strong:

- low-level fixed-sp fit now reproduces audited `mgcv` inner-fit results at optimized `sp`
- but high-level snapshot `fit$Vp` still differs materially from NAMpy final exported `cov_bayes`

So remaining bug sits in high-level final parameterization of the fitted model, not in the low-level solve.

## Strongest Evidence For Missing Fit-to-Prediction Parameterization

### Evidence from `mgcv`

Direct R probe on `t2(x0, x1, bs = c("tp", "cr"), k = c(6, 6))` showed:

- `smoothCon(..., absorb.cons = TRUE)[[1]]$X` has shape `160 x 35`
- `Predict.matrix(sc, data.frame(...))` has shape `160 x 36`

That means upstream fit-time and prediction-time parameterizations differ for this `t2` surface.

### Evidence from NAMpy

Local probe on current branch for `gaulss_t2_full_false` showed:

- snapshot coefficient length: `37`
- NAMpy compiled model `n_coef`: `35`
- compiled model `fit_to_prediction_parameterization_map`: `None`
- predictor 0 fit design shape: `(160, 35)`
- predictor 0 prediction-on-training-data shape: `(160, 35)`
- predictor 0 fit/predict matrices match exactly
- `TensorANOVASplineTerm.transform_new(...)` returns `(160, 35)` on training data
- term-level `predict_coefficient_map`: `None`

This is wrong relative to upstream `mgcv` behavior if `Predict.matrix.t2.smooth` should expose the larger raw prediction parameterization.

## Concrete Symptom at Final Export Surface

Current audited `gaulss_t2_full_false` final result:

- `fit_result.coef_space == "prediction"`
- but there is no nontrivial fit-to-prediction map
- exported coefficients/covariances therefore remain in low-level fit parameterization, not true high-level `mgcv::gam(...)` prediction parameterization

Example coefficient drift from local probe:

Current NAMpy first 8 coefficients:

```text
[ 1.11186565e+00  8.23940781e-07  1.80143858e-06 -1.09139749e-06
 -4.33140748e-06  2.82251135e-06 -1.97347622e-06 -3.08531820e-06]
```

Snapshot first 8 coefficients:

```text
[ 4.47866172e-01  1.80190893e-06 -1.08425458e-06 -4.35573818e-06
  2.82638025e-06 -1.95768069e-06 -3.10172944e-06  3.60172090e-06]
```

This is consistent with parameterization mismatch, not with tiny numeric drift.

## Current Best Diagnosis

Likely missing behavior:

- `t2` should preserve separate fit-time and prediction-time parameterizations like upstream
- prediction path should expose raw `Predict.matrix.t2.smooth`-style basis
- compiled model should then derive a nontrivial `fit_to_prediction_parameterization_map`
- final coefficient and covariance export should use that map before parity comparison against snapshot `fit$coef_full` / `fit$Vp`

In short:

- NAMpy currently makes `t2` fit basis and prediction basis identical too early
- `mgcv` does not

## Most Relevant Files To Continue In

- [nampy/gam/smooths/tensor/t2.py](/home/ad32/projects/package/NAMpy/nampy/gam/smooths/tensor/t2.py)
- [nampy/gam/smooths/tensor/t2_basis.py](/home/ad32/projects/package/NAMpy/nampy/gam/smooths/tensor/t2_basis.py)
- [nampy/gam/compiler/construct.py](/home/ad32/projects/package/NAMpy/nampy/gam/compiler/construct.py)
- [nampy/gam/compiler/compile_model.py](/home/ad32/projects/package/NAMpy/nampy/gam/compiler/compile_model.py)
- [nampy/gam/compiler/structures.py](/home/ad32/projects/package/NAMpy/nampy/gam/compiler/structures.py)
- [nampy/gam/fit/parameterization.py](/home/ad32/projects/package/NAMpy/nampy/gam/fit/parameterization.py)
- [nampy/gam/fit/postprocess/unconditional_covariance.py](/home/ad32/projects/package/NAMpy/nampy/gam/fit/postprocess/unconditional_covariance.py)

Upstream reference file:

- [mgcv/R/smooth.r](/home/ad32/projects/package/NAMpy/mgcv/R/smooth.r)

## Recommended Next Steps

1. Make `TensorANOVASplineTerm.transform_new(...)` mirror upstream `Predict.matrix.t2.smooth` raw width/order exactly, even when fit-time basis already absorbed null constraint.
2. Preserve explicit local map from raw prediction parameterization to fitted parameterization instead of collapsing them to same matrix at runtime.
3. Ensure compiler sees distinct fit and prediction parameterizations and produces nontrivial `fit_to_prediction_parameterization_map`.
4. Re-run:

```bash
pytest tests/optimization/test_mgcv_postprocessing_final_fit_parity.py -k 'gaulss_t2_full_false' -v --runxfail
```

5. If that case turns green, expand only to closely neighboring `t2` general-family cases.

## Important Non-Conclusions

Not resolved yet:

- P1 is not fixed
- `xfails.md` should not be reduced yet
- broader `t2_full_true` and other family cases remain unverified after this investigation

Still uncertain:

- exact place where high-level `mgcv::gam(...)` injects or stores its `G$P`-equivalent map in this surface
- whether current compiler contract needs small extension to let raw prediction basis survive long enough for model-level map construction

## Bottom Line

Investigation narrowed P1 from broad `t2` general-family post-fit drift to one much smaller problem:

- missing high-level fit-to-prediction parameterization parity for `t2`

Everything upstream of that now looks substantially healthier for the audited `gaulss_t2_full_false` case.
