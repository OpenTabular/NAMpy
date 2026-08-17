# NAMpy TODO

- Snapshot date: 2026-08-17 (branch `mgcv`, implementation snapshot through
  `e750f6d`; audit reconciliation in
  [REVIEW.md](REVIEW.md), current-state summary in [tldr.md](tldr.md))
- Status report: [PROJECT_STATUS.md](PROJECT_STATUS.md)

## Verification rule

Do not rerun tests already recorded as passing in `PROJECT_STATUS.md` unless one of
their owning files changes. Run the smallest pending slice below, record its exact
command and result in the status report, and then check it off here.

Use the configured environment explicitly while the system interpreter lacks project
dependencies:

```bash
/home/ad32/miniconda3/envs/nampy/bin/pytest ...
```

Do not run the full suite by default.

## Completed — do not repeat without invalidation

- [x] Port the relevant `gdi1` stages (`multSk`, `applyPt`, `applyP`, `ift1`, and
  `get_bSb`) with upstream operation order.
- [x] Preserve pre-`gdiPK` `eta`/`mu` for deviance derivatives while retaining the
  refreshed coefficient representative.
- [x] Verify signed-weight `gdiPK`/`ift1` behavior.
- [x] Verify Gamma REML BFGS lifecycle parity.
- [x] Verify Gamma ML BFGS lifecycle parity.
- [x] Verify the neighboring Gamma REML Newton, Poisson REML BFGS, and Binomial REML
  BFGS cases.
- [x] Verify the Poisson fixed-smoothing `gam.fit3` inner state.
- [x] Verify lifecycle-registry coverage, zero known-gap entries, and unique IDs.
- [x] Run focused Ruff, isort, compilation, and diff checks for the final `gdi1`
  touched-file set.
- [x] (2026-08-14 pm) Fix multivariate `s()` default `k` to defer to the
  constructor rule via `k=-1` like upstream `smooth.r:1316-1318`.
- [x] (2026-08-14 pm) Guard `pc=` on `te/ti/re/fs/sz`, guard `t2()`, remove
  `nei=`, remove the `gacv.cp` alias, remove orphaned `gp` primitives.
- [x] (2026-08-14 pm) Mirror upstream multi-`offset()` behavior (first offset
  kept + R warning; verified in `debug/multi_offset_probe.R`).
- [x] (2026-08-14 pm) Mirror upstream `Vc` availability for efs/optim
  (GLM/extended: `Vc`/`edf2` absent; general families: `Vc == Vb`, correction
  suppressed at `gam.fit5.post.proc` gate).
- [x] (2026-08-14 pm) Wire Gaussian noncanonical-link ML/REML profiled-scale
  criterion/gradient/Hessian through dispatch; 4 new lifecycle cases pass.
- [x] (2026-08-14 pm) Lift stale negbin joint-theta guards (offset, weights,
  non-formula); weighted lifecycle case passes after adding weights support to
  `mgcv_outer_trace.R`.
- [x] (2026-08-14 pm) Promote `factor_smooth_sz` to strict coefficient/SE
  comparison (stale relaxation; no rank drop occurs on either side).
- [x] (2026-08-14 pm) Verify `gaulss_select_true_cr` endpoint is
  orientation-indeterminate inside mgcv itself
  (`debug/gaulss_select_initial_spg_probe.py`: mirrored basis reproduces
  NAMpy's endpoint exactly).
- [x] (2026-08-14 pm) Write `GAM_IMPLEMENTED.md` / `GAM_NOT_IMPLEMENTED.md`.

## P0 — Stabilize and validate the current working tree

### Neural subsystem: first validation — complete 2026-08-17

- [x] Run the architecture forward/backward and penalty contracts (25 passed).

  ```bash
  /home/ad32/miniconda3/envs/nampy/bin/pytest tests/neural/test_neural_architecture_smoke.py -v
  ```

- [x] Run task semantics, multi-output regression, and metric contracts (2 passed).

  ```bash
  /home/ad32/miniconda3/envs/nampy/bin/pytest tests/neural/test_neural_task_model.py -v
  ```

- [x] Run scikit-learn constructor, clone, parameter, QNAM, and positional-data
  contracts (41 passed).

  ```bash
  /home/ad32/miniconda3/envs/nampy/bin/pytest tests/neural/test_neural_sklearn_contracts.py -v
  ```

- [x] Run focused SplineNAM basis, interaction, gradient, naming, and public-fit
  coverage (7 passed).

  ```bash
  /home/ad32/miniconda3/envs/nampy/bin/pytest tests/neural/test_neural_spline_nam.py -v
  ```

- [x] Run the public estimator fit/predict smoke matrix last because it is the most
  expensive neural slice (73 passed).

  ```bash
  /home/ad32/miniconda3/envs/nampy/bin/pytest tests/neural/test_neural_estimator_fit_smoke.py -v
  ```

- [x] No architecture failed; no parameter-ID reduction or repeat matrix run was
  needed. Total retained first-validation result: 148 passed.

### GAM joint optimizer branch validation — complete

- [x] Validate the newly added Gamma Newton/`optim` cases (2026-08-15: 3 passed,
  rerun after the PIRLS rank-tol change: 7 passed with negbin cases).

  ```bash
  /home/ad32/miniconda3/envs/nampy/bin/pytest \
    tests/optimization/test_mgcv_optimization_lifecycle_parity.py \
    -k 'gamma_ml_newton_joint_scale_cr or gamma_ml_optim_joint_scale_cr or gamma_reml_optim_joint_scale_cr' -v
  ```

- [x] Validate supported estimated-theta negative-binomial lifecycle cases
  (2026-08-15: 4 passed; re-validated after rank-tol change).

  ```bash
  /home/ad32/miniconda3/envs/nampy/bin/pytest \
    tests/optimization/test_mgcv_optimization_lifecycle_parity.py \
    -k 'negbin_est_ml_newton_joint_theta_cr or negbin_est_ml_bfgs_joint_theta_cr or negbin_est_reml_bfgs_joint_theta_cr or negbin_est_reml_optim_joint_theta_cr' -v
  ```

- [x] Add or identify the exact Gaussian noncanonical joint-scale lifecycle cases,
  then run only those parameter IDs. Done 2026-08-14 pm:
  `gaussian_log_reml_newton_joint_scale_cr`, `gaussian_log_ml_newton_joint_scale_cr`,
  `gaussian_log_reml_bfgs_joint_scale_cr`, `gaussian_inverse_reml_newton_joint_scale_cr`
  all pass strict trace + final-fit parity.

- [x] Run the explicit negative-binomial ML `optim` guard contract (passed
  2026-08-14/15).

  ```bash
  /home/ad32/miniconda3/envs/nampy/bin/pytest \
    tests/optimization/test_gam_unsupported_branch_guards.py::test_negbin_estimated_theta_ml_optim_guard_raises_explicitly -v
  ```

### GAM prediction and public-surface validation — complete

- [x] Validate ordinary `terms=`/`exclude=` filtering and unknown-term behavior
  (2026-08-15: passed).

  ```bash
  /home/ad32/miniconda3/envs/nampy/bin/pytest \
    tests/diagnostics/test_gam_plot_and_public_api_contracts.py \
    -k 'predict_terms_and_exclude_filters or predict_unknown_term_filter' -v
  ```

- [x] Validate `iterms`, factor-smooth feature contributions, and mixed fixed/free
  smoothing prediction (2026-08-15: passed).

  ```bash
  /home/ad32/miniconda3/envs/nampy/bin/pytest \
    tests/diagnostics/test_gam_plot_and_public_api_contracts.py \
    -k 'predict_iterms or factor_smooth or mixed_fixed_and_free' -v
  ```

- [x] Validate the general-family term-filter guard (2026-08-15: passed).

  ```bash
  /home/ad32/miniconda3/envs/nampy/bin/pytest \
    tests/optimization/test_gam_unsupported_branch_guards.py::test_general_family_prediction_term_filter_guard_raises_explicitly -v
  ```

- [x] Validate the public plot, summary, BIC, formula metadata, `gam_check`, and
  `lpmatrix` contract file (2026-08-15: 7 passed after the summary.gam port).

  ```bash
  /home/ad32/miniconda3/envs/nampy/bin/pytest \
    tests/diagnostics/test_gam_plot_and_public_api_contracts.py \
    -k 'bic or formula_metadata or gam_check or lpmatrix or summary' -v
  ```

### GAM constructor and formula-preprocessing validation — complete

- [x] Validate transformed numeric smooth `by=` behavior and logical-expression
  rejection (2026-08-15: passed).

  ```bash
  /home/ad32/miniconda3/envs/nampy/bin/pytest \
    tests/parity/test_mgcv_remaining_gap_xfails.py \
    -k 'transformed_numeric_by or logical_transformed_by' -v
  ```

- [x] Validate vector-valued tensor `fx=` across the promoted regression snapshot
  cases (2026-08-15: passed).

  ```bash
  /home/ad32/miniconda3/envs/nampy/bin/pytest \
    tests/parity/test_mgcv_remaining_gap_xfails.py \
    -k 'tensor_te_vector_fx or tensor_ti_vector_fx' -v
  ```

- [x] Validate the natural-parameterization port. Done 2026-08-15: promoted
  `debug/nat_param_type1_probe.py` into
  `tests/smooths/test_mgcv_nat_param_parity.py` (column-sign invariants for
  simple eigenvalues; projector invariants inside the repeated null
  eigenspace); 2 passed.

- [x] Reconcile the now-empty raw known-gap registry. Done 2026-08-15: dead
  registry scaffolding removed; targeted slices `-k 'fs or sz'` (20 passed) and
  `-k 'tp or cr'` (31 passed) after the nat-param and factor-smooth cases
  passed independently.

### GAM post-fit and covariance validation — complete

- [x] Run the new non-Gaussian unconditional covariance tests (2026-08-15:
  4 passed; re-run after the rank-tol change).

  ```bash
  /home/ad32/miniconda3/envs/nampy/bin/pytest \
    tests/optimization/test_mgcv_postprocessing_final_fit_parity.py \
    -k 'non_gaussian_unconditional or gamma_hat_diag' -v
  ```

- [x] Validate joint-scale inclusion in unconditional covariance. Covered by the
  `gamma_hat_diag`/`non_gaussian_unconditional` parity slice (joint-scale Gamma
  Vc vs mgcv) plus the efs-gate owner contract added 2026-08-14 in
  `tests/optimization/test_gam_fit_backend_owner_contracts.py`.

### Documentation and worktree hygiene

- [x] Update `tests/SUBSYSTEM_COVERAGE.md` (done 2026-08-15: Stage 6-7/14 claims
  corrected, Stage 21 guard attribution fixed).
- [x] Retain the portability test and evidence probes. The production guard is in
  `c25af71`; the six cited cs, near-singular REML, and rank-deficiency probes and
  the CI portability job are in `e750f6d`. All five `tests/neural/` files were
  already tracked.
- [x] Split the dirty tree into reviewable commits by concern:

  1. `46bacbc` — upstream parity and heuristic-removal changes,
  2. `c25af71` — portable numerical/stacked-QR simplification,
  3. `e750f6d` — regression/API tests, retained debug evidence, and CI.

- [x] Record the implementation commit hashes in `PROJECT_STATUS.md`.
- [ ] Standardize the documented developer environment. The verified
  `/home/ad32/miniconda3/envs/nampy` environment contains `pretab 0.0.3`, and
  `pretab` is mandatory in both `pyproject.toml` and `requirements.txt`; retain
  explicit environment-qualified commands until a reproducible setup section is
  added for contributors.

## P1 — Reconciled GAM parity findings and API policy

### Factor-smooth penalty ordering (resolved)

- [x] Resolved 2026-08-15. Investigation (debug/fs_null_order_probe.py,
  debug/fs_null_order_stability_probe.py) proved the "swap" is mgcv-internal
  indeterminacy, not a NAMpy ordering bug: upstream assigns one sp per
  nat.param(type=1) null column (mgcv/R/smooth.r:2067-2075), and R's eigen
  orders those numerically-zero RSR eigenvalues by roundoff — mgcv itself
  flips the order under a row permutation of the same data, and the two null
  directions are identical between NAMpy and mgcv (cross-correlation is an
  exact permutation matrix). There is no deterministic upstream rule to port;
  forcing one would require platform-specific LAPACK behavior, which policy
  forbids. Treatment: the lifecycle registry gained declared
  `exchangeable_sp_groups` (+ coefficient-column mapping), and the harness
  canonicalizes each side independently by descending final log-sp inside the
  group before the otherwise-strict comparison — the full Newton trace, sp
  endpoint, Vp, Ve, EDF, scale, hat, and outer info all match strictly after
  alignment. Vc / edf2_total / AIC are excluded for this case with recorded
  evidence: mgcv's own row-permuted fit moves them by the same amounts, and
  NAMpy's values equal mgcv's other branch to 7 digits (Vc[0,0] 0.0695778077,
  edf2_total 15.9940558; AIC diff = exactly 2x edf2 spread).
  `gaussian_reml_newton_fs_xt_ps` passes; the fs+select snapshot case
  (test_gaussian_fs_ps_marginal_select_reml_matches_mgcv) already documents
  its flat-ridge tolerances and passes.

### cs shrinkage parity (unmasked by the cache bump)

- [x] Fixed 2026-08-16 from the upstream constructor, without a fallback.
  `add_full_rank_shrinkage` now mirrors
  `mgcv/R/smooth.r::smooth.construct.cr.smooth.spec` exactly: form
  `(S + t(S))/2`, take the descending symmetric eigensystem, then replace
  eigenvalues `nk-1` and `nk` successively by `.1` times the preceding value
  (`smooth.construct.cs.smooth.spec` sets `shrink=.1` and delegates). The
  ordinary `cs` terms tests now pass with and without SEs; added transformed
  `cr`/`cs` raw-constructor cases pass. The one remaining transformed-terms
  numeric difference is not a second constructor bug: preserved probes show
  the raw CR penalty agrees to `1.4e-14`, while R and every SciPy symmetric
  eigensolver driver choose different orientations for the repeated
  two-dimensional zero eigenspace before upstream assigns unequal `.1`/`.01`
  shrinkage. Per project policy this is left as platform/LAPACK orientation,
  with no solver hook, heuristic canonicalization, or tolerance change.

### Contract drift and audit follow-ups

- [x] Simplified stacked QR (2026-08-17): removed raw `ctypes` BLAS/LAPACK
  loading, compact Householder work-buffer helpers, `JPVT` reuse, and `dsyrk`
  accumulation. The supported implementation now uses explicit factors from
  SciPy's public pivoted-QR interface while preserving `pls_fit1`/`gdiPK`
  rank, signed-weight, coefficient, Hessian, and covariance behavior.
- [x] Audit the production package for direct native numeric bindings. The AST
  regression forbids `ctypes`, CFFI, direct SciPy BLAS/LAPACK modules,
  `get_lapack_funcs`, and explicit solver-driver selection; its exact test passes
  on 2026-08-17. The CI portability job runs the guard and focused numerical
  slices on Linux, macOS, and Windows.
- [x] Repaired the two stale unit tests in
  `tests/regressions/test_optimize_driver_mgcv_parity.py`
  (`test_all_fixed_smoothing_params_still_optimizes_unknown_gaussian_scale`:
  mock now supplies the strict initial.spg design state;
  `test_negbin_reml_native_all_fixed_optimizes_theta_first`: missing the
  keyword-only `optimizer` argument is now supplied). Both pass (2026-08-16).
- [x] Removed `_fallback_single_smooth_edf`; Gaussian post-fit EDF now comes
  only from the upstream `gdi1` `rV`/`K` construction.
- [x] Removed `_fs_term_penalty_adjustment` and the associated least-squares
  contribution shift (2026-08-16). `predict.gam` forms term contributions as
  the direct prediction-matrix coefficient block product; NAMpy now does the
  same. The strict `test_output_parity_terms[fs-no_se]` case passes.
- [x] Fix the gammals `select=True` optimized-prediction gap (2026-08-17).
  The mirrored-basis probe localized the path but the strict `initial.spg`
  regression found the defect: `_sl_multi_penalty_block` consumed the upper
  triangle, whereas `mgcv/R/fast-REML.r::Sl.setup` uses the lower-triangle
  convention. With `lower=True`, initial and final smoothing parameters match
  mgcv, optimized link/response/terms differences are at most `3.9e-9`, SE
  differences are at most `8.2e-10`, and every gammals xfail is now a strict
  pass. No tolerance, heuristic, or solver-driver selection was added.
- [x] Resolve the public-export mismatch (2026-08-17). Following the repository
  contract, `nampy.gam.__all__` now contains only `fit_model_core`, `solve_fit`,
  and `FitCoreSolution`; `GAM` is no longer re-exported from `nampy.gam` or
  top-level `nampy`. Tests and retained probes import the implementation from
  `nampy.gam.model.api`. The direct export-contract regression and the strict
  gammals initialization neighbor both pass.

### Side conditions

- [x] Align side-condition scope with upstream `gam.side` for exactly aliased
  parametric columns. Done 2026-08-15: parametric terms now pass through
  untouched and are used only for upstream's intercept-equivalence check;
  one-smooth/no-nesting designs preserve their compiled matrices byte-for-byte.
  Default and direct solver regressions cover Gaussian, Poisson, binomial, and
  Gamma rank drops. Factor-by/no-intercept nesting and linked factor-by parity
  remain strict. `debug/rank_deficient_side_condition_probe.py` records that
  mgcv itself switches the zeroed member of an exact alias under a 2.6e-12
  log-sp endpoint shift, while matching NAMpy at each shared endpoint.

### Optimizers and endpoints

- [x] Route every supported model-level PIRLS iteration through the current-SP
  `gam.reparam` state from `mgcv/R/gam.fit3.r`: transform starts with `t(T)`,
  solve on `X %*% T` with exact `St`, `Sr`, and `Eb`, then restore the public
  coefficient/covariance gauge together. Removed the Poisson-identity
  forced-Fisher endpoint and tightened-tolerance helpers; noncanonical GLMs now
  use upstream full Newton with only the local indefinite-system Fisher retry
  (2026-08-17).
- [x] Ported the remaining near-singular Gaussian joint REML Hessian behavior
  on 2026-08-16. The divergence was caused by reconstructing the coefficient
  deviance Hessian as `X'WX`, which is algebraically equivalent but does not
  preserve upstream cancellation at this boundary. The strict port now mirrors
  `mgcv/src/gdi.c::gdiPK` and `gdi1`: build the first Hessian half from the
  unpenalized weighted QR factor with `mat.c::getXtX`, include the signed-weight
  correction, double it in `gdi1`, and accumulate the smoothing Hessian with
  `mat.c::getXtMX` ordering. The exact random-effect regression passes in
  87.20s (EDF within its boundary tolerance; NAMpy endpoint `log(sp)=-66.5044`,
  EDF `2.0003662`, versus mgcv `-64.5252` and `2.0001068`). Fixed-endpoint
  Gaussian, Poisson, Gamma, and signed-weight neighbor tests also pass. See
  `debug/near_singular_reml_endpoint_probe.py` and
  `debug/near_singular_reml_derivatives_probe.py`.
- [x] Keep estimated-theta negative-binomial ML + `optim` outside the declared
  surface. The explicit guard is covered and preferable to approximating R's
  flat-boundary L-BFGS-B behavior; a port becomes backlog only if scope expands.
- [x] Classify the remaining `gaulss_select_true_cr` optimized endpoint
  (2026-08-17). `mgcv/R/mgcv.r::estimate.gam` reparameterizes `G$X` with
  `Sl.setup`'s symmetric-eigen vectors but calls `initial.spg` with the
  separately constructed, unreparameterized `G$Eb`. Base R leaves real
  `DSYEVR` eigenvector signs arbitrary; NAMpy and this R build differ by one
  legal column sign, which changes this flat-boundary start. A strict
  `pen.reg` QR experiment did not change the result and was removed. Matching
  one R build would require forbidden sign forcing/platform selection, so the
  evidence-backed local xfail and strict shared-endpoint test remain visible.
- [x] Fix the `gammals_select_true_cr` `edf2_total` failure. It shared the same
  incorrect upper-triangle `Sl.setup` start as the optimized predictions, not
  the remaining gaulss endpoint class. At the corrected mgcv endpoint the
  optimized fit5 post-processing test passes strictly and gammals was removed
  from `_GENERAL_OPTIMIZED_ENDPOINT_KNOWN_GAP_TAGS`.

### Underdetermined factor-smooth covariance

- [x] `factor_smooth_sz` resolved 2026-08-14 pm: the premise was stale — the
  balanced penalty fills `null(X)`, both sides keep the augmented system at full
  rank (rank 25, no drops), and raw coefficients/SEs agree strictly. The
  relaxation (`skip_coef_comparison`, `se_tol_scale=2e-3`) was removed and the
  strict case passes. Remaining latent (unexercised) deviations under genuine
  rank deficiency are documented in `GAM_NOT_IMPLEMENTED.md`.

### General-family unsupported surfaces

- [x] Keep multi-predictor general-family `terms=`/`exclude=` outside the declared
  surface. The public path raises explicitly and has an owner guard; port exact
  `predict.gam` coefficient-block selection only if scope expands.
- [x] Port or explicitly scope the remaining general-family `Sl` branches.
  Scoped 2026-08-14 (second pass): non-reparameterized single-/multi-penalty
  blocks stay explicitly guarded (`NotImplementedError`); the nonlinear
  (`linear=FALSE`) machinery is kept as **extension-only** — it mirrors the
  upstream adaptive-smooth `Sl` structure and is covered by
  `tests/families/test_mgcv_gamlss_nonlinear_sl.py`, but no in-tree smooth
  emits `general_family_nonlinear_sl`, so no real fit reaches it (documented
  in `GAM_NOT_IMPLEMENTED.md`).

- [x] Keep formula-list/general-family data-aware `.` shorthand outside the
  declared surface. The explicit guard and documentation are retained; there is
  no heuristic expansion.
- [x] Decide whether multiple `offset(...)` terms per predictor should be ported or
  remain explicitly unsupported. Resolved 2026-08-14: upstream keeps only the
  first offset (interpret.gam0 single-slot assignment, verified in
  `debug/multi_offset_probe.R`); NAMpy mirrors that including the R warning.

## P2 — Broaden confidence after P0/P1 are green

- [ ] Run touched-file `ruff`, `isort --check-only`, and `py_compile` for every logical
  commit. Do not run Black as a check; format only the files that actually need it.
  The five Python files changed by the completed 2026-08-17 gammals fix pass all
  three checks; the remaining dirty-tree files still need commit-scoped checks.
- [x] No architecture-specific failure was discovered by the 148-test neural
  validation, so no additional owner regression is required from this run.
- [ ] Add explicit multi-output regression coverage beyond LinReg if multi-output is
  intended for every public regressor.
- [x] Run an installed-wheel import/one-estimator smoke (2026-08-17). With the
  network unavailable, `python -m build --wheel --no-isolation` built
  `nampy-0.1.0-py3-none-any.whl`; installation with `pip install --no-deps` into a
  temporary venv succeeded, `nampy` imported from that venv rather than the source
  tree, mandatory `pretab 0.0.3` imported, `LinRegRegressor()` instantiated, and
  the installed package exposed exactly `fit_model_core`, `solve_fit`, and
  `FitCoreSolution` from `nampy.gam` with no top-level `GAM` alias.
- [ ] Optionally automate that built-artifact installation/import smoke in the
  build CI job. The package itself has now passed the manual clean-artifact check.
- [x] Update user documentation for (2026-08-17, `README.md`):

  - supported GAM optimizers/families,
  - intentional unsupported GAM branches,
  - prediction `terms`/`exclude`/`iterms`,
  - neural estimator clone/parameter behavior,
  - multi-output regression,
  - and SplineNAM preprocessing requirements.

  The multi-output section states the precise evidence boundary: the shared
  wrapper accepts multi-output targets, but only `LinRegRegressor` currently has
  an end-to-end multi-output fit regression.

- [ ] Only after all affected owner-level slices pass, run a justified broader
  subsystem grouping. Record the exact command and result in `PROJECT_STATUS.md`.
  Do not default to `pytest` or `pytest tests`.

## Intentional non-tasks

- Do not add `pyblas` merely to address the resolved Gamma lifecycle issue.
- Do not force raw eigenvector orientation inside repeated eigenspaces.
- Do not implement random-effect `id=` linkage unless upstream `mgcv` begins to
  support it.
- Do not enable full-rank `bs="fs"` shrinkage bases that upstream `mgcv` rejects.
- Do not replace explicit unsupported guards with heuristic fallbacks.
