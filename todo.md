# NAMpy TODO

- Snapshot date: 2026-08-14
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

### Neural subsystem: first validation

- [ ] Run the architecture forward/backward and penalty contracts.

  ```bash
  /home/ad32/miniconda3/envs/nampy/bin/pytest tests/neural/test_neural_architecture_smoke.py -v
  ```

- [ ] Run task semantics, multi-output regression, and metric contracts.

  ```bash
  /home/ad32/miniconda3/envs/nampy/bin/pytest tests/neural/test_neural_task_model.py -v
  ```

- [ ] Run scikit-learn constructor, clone, parameter, QNAM, and positional-data
  contracts.

  ```bash
  /home/ad32/miniconda3/envs/nampy/bin/pytest tests/neural/test_neural_sklearn_contracts.py -v
  ```

- [ ] Run focused SplineNAM basis, interaction, gradient, naming, and public-fit
  coverage.

  ```bash
  /home/ad32/miniconda3/envs/nampy/bin/pytest tests/neural/test_neural_spline_nam.py -v
  ```

- [ ] Run the public estimator fit/predict smoke matrix last because it is the most
  expensive neural slice.

  ```bash
  /home/ad32/miniconda3/envs/nampy/bin/pytest tests/neural/test_neural_estimator_fit_smoke.py -v
  ```

- [ ] If one architecture fails, reduce to its exact parameter ID before debugging;
  do not repeatedly rerun the entire estimator matrix.

### GAM joint optimizer branches not yet in the verification ledger

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

### GAM prediction and public surfaces

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

### GAM constructors and formula preprocessing

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

### GAM post-fit and covariance

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
- [ ] Add the new untracked source, test, and debug files intentionally or remove them
  if obsolete. Do not leave `tests/neural/` outside version control after validation.
- [ ] Split the dirty tree into reviewable commits by concern:

  1. `gdi1`/PIRLS/QR state and Gamma lifecycle,
  2. joint Gaussian/Gamma/negative-binomial optimization,
  3. GAM prediction/formula/natural-parameterization work,
  4. neural estimator and architecture contracts,
  5. tests/debug/documentation.

- [ ] Record each commit hash in `PROJECT_STATUS.md`; the current base hash alone is
  insufficient for reproducing uncommitted changes.
- [ ] Standardize the documented developer environment. Either activate the `nampy`
  conda environment or install the project plus dev dependencies into the system
  interpreter; avoid ambiguous bare `pytest` commands.

## P1 — Resolve remaining parity gaps

### Optimizers and endpoints

- [ ] Port the exact estimated-theta negative-binomial ML `optim`/L-BFGS-B boundary
  behavior from R before removing its explicit guard. Add a targeted lifecycle case
  first.
- [x] Resolve the `gaulss_select_true_cr` optimized-endpoint attribution. Verified
  2026-08-14 pm (`debug/gaulss_select_initial_spg_probe.py`): the endpoint is
  orientation-indeterminate inside mgcv itself — mgcv refit on the mirrored basis
  (`x -> -x`) reproduces NAMpy's endpoint exactly (log sp 11.79338762, score
  159.6076681047); both orientations' criteria agree to 5e-6 with near-zero
  gradients. The endpoint will never "match" a single arbitrary orientation, so
  the local `xfail` stays with this evidence recorded in its reason string.
- [x] Investigate the `gammals_select_true_cr` `edf2_total` borderline failure.
  Resolved 2026-08-14 pm (`debug/gammals_select_edf2_probe.py`): same
  orientation-indeterminate `initial.spg` endpoint class as
  `gaulss_select_true_cr` — mgcv on the mirrored basis reproduces NAMpy's
  endpoint and edf1/edf2 sums to 2.3e-7; the sum-cap (`gam.fit4.r:1715`)
  fires on both sides so `edf2 == edf1`, making edf2 the endpoint-sensitive
  scalar. Tagged in `_GENERAL_OPTIMIZED_ENDPOINT_KNOWN_GAP_TAGS`; a strict
  fixed-endpoint post-processing test now covers gammals alongside gaulss
  (Vc + `edf2_total` at 5e-6).

### Underdetermined factor-smooth covariance

- [x] `factor_smooth_sz` resolved 2026-08-14 pm: the premise was stale — the
  balanced penalty fills `null(X)`, both sides keep the augmented system at full
  rank (rank 25, no drops), and raw coefficients/SEs agree strictly. The
  relaxation (`skip_coef_comparison`, `se_tol_scale=2e-3`) was removed and the
  strict case passes. Remaining latent (unexercised) deviations under genuine
  rank deficiency are documented in `GAM_NOT_IMPLEMENTED.md`.

### General-family unsupported surfaces

- [ ] Port multi-predictor general-family `terms=`/`exclude=` coefficient-block
  selection from `predict.gam` before enabling the public surface.
- [x] Port or explicitly scope the remaining general-family `Sl` branches.
  Scoped 2026-08-14 (second pass): non-reparameterized single-/multi-penalty
  blocks stay explicitly guarded (`NotImplementedError`); the nonlinear
  (`linear=FALSE`) machinery is kept as **extension-only** — it mirrors the
  upstream adaptive-smooth `Sl` structure and is covered by
  `tests/families/test_mgcv_gamlss_nonlinear_sl.py`, but no in-tree smooth
  emits `general_family_nonlinear_sl`, so no real fit reaches it (documented
  in `GAM_NOT_IMPLEMENTED.md`).

- [ ] Decide whether formula-list/general-family data-aware `.` shorthand is in the
  intended supported surface. If yes, port it exactly; otherwise retain the explicit
  guard and document it publicly.
- [x] Decide whether multiple `offset(...)` terms per predictor should be ported or
  remain explicitly unsupported. Resolved 2026-08-14: upstream keeps only the
  first offset (interpret.gam0 single-slot assignment, verified in
  `debug/multi_offset_probe.R`); NAMpy mirrors that including the R warning.

## P2 — Broaden confidence after P0/P1 are green

- [ ] Run touched-file `ruff`, `isort --check-only`, and `py_compile` for every logical
  commit. Do not run Black as a check; format only the files that actually need it.
- [ ] Add neural owner tests for any architecture-specific failure discovered by the
  smoke matrix rather than growing one monolithic test.
- [ ] Add explicit multi-output regression coverage beyond LinReg if multi-output is
  intended for every public regressor.
- [ ] Add packaging/install smoke coverage ensuring `pretab` and neural optional
  dependencies are present in the documented environment.
- [ ] Update user documentation for:

  - supported GAM optimizers/families,
  - intentional unsupported GAM branches,
  - prediction `terms`/`exclude`/`iterms`,
  - neural estimator clone/parameter behavior,
  - multi-output regression,
  - and SplineNAM preprocessing requirements.

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
