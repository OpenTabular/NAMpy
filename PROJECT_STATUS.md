# NAMpy Project Status and Verification Ledger

- Snapshot date: 2026-08-14
- Branch: `mgcv`
- Base commit: `97a253021c4ddb655725358962c0bb1db873d271`

## Purpose of this document

This is the durable handoff for the current NAMpy working tree. It records:

- what is implemented,
- what has actually been executed and observed to pass,
- which apparent gaps have been removed from test metadata,
- which surfaces are implemented but do not yet have a retained passing run,
- which unsupported branches are deliberate,
- and exactly when an already-passing test needs to be rerun.

The verification ledger is intentionally stricter than the test source. A test being
present, collected, or absent from an `xfail` registry is not proof that it passed.
Only commands with a retained result are listed as verified.

## Status vocabulary

- **Verified**: a targeted command was run against this working tree and passed.
- **Implemented, not revalidated**: code and tests exist, but there is no retained
  passing result for the current implementation snapshot.
- **Known gap**: NAMpy intends to support the surface, but strict `mgcv` parity is not
  yet complete.
- **Explicitly unsupported**: the code raises deliberately instead of using a
  heuristic or partial implementation.
- **Upstream unsupported**: `mgcv` itself rejects the surface; this is not a NAMpy
  implementation backlog unless upstream behavior changes.

## Executive summary

NAMpy currently has two active development streams:

1. The GAM subsystem is a broad Python port of `mgcv`, with owner-level tests from
   formula parsing through constructors, fitting, smoothing optimization, prediction,
   inference, diagnostics, and parity serialization.
2. The neural-model subsystem is undergoing a substantial behavioral and
   scikit-learn-contract cleanup, with new architecture, wrapper, and fit-smoke tests.

The most recently completed GAM issue was the strict port of the relevant
`mgcv::gdi1()` path. The Gamma REML and ML BFGS lifecycle cases that previously
diverged now pass strict lifecycle parity. The cause was an incorrect state boundary
around `gdiPK()`, not an absent BLAS package.

The current working tree is large and uncommitted: it includes changes across GAM,
neural models, tests, and debug probes. Consequently, the verification entries below
apply to this exact snapshot. A later edit to an owning subsystem invalidates only the
related entries, as described under [When to rerun](#when-to-rerun).

No full test suite was run as part of this recorded workstream, in accordance with
the repository policy requiring the smallest sufficient pytest slice.

## Repository architecture

### GAM subsystem

The GAM data flow remains:

`TermSpec` → runtime term → constructed term → compiled predictor → fit core →
post-fit/prediction/diagnostics.

The vendored sources under `mgcv/` are the behavioral specification. Raw constructor
representations are compared through `mgcv`-relevant invariants when eigenspace
orientation is mathematically indeterminate; behavioral surfaces remain strict.

The primary owner map remains [tests/SUBSYSTEM_COVERAGE.md](tests/SUBSYSTEM_COVERAGE.md),
but its Stage 6–7 and Stage 14 backlog text is stale relative to the current working
tree. In particular, the current raw-constructor known-gap sets are empty and the
joint trace test is strict rather than marked as an expected failure. Updating that
document is an explicit TODO.

### Neural-model subsystem

The neural models retain the layered structure:

- `nampy/basemodels/`: PyTorch modules and the Lightning task wrapper,
- `nampy/models/`: scikit-learn-style public estimators,
- `nampy/configs/`: configuration dataclasses,
- `nampy/arch_utils/`: shared embeddings, splines, attention, and architecture helpers.

New focused tests live under `tests/neural/`. They are included in pytest discovery by
the current `pyproject.toml`, but they have not yet been promoted to **Verified** in
this ledger.

## Implemented work in the current tree

### 1. `gdi1` operation-order and state-boundary parity

Status: **Verified for the targeted cases listed below**.

The port now follows the relevant upstream operation structure:

- `mgcv/src/gdi.c::multSk`,
- `mgcv/src/gdi.c::applyPt`,
- `mgcv/src/gdi.c::applyP`,
- `mgcv/src/gdi.c::ift1`,
- `mgcv/src/gdi.c::get_bSb`,
- `mgcv/src/gdi.c::gdi1`,
- `mgcv/src/mat.c::mgcv_mmult`,
- `mgcv/src/mat.c::mgcv_forwardsolve`,
- `mgcv/src/mat.c::mgcv_backsolve`,
- and the call/refresh lifecycle in `mgcv/R/gam.fit3.r::gam.fit3` and `bfgs`.

Implemented details:

- The penalized QR state retains `R`, `Vt`, and the negative-weight count so
  `applyP`/`applyPt` can follow the upstream factor path.
- Matrix multiplication and triangular solves use SciPy's low-level `dgemm` and
  `dtrsm` wrappers.
- `ift1` first- and second-derivative blocks preserve upstream packing and operation
  order.
- `get_bSb` uses the upstream accumulation structure instead of algebraically
  condensed NumPy expressions.
- PIRLS preserves the pre-refresh `eta` and `mu` supplied to `gdi1` while allowing
  `gdiPK` to refresh the coefficient representative returned to the caller.
- The pre-refresh state is retained in `FitState` so derivative evaluation does not
  silently switch to the refreshed fitted state.

Root cause fixed:

`mgcv::gam.fit3()` passes pre-refresh `etag` and `mug` into `gdi1()`. The internal
`gdiPK()` call refreshes the coefficient representative, but deviance derivatives in
that call remain tied to the original `etag`/`mug`. NAMpy previously refreshed the
coefficients and then evaluated those derivatives with refreshed `eta`/`mu`. The
cold-start Gamma derivative changed by roughly `4.23e-9`; BFGS finite-difference
initialization amplified this enough to change the lifecycle trace.

This explains why adding a separate `pyblas` dependency would not have fixed the
issue. BLAS call order matters for last-bit behavior, but the observed lifecycle gap
was primarily a state-staging error.

### 2. Joint scale and family-parameter optimization

Status: **partly verified; broader branch matrix not yet revalidated**.

The current tree contains:

- joint Gaussian smoothing/scale ML and REML objective, gradient, and Hessian paths,
- Gamma joint-scale objective plumbing with a scale reference passed into PIRLS,
- native estimated-theta negative-binomial joint optimization for supported Newton,
  BFGS, and `optim` branches,
- preservation of joint scale information in final-fit/post-fit state,
- and expanded lifecycle registry cases for Gaussian/Gamma/negative-binomial branch
  coverage.

Verified within this stream:

- Gamma REML BFGS,
- Gamma ML BFGS,
- Gamma REML Newton as a neighboring lifecycle check,
- Poisson REML BFGS,
- Binomial REML BFGS,
- and a Poisson fixed-smoothing `gam.fit3` inner-state comparison.

Not yet verified by a retained run:

- Gamma ML Newton,
- Gamma ML `optim`,
- Gamma REML `optim`,
- estimated-theta negative-binomial ML Newton,
- estimated-theta negative-binomial ML BFGS,
- estimated-theta negative-binomial REML BFGS,
- estimated-theta negative-binomial REML `optim`,
- and the new Gaussian noncanonical joint-scale branches.

Estimated-theta negative-binomial ML `optim` is explicitly guarded because the exact
R `stats::optim` L-BFGS-B behavior at its flat joint boundary has not been ported.

### 3. GAM prediction, term labeling, and public API work

Status: **implemented, not revalidated as a group**.

The current tree contains:

- canonical `mgcv` term-label normalization in `nampy/gam/term_labels.py`,
- `terms=` and `exclude=` coefficient-block filtering for ordinary GAM prediction,
- `type="iterms"` support and separate term standard errors,
- factor-smooth term-contribution handling,
- updated formula metadata for transformed terms and offsets,
- public API delegation/plot/summary/BIC contract coverage,
- and unconditional covariance assembly that includes an optimized joint scale when
  the upstream parameterization does so.

The prediction/inference/diagnostics test module currently has empty explicit gap
dictionaries for prediction, unconditional prediction, ANOVA, residuals, and
`k.check`. This is test-metadata state, not a retained full-module passing result.

Multi-predictor general-family `terms=`/`exclude=` filtering remains explicitly
unsupported until coefficient-block selection can mirror `predict.gam` exactly.

### 4. Formula, tensor, and natural-parameter constructor work

Status: **implemented, not revalidated as a group**.

The current tree contains:

- vector-valued `fx=` for `te()`/`ti()`, including upstream-style wrong-dimension
  warning behavior,
- transformed numeric smooth `by=` expressions materialized into hidden columns,
- tensor marginal iteration hardened with strict length matching,
- a base-R/LINPACK-style QR path for natural parameterization,
- explicit Netlib-style triangular-solve operation order,
- and preserved eigenspace-invariant handling where raw orientation is not unique.

The current raw-constructor registry has no active `KNOWN_GAP_REASONS`. Three
factor-smooth full-rank shrinkage bases are excluded because upstream `mgcv` rejects
them as well. The empty registry should not be interpreted as a recorded pass of the
entire raw-constructor matrix in this snapshot.

### 5. Neural task and architecture work

Status: **implemented, not yet verified**.

The current tree contains:

- explicit task semantics separate from output width in the Lightning `TaskModel`,
- multi-output regression support,
- regression target/prediction shape validation,
- unregularized RMSE reporting while penalties remain part of the training objective,
- canonical aggregation of `*_penalty` and `*_regularizer` outputs,
- fixes to NATT and NAMformer feature-token and interaction routing,
- isolated activation-module defaults in transformer configuration dataclasses,
- NodeGAM penalty naming aligned with the shared contract,
- SplineNAM boundary behavior, learnable-knot gradients, multi-output terms,
  interactions, feature-name guards, and smoothness penalties,
- shared scikit-learn data validation in `nampy/models/_sklearn_data.py`,
- shared `get_params`/`set_params`/clone ownership in
  `nampy/models/_sklearn_params.py`,
- QNAM default quantile-family ownership,
- and new architecture, wrapper, task, spline, and estimator fit-smoke tests under
  `tests/neural/`.

These test files are new and currently untracked. They need their first focused run
before any neural behavior is marked **Verified**.

## Current test-metadata state

The following describes the current source registries, not execution evidence:

| Registry/surface | Current metadata state |
| --- | --- |
| Optimization lifecycle registry | 26 cases; zero `status="known_gap"` entries |
| Raw constructor registry | Empty active known-gap sets |
| General-family parity registry | Empty `_GENERAL_KNOWN_GAP_TAGS` |
| Prediction/inference/diagnostics gaps | All five explicit gap dictionaries are empty |
| Joint branch trace tests | Strict tests; no `xfail` marker in the file |
| Post-fit general-family endpoint | `gaulss_select_true_cr` optimized endpoint still has a local `xfail`; fixed-endpoint post-processing is tested strictly |
| Tracked underdetermined fit | `factor_smooth_sz` retains relaxed/representation-aware handling because its coefficient/covariance representative is underdetermined |

## Verification ledger: tests that have passed

### Test environment

The authoritative final rerun used:

- Python: `/home/ad32/miniconda3/envs/nampy/bin/python` (Python 3.11),
- pytest: `/home/ad32/miniconda3/envs/nampy/bin/pytest`,
- R/mgcv through the repository parity harness,
- repository working directory: `/home/ad32/projects/package/NAMpy`.

The unqualified system `pytest` currently uses `/usr/bin/python3` and could not collect
the same tests because that interpreter does not have the declared `pretab` dependency
installed. That collection error is an environment mismatch, not a GAM test failure.
Use the `nampy` environment or install the project dependencies before reproducing the
ledger.

### Final retained focused run

```bash
/home/ad32/miniconda3/envs/nampy/bin/pytest \
  tests/regressions/test_gam_mgcv_patch_regressions.py::test_gdi_pk_setup_and_ift1_match_signed_weight_inverse_root \
  'tests/optimization/test_mgcv_optimization_lifecycle_parity.py::test_supported_optimization_lifecycle_matches_mgcv[gamma_reml_bfgs_joint_scale_cr]' \
  'tests/optimization/test_mgcv_optimization_lifecycle_parity.py::test_supported_optimization_lifecycle_matches_mgcv[gamma_ml_bfgs_joint_scale_cr]' \
  tests/regressions/test_gam_optimization_lifecycle_contracts.py -v
```

Result: **6 passed in 20.36 seconds**.

This verifies:

- signed/negative-weight `gdiPK` plus `ift1` inverse-factor behavior,
- the complete Gamma REML BFGS lifecycle,
- the complete Gamma ML BFGS lifecycle,
- lifecycle-registry branch coverage expectations,
- absence of active lifecycle known-gap entries,
- and lifecycle case-ID uniqueness.

### Earlier targeted neighboring runs retained from the same workstream

```bash
pytest tests/optimization/test_mgcv_fixed_inner_fit_parity.py::test_poisson_gam_fit3_fixed_sp_inner_state_matches_mgcv -v
```

Result: **passed**.

```bash
pytest \
  'tests/optimization/test_mgcv_optimization_lifecycle_parity.py::test_supported_optimization_lifecycle_matches_mgcv[gamma_reml_newton_joint_scale_cr]' \
  'tests/optimization/test_mgcv_optimization_lifecycle_parity.py::test_supported_optimization_lifecycle_matches_mgcv[poisson_reml_bfgs_two_cr]' \
  'tests/optimization/test_mgcv_optimization_lifecycle_parity.py::test_supported_optimization_lifecycle_matches_mgcv[binomial_reml_bfgs_cr]' -v
```

Result: **3 passed**.

The signed-weight regression, both Gamma BFGS lifecycle cases, and the lifecycle
registry contracts were also run separately while localizing the issue; the combined
six-test result above supersedes those duplicate runs.

### Static validation retained from the final `gdi1` snapshot

The touched Python files for the `gdi1` change passed:

- `ruff check`,
- `isort --check-only`,
- `python -m py_compile`,
- and `git diff --check`.

These checks covered the `gdi1` derivative implementation, QR state, PIRLS state,
lifecycle registry/tests, and the preserved Gamma debug probe. They do not certify all
other modified files in the large working tree.

### Preserved diagnostic evidence

`debug/gamma_bfgs_initial_probe.R` and `debug/gamma_bfgs_probe.py` preserve the
pre-/post-refresh derivative split and BFGS initialization probe. After the fix, the
observed Gamma trace differences were at floating-point noise scale, with the largest
recorded rollback-row log-smoothing-parameter difference around `4.6e-10`, well within
the strict lifecycle tolerance of `5e-7`.

The debug scripts are diagnostic evidence; the strict pytest lifecycle cases are the
regression authority.

## What does not need to be rerun now

Do not rerun the final six-test `gdi1`/Gamma BFGS command merely to reconfirm this
unchanged snapshot. Likewise, the three neighboring lifecycle cases and Poisson
fixed-inner case do not need another run unless an invalidating file changes.

This saves the following known-good checks:

| Verified surface | Retained result | Primary invalidators |
| --- | --- | --- |
| Signed-weight `gdiPK`/`ift1` | Passed | `stacked_qr.py`, `derivatives.py`, signed-weight QR helpers |
| Gamma REML BFGS lifecycle | Passed | PIRLS state, Gamma objective, `gdi1`, BFGS, driver, lifecycle harness |
| Gamma ML BFGS lifecycle | Passed | Same as above plus ML score assembly |
| Gamma REML Newton lifecycle | Passed | Joint-scale objective/Hessian, Newton optimizer, driver |
| Poisson REML BFGS lifecycle | Passed | PIRLS derivatives, BFGS, common outer-driver code |
| Binomial REML BFGS lifecycle | Passed | PIRLS derivatives, BFGS, common outer-driver code |
| Poisson fixed-sp inner state | Passed | PIRLS loop, working-state construction, post-loop refresh |
| Lifecycle registry contracts | Passed | `_optimization_lifecycle_registry.py` or its contract test |

## When to rerun

Rerun only the smallest affected slice when one of these conditions occurs:

1. Changes to `nampy/gam/smoothing_selection/criteria/pirls/derivatives.py`,
   `nampy/gam/fit/solvers/irls_core.py`, or
   `nampy/gam/fit/linalg/stacked_qr.py` invalidate the signed-weight and Gamma
   lifecycle entries.
2. Changes to `bfgs_strict.py`, `driver.py`, `objectives.py`, or joint scale assembly
   invalidate the lifecycle entries but not necessarily the standalone signed-weight
   regression.
3. Changes to the lifecycle registry or parity serializer invalidate only the
   lifecycle/registry assertions.
4. A change in R, `mgcv`, SciPy, BLAS, or LAPACK versions invalidates last-bit numeric
   claims and requires the relevant parity slices again.
5. Changes limited to neural code do not invalidate the retained GAM results.
6. Documentation-only changes do not invalidate any test entry.

## Implemented surfaces still awaiting first validation

The following work must not be reported as passing yet:

- all new `tests/neural/` architecture, sklearn, task, spline, and fit-smoke coverage,
- Gaussian joint-scale lifecycle branches,
- the newly registered Gamma ML Newton/`optim` and Gamma REML `optim` branches,
- the newly registered estimated-theta negative-binomial lifecycle branches,
- transformed numeric smooth `by=` parity,
- vector-valued tensor `fx=` parity,
- the new base-R/LINPACK natural-parameterization path,
- factor-smooth term contribution filtering,
- `terms=`/`exclude=`/`iterms` public prediction behavior,
- new public plotting/summary/BIC contracts,
- and the broadened post-fit/unconditional covariance slices.

Targeted commands for these are maintained in [todo.md](todo.md).

## Known gaps and deliberate unsupported behavior

### Genuine remaining parity work

- **Estimated-theta negative-binomial ML `optim`**: guarded until the exact R
  L-BFGS-B flat-boundary behavior is ported.
- **`gaulss_select_true_cr` optimized endpoint**: the current local `xfail` attributes
  the difference to `mgcv::initial.spg()` eigenspace/solver orientation. Fixed-endpoint
  post-processing is strict, so the remaining issue is endpoint selection rather than
  downstream assembly.
- **`factor_smooth_sz` covariance representative**: fit/prediction/criterion behavior
  is covered with invariant or relaxed comparisons, but the underdetermined raw
  coefficient/covariance representative remains a sensitive surface.
- **General-family term filters**: `terms=` and `exclude=` remain unsupported for
  multi-predictor general-family models.
- **General-family `Sl` variants**: several non-reparameterized single/multi-penalty
  and nonlinear reparameterized blocks still raise explicitly.
- **Formula-list dot shorthand**: data-aware `.` is supported for ordinary formulas
  but remains explicitly unsupported for formula-list/general-family models.
- **Multiple offsets in one predictor**: still explicitly unsupported.

### Deliberate non-backlog behavior

- Random-effect smooths linked with `id=` remain unsupported because upstream `mgcv`
  also rejects them.
- Full-rank shrinkage bases under `bs="fs"` remain excluded where upstream `mgcv`
  rejects the same construction.
- Repeated-eigenspace raw orientation should continue to be tested with invariant
  comparisons rather than platform-specific solver hooks.
- A separate `pyblas` dependency is not currently warranted; the port uses SciPy BLAS
  wrappers where operation order matters.

## Worktree and release readiness

The project is not release-ready from this snapshot because:

- the worktree contains a broad mixture of uncommitted GAM and neural changes,
- several new source files and all `tests/neural/` files are untracked,
- only a narrow subset of the changed surfaces has retained passing evidence,
- the system Python environment is incomplete relative to project dependencies,
- and the existing subsystem coverage backlog needs reconciliation with current test
  metadata.

The next work should follow [todo.md](todo.md), starting with first-time targeted
validation of the new neural and unverified GAM surfaces, then resolving the explicit
parity gaps, and finally splitting the working tree into reviewable commits.

---

## 2026-08-14 (afternoon) — strict-parity fix workstream

Companion documents produced by this workstream:
[GAM_IMPLEMENTED.md](GAM_IMPLEMENTED.md) and
[GAM_NOT_IMPLEMENTED.md](GAM_NOT_IMPLEMENTED.md).

### Fixes (with upstream references)

1. **Multivariate `s()` default `k`** — omitted `k` on `tp`/`ts` now defers to
   the constructor via `k = -1`, resolved by `default_tprs_k` exactly as
   `mgcv/R/smooth.r:1316-1323` (`M + c(8,27,100)[min(d,3)]`, minimum `M+1`).
   Previously a flat default of 10 silently changed all `d > 1` models.
   Owner test: `test_formula_multivariate_tp_default_k_defers_to_mgcv_constructor_rule`.
2. **`pc=` guards** — explicitly unsupported on `te`/`ti`/`re`/`fs`/`sz`
   (previously silently dropped); univariate `cc/cr/cs/ps/tp/ts` support
   unchanged. Guard tests added.
3. **Multiple `offset()`** — upstream `interpret.gam0` (`mgcv/R/mgcv.r:387-389`)
   keeps only the first offset with base R's vector-assignment warning;
   verified against mgcv 1.9-4 (`debug/multi_offset_probe.R`). NAMpy now emits
   the same warning at truncation. Owner test added.
4. **`nei=` removed** (NCV remnant); **`gacv.cp` alias removed** (now rejected;
   mgcv maps it to the unimplemented GACV criterion); **`t2()` guarded** with a
   clear message; **orphaned `gp` primitives deleted**.
5. **Unconditional `Vc` for efs/optim** — mirrors the upstream post-processor
   split: GLM/extended (`gam.fit3.post.proc`) leaves `V.sp <- edf2 <- Vc <- NULL`
   without `db.drho` (`mgcv/R/gam.fit3.r:1053`); general families
   (`gam.fit5.post.proc`) always return `Vc`, degenerating to `Vc == Vb` at
   deriv=0 (`mgcv/R/gam.fit4.r:1648,1685-1690,1714-1715`). The general-family
   postprocess now suppresses the smoothing-uncertainty correction for
   efs/optim (locally available `REML2`/`db_drho` state must not substitute
   for the absent upstream `outer.info$hess`).
6. **Gaussian noncanonical-link ML/REML** — profiled-scale branch added
   (`_gdi2_gaussian_joint_kernel`; closed form `phi = Dp/(n - gamma*Mp*remlInd)`
   from `gam.fit3.r:628-637` + `:2503-2508`), the PIRLS criterion value for
   gaussian now evaluates the joint `gam.fit3` score at the profiled scale
   (replacing the Pearson/(n-Mp) plug-in, which is mgcv's P-REML scale), and
   dispatch routes gaussian through the exact PIRLS gradient/Hessian path.
   The joint (log sp, log scale) optimizer path was already upstream-exact.
7. **Negbin estimated-theta joint path** — stale guards removed (offset, prior
   weights, non-formula construction; vestiges of a removed Rscript shim).
   Upstream threads both through `gam.fit4` with no special-casing
   (`mgcv/R/gam.fit4.r:240-244`, `:509`, `:561`, `:730`). The ML+`optim`
   L-BFGS-B boundary guard is retained. `tests/parity/mgcv_outer_trace.R` and
   `_run_mgcv_outer_trace` gained weights support (the harness previously ran
   the expected trace unweighted).
8. **`factor_smooth_sz`** — stale relaxation removed; strict coefficient/SE
   comparison passes (no rank drop occurs on either side; deltas ~2.7e-11).

### Verification evidence

- `gaulss_select_true_cr` — endpoint verified orientation-indeterminate inside
  mgcv itself (`debug/gaulss_select_initial_spg_probe.py`): mgcv on the
  mirrored basis reproduces NAMpy's endpoint exactly; criteria at both
  endpoints agree to 4.7e-6 with gradients ~4e-5. The xfail stays, now with
  quantitative evidence in its reason.
- Gaussian noncanonical probe (`debug/gaussian_noncanonical_reml_probe.py`):
  newton REML/ML scores match mgcv to 8 decimals; EFS sp matches to 1e-7.

### Retained passing runs (this snapshot)

```bash
pytest tests/optimization/test_gam_unsupported_branch_guards.py -k 't2_smooth or pc_guard or gacv_cp' -v   # 7 passed
pytest tests/parity/test_gam_spec_build_owner_contracts.py -k 'multiple_offsets or default_k' -v          # 3 passed
pytest tests/parity/test_mgcv_parity_failing_and_warnings.py -k factor_smooth_sz -v                       # 1 passed (strict)
pytest 'tests/optimization/test_mgcv_optimization_lifecycle_parity.py::test_supported_optimization_lifecycle_matches_mgcv[negbin_est_reml_newton_joint_theta_weighted_cr]' -v  # passed
pytest 'tests/optimization/test_mgcv_optimization_lifecycle_parity.py::test_supported_optimization_lifecycle_matches_mgcv[negbin_est_reml_newton_joint_theta_cr]' -v          # passed (regenerated cache)
pytest tests/regressions/test_gam_mgcv_patch_regressions.py::test_negbin_estimated_theta_joint_path_accepts_arrays_offset_and_weights -v  # passed
pytest tests/optimization/test_gam_fit_backend_owner_contracts.py -v                                       # 10 passed (incl. new Vc efs-gate contract)
pytest tests/regressions/test_gam_optimization_lifecycle_contracts.py tests/optimization/test_mgcv_optimization_lifecycle_parity.py -k 'contract or registry or gaussian_log or gaussian_inverse' -v  # 7 passed
pytest <gdi1 signed-weight regression + gamma_reml_bfgs/gamma_ml_bfgs/gamma_reml_newton lifecycle> -v      # 4 passed (revalidation after derivatives.py edits)
pytest tests/parity/test_gam_spec_build_owner_contracts.py tests/optimization/test_gam_unsupported_branch_guards.py  # 34 passed
pytest tests/optimization/test_mgcv_postprocessing_final_fit_parity.py -k gam_fit5  # 6 passed, 1 xfail, 1 FAIL (pre-existing, see below)
```

All commands used `/home/ad32/miniconda3/envs/nampy/bin/pytest`. Touched files
passed `ruff check`, `isort --check-only`, and `py_compile`.

### New failure surfaced by first validation (pre-existing, not caused here)

`test_gam_fit5_postprocessing_final_fit_matches_mgcv[gammals_select_true_cr]`:
`edf2_total` 3.690821 vs mgcv 3.691378 at tolerance 5e-4 (diff 5.6e-4).
Verified independent of the new efs/optim Vc gate by disabling the gate and
re-running (identical numbers). Tracked in todo.md P1.

### Registry state after this workstream

31 lifecycle cases (was 26): +`negbin_est_reml_newton_joint_theta_weighted_cr`,
+4 Gaussian noncanonical joint-scale cases. Zero known-gap entries. The
`_make_negbin_data` factory gained a `w` column (y draws unchanged; cached
negbin snapshots regenerate).

### Addendum (2026-08-14, later): `gammals_select_true_cr` edf2_total resolved

Localized via `debug/gammals_select_edf2_probe.py`. The edf2 assembly is
exact — the 5.6e-4 difference is entirely the optimized endpoint:

- Both sides fire the `sum(edf2) > sum(edf1)` cap (`mgcv/R/gam.fit4.r:1715`),
  so `edf2 == edf1`, and edf1 is the endpoint-sensitive scalar.
- Endpoints differ in the select-penalty direction (sp 1387.572 vs 1385.902).
  mgcv refit on the mirrored basis (`x -> -x`) lands on NAMpy's endpoint and
  reproduces NAMpy's edf1/edf2 sums to 2.3e-7. NAMpy's endpoint is the
  better-converged optimum (|grad| 3.8e-7 vs 1.3e-4; criterion diff 7e-8).
- Classification: same `initial.spg` orientation-indeterminacy as
  `gaulss_select_true_cr`. Added `gammals_select_true_cr` to
  `_GENERAL_OPTIMIZED_ENDPOINT_KNOWN_GAP_TAGS` with the evidence in the xfail
  reason, and generalized the strict fixed-endpoint post-processing test to
  cover both select cases
  (`test_gam_fit5_select_true_postprocessing_at_mgcv_endpoint_matches_mgcv`),
  asserting Vc at covariance tolerance and `edf2_total` at 5e-6.

Retained run:

```bash
/home/ad32/miniconda3/envs/nampy/bin/pytest \
  tests/optimization/test_mgcv_postprocessing_final_fit_parity.py -k 'gam_fit5' -v
```

Result: **7 passed, 2 xfailed, 0 failed** (previously 6 passed, 1 xfailed,
1 failed).

---

## 2026-08-15 — release-fix workstream (approved plan)

Plan: correctness + hygiene fixes, summary.gam port, P0 validation ladder,
worktree commit split. All commands via `/home/ad32/miniconda3/envs/nampy/bin/pytest`.

### Landed changes

1. **Packaging**: `scipy>=1.16` bound (pyproject.toml, requirements.txt) —
   `dstevd` absent before 1.16.
2. **Kwarg validation**: `GAM.__init__` rejects unknown kwargs (23-key
   allowlist); guard test added.
3. **Dead scaffolding removed** in 4 test files (empty gap registries, dead
   xfail plumbing); `tests/SUBSYSTEM_COVERAGE.md` refreshed (Stage 6-7/14/21).
4. **Rank-deficient Gaussian representative** (upstream gauge,
   mgcv/src/gdi.c:2253-2292): Vp now built from the canonical gdi1 rV
   (same factorization as coefficients); NAMpy-only gauge pin disarmed on the
   gaussian path; dead `Mp` parameter removed. New strict drop-gauge
   regression vs live mgcv (exact zero coef AND zero Vp row at mgcv's dropped
   position). NEW FINDING: default side conditions delete aliased parametric
   columns (upstream gam.side does not) — tracked as an open item; with
   `apply_side_conditions=False` the solver path is upstream-exact.
5. **PIRLS rank tolerance**: `irls_core` default now `eps*100`
   (gam.fit3.r:131), consistent with the gdiPK kernel; stale docstring fixed.
   Re-validated: gdi1 six-test command (6 passed), Gamma+negbin lifecycle
   (7 passed), non-Gaussian unconditional covariance (4 passed).
6. **Result schema**: `GAMFitResult` carries `cov_unconditional`,
   `cov_unconditional_space`, `edf2`; public `GAM.edf1()`; pinned schema test
   updated. logLik/AIC/BIC parity-tested for poisson, negbin (fixed + est
   theta), gaulss (`test_loglik_aic_bic_match_mgcv`, 4 passed).
7. **summary.gam port**: `nampy/gam/inference/null_deviance.py` (GLM closed
   form, offset-corrected intercept-only IRLS refit, negbin find.null.dev
   port, gaulss/gammals postproc hooks — probe matched mgcv at 1e-15/1e-7),
   `nampy/gam/inference/summary.py` (`GAMSummary` + `summary_gam` with
   dispersion/freq/re_test per mgcv.r:3890-4025; `_term_table` gained
   `re_test=` and a dispersion-rescale fix), `diagnostics/summary.py`
   rewritten to the print.summary.gam layout, `GAM.summary()` returns the
   object. `mgcv_snapshot.R` gained the summary block + null_deviance/rank/
   scale_estimated (`_SNAPSHOT_CACHE_VERSION` bumped to 7).

### Verification ledger additions (all passed)

- `tests/optimization/test_mgcv_optimization_lifecycle_parity.py` A1+A2 slice: 7 passed (twice: baseline + post rank-tol).
- B1/B2 prediction slices: 9 passed; B3 guard + C1/C2 xfails file slice: 5 passed; D1: 4 passed (twice).
- `tests/regressions/test_gam_mgcv_patch_regressions.py::test_rank_deficient_gaussian_fit_matches_mgcv_drop_gauge`: passed (after fix; failed before, as designed).
- `tests/parity/test_gam_results_api_stage_owner_contracts.py`: 3 passed (schema).
- `tests/parity/test_mgcv_remaining_gap_xfails.py -k 'loglik_aic_bic or bic_matches'`: 5 passed.
- `tests/diagnostics/test_gam_summary_owner_contracts.py`: 5 passed (null-deviance branches + dispersion/freq/re_test contracts).
- `tests/parity/test_mgcv_summary_parity.py`: 6 passed (gaussian, gaussian+offset, poisson, gamma joint scale, negbin est theta, gaulss — p.table at 1e-6, scalars at 1e-15..1e-5).
- `tests/parity/test_mgcv_prediction_inference_diagnostics_parity.py -k anova`: 20 passed (post `_term_table` refactor, cache v7 regeneration).
- B4 slice: 7 passed (post summary port).
- `tests/smooths/test_mgcv_nat_param_parity.py` (new, C3): 2 passed.
- C4 raw-constructor slices: `-k 'fs or sz'` 20 passed; `-k 'tp or cr'` 31 passed.
- Kwarg guard test, Vc efs-gate owner file (10), diagnostics owner file (15 with new layout tests), general-family summary parity test: passed.
- WS2c blast radius: covariance owners (4), tracked model cases (2), promoted re/fs/sz snapshots (6), magic postprocessing (12): passed.

Touched files passed `ruff check`, `isort --check-only`, `py_compile`.

### Commits

(Hashes recorded below after the split.)
- `913ff3e` — gam/fit+smoothing: gdi1 state boundary, joint outer
  optimization, upstream Vc and rank handling.
- `21291a0` — gam/constructors: mgcv-exact defaults and guards for formula
  smooths.
- `56eac49` — gam/public surface: summary.gam port, null deviance, prediction
  filters, schema.
- `70ac3e1` — neural: task semantics, sklearn contracts, first-validation
  test suite.
- Tests/debug/docs/packaging land as the commit that contains this ledger
  entry (the fifth, HEAD of this sequence on branch `mgcv`).

### Addendum (2026-08-15, later): audit reconciliation

REVIEW.md (independent pre-commit audit) reconciled against the committed
tree; full status table prepended there. Verification runs:

- `gaussian_reml_newton_fs_xt_ps` lifecycle: **FAILED** (confirmed audit
  finding 1 — fs null-space sp order swapped; registry still marks it
  stable). Highest-priority open defect (todo P1).
- `tests/parity/test_mgcv_output_parity.py -k cs`: **3 FAILED, 1 passed**
  (terms max diff 8.6e-5 vs fresh live-R references). Bisect vs base
  `97a2530` (which passes): plain-cs failures introduced by the
  prior-session `symmetrize_lower_triangle=True` change in
  `nampy/splines/univariate/cr.py`; `transformed_cs` has a second
  unlocalized cause in the same constructor changes. This session's fixes
  exonerated by consistent-group bisect. The defect was previously masked
  by stale v6 snapshot-cache entries (mgcv_snapshot.R had been modified
  without a cache version bump); the v7 bump exposed it.
- `tests/regressions/test_optimize_driver_mgcv_parity.py`: **2 FAILED,
  12 passed** (stale mocks/contract drift; todo P1).
- `_fallback_single_smooth_edf` (fit/state.py:296) and
  `_fs_term_penalty_adjustment` (predict/predictions.py) confirmed present;
  tracked as policy items in todo P1.

REVIEW.md, tldr.md, and todo.md updated accordingly.
