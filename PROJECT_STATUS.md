# NAMpy Project Status and Verification Ledger

- Snapshot date: 2026-08-17
- Branch: `mgcv`
- Implementation snapshot: `e750f6d`
- Reviewable implementation commits: `46bacbc`, `c25af71`, `e750f6d`

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

The later GAM work has also resolved the fs/cs audit findings, rank-deficient and
near-singular REML behavior, current-SP PIRLS coordinates, and the remaining
known heuristics/silent fallbacks. Stacked QR now uses supported public SciPy
interfaces, with no direct native-library plumbing. The only visible GAM xfail
is the separate `gaulss(select=True)` optimized endpoint; gammals select=True
now passes initialization, optimized fit, prediction, SE, and post-processing
parity strictly.

The GAM parity, portability, tests, CI, API, and retained-probe changes are split
across the three implementation commits listed above. The verification entries
below apply to that snapshot. A later edit to an owning subsystem invalidates
only the related entries, as described under [When to rerun](#when-to-rerun).

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

The primary owner map remains [tests/SUBSYSTEM_COVERAGE.md](tests/SUBSYSTEM_COVERAGE.md)
and has been reconciled with the current stage-local coverage and explicit guards.

### Neural-model subsystem

The neural models retain the layered structure:

- `nampy/basemodels/`: PyTorch modules and the Lightning task wrapper,
- `nampy/models/`: scikit-learn-style public estimators,
- `nampy/configs/`: configuration dataclasses,
- `nampy/arch_utils/`: shared embeddings, splines, attention, and architecture helpers.

The five focused files under `tests/neural/` are tracked and **Verified** as of
2026-08-17: 148 tests pass across architecture, task/multi-output, sklearn,
SplineNAM, and public estimator fit/predict contracts.

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
- Matrix multiplication preserves upstream operand order through a small NumPy
  wrapper, while triangular solves use SciPy's supported public
  `solve_triangular` interface; no direct BLAS/LAPACK wrapper is selected.
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

Status: **Verified for the registered supported lifecycle branches**.

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

Later retained runs cover Gamma ML Newton/`optim`, Gamma REML `optim`, the
supported estimated-theta negative-binomial Newton/BFGS/`optim` combinations,
and the Gaussian log/inverse joint-scale branches. The lifecycle registry now
contains 31 strict supported cases.

Estimated-theta negative-binomial ML `optim` is explicitly guarded because the exact
R `stats::optim` L-BFGS-B behavior at its flat joint boundary has not been ported.

### 3. GAM prediction, term labeling, and public API work

Status: **Verified through focused owner/public-surface slices**.

The current tree contains:

- canonical `mgcv` term-label normalization in `nampy/gam/term_labels.py`,
- `terms=` and `exclude=` coefficient-block filtering for ordinary GAM prediction,
- `type="iterms"` support and separate term standard errors,
- factor-smooth term-contribution handling,
- updated formula metadata for transformed terms and offsets,
- public API delegation/plot/summary/BIC contract coverage,
- and unconditional covariance assembly that includes an optimized joint scale when
  the upstream parameterization does so.

Retained focused runs cover ordinary `terms=`/`exclude=`, `iterms`, factor-smooth
contributions, mixed fixed/free smoothing, `lpmatrix`, summary/BIC/formula
metadata/gam_check, and general-family guard behavior. The empty explicit gap
dictionaries are therefore accompanied by execution evidence for the changed
owners rather than inferred from metadata alone.

Multi-predictor general-family `terms=`/`exclude=` filtering remains explicitly
unsupported until coefficient-block selection can mirror `predict.gam` exactly.

### 4. Formula, tensor, and natural-parameter constructor work

Status: **Verified for the changed constructor/formula owners**.

The current tree contains:

- vector-valued `fx=` for `te()`/`ti()`, including upstream-style wrong-dimension
  warning behavior,
- transformed numeric smooth `by=` expressions materialized into hidden columns,
- tensor marginal iteration hardened with strict length matching,
- a base-R/LINPACK-style QR path for natural parameterization,
- explicit Netlib-style triangular-solve operation order,
- and preserved eigenspace-invariant handling where raw orientation is not unique.

Retained runs cover transformed numeric `by=`, vector-valued tensor `fx=`, the
natural-parameterization owner tests, and the affected raw-constructor fs/sz and
tp/cr slices. Three factor-smooth full-rank shrinkage bases remain excluded because
upstream `mgcv` rejects them as well.

### 5. Neural task and architecture work

Status: **Verified for the focused first-validation matrix (2026-08-17)**.

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

All five test files are tracked. Their exact focused runs produced 25, 2, 41, 7,
and 73 passes respectively (148 total); warnings were limited to expected
Lightning logging/worker notices, transformer nested-tensor notices, small-batch
NodeGAM initialization, and small smoke-sample metric notices.

## Current test-metadata state

The following describes the current source registries, not execution evidence:

| Registry/surface | Current metadata state |
| --- | --- |
| Optimization lifecycle registry | 31 cases; zero `status="known_gap"` entries; fs exchangeable block declared as an invariant |
| Raw constructor registry | Empty active known-gap sets |
| General-family parity registry | Empty `_GENERAL_KNOWN_GAP_TAGS` |
| Prediction/inference/diagnostics gaps | All five explicit gap dictionaries are empty |
| Joint branch trace tests | Strict tests; no `xfail` marker in the file |
| General-family optimized endpoint | Only `gaulss_select_true_cr` retains a local xfail; `gammals_select_true_cr` is strict after the upstream `Sl.setup` lower-triangle correction |
| Factor smooth | `factor_smooth_sz` uses strict coefficient/SE comparison; fs zero-eigenspace smoothing-parameter order is compared through its declared exchangeable invariant |

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

## Implemented surfaces still awaiting additional validation

The first-validation GAM and neural owner slices listed in `todo.md` are now
recorded. Remaining validation work is narrower:

- optionally automate the now-passing built-wheel install/import smoke in CI
  (the manual temporary-venv check imported mandatory `pretab` and instantiated
  `LinRegRegressor`),
- exercise multi-output fitting beyond LinReg only if that is intended as a
  contract for every public neural regressor,
- and obtain the configured Linux/macOS/Windows CI results for the new
  portability job after the guard and workflow are committed.

Targeted commands and commit hygiene remain in [todo.md](todo.md).

## Known gaps and deliberate unsupported behavior

### Supported-surface deviations

- **`gaulss(select=True)` optimized endpoint**: this is the only remaining
  visible GAM xfail; fixed/shared-endpoint post-processing is strict. The
  former gammals endpoint/prediction gap was a real multi-penalty `Sl.setup`
  triangle-convention defect and is now fixed. The gaulss start has now been
  localized to an upstream sign indeterminacy: `estimate.gam` transforms
  `G$X` using `Sl.setup`'s arbitrary-sign symmetric-eigen vectors, then passes
  unreparameterized `G$Eb` to `initial.spg`. No platform/sign-forcing fix is
  permitted, so this evidence-backed xfail remains explicit.

No unclassified algorithmic defect in the currently declared GAM surface is
left by this audit.

### Deliberate non-backlog behavior

- Random-effect smooths linked with `id=` remain unsupported because upstream `mgcv`
  also rejects them.
- Full-rank shrinkage bases under `bs="fs"` remain excluded where upstream `mgcv`
  rejects the same construction.
- Repeated-eigenspace raw orientation should continue to be tested with invariant
  comparisons rather than platform-specific solver hooks.
- Stacked penalized QR uses SciPy's supported high-level pivoted-QR and triangular-solve
  interfaces. Raw LAPACK work arrays, input-`JPVT` reuse, `ctypes` library loading, and
  BLAS-specific accumulation order are deliberately outside the behavioral parity
  contract.
- Estimated-theta negative-binomial ML + `optim`, general-family term filters,
  formula-list data-aware dot shorthand, unported `Sl` layouts, plot.gam, and
  absent bases/families remain explicit unsupported scope, not active parity
  bugs.

## Worktree and release readiness

The parity/portability work is committed in reviewable units, the portability
guard and six evidence probes are tracked, and the local Linux slices pass. The
remaining release evidence is the hosted macOS/Windows (and clean hosted Linux)
CI result. The public-export contract, manual built-wheel smoke, and neural
first-validation matrix pass; all five neural files are tracked.

The next work should follow [todo.md](todo.md): obtain hosted CI results and
optionally automate the passing wheel-install smoke. The user-facing GAM support
matrix and neural estimator/preprocessing contracts are now documented in
`README.md`.

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
  mgcv itself (`debug/gaulss_select_initial_spg_probe.py`): a mirrored input
  moves mgcv's endpoint to second log-sp `11.79338762`, versus NAMpy
  `11.81049973` and ordinary mgcv `11.91107097`. Criteria at the NAMpy and
  ordinary-mgcv endpoints agree to `3.98e-6`, with gradients about `4e-5`.
  Source tracing localizes the sensitivity to arbitrary `DSYEVR` signs in the
  `Sl.setup` transform of `G$X` combined with unreparameterized `G$Eb` in
  `initial.spg`; the xfail stays because sign/platform forcing is out of scope.
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

### Historical failure surfaced by first validation (now fixed)

`test_gam_fit5_postprocessing_final_fit_matches_mgcv[gammals_select_true_cr]`:
`edf2_total` 3.690821 vs mgcv 3.691378 at tolerance 5e-4 (diff 5.6e-4).
Verified independent of the new efs/optim Vc gate by disabling the gate and
re-running (identical numbers). The 2026-08-17 strict `initial.spg` regression
later localized and fixed its actual `Sl.setup` cause.

### Registry state after this workstream

31 lifecycle cases (was 26): +`negbin_est_reml_newton_joint_theta_weighted_cr`,
+4 Gaussian noncanonical joint-scale cases. Zero known-gap entries. The
`_make_negbin_data` factory gained a `w` column (y draws unchanged; cached
negbin snapshots regenerate).

### Addendum (2026-08-14, superseded 2026-08-17): gammals endpoint

The original probe correctly localized EDF2 to the optimized endpoint, but the
orientation-only classification was incomplete. The later strict
`initial.spg` regression found that `_sl_multi_penalty_block` used the upper
triangle while `mgcv/R/fast-REML.r::Sl.setup` uses the lower triangle. After
that correction the initial/final smoothing parameters, EDF2, predictions, and
post-processing match ordinary mgcv and the gammals xfail was removed.

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
   position). Resolved follow-up: default side conditions now leave aliased
   parametric columns to the solver, matching `mgcv::gam.side`; direct and
   default paths are covered across Gaussian, Poisson, binomial, and Gamma.
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
- `tests/regressions/test_gam_mgcv_patch_regressions.py::test_rank_deficient_pirls_fit_matches_mgcv_drop_gauge`: 6 passed (direct + default side conditions across Poisson/binomial/Gamma).
- `tests/optimization/test_mgcv_gam_side_parity.py::test_gam_side_matches_mgcv_nested_side_condition_cases`: 12 passed.
- Current non-general `gam.side` one-smooth/no-op slice (`-k 'uni or random_effect or random_slope_re or numeric_by_cr or factor_by_cr'`): 9 passed, 20 deselected.
- `tests/parity/test_mgcv_parity_failing_and_warnings.py::test_strict_factor_by_link_parity`: passed.
- Gaussian smoothness post-process owner checks (single CR + noisy random effect): 2 passed after removal of the single-smooth EDF heuristic.
- The 2026-08-15 `test_gaussian_re_reml_intercept_edf_attribution_matches_mgcv`
  failure (EDF `2.03125` versus mgcv `2.00010681`) is superseded by the
  2026-08-16 strict `gdiPK`/`gdi1` Hessian port recorded in the latest addendum.
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
- `913ff3e` — gam/fit+smoothing: gdi1 state boundary, joint outer
  optimization, upstream Vc and rank handling.
- `21291a0` — gam/constructors: mgcv-exact defaults and guards for formula
  smooths.
- `56eac49` — gam/public surface: summary.gam port, null deviance, prediction
  filters, schema.
- `70ac3e1` — neural: task semantics, sklearn contracts, first-validation
  test suite.
- `46bacbc` — GAM: upstream fitting/parity behavior and heuristic removal.
- `c25af71` — GAM: portable numerical boundary and stacked-QR simplification.
- `e750f6d` — tests/API/CI: retained portability and parity evidence.

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
- At this audit point,
  `tests/regressions/test_optimize_driver_mgcv_parity.py` had **2 FAILED,
  12 passed** from stale mock/contract drift. Both tests were repaired and
  pass as of 2026-08-16.
- `_fallback_single_smooth_edf` (fit/state.py:296) and
  `_fs_term_penalty_adjustment` (predict/predictions.py) were confirmed present
  at this audit point. Both have since been removed.

REVIEW.md, tldr.md, and todo.md updated accordingly.

### Addendum (2026-08-15, later): fs null-space penalty ordering resolved

Investigated `gaussian_reml_newton_fs_xt_ps` (audit finding 1). Probes
(`debug/fs_null_order_probe.py`, `debug/fs_null_order_stability_probe.py`):

- The two nat.param null directions are IDENTICAL between NAMpy and mgcv
  (cross-correlation is an exact permutation matrix); only the ordering of
  the numerically-zero RSR eigenvalues differs (NAMpy [2.2e-14, 3.8e-16] vs
  R [1.9e-15, -6.7e-15], both descending-sorted roundoff).
- mgcv's own order is NOT deterministic: row-permuting the data (identical
  model) flips it (seeds 1711/1 flip; seed set shows no consistent rule).
  No deterministic upstream rule exists to port.
- Treatment per invariant policy: lifecycle registry gained
  `exchangeable_sp_groups`/`exchangeable_sp_coef_cols`/`compare_unconditional`;
  the harness canonicalizes each side independently (descending final log-sp
  within declared groups, extended with identity over trailing joint
  scale/theta coordinates, and the induced coefficient permutation for
  covariance diagonals) before the otherwise-strict comparison.
- Post-alignment: full Newton trace, endpoint, Vp, Ve, EDF, scale, hat,
  outer_info all strict. Vc/edf2_total/AIC excluded for this case with
  evidence: mgcv row-permuted moves Vc[0,0] 0.0696214->0.0695778 and
  edf2_total 15.9964324->15.9940558, and NAMpy equals the row-permuted
  branch to 7 digits in both; AIC diff = exactly 2x the edf2 spread.

Runs: `[gaussian_reml_newton_fs_xt_ps]` PASSED; registry contracts 3 passed;
neighbors `gaussian_reml_newton_random_effect`, `gaussian_reml_newton_two_cr`,
`gamma_reml_newton_joint_scale_cr`, `negbin_est_reml_newton_joint_theta_weighted_cr`,
`gaussian_log_reml_newton_joint_scale_cr` all PASSED (canonicalization is a
no-op without declared groups). smoothCon fs slice 4 passed. The fs+select
snapshot case passes under its documented flat-ridge tolerances.

### Addendum (2026-08-16): cs and near-singular Gaussian REML

- `cs` shrinkage now directly mirrors
  `mgcv/R/smooth.r::smooth.construct.cr.smooth.spec` and
  `smooth.construct.cs.smooth.spec`: explicit `(S+t(S))/2`, descending
  symmetric eigensystem, and the two ordered `.1` shrinkage assignments.
  Ordinary `cs` term output passes with and without SEs; transformed raw `cr`
  and `cs` constructor cases pass.
- The residual transformed-terms and representative Gaussian GCV differences
  are confined to LAPACK selection of a basis inside the repeated
  two-dimensional zero eigenspace. `debug/transformed_cs_penalty_probe.py` and
  `debug/gaussian_cs_gcv_probe.py` show raw CR penalties agreeing at
  `1.4e-14`, while all SciPy symmetric eigensolver drivers select a different
  zero-space orientation from R before the unequal `.1`/`.01` penalties are
  assigned. No platform hook, heuristic, or tolerance relaxation was added.
- Near-singular Gaussian REML now uses the upstream numerical path from
  `mgcv/src/gdi.c::gdiPK`/`gdi1` and `mgcv/src/mat.c::getXtX`/`getXtMX`:
  the deviance Hessian comes from the unpenalized weighted QR factor (including
  signed-weight correction), and its smoothing-derivative contraction follows
  upstream scalar accumulation order. The former `X'WX` reconstruction was
  removed from this path.
- Exact regression:
  `test_gaussian_re_reml_intercept_edf_attribution_matches_mgcv` passes with
  the expected iteration-limit warning. The preserved endpoint probe gives
  NAMpy `log(sp)=-66.50440143`, EDF `2.00036621`, against mgcv `-64.52515079`,
  EDF `2.00010681`, within the declared near-boundary tolerances.

### Addendum (2026-08-17): stacked-QR implementation boundary

- `nampy/gam/fit/linalg/stacked_qr.py` no longer loads BLAS/LAPACK symbols via
  `ctypes`, manages compact Householder buffers, or reuses the raw `JPVT`
  workspace from an earlier factorization. Weighted-design, rank-reveal, and
  augmented-system factorizations now use SciPy's supported economic pivoted-QR
  interface and explicit `Q`/`R` factors.
- The behavioral port remains anchored to `mgcv/src/gdi.c::pls_fit1` and
  `gdiPK`: rank dropping and zero restoration, signed-weight correction,
  coefficient solves, deviance-Hessian construction, covariance roots, and
  determinant corrections are unchanged. Pivot/gauge choices that depend on a
  particular LAPACK build are treated as representation details.
- Fixed-endpoint Gaussian and non-Gaussian, rank-deficient, signed-weight,
  near-singular REML, unconditional-covariance, and outer-Newton neighboring
  tests pass with the supported QR interface.

### Addendum (2026-08-16): removal of parity heuristics and silent fallbacks

- Factor-smooth term contributions now use the direct `PredictMat` coefficient
  block product from `mgcv/R/mgcv.r::predict.gam`; the least-squares constant
  shift was removed and the strict `fs-no_se` output-parity case passes.
- Gaussian fits no longer switch algebraic backends based on condition numbers,
  rank estimates, or term types. The supported exact path always follows the
  `gam.fit3` current-sp `pls_fit1` state and `gdi1` coefficient/covariance
  overwrite.
- Dormant coefficient-method and null-space gauge controls, the unused
  design-balance initializer, and their public exports were removed.
- Exact derivative, Gaussian score-refresh, and unconditional-covariance paths
  no longer catch arbitrary exceptions and substitute alternate state. Missing
  canonical state now raises explicitly; targeted regression guards cover the
  `gdiPK` design owner and stored `gam.fit3` deviance requirements.

### Addendum (2026-08-17): canonical PIRLS working system

- Every supported `gam.fit3`/PIRLS inner solve now uses the current smoothing-
  parameter reparameterization before iteration: coefficients and warm starts
  are mapped by `t(T)`, the design is `X %*% T`, and `St`, `Sr`, and `Eb` are
  passed directly to the `pls_fit1` mirror. The dense public-coordinate penalty
  root reconstruction is no longer used by model-level PIRLS fits.
- The Poisson-identity forced-Fisher endpoint and tightened-tolerance helpers
  were removed. Noncanonical GLMs follow `gam.fit3`: full Newton by default,
  with Fisher scoring only for the local indefinite-system retry.
- Final coefficients, covariance, penalty, and design are restored to public
  coordinates together. Invalid final working systems now raise explicitly
  instead of returning a partially canonical solution.
- Targeted validation passed for the canonical-state regression, all six
  rank-deficient Poisson/binomial/Gamma variants, fixed-SP inner parity (13
  cases), Poisson identity/sqrt ML and REML, outer Newton and negative-binomial
  traces, GLM prediction covariance, and fit-result ownership contracts.

### Addendum (2026-08-17): remaining-issue investigation and neural validation

- `gammals_select_true_cr` optimized new-data prediction was reproduced with:

  ```bash
  /home/ad32/miniconda3/envs/nampy/bin/pytest \
    tests/families/test_general_family_mgcv_parity.py \
    -k 'gammals_select_true_cr and newdata_prediction_surfaces' -v
  ```

  Before the fix: 3 failed (link/response/terms) and `lpmatrix` passed.
  `debug/gammals_select_edf2_probe.py` was extended to evaluate mgcv on the
  original and mirrored representations at the same physical newdata. Original
  mgcv maximum prediction differences were `3.937e-5`, `5.497e-5`, and
  `3.707e-5`; mirrored mgcv differences were `1.492e-8`, `2.083e-8`, and
  `1.537e-8`, with all mirrored SE differences below `9.7e-9`.

  A strict follow-up compared `initial.spg` itself and found the real cause:
  `_sl_multi_penalty_block` passed the upper triangle to the symmetric
  eigensolver, unlike `mgcv/R/fast-REML.r::Sl.setup` and NAMpy's singleton
  branch. With the lower-triangle convention, both sides start at
  `[3.62377581, 4.48603160]` and finish at
  `[1385.90211256, 11.26504219]`. Maximum link/response/terms prediction
  differences are now `3.19e-9`, `3.90e-9`, and `2.44e-10`; all SE differences
  are below `8.2e-10`. The prediction rerun is **4 passed**, the full gammals
  select slice is **10 passed**, and the optimized fit5 post-processing test
  passes. All gammals xfails were removed.

  Strict retained commands:

  ```bash
  /home/ad32/miniconda3/envs/nampy/bin/pytest \
    tests/optimization/test_mgcv_general_family_preoptimization_parity.py -v
  /home/ad32/miniconda3/envs/nampy/bin/pytest \
    tests/families/test_general_family_mgcv_parity.py \
    -k 'gammals_select_true_cr' -v
  /home/ad32/miniconda3/envs/nampy/bin/pytest \
    tests/optimization/test_mgcv_postprocessing_final_fit_parity.py \
    -k 'gammals_select_true_cr' -v
  ```

  Results: **8 passed**, **10 passed**, and **1 passed**, respectively.

- Production portability guard:

  ```bash
  /home/ad32/miniconda3/envs/nampy/bin/pytest \
    tests/regressions/test_no_platform_numeric_bindings.py -v
  ```

  Result: **1 passed**. The guard covers direct `ctypes`/CFFI,
  BLAS/LAPACK-module imports, `get_lapack_funcs`, and explicit driver keywords.

  The exact local Linux CI slices were also executed with `MGCV_CACHE_ONLY=1`.
  The guard plus GAM linalg exports, parity invariants, natural-parameter, and
  chi-square-mixture tests passed **17/17**; the focused `cr`, `cs`, `tp`, `fs`,
  and `ti` raw-constructor matrix passed **5/5**. Hosted macOS and Windows results
  remain unverified until the workflow is committed and run on those systems.

- Neural first-validation files:

  ```text
  tests/neural/test_neural_architecture_smoke.py       25 passed
  tests/neural/test_neural_task_model.py                2 passed
  tests/neural/test_neural_sklearn_contracts.py        41 passed
  tests/neural/test_neural_spline_nam.py                7 passed
  tests/neural/test_neural_estimator_fit_smoke.py      73 passed
  total                                                148 passed
  ```

  No architecture-specific failure required a reduced rerun. All five files are
  tracked.

- Packaging inspection confirms `pretab` is mandatory in both
  `pyproject.toml` and `requirements.txt`, installed as version `0.0.3` in the
  verified environment, and exercised by the editable-install neural tests. A
  no-isolation wheel build also succeeded; the wheel was installed with
  `--no-deps` into a temporary venv, imported `nampy` and `pretab` from outside
  the source tree, instantiated `LinRegRegressor`, and verified the installed
  three-symbol GAM API with no top-level `GAM` alias. Only optional CI automation
  of that artifact smoke remains. (The first isolated build attempt could not
  download its build requirement because the environment had no network access;
  that was not a package failure.)

- Public exports now follow the repository contract: `nampy.gam.__all__`
  contains only `fit_model_core`, `solve_fit`, and `FitCoreSolution`; neither
  `nampy.gam` nor top-level `nampy` re-exports `GAM`. Internal tests and probes
  import `GAM` explicitly from `nampy.gam.model.api`, and the direct contract
  regression passes.
