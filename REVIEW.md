# GAM / mgcv parity audit

> ## Status update — 2026-08-17, committed implementation through `e750f6d`
>
> The original audit below was run before the later strict-parity and
> portability work. This table is the reconciled current status; the detailed
> audit body is retained as historical localization evidence.
>
> | Audit finding | Current status |
> | --- | --- |
> | 1. fs penalty/null-space ordering (`gaussian_reml_newton_fs_xt_ps` swapped log-sp; fs+select endpoint off) | **RESOLVED AS AN INVARIANT (2026-08-15).** mgcv itself reverses the two numerically-zero natural-parameter directions under a row permutation. The lifecycle harness now declares that smoothing-parameter block exchangeable and otherwise retains strict trace, endpoint, covariance, EDF, scale, hat, and outer-info comparisons. |
> | 2. cs GCV/Cp criterion divergence | **FIXED for the algorithmic surface (2026-08-16).** `add_full_rank_shrinkage` is now a direct port of `smooth.construct.cr.smooth.spec`/`smooth.construct.cs.smooth.spec`; ordinary cs output passes with and without SEs. Preserved probes localize the remaining transformed/Gaussian-GCV numeric difference exclusively to R-versus-SciPy orientation of the repeated two-dimensional zero eigenspace (raw CR agrees to `1.4e-14`; every SciPy symmetric driver takes the other orientation). Per policy, no platform hook, heuristic, or tolerance change was added. |
> | 3. gammals(select=True) optimized predictions | **FIXED (2026-08-17).** The mismatch was not an unavoidable endpoint invariant: the multi-penalty `Sl.setup` port used the upper triangle for `eigen(St, symmetric=TRUE)`, unlike upstream's lower-triangle convention and NAMpy's singleton branch. The one-line correction makes `initial.spg` starts and final smoothing parameters match mgcv; link/response/terms predictions now agree within `3.9e-9`, their SEs within `8.2e-10`, and all gammals xfails were removed. |
> | 4. gaulss(select=True) xfail | **CLASSIFIED AS UPSTREAM SIGN-INDETERMINATE (2026-08-17).** `estimate.gam` reparameterizes `G$X` with `Sl.setup`'s symmetric-eigen vectors, but `initial.spg` receives the unreparameterized `G$Eb`. Base R uses `DSYEVR` and explicitly leaves real eigenvector signs arbitrary, so one legal sign difference changes the start and the flat-boundary endpoint. Matching one R build would require forbidden sign forcing or a platform hook. The local xfail and strict shared-endpoint test remain; no heuristic fix was added. |
> | 5. Rank-deficiency behavior | **FIXED**: PIRLS rank_tol is eps*100; Vp is built from the canonical gdi1 rV; coefficients and covariance share mgcv's dropped-coordinate gauge. The NAMpy-only null-space pin and its routing controls were removed from Gaussian and PIRLS paths, and `gam.side` no longer deletes aliased parametric columns. |
> | Platform-sensitive findings | **CLASSIFIED, with no production platform hook.** Repeated-eigenspace orientation is compared through invariants or retained as an evidence-backed endpoint xfail. The production package contains no direct `ctypes`, CFFI, BLAS/LAPACK-module, solver-driver, or `get_lapack_funcs` binding. The two local Linux portability slices pass (17 guard/linalg/inference tests and 5 raw-constructor cases); a Linux/macOS/Windows portability job is present in CI, but its hosted results require the workflow to be committed and run. |
> | Policy 1: `_fallback_single_smooth_edf` heuristic | **FIXED**: the fallback was removed; Gaussian EDF is produced only by the upstream `gdi1` `rV`/`K` construction. |
> | Policy 2: fs contribution shift via least squares | **FIXED (2026-08-16)**: the NAMpy-only adjustment was removed. Term contributions are the direct `PredictMat` coefficient block products used by `mgcv/R/mgcv.r::predict.gam`; the strict `fs-no_se` output-parity case passes. |
> | Policy 3: unknown kwargs swallowed | **FIXED** (commit `56eac49`): 23-key allowlist, `TypeError` on unknown arguments. |
> | Policy 4: bespoke summary() | **FIXED** (commit `56eac49`): full summary.gam port incl. null.deviance; machine-precision parity on 6 cases. |
> | Policy 5: bespoke plot() | Open by scope decision (documented). |
> | Policy 6: public-coordinate PIRLS plus Poisson forced-Fisher/tolerance overrides | **FIXED (2026-08-17)**: `gam.fit3` iterations now consume the exact current-SP `T`/`St`/`Sr`/`Eb` state. Noncanonical GLMs use full Newton with only upstream's local indefinite-system Fisher retry; the endpoint/tolerance heuristics were removed. |
> | Doc claim: multiple offsets raise | **Partially a misreading**: the single-formula path warns and keeps the first offset exactly as upstream (strict test + `debug/multi_offset_probe.R`); only multi-formula lpi aggregation raises, matching upstream's `"shared offsets not allowed"` stop. |
> | Doc claim: eps**0.66 vs docs | **FIXED** (rank_tol change above); docs now accurate. |
> | SUBSYSTEM_COVERAGE drift | **FIXED** (commit `c4586dd`). |
> | Stale unit tests | **FIXED (2026-08-16)**: both optimization-driver tests now supply the current strict contracts and pass. |
> | Public exports inconsistency | **RESOLVED (2026-08-17).** `nampy.gam.__all__` now contains only `fit_model_core`, `solve_fit`, and `FitCoreSolution`; `GAM` is no longer re-exported by `nampy.gam` or top-level `nampy`. Internal tests and retained probes import the implementation explicitly from `nampy.gam.model.api`. A direct package-contract regression passes. |
> | Neural first-validation matrix | **VERIFIED (2026-08-17).** The five focused files pass: 25 architecture, 2 task/multi-output, 41 sklearn-contract, 7 SplineNAM, and 73 estimator-smoke tests (148 total). |
>
> No confirmed algorithmic defect remains in the currently declared GAM
> surface from this audit. Remaining work is hosted CI evidence, optional
> packaging-smoke automation, and deliberately unsupported feature scope recorded in
> `GAM_NOT_IMPLEMENTED.md`.


  Bottom line: nampy/gam is a substantial, broad port of mgcv 1.9-4. The
  algorithmic defects found by the original audit have been fixed from
  upstream or classified through mgcv-relevant invariants. It is intentionally
  incomplete: unsupported surfaces remain guarded instead of approximated.

  ## Reference and method

  - Vendored and installed R mgcv versions are both 1.9-4.
  - I audited the implementation now split across `46bacbc`, `c25af71`, and
    `e750f6d`, not merely the older `9b5ca0d` base.
  - Live R comparisons ran in a disposable copy under /tmp, avoiding writes to the repository and its snapshot cache.
  - More than 1,300 targeted test invocations were exercised, including deliberate reruns of failures on a second numerical stack.
  - No full test-suite run was performed, following repository policy.

  Primary upstream references were:

  - Formula, setup, side conditions, prediction and inference: mgcv/R/mgcv.r:292
  - Smooth constructors and prediction matrices: mgcv/R/smooth.r:741
  - gam.fit3, Newton/BFGS and post-processing: mgcv/R/gam.fit3.r:67
  - gam.fit4, gam.fit5, EFS and general-family post-processing: mgcv/R/gam.fit4.r:941
  - Sl construction/reparameterization: mgcv/R/fast-REML.r:68
  - General families: mgcv/R/gamlss.r:862
  - Extended families: mgcv/R/efam.r:1
  - Low-level fitting derivatives and QR: mgcv/src/gdi.c:52, mgcv/src/mat.c:348
  - Chi-square mixtures: mgcv/src/davies.c:198

  ## Current subsystem summary

  Passing counts originated in the 2026-08-15 audit and are supplemented by
  the dated follow-up evidence below.

   Subsystem                                 Status                                    Findings
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Formula parsing/spec building             Strong                                    30 formula-parity and 15 ownership tests passed. Formula lists, R operators, transformed responses/covariates, factors, numeric/factor by, offsets, id, pc, fx, m, xt, d, mc, sp, knots, min_sp and
                                                                                       drop_intercept are implemented for the declared surface.
  ────────────────────────────────────────  ────────────────────────────────────────  ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   Raw smooth constructors                   Strong                                    All 74 live-R raw-constructor cases passed, using invariant comparisons where eigenspace orientation is indeterminate.
  ────────────────────────────────────────  ────────────────────────────────────────  ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   smoothCon behavior                        Strong                                    All 43 targeted cases passed. Constraint absorption, penalty scaling and prediction transforms are broadly aligned.
  ────────────────────────────────────────  ────────────────────────────────────────  ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   Linked IDs, point constraints, tensors    Strong                                    68 focused pc/id/te/ti cases passed.
  ────────────────────────────────────────  ────────────────────────────────────────  ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   Design and side conditions                Strong                                    gam.setup, preoptimization, reparameterization and gam.side coverage is green for the declared surface; raw indeterminate directions use declared invariants.
  ────────────────────────────────────────  ────────────────────────────────────────  ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   Ordinary families                         Strong                                    Gaussian, binomial, Poisson, Gamma and supported links pass broad derivative, fit and prediction coverage; the cs constructor defect was fixed directly from upstream.
  ────────────────────────────────────────  ────────────────────────────────────────  ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   Negative binomial                         Strong but partial                        Fixed and estimated theta, REML joint theta and EFS cases pass representative coverage. Estimated-theta ML with optimizer="optim" is deliberately unsupported.
  ────────────────────────────────────────  ────────────────────────────────────────  ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   gaulss / gammals                          Strong; one gaulss endpoint gap           Gammals select=True initialization, optimized fit, prediction, SEs, and post-fit now pass strictly; only the separate gaulss select=True endpoint retains an evidence-backed xfail.
  ────────────────────────────────────────  ────────────────────────────────────────  ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   Gaussian/IRLS solvers                     Strong                                    Fixed inner fits, signed-weight QR, near-singular REML, rank-deficient gauge, post-fit covariance, and current-SP PIRLS coordinates follow the upstream routines.
  ────────────────────────────────────────  ────────────────────────────────────────  ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   Smoothing criteria                        Strong within declared surface            ML/REML, GCV, UBRE/AIC and general-family routes use exact derivative paths; repeated-eigenspace representation is not platform-forced.
  ────────────────────────────────────────  ────────────────────────────────────────  ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   Optimizers                                Strong within declared surface            Newton, BFGS, EFS and supported optim branches match representative lifecycle traces; exchangeable fs coordinates are compared by their declared invariant.
  ────────────────────────────────────────  ────────────────────────────────────────  ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   Post-fit covariance/EDF                   Strong with declared endpoint invariant   Vp, Vf/Ve, Vc, EDF1/EDF2, sp_vcov and gam_vcomp have broad passing evidence; selected general-family endpoint scalars retain visible orientation xfails.
  ────────────────────────────────────────  ────────────────────────────────────────  ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   Prediction                                Strong for ordinary GAMs                  link, response, terms, iterms, lpmatrix, SEs, unconditional covariance, terms= and exclude= passed broad coverage. General-family filters remain unsupported.
  ────────────────────────────────────────  ────────────────────────────────────────  ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   ANOVA/residuals/diagnostics               Strong within declared numeric surface    228 broad prediction/inference/diagnostic cases passed, including single/multi-model ANOVA, residuals, concurvity and k_check. summary is a port; plot and gam_check remain deliberately narrower.
  ────────────────────────────────────────  ────────────────────────────────────────  ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   Public API                                Aligned                                   `nampy.gam` exports only fit_model_core, solve_fit and FitCoreSolution; the internal GAM implementation is imported from nampy.gam.model.api by tests/probes.

  ## Original parity findings and resolutions

  ### 1. fs penalty/null-space ordering

  The original lifecycle comparison reported the final two log smoothing
  parameters in the opposite order. The retained probes subsequently proved
  that these are the two exchangeable natural-parameter directions associated
  with numerically-zero eigenvalues. mgcv reverses them under a row
  permutation of the same model, while the two NAMpy/mgcv directions agree as
  an exact permutation.

  There is therefore no deterministic upstream order to port. The lifecycle
  registry declares only this block exchangeable, applies the induced
  coefficient mapping, and keeps the remaining trace and fitted-behavior
  assertions strict. The lifecycle and fs+select snapshot cases pass under
  that invariant; no solver hook or heuristic basis canonicalization was
  added.

  ### 2. cs GCV/Cp criterion values

  The algorithmic defect was the use of the lower triangle rather than
  upstream's explicit `(S + t(S))/2` before the cs eigendecomposition. The
  constructor now ports the two upstream eigenvalue replacements literally;
  ordinary cs term output passes with and without SEs, and transformed raw
  cr/cs constructor cases pass.

  A representative Gaussian fixed-SP GCV value can still differ by `3.05e-6`
  on this numerical stack. `debug/gaussian_cs_gcv_probe.py` proves this is not
  downstream scoring: its raw CR penalty matches R to `1.4e-14`, while the raw
  cs penalty changes because R and SciPy select different bases in the repeated
  zero eigenspace before assigning unequal `.1` and `.01` shrinkage. All SciPy
  symmetric eigensolver drivers exhibit the same issue. This is intentionally
  not forced with platform-specific LAPACK behavior or heuristic
  canonicalization.

  ### 2a. Stacked-QR raw solver plumbing

  The earlier stacked-QR port crossed the intended parity boundary by loading
  LAPACK through `ctypes`, managing `dgeqp3`/`dormqr` compact buffers, reusing
  raw `JPVT` workspace, and calling BLAS `dsyrk` solely to reproduce a backend
  accumulation path. This has been removed. The implementation now uses
  SciPy's public pivoted-QR and triangular-solve interfaces while preserving
  the `pls_fit1`/`gdiPK` behavioral state, signed-weight correction, rank drop,
  and covariance construction.

  ### 3. Optimized gammals(select=True) predictions

  The earlier mirrored-basis result localized the mismatch but classified it
  too early. A strict `initial.spg` regression exposed the actual divergence:
  `nampy/gam/fit/solvers/general_family/fixed_smoothing.py::_sl_multi_penalty_block`
  called the symmetric eigensolver with the upper triangle, while
  `mgcv/R/fast-REML.r::Sl.setup` and NAMpy's singleton path use the lower
  triangle. After changing only that operand convention, NAMpy and mgcv start
  at `[3.62377581, 4.48603160]` and finish at
  `[1385.90211256, 11.26504219]`.

  The optimized link/response/terms maximum prediction differences are now
  `3.19e-9`, `3.90e-9`, and `2.44e-10`; corresponding SE differences are at
  most `8.17e-10`. The full 10-test gammals select slice and strict fit5
  post-processing comparison pass. No tolerance or solver-driver selection
  was added, and no gammals xfail remains.

  ### 4. gaulss(select=True) remains an xfail

  The post-fit suite intentionally xfails the optimized
  `gaulss_select_true_cr` endpoint. After the strict `Sl.setup` fix, NAMpy's
  second log smoothing parameter is `11.81049973` versus mgcv `11.91107097`,
  with criteria differing by `3.98e-6`; optimized Vc remains outside tolerance.
  The retained strict-initialization experiment confirms that changing
  `pen.reg`'s secondary QR does not change this result.

  The controlling upstream sequence is
  `mgcv/R/mgcv.r::estimate.gam`: lines 1899-1901 form `G$Sl` and replace
  `G$X` by `Sl.initial.repara(...)`, while the call near line 1998 passes
  `G$S` and the separately constructed, unreparameterized `G$Eb` to
  `initial.spg`. `mgcv/R/fast-REML.r::Sl.setup` obtains the transform from
  `eigen(St, symmetric=TRUE)`. Base R's `eigen()` uses `DSYEVR` and does not
  define real eigenvector signs. In this case NAMpy and R differ by exactly
  one legal column sign in that transform; because `G$Eb` is not transformed
  with the same sign, upstream initialization itself is sign-sensitive.

  Mirroring the input basis moves mgcv's endpoint to `11.79338762`, confirming
  the sensitivity but no longer reproducing NAMpy exactly. Forcing the sign
  seen from one R/LAPACK build would violate the no-platform-hook and
  representation-invariant policy. The evidence-backed xfail therefore stays
  visible, and fixed/shared-endpoint post-processing remains strict.

  ### 5. Rank-deficient/random-effect performance and gauge behavior

  The near-singular random-intercept regression now passes. The remaining
  Hessian divergence was fixed by porting the exact `gdiPK`/`gdi1` route:
  construct the coefficient deviance Hessian from the unpenalized weighted QR
  factor through `getXtX`, apply the signed-weight correction, and use
  `getXtMX` accumulation for the smoothing Hessian. Reconstructing the same
  matrix as `X'WX` had changed cancellation at the REML boundary.

  The related audit concerns are also superseded: PIRLS uses upstream's
  `eps*100` tolerance, the Gaussian penalty-minimizing gauge is disarmed, and
  coefficients/covariance use the same dropped-coordinate `gdi1` gauge.

  ## Platform-sensitive findings

  These passed on one numerical stack and failed narrowly on another:

  - gaussian_fs_numeric_by: penalty matrix maximum difference 1.2316e-12 against 1e-12.
  - gaulss_select_true_cr preoptimization: ldetS difference 1.95e-10 against 1e-10.
  - gammals_select_true_cr: the former EDF2 endpoint difference is fixed by the
    upstream `Sl.setup` lower-triangle convention.

  Follow-up probes classify the remaining fs/cs/gaulss cases through explicit
  invariants or a visible evidence-backed xfail. The gammals case instead
  exposed and fixed a real upstream operand-convention mismatch. A
  production-source AST guard passes and prevents direct native numeric
  bindings or explicit solver-driver selection from returning.

  ## Correctly guarded or absent surfaces

  The following are not implemented and generally fail explicitly, which is preferable to approximation:

  - Smooths: t2, cp, bs, ds, gp, ad, mrf, sos, soap bases, sc, scad.
  - Matrix covariates, matrix by, linear-functional/signal terms, user constraint matrices and paraPen.
  - pc= for tensor, random-effect and factor-smooth surfaces.
  - General-family terms=/exclude=.
  - Reparameterized nonlinear and several non-reparameterized Sl layouts.
  - Formula-list data-aware dot shorthand.
  - Ordered parametric factors.
  - GACV, P-ML/P-REML, NCV/QNCV, nlm, known-scale scale= workflows, and a distinct magic optimizer identity.
  - Prediction arguments such as block.size, newdata.guaranteed, na.action, iterms.type.
  - gam, bam, gamm, jagam, vis.gam.

  Families absent from the Python registry include:

  - quasi, quasi-Poisson, quasi-binomial, inverse Gaussian.
  - Extended families such as Tweedie, beta regression, scaled-t, ordered categorical, censored/count and zero-inflated families.
  - General families including multinomial, ziplss, gevlss, twlss, gumbls, shash, mvn, Cox and gfam.

  The actual registry is visible in nampy/gam/families/registry.py:1.

  Upstream-unsupported cases are correctly excluded rather than treated as backlog:

  - Random-effect smooths with id=.
  - fs over full-rank shrinkage bases such as cs/ts.

  ## Architectural and policy concerns

  1. Resolved: `_fallback_single_smooth_edf` was removed. Invalid EDF is no
     longer replaced by a design-rank or `trace(H)-nsdf` heuristic.

  2. Resolved: the NAMpy-only fs least-squares contribution shift was removed.
     Term contributions now use the direct prediction-matrix coefficient block
     multiplication in `mgcv/R/mgcv.r::predict.gam`.

  3. Resolved: unknown GAM constructor keywords are checked against the
     supported allowlist and raise `TypeError` instead of being stored and
     ignored.

  4. Resolved: `summary()` now mirrors `summary.gam`, including coefficient and
     smooth tables, adjusted R-squared, deviance explained, null deviance, and
     family/scale-specific behavior.

  5. Scope decision: `plot()` remains a deliberately narrower matplotlib
     contribution plot rather than a `plot.gam` port.

  6. Scope decision: absent family-specific covariance and diagnostic hooks
     remain unsupported and documented; supported general-family surfaces have
     owner and live-R parity coverage.

  ## Documentation and test-suite drift

  The implementation inventories and subsystem coverage map have been
  reconciled with the fixes above. Multiple offsets, PIRLS coordinates,
  lifecycle status, known endpoint invariants, and stale unit-test contracts
  are now described consistently. The public-export conflict is also resolved:
  package `__all__` follows the three-symbol fit-core contract and internal
  parity code imports `GAM` from its implementation module.

  ## Validation summary

  The counts below are retained from the original audit. Current follow-up
  evidence includes the 2026-08-17 strict `gammals` initialization and
  optimized-prediction probes, the
  passing direct-native-binding guard, and 148 passing neural owner/smoke
  tests. Exact commands and results are recorded in `PROJECT_STATUS.md`.

  Notable live-R results:

  - Formula parser: 30 passed.
  - Formula/spec owner contracts: 15 passed.
  - Raw constructors: 74 passed.
  - smoothCon: 43 passed.
  - Point constraints, linked IDs and tensors: 68 passed.
  - Family derivative/core tests: 53 passed.
  - Representative end-to-end model matrix: 13 passed.
  - Extended snapshot matrix: 48 passed.
  - Main output parity: 54 passed.
  - Staged general prediction/inference/diagnostics: 50 passed.
  - Broad prediction/inference/diagnostics: 228 passed.
  - Historical known-gap tests: 9 passed.
  - Fixed inner/low-level optimization: 27 passed.
  - Optimization/trace group: 47 passed, 1 failed.
  - Tracked failing/warning group: 14 passed, 1 failed.
  - General preoptimization/backend/chi-square/ANOVA slice: 16 passed, 1 failed.
  - Post-fit slice on one stack: 51 passed, 1 xfailed, 1 failed; its failure passed when rerun under the other numerical stack.

  The bare /home/ad32/.local/bin/pytest environment currently lacks the declared pretab dependency. I therefore used /home/ad32/miniconda3/bin/pytest for the final live-R runs. Standardizing the documented development environment is necessary for reproducible parity evidence.

  ## Recommended next work

  1. Obtain the configured hosted Linux/macOS/Windows portability results; the
     guard and CI matrix are now committed.
  2. Optionally automate the passing manual installed-wheel smoke in CI. The
     built wheel imports `nampy` and mandatory `pretab` from a temporary venv and
     can instantiate `LinRegRegressor`; the installed artifact also satisfies the
     three-symbol GAM export contract.
  3. Add multi-output fit coverage beyond LinReg only if every public neural
     regressor is intended to promise that contract.

  Ordinary GAMs and the declared constructor/optimizer surface have strong
  strict-parity evidence. The remaining numerical differences are explicitly
  classified representation/endpoint choices inside mathematically
  indeterminate eigenspaces, not candidates for platform hooks or heuristics.
