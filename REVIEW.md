# GAM / mgcv parity audit

> ## Status update — 2026-08-15, after the release-fix commits
>
> This audit ran against the pre-commit dirty tree. Each finding was re-verified
> against the committed tree (branch `mgcv`, commits `913ff3e..e8c9b21`):
>
> | Audit finding | Current status |
> | --- | --- |
> | 1. fs penalty/null-space ordering (`gaussian_reml_newton_fs_xt_ps` swapped log-sp; fs+select endpoint off) | **CONFIRMED OPEN — highest priority.** Reproduced on the committed tree; the lifecycle registry still marks the case stable, so the strict suite has one known-failing unmarked case. Tracked in todo.md P1. |
> | 2. cs GCV/Cp criterion divergence | **CONFIRMED OPEN, now visible in the cached suite too.** `tests/parity/test_mgcv_output_parity.py` cs cases fail at 8.6e-5 (terms) after the snapshot-cache version bump forced fresh live-R references. Bisect: plain-cs failures are introduced by the prior-session `symmetrize_lower_triangle=True` change in `nampy/splines/univariate/cr.py::add_full_rank_shrinkage` (its comment claims upstream parity; live mgcv disagrees); `transformed_cs` has a second, not-yet-localized cause in the same constructor commit. All three passed at the old base `97a2530`. This session's fixes are exonerated by a consistent-group bisect. |
> | 3. gammals(select=True) optimized predictions | **OPEN, classified.** Same select-endpoint family as finding 4: the gammals endpoint difference was shown to be mgcv-internal initial.spg orientation indeterminacy (mirrored basis reproduces NAMpy's endpoint/edf to 2.3e-7, `debug/gammals_select_edf2_probe.py`); the optimized-prediction surface still needs the same mirrored-basis verification before its gap is tagged. |
> | 4. gaulss(select=True) xfail | **AGREED, evidence recorded.** The xfail remains visible with the quantitative mirrored-basis evidence in its reason string; strict fixed-endpoint post-processing covers both select cases at 5e-6. |
> | 5. Rank-deficiency behavior | **FIXED** (commit `913ff3e`): PIRLS rank_tol now eps*100; Vp built from the canonical gdi1 rV (coef and covariance share mgcv's dropped-coordinate gauge, strict live-R drop regression); gaussian gauge pin disarmed. Still open: pirls.py arms the pin for non-Gaussian fits; side conditions delete aliased parametric columns (new finding, todo.md P1). |
> | Platform-sensitive trio | gammals EDF2 resolved as endpoint orientation (see 3/4); fs numeric-by 1.23e-12 and gaulss ldetS 1.95e-10 remain unexplained micro-sensitivities. |
> | Policy 1: `_fallback_single_smooth_edf` heuristic | **CONFIRMED OPEN** (`nampy/gam/fit/state.py:296`, used at `:418`). |
> | Policy 2: fs contribution shift via least squares | **CONFIRMED OPEN** (`nampy/gam/predict/predictions.py::_fs_term_penalty_adjustment`) — needs an upstream predict.gam citation or exact replacement. |
> | Policy 3: unknown kwargs swallowed | **FIXED** (commit `56eac49`): 23-key allowlist, `TypeError` on unknown arguments. |
> | Policy 4: bespoke summary() | **FIXED** (commit `56eac49`): full summary.gam port incl. null.deviance; machine-precision parity on 6 cases. |
> | Policy 5: bespoke plot() | Open by scope decision (documented). |
> | Doc claim: multiple offsets raise | **Partially a misreading**: the single-formula path warns and keeps the first offset exactly as upstream (strict test + `debug/multi_offset_probe.R`); only multi-formula lpi aggregation raises, matching upstream's `"shared offsets not allowed"` stop. |
> | Doc claim: eps**0.66 vs docs | **FIXED** (rank_tol change above); docs now accurate. |
> | SUBSYSTEM_COVERAGE drift | **FIXED** (commit `c4586dd`). |
> | Stale unit tests | **CONFIRMED OPEN**: `test_optimize_driver_mgcv_parity.py::test_all_fixed_smoothing_params_still_optimizes_unknown_gaussian_scale` and `::test_negbin_reml_native_all_fixed_optimizes_theta_first` fail (mock/contract drift). |
> | Public exports inconsistency | **OPEN** (CLAUDE.md names three exports; `nampy/gam/__init__.py` exports more) — needs a documentation-or-code decision. |
>
> Remaining open items are tracked in `todo.md` P1 with the fs ordering first.


  Bottom line: nampy/gam is a substantial, broad port of mgcv 1.9-4, and most of the declared supported surface works at the repository’s parity tolerances. It is not yet an exact or complete mgcv mirror. I found reproducible parity defects in factor-smooth penalty ordering, cs GCV/Cp scoring,
  and an optimized gammals(select=True) endpoint, plus known rank-deficient and general-family limitations.

  I made no repository edits. The worktree remains in its pre-existing dirty state.

  ## Reference and method

  - Vendored and installed R mgcv versions are both 1.9-4.
  - I audited the current uncommitted tree, not merely the committed base.
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

  ## Subsystem status

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
   Design and side conditions                Mostly correct                            gam.setup, preoptimization, reparameterization and gam.side coverage was overwhelmingly green; one fs numeric-by matrix comparison differed at approximately 1.23e-12 on one stack and passed on another.
  ────────────────────────────────────────  ────────────────────────────────────────  ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   Ordinary families                         Mostly correct                            Gaussian, binomial, Poisson, Gamma and their supported links passed broad derivative, fit and prediction coverage. Two cs GCV/Cp criterion cases fail consistently.
  ────────────────────────────────────────  ────────────────────────────────────────  ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   Negative binomial                         Strong but partial                        Fixed and estimated theta, REML joint theta and EFS cases pass representative coverage. Estimated-theta ML with optimizer="optim" is deliberately unsupported.
  ────────────────────────────────────────  ────────────────────────────────────────  ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   gaulss / gammals                          Partial                                   Likelihood derivatives, fixed-endpoint fitting and most prediction/post-fit surfaces pass. Optimized selection endpoints retain gaps.
  ────────────────────────────────────────  ────────────────────────────────────────  ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   Gaussian/IRLS solvers                     Strong in regular cases                   Fixed inner fits, signed-weight QR, post-fit covariance and regular random-effect fits pass. Genuine rank-deficient cases remain problematic.
  ────────────────────────────────────────  ────────────────────────────────────────  ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   Smoothing criteria                        Mostly correct                            ML/REML, GCV, UBRE/AIC and general-family routes are implemented with exact derivative paths. cs GCV/Cp values expose a remaining discrepancy.
  ────────────────────────────────────────  ────────────────────────────────────────  ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   Optimizers                                Mostly correct                            Newton, BFGS, EFS and much of optim match representative lifecycle traces. One fs(xt="ps") lifecycle has swapped smoothing-parameter order.
  ────────────────────────────────────────  ────────────────────────────────────────  ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   Post-fit covariance/EDF                   Mostly correct                            Vp, Vf/Ve, Vc, EDF1/EDF2, sp_vcov and gam_vcomp have broad passing evidence. One borderline gammals EDF2 result varies by numerical stack.
  ────────────────────────────────────────  ────────────────────────────────────────  ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   Prediction                                Strong for ordinary GAMs                  link, response, terms, iterms, lpmatrix, SEs, unconditional covariance, terms= and exclude= passed broad coverage. General-family filters remain unsupported.
  ────────────────────────────────────────  ────────────────────────────────────────  ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   ANOVA/residuals/diagnostics               Strong within declared numeric surface    228 broad prediction/inference/diagnostic cases passed, including single/multi-model ANOVA, residuals, concurvity and k_check. summary, plot and gam_check are intentionally narrower Python interfaces, not
                                                                                       complete ports.
  ────────────────────────────────────────  ────────────────────────────────────────  ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   Public API                                Inconsistent                              The repository instructions describe only three intended public fit exports, while nampy/gam/__init__.py:1 also exports GAM, families and parity.

  ## Reproducible parity defects

  ### 1. fs penalty/null-space ordering

  The clearest behavioral defect is:

  gaussian_reml_newton_fs_xt_ps
  NAMpy log_sp: [-3.81338665, -5.60089480, -5.42249433]
  mgcv  log_sp: [-3.81338665, -5.42249433, -5.60089480]

  The last two null-space smoothing parameters are reversed. This violates the strict ordering requirement even if the model fit is invariant after the corresponding penalty permutation.

  The construction currently appends the range penalty followed by null-space penalties in nampy/gam/smooths/categorical/fs.py:712. The lifecycle failure indicates that the two P-spline null directions are not being mapped to the same upstream order.

  The related optimized fs + select=True case also fails:

  NAMpy log_sp: [-18.2641, -20.5063, -24.3523]
  mgcv  log_sp: [-18.1930, -24.4520, -19.3043]

  Two coordinates differ by roughly four to five log units. Because the assertion stops at smoothing parameters, its optimized EDF, criterion and prediction comparisons are not established by that test. Fixed-SP prediction/inference at the mgcv endpoint does pass.

  Priority: highest. Fix penalty identity/order before attempting endpoint tuning.

  ### 2. cs GCV/Cp criterion values

  Two fixed-SP scoring cases fail consistently:

  - Gaussian s(x, bs="cs"): 0.09439735 versus 0.09439430, difference 3.05e-6.
  - Binomial s(x, bs="cs"): 0.35554383 versus 0.35554659, difference 2.76e-6.

  Tolerance was 2e-7. Raw cs construction passes, so the remaining divergence is downstream in fit/scoring state rather than basic basis construction. The failing assertion precedes gradient and Hessian checks, leaving those derivative assertions unexecuted for these two cases.

  Relevant owners are nampy/gam/smoothing_selection/criteria/gaussian.py:31, nampy/gam/smoothing_selection/criteria/pirls/value.py:216, and the gdi1 derivative path.

  ### 3. Optimized gammals(select=True) predictions

  For gammals_select_true_cr, optimized new-data predictions exceed the 1e-5 tolerance:

  - Link: maximum difference approximately 2.02e-5 to 3.94e-5, depending on numerical stack.
  - Response: approximately 2.82e-5 to 5.50e-5.
  - Terms: approximately 1.90e-5 to 3.71e-5.
  - lpmatrix passes.

  Because the design matrix agrees, this is an optimized coefficient/endpoint issue, not a new-data transform error. Standard-error assertions are not reached after the prediction failure.

  ### 4. gaulss(select=True) remains an xfail

  The post-fit suite intentionally xfails the optimized gaulss_select_true_cr endpoint. The recorded endpoints have log smoothing parameters around 11.7934 versus 11.9111, while criteria differ by about 4.7e-6.

  The repository has evidence that reversing an indeterminate basis orientation makes R land on NAMpy’s endpoint. That is good evidence against a simple algebra bug, but behavioral endpoint parity is still not proven for the original input, so the xfail should remain visible.

  ### 5. Rank-deficient/random-effect performance and gauge behavior

  A regular random-intercept case passed, but the near-singular row-space snapshot did not finish within a 360-second combined timeout. Earlier broad snapshot attempts stalled on the same random-effect cluster.

  More importantly, current code has known non-upstream behavior:

  - PIRLS defaults to eps**0.66, while upstream uses eps * 100 on this path.
  - A NAMpy-only penalty-minimizing null-space gauge is enabled for forced stacked QR.
  - Covariance and coefficients can use different gauges after genuine column drops.

  These are documented in GAM_NOT_IMPLEMENTED.md:86 and visible in nampy/gam/fit/linalg/stacked_qr.py:1210. They are inert in most current tests but mean genuine rank-deficient parity is not complete.

  ## Platform-sensitive findings

  These passed on one numerical stack and failed narrowly on another:

  - gaussian_fs_numeric_by: penalty matrix maximum difference 1.2316e-12 against 1e-12.
  - gaulss_select_true_cr preoptimization: ldetS difference 1.95e-10 against 1e-10.
  - gammals_select_true_cr: EDF2 total 3.690821 versus 3.691378, difference 5.56e-4 against 5e-4.

  These should not be dismissed as ordinary tolerance noise under the repository’s strict parity policy. They indicate operation-order or eigensolver sensitivity, but are less urgent than the material failures above.

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

  1. nampy/gam/fit/state.py:296 contains _fallback_single_smooth_edf. If computed EDF is invalid, it substitutes either a numerical design rank for a random effect or trace(H) - nsdf for a single smooth. This is heuristic fallback behavior and conflicts with the repository’s “fail explicitly,
     do not approximate parity” rule.

  2. nampy/gam/predict/predictions.py:157 derives an fs contribution shift through a least-squares constant vector and penalty ratio. It passes current output tests, but the upstream mapping is not documented and deserves a direct predict.gam source citation or replacement by the exact upstream
     transformation.

  3. GAM.__init__ stores arbitrary unknown keyword arguments in hparams without rejecting them. Unsupported mgcv-looking arguments can therefore be silently ignored. See nampy/gam/model/api.py:47.
  4. summary() is a bespoke text summary, not summary.gam; nampy/gam/diagnostics/summary.py:87 lacks coefficient tables, smooth significance tables, R² and deviance explained.
  5. plot() is a basic one-/two-dimensional contribution plot, not plot.gam; terms with more dimensions receive a “not implemented” panel. See nampy/gam/diagnostics/plots.py:10.
  6. Extended-family sandwich covariance is not implemented, and general-family residual/diagnostic surfaces depend on family-specific hooks.

  ## Documentation and test-suite drift

  The new GAM_IMPLEMENTED.md:1 and GAM_NOT_IMPLEMENTED.md:1 are valuable inventories, but they are not yet fully accurate:

  - They claim multiple offsets keep the first with an R warning. Current extraction instead raises NotImplementedError in nampy/gam/formula/extract.py:65.
  - GAM_IMPLEMENTED.md describes the PIRLS QR path as using eps * 100, while irls_core defaults to eps**0.66; only the Gaussian exact calls override it.
  - tests/SUBSYSTEM_COVERAGE.md still names obsolete paths and old tensor/joint-trace gaps.
  - The optimization lifecycle registry marks every case stable even though gaussian_reml_newton_fs_xt_ps currently fails.
  - Two unit tests are stale relative to code contracts:
      - The all-fixed Gaussian-scale mock lacks the strict initial.spg design state now required.
      - The negative-binomial helper test omits its new keyword-only optimizer argument.

  These two unit failures are primarily test/code contract drift, not direct evidence that real fitted models fail.

  ## Validation summary

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

  ## Recommended implementation order

  1. Correct fs null-space penalty identity/order and rerun the fs(xt="ps") lifecycle plus fs + select endpoint tests.
  2. Localize the two cs GCV/Cp criterion differences before examining their gradients/Hessians.
  3. Trace gammals_select_true_cr from initial.spg through the optimized coefficient endpoint; the prediction design itself is already correct.
  4. Align PIRLS rank tolerance and replace the NAMpy-only rank-deficient gauge with the actual gdi1 dropped-coordinate behavior.
  5. Remove or explicitly guard the single-smooth EDF heuristic.
  6. Resolve the remaining general-family Sl, prediction-filter and family catalogue gaps.
  7. Reconcile public exports, unknown keyword handling, documentation, lifecycle status metadata and stale unit tests.

  Overall parity is strong for ordinary, well-conditioned GAMs and the declared constructor matrix, but exact parity remains unresolved for factor-smooth selection/order, two shrinkage-spline score cases, selected general-family endpoints, and genuinely rank-deficient fits.
