If I were ordering the remaining work, the major issues are:

1. predict.gam(type="iterms")
This is still not implemented in NAMpy, and it is a real missing surface, not just
a tolerance issue. See tests/parity/
test_mgcv_prediction_inference_diagnostics_parity.py:1.
2. Non-Gaussian unconditional prediction parity
The main blocker is the known mgcv-style Vc / edf2 post-proc gap for non-Gaussian
PIRLS/final-fit objects. Until that is exact, unconditional se.fit parity will stay
incomplete. See tests/optimization/test_mgcv_postprocessing_final_fit_parity.py:1.
3. predict(type="terms") edge cases
The main concrete mismatch now is factor-by term aggregation/splitting. Random-
effect unconditional termwise SE is also still under triage. See tests/parity/
test_mgcv_prediction_inference_diagnostics_parity.py:1.
4. anova.gam parity
This is still a big bucket.
  - Single-model anova(freq=FALSE) is still tracked as a gap for several surfaces:
  binomial_logit, poisson, gamma_log, gaussian_random_intercept_re,
  gaussian_by_factor, gaussian_fs_by_factor, gaussian_select_true,
  gaussian_weights, gaussian_formula_offset.
  - Non-Gaussian model-comparison anova(..., test="Chisq") is also still a tracked
  gap.
  Same file as above.
5. Working residuals with remembered formula offsets
gaussian_formula_offset working residual parity is still wrong and worth debugging
directly. Same file.
6. k.check edge cases
Still tracked for gaussian_by_factor, gaussian_fs_by_factor,
gaussian_fs_select_reml, plus the broader failing-warning cases. Same file.
7. Final-fit post-processing parity
The big remaining implementation items are:
  - non-Gaussian PIRLS Vc / edf2
  - advanced general-family final-fit parity for select/by/tensor/shashlss-type
  surfaces
  - exact weighted and general-family outer.info trace/count parity
  See tests/optimization/test_mgcv_postprocessing_final_fit_parity.py:1.

  If you want the best next sequence, I’d do:

1. iterms
2. non-Gaussian Vc/edf2 post-proc
3. anova.gam
4. factor-by terms
5. residual/k-check edge cases
6. remaining scenario-case mismatches

  After those, the main “add” items would be extra predict.gam coverage for terms=,
  exclude=, iterms.type, na.action, and factor-level edge cases.







Listed “still-failing model-level scenarios” in tests/parity/

  test_mgcv_parity_failing_and_warnings.py:69 not actually failing on current tree. All

  7 pass. Zero fit warnings captured. File/comment now stale as model-fit tracker.

  Shared assertion helper in tests/_mgcv_parity_requested_shared.py:42 also shows which

  cases still use relaxed/skipped checks. Some of those relaxations now look stale too.

  Triage

  - gaussian_ti_mc: fit parity exact. Link newdata parity passes. anova.gam still fails:

    p-value 0.380306 vs 0.380213. Looks like output-surface gap in nampy/gam/inference/

    anova.py:503, not model-fit gap.

  - gaussian_t2_full_false: fit parity exact. Output SE slice passes. Setup/preopt/

    reparam slices pass. Link newdata parity passes. anova.gam still fails: smooth stat

    23.6368 vs 24.5493. Again looks like anova.gam, not core t2() fit. Local owner still

    nampy/gam/smooths/tensor/t2.py:136 for term construction.

  - binomial_separation: fit parity exact enough that current skip_coef_comparison=True

    looks stale. Raw coef max err only 1.84e-9. Link newdata parity passes. anova.gam

    passes. Remaining tested gap is k.check p-value: 0.75 vs 0.0 in nampy/gam/

    diagnostics/k_check.py:125. PIRLS owner still nampy/gam/fit/solvers/

    irls_core.py:347.

  - mrf_lattice: one real model-level gap remains. Coefs/link exact, but REML criterion

    still off by 0.3494, log-sp by 0.1398; test still needs criterion_atol=5e-1.

    anova.gam also fails badly: ref_df 0 vs 2. Main owner nampy/gam/smooths/categorical/

    mrf.py:217 plus anova retest path.

  - factor_smooth_sz: model fit now very close; skip_coef_comparison=True also looks

    stale here. Raw coef max err 5.09e-9. Link newdata parity passes. k.check passes.

    Remaining tested gap is anova.gam: smooth stat 2.3733 vs 1.6726. Local owner nampy/

    gam/smooths/categorical/factor_smooth.py:883.

  - gaussian_fs_select_reml: fit parity very close; raw coef max err 1.23e-9, so skip

    also looks stale. Link newdata parity passes. k.check passes. Remaining tested gap

    is anova.gam p-value 0.0 vs 0.5. Owner same nampy/gam/smooths/categorical/

    factor_smooth.py:615.

  - gaussian_t2_ts_cr_reml: strongest “graduation” candidate. Model fit passes. Link

    newdata passes. anova.gam passes. k.check passes. Only raw coef vector differs a lot

    (max err 10.8), so prediction-based comparison still correct for this one.

  Big finding

  Blanket xfail bucket in tests/parity/

  test_mgcv_prediction_inference_diagnostics_parity.py:79 too coarse now. It xfails all

  tracked cases on all prediction/anova/residual/k.check surfaces before running. Spot-

  checks with --runxfail show:

  - link newdata predictions: 7/7 pass

  - k.check: 6/7 pass, only binomial_separation fails

  - anova.gam: 2/7 pass, 5/7 fail

  So tracking should move from “scenario-wide failing” to “surface-specific failing”.