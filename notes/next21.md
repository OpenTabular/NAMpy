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
  8. The still-failing model-level scenarios
     These are the core parity cases still under triage and should stay visible until
     fixed:
      - gaussian_ti_mc
      - gaussian_t2_full_false
      - binomial_separation
      - mrf_lattice
      - factor_smooth_sz
      - gaussian_fs_select_reml
      - gaussian_t2_ts_cr_reml
        See tests/parity/test_mgcv_parity_failing_and_warnings.py:1.

  If you want the best next sequence, I’d do:

  1. iterms
  2. non-Gaussian Vc/edf2 post-proc
  3. anova.gam
  4. factor-by terms
  5. residual/k-check edge cases
  6. remaining scenario-case mismatches

  After those, the main “add” items would be extra predict.gam coverage for terms=,
  exclude=, iterms.type, na.action, and factor-level edge cases.