Mostly Green, With Specific Remaining Gaps

- gaussian_t2_full_false: overall fit matches; predict(link/response/lpmatrix)
matches.
Remaining: predict(type="terms"), unconditional SE for link/response/terms, iterms,
iterms_type=2, and single-model anova.gam.
- factor_smooth_sz: overall fit matches; ordinary prediction surfaces, residuals, and
k.check match.
Remaining: unconditional SE for link/response/terms, iterms, iterms_type=2, and
single-model anova.gam.
- gaussian_fs_select_reml: fit and most prediction/diagnostic surfaces match.
Remaining: unconditional predict(type="terms") and single-model anova.gam.
- gaussian_t2_ts_cr_reml: fit is green on invariant/prediction-based checks; link/
response, unconditional link/response, anova.gam, residuals, and k.check match.
Remaining: raw coefficients are still not a reliable parity signal here, plus
predict(type="terms"), lpmatrix, unconditional terms, iterms, and iterms_type=2.

  Still Not Fully Green At Model/Inference Level

- mrf_lattice: many prediction surfaces now match, and k.check matches.
Remaining: model criterion still needs relaxed tolerance, single-model anova.gam
still differs, and scaled.pearson residuals still differ.

  Cross-Cutting Summary

- Ordinary predict(type="link") and predict(type="response"): green for all 7.
- k.check: green for all 7.
- Residuals: green except mrf_lattice scaled Pearson.
- Biggest remaining cluster: tensor/factor-smooth termwise surfaces and anova.gam.

  If you want, I can turn this into a small checklist in FAILING_TESTS.md style so each
  remaining bug has one owner surface and one exact failing test.



 Still Open

  - test_gam_fit5_postprocessing_final_fit_matches_mgcv[gevlss_cr] still real failure.

    Mismatch still in unconditional covariance Vc diagonal. Best code pointer: nampy/

    gam/fit/solvers/general_fit5.py:903.

  - test_negbin_estimated_theta_reml_matches_mgcv still real failure. Drift still

    0.0127992 on log smoothing parameter. Strong suspect: existing dirty user edit in

    nampy/gam/smoothing_selection/criteria/pirls_deriv.py:472, specifically

    *gdi*pk_setup feeding *gdi2*negbin_joint_kernel. I left that file untouched.

  - test_gaussian_fs_select_reml_matches_mgcv is harness issue, not model regression.

    tests/mgcv_parity_utils.py:97 defaults MGCV_CACHE_ONLY=1; missing cache key then

    hard-fails at tests/mgcv_parity_utils.py:330.

  - XPASS(strict) items are stale test metadata, not regressions. I confirmed

    representative cases for unconditional RE SE and k.check fs-by-factor now pass;

    xfail markers need cleanup.