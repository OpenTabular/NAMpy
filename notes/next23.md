Big picture: core ordinary gam.fit3-style parity looks far along. Biggest missing work sits in gam.fit5 general-family path, post-fit output surfaces, tensor/factor-smooth edge cases, and full mgcv formula/basis surface.

  Findings

- High: general-family gam.fit5 parity still partial. Python still rejects non-reparameterized multi-penalty Sl blocks, contiguous-block assumptions leak through, and outer ML/REML needs analytic gradient/Hessian only. nampy/gam/fit/
solvers/general_fit5.py:260, nampy/gam/fit/solvers/general_fit5.py:393, nampy/gam/fit/solvers/general_fit5.py:844, tests/optimization/test_mgcv_postprocessing_final_fit_parity.py:731. Upstream target: mgcv/R/gam.fit4.r:941, mgcv/R/
gam.fit4.r:1571, mgcv/R/fast-REML.r:68, mgcv/R/fast-REML.r:517, mgcv/src/gdi.c:1953.
- High: non-Gaussian final-fit post-processing still not exact mgcv. Test suite still marks missing unconditional Vc / edf2 carry-through for non-Gaussian PIRLS, separate gamma AIC gap, and exact outer.info trace/count mismatches for
weighted and general-family outer_newton. tests/optimization/test_mgcv_postprocessing_final_fit_parity.py:696, tests/optimization/test_mgcv_postprocessing_final_fit_parity.py:701, tests/optimization/
test_mgcv_postprocessing_final_fit_parity.py:779, tests/optimization/test_mgcv_postprocessing_final_fit_parity.py:789. Upstream target: mgcv/R/gam.fit3.r:955, mgcv/R/gam.fit4.r:1571.
- High: tensor constructor parity not finished. Raw constructor file says several t2 surfaces already leak into fixed-sp behavior, not only raw representation. tests/smooths/test_mgcv_raw_constructor_parity.py:858. Upstream target:
mgcv/R/smooth.r:741, mgcv/R/smooth.r:1033.
- Medium: prediction/inference parity still has tracked gaps on advanced surfaces. Unconditional SE for t2(full=False) still differs, sz unconditional SE and iterms still differ, k.check for factor-by smooths still under triage, and
non-Gaussian model-comparison anova.gam still has tiny drift. tests/parity/test_mgcv_prediction_inference_diagnostics_parity.py:127, tests/parity/test_mgcv_prediction_inference_diagnostics_parity.py:153, tests/parity/
test_mgcv_prediction_inference_diagnostics_parity.py:158, tests/parity/test_mgcv_prediction_inference_diagnostics_parity.py:726. Upstream target: mgcv/R/mgcv.r:2692, mgcv/R/mgcv.r:4102, mgcv/R/plots.r:182.
- Medium: formula/build surface still much narrower than mgcv. Transformed parametric terms, transformed smooth covariates, transformed by, transformed response/offset, multiple predictor-specific offsets, non-numeric offsets, and
extra smooth args still stop at build stage. nampy/gam/specs/build.py:344, nampy/gam/specs/build.py:762, nampy/gam/specs/build.py:791, nampy/gam/specs/build.py:864, nampy/gam/specs/build.py:984, nampy/gam/specs/build.py:1011, nampy/
gam/specs/build.py:1026, nampy/gam/formula/parse.py:606, nampy/gam/formula/parse.py:628. Upstream target: mgcv/R/mgcv.r:292.
- Medium: basis catalog still smaller than mgcv. Builder only accepts cr/cs/cc/ps/tp/ts/gp/mrf/re/fs/sz plus te/ti/t2; anything else hard-stops. nampy/gam/specs/build.py:209, nampy/gam/specs/build.py:301, nampy/gam/specs/build.py:305.
Upstream mgcv has many more constructor families in smooth.r.
- Medium: categorical smooths still narrower than upstream. fs/sz require singly penalized base smooths and limited base bases; re rejects id=; mrf supports exactly one area variable. nampy/gam/smooths/categorical/factor_smooth.py:636,
nampy/gam/smooths/categorical/factor_smooth.py:912, nampy/gam/smooths/categorical/random_effect.py:129, nampy/gam/smooths/categorical/mrf.py:58. Upstream target: mgcv/R/smooth.r:1996, mgcv/R/smooth.r:2187, mgcv/R/smooth.r:2571, mgcv/
R/smooth.r:2726.
- Medium: diagnostics for some general families still explicitly unsupported. shashlss deviance residuals and k_check still raise, and generic predictive SE path for general families depends on family-specific predict(..., Vb=...).
nampy/gam/diagnostics/residuals.py:99, nampy/gam/predict/general.py:188, tests/families/test_general_family_mgcv_parity.py:1154. Upstream target: mgcv/R/mgcv.r:3426, mgcv/R/mgcv.r:2692, family residual methods in mgcv/R/
gamlss.r:3376.
- Medium: NCV/QNCV coverage still incomplete. Code only supports selected family classes, rejects some extended/general branches, and keeps joint negative-binomial NCV as missing. nampy/gam/smoothing_selection/criteria/ncv.py:1599,
nampy/gam/smoothing_selection/criteria/ncv.py:1606, nampy/gam/smoothing_selection/criteria/ncv.py:1646, nampy/gam/smoothing_selection/criteria/ncv.py:1650. Upstream target: mgcv/R/gamlss.r:114, mgcv/R/fast-REML.r:1492, mgcv/src/
discrete.c:2810.
- Low: raw constructor parity still has many gaps, but not all equal. Some are behavior-affecting t2 gaps; many others are raw-only representation mismatches; some branches mgcv itself does not fit. tests/smooths/
test_mgcv_raw_constructor_parity.py:804, tests/smooths/test_mgcv_raw_constructor_parity.py:813, tests/smooths/test_mgcv_raw_constructor_parity.py:858. This means: do not treat whole raw-constructor list as same priority.

  Priority

1. Finish gam.fit5.post.proc parity for non-Gaussian models: Vc, edf2, AIC, outer.info, weighted/general-family exact traces.
2. Finish remaining gam.fit5/gdi2/Sl.setup parity for general families and NCV.
3. Fix behavior-affecting tensor gaps first, especially t2 surfaces.
4. Close output-surface gaps: unconditional SE, iterms, anova.gam, k.check.
5. Broaden formula/runtime surface only after behavior gaps: transformed terms, offsets, missing smooth families, richer categorical smooth cases.

What changed

  - Ported core mgcv::[gam.fit5.post](http://gam.fit5.post).proc shape into nampy/gam/fit/solvers/gam_fit5.py:799.

  - Added upstream-style split between final Vc and edge/unshifted Vc1 for edf2.

  - Threaded outer_info into general-family post-proc in nampy/gam/fit/solvers/general_fit5.py:896.

  - Added [outer.info](http://outer.info) metadata fields convergence/message/counts in nampy/gam/smoothing_selection/optimize/newton_mgcv.py:647 and nampy/gam/smoothing_selection/optimize/bfgs_mgcv.py:719.

  - Added final selected-sp derivative refresh hook in nampy/gam/smoothing_selection/optimize/driver.py:155.

  - Tightened ordinary-family loglik() scale choice in nampy/gam/model/api.py:645.

  Upstream refs used

  - mgcv/R/gam.fit4.r:1571 [gam.fit5.post](http://gam.fit5.post).proc

  - mgcv/R/gam.fit3.r outer optimizer logic

  - mgcv/R/mgcv.r:4420 logLik.gam

  What now passes

  - Vc/edf2 post-proc gap moved forward.

  - pytest tests/optimization/test_mgcv_postprocessing_final_fit_[parity.py](http://parity.py)::test_gam_fit3_non_gaussian_unconditional_postproc_matches_mgcv[gamma_log] -v

    passed.

  What still fails

  - Weighted Gaussian exact [outer.info](http://outer.info) trace still off:

      - score_hist mismatch remains

  Targeted tests run

  - pytest tests/optimization/test_mgcv_postprocessing_final_fit_[parity.py](http://parity.py)::test_gam_fit3_non_gaussian_unconditional_postproc_matches_mgcv[gamma_log] -v

    passed

  - pytest tests/optimization/test_mgcv_postprocessing_final_fit_[parity.py](http://parity.py)::test_gam_fit5_postprocessing_final_fit_matches_mgcv[gaulss_cr] -v

    failed on outer_info hess

  - pytest --runxfail tests/optimization/test_mgcv_postprocessing_final_fit_[parity.py](http://parity.py)::test_gam_fit5_outer_info_trace_exact_known_gap[gaulss_cr_outer_info_exact] -v

    failed on outer_info hess

  - pytest --runxfail tests/optimization/test_mgcv_postprocessing_final_fit_[parity.py](http://parity.py)::test_gam_fit5_outer_info_trace_exact_known_gap[gaussian_weights_outer_info_exact] -v

    failed on score_hist

    still fails on aic

  Bottom line

  - [gam.fit5.post](http://gam.fit5.post).proc parity for non-Gaussian Vc/edf2 improved materially.

  - Exact parity not finished.

  - Remaining work is not in broad post-proc glue now. It is in:

      1. exact outer-derivative trace reproduction for weighted/general-family optimizer history,

      2. exact mgcv logLik/AIC path for gamma_log.