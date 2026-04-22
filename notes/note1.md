General-family prediction parity is still incomplete. nampy/gam/predict/
     general.py:118 implements terms, lpmatrix, offsets, and single-predictor iterms, but
     direct mgcv parity in tests/families/test_general_family_mgcv_parity.py:1083 and
     tests/families/test_general_family_mgcv_parity.py:1245 only covers in-sample link/
     response plus their SEs; tests/families/test_general_family_mgcv_parity.py:1202.
     There is no general-family analog of the ordinary-family newdata and unconditional-
     SE parity in tests/parity/test_mgcv_prediction_inference_diagnostics_parity.py:387
     and tests/parity/test_mgcv_prediction_inference_diagnostics_parity.py:442.
  2. General-family secondary diagnostics are mostly unpinned even though the snapshot
     harness already serializes them. tests/parity/mgcv_snapshot.R:239 captures
     concurvity, sp.vcov, gam.vcomp, one_se_rule, and tests/parity/mgcv_snapshot.R:263,
     but the only direct parity file for those surfaces is tests/parity/
     test_mgcv_secondary_diagnostics_parity.py:17, and it is Gaussian/Poisson-only. In
     the general-family file, tests/families/test_general_family_mgcv_parity.py:1321, so
     concurvity, sp_vcov, one_se_rule, gam_vcomp, and full k_check parity still need
     direct coverage for general families.
  3. NCV/QNCV parity breadth is behind the implementation. nampy/gam/smoothing_selection/
     criteria/ncv.py:1581 supports gaulss, gammals, gevlss, shashlss, and ziplss, but
     tests/optimization/test_mgcv_ncv_qncv_parity.py:287 only exercises gaulss on the
     general-family side, plus a single tests/optimization/
     test_mgcv_ncv_qncv_parity.py:385. The missing cases are gammals, gevlss, shashlss,
     and ziplss for both NCV and QNCV.
  4. Outer-optimizer trace parity is still too narrow. tests/optimization/
     test_mgcv_outer_optimization_parity.py:230 is effectively Poisson-only, while nampy/
     gam/smoothing_selection/optimize/driver.py:380 has distinct joint branches for Gamma
     scale and negative-binomial theta. Endpoint parity exists elsewhere for some of
     those paths, but exact trace / outer-info row parity is not pinned for the joint
     Gamma and joint negbin branches.
  5. gam_vcomp parity coverage is much thinner than the implementation. nampy/gam/
     smoothing_selection/postfit.py:166 supports both rescale=False and rescale=True, and
     the R helper already supports Gaussian plus several general families in tests/
     mgcv_parity_utils.py:1039, but tests/optimization/test_mgcv_vcomp_parity.py:96 only
     checks Gaussian rescale=True. Default rescale=False, non-Gaussian ordinary families,
     and general families are still untested for parity.
  6. Some explicit unsupported branches still lack strict negative tests. I did not find
     tests for the general-family contiguous-penalty guards in nampy/gam/fit/solvers/
     general_family_solver.py:421 and nampy/gam/fit/solvers/general_family_solver.py:527,
     the non-reparameterized Sl rejection in nampy/gam/fit/solvers/
     general_newton_solver.py:1718, or the general-family terms/iterms reject branches in
     nampy/gam/predict/general.py:133. Given the repo rule that unsupported behavior
     should raise explicitly, those should be pinned.