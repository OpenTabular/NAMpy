FAILED tests/parity/test_mgcv_parity.py::test_requested_mgcv_parity_models[gaussian_weights] - AssertionError: gaussian_weights: |beta-beta_mgcv| exceeded tolerance; max_err=8.829e-04, max_tol=2.180e-06
FAILED tests/test_gam_mgcv_patch_regressions.py::test_pirls_step_halving_exhaustion_returns_failure_without_accepting_bad_step - TypeError: _FailingStepFamily.deviance() got an unexpected keyword argument 'weights'
FAILED tests/test_gam_mgcv_patch_regressions.py::test_disjoint_multi_penalty_term_is_accepted_and_reparameterized - assert False
FAILED tests/test_gam_mgcv_patch_regressions.py::test_overlapping_null_space_penalties_on_one_term_are_accepted - assert False
FAILED tests/test_gam_mgcv_patch_regressions.py::test_dynamic_reparam_design_depends_on_current_sp - AssertionError: assert (60, 5) == (60, 1)
FAILED tests/test_gam_mgcv_patch_regressions.py::test_tensor_id_metadata_maps_one_smoothing_id_to_multiple_sp_indices - AttributeError: 'GAM' object has no attribute 'design_'
FAILED tests/test_mgcv_additional_scenarios.py::TestAdditionalScenarioParity::test_gaussian_sz_select_reml_matches_mgcv - ValueError: _flapack._flapack.dormqr: failed to create array from the 4th argument `tau` -- 0-th dimension must be fixed to 13 but got 8
FAILED tests/test_mgcv_additional_scenarios.py::TestAdditionalScenarioParity::test_weighted_poisson_fixed_sp_matches_mgcv - AssertionError: 
FAILED tests/test_mgcv_additional_scenarios.py::TestAdditionalScenarioParity::test_negbin_estimated_theta_reml_matches_mgcv - AssertionError: 
FAILED tests/test_mgcv_additional_scenarios.py::TestFsSzMoreFactors::test_gaussian_fs_4levels_reml_matches_mgcv - ValueError: _flapack._flapack.dormqr: failed to create array from the 4th argument `tau` -- 0-th dimension must be fixed to 25 but got 24
FAILED tests/test_mgcv_additional_scenarios.py::TestFsSzMoreFactors::test_gaussian_sz_3x3_reml_matches_mgcv - ValueError: _flapack._flapack.dormqr: failed to create array from the 4th argument `tau` -- 0-th dimension must be fixed to 25 but got 18
FAILED tests/test_mgcv_additional_scenarios.py::TestDistributionalRegressionMultiPredictor::test_two_predictors_are_structurally_independent - TypeError: 'module' object is not callable
FAILED tests/test_mgcv_known_gaps.py::test_negbin_estimated_theta_reml_endpoint_matches_mgcv - AssertionError: 
FAILED tests/test_mgcv_known_gaps.py::test_negbin_estimated_theta_reml_two_smooth_theta2_matches_mgcv - AssertionError: 
FAILED tests/test_mgcv_known_gaps.py::test_negbin_estimated_theta_reml_two_smooth_theta05_gap_tracked - Failed: DID NOT RAISE <class 'NotImplementedError'>
FAILED tests/test_mgcv_newton_exact_parity.py::test_newton_score_hist_matches_r_exact[gaussian-321-0.0] - AssertionError: 
FAILED tests/test_mgcv_newton_parity.py::TestMgcvNewtonParity::test_newton_score_hist_gaussian_reml_matches_r_exact - AssertionError: 
FAILED tests/test_mgcv_output_parity.py::test_output_parity_terms[fs-no_se] - ValueError: _flapack._flapack.dormqr: failed to create array from the 4th argument `tau` -- 0-th dimension must be fixed to 19 but got 18
FAILED tests/test_mgcv_output_parity.py::test_output_parity_terms[fs-with_se] - ValueError: _flapack._flapack.dormqr: failed to create array from the 4th argument `tau` -- 0-th dimension must be fixed to 19 but got 18
FAILED tests/test_mgcv_output_parity.py::test_output_parity_terms[sz-no_se] - ValueError: _flapack._flapack.dormqr: failed to create array from the 4th argument `tau` -- 0-th dimension must be fixed to 13 but got 8
FAILED tests/test_mgcv_output_parity.py::test_output_parity_terms[sz-with_se] - ValueError: _flapack._flapack.dormqr: failed to create array from the 4th argument `tau` -- 0-th dimension must be fixed to 13 but got 8
FAILED tests/test_mgcv_parity_failing_and_warnings.py::test_requested_mgcv_parity_models_failing_or_warning[mrf_lattice] - AssertionError: mrf_lattice: |REML-REML_mgcv|=2.291e+00 >= 5.000e-01
FAILED tests/test_mgcv_parity_failing_and_warnings.py::test_requested_mgcv_parity_models_failing_or_warning[factor_smooth_sz] - ValueError: _flapack._flapack.dormqr: failed to create array from the 4th argument `tau` -- 0-th dimension must be fixed to 25 but got 18
FAILED tests/test_mgcv_parity_failing_and_warnings.py::TestAdditionalScenarioParityFailingOrWarning::test_gaussian_fs_select_reml_matches_mgcv - ValueError: _flapack._flapack.dormqr: failed to create array from the 4th argument `tau` -- 0-th dimension must be fixed to 19 but got 18
FAILED tests/test_mgcv_parity_failing_and_warnings.py::test_negbin_theta_estimation_reml_matches_mgcv - AssertionError: 
FAILED tests/test_mgcv_snapshot_parity.py::TestMgcvParity::test_snap_coef_to_reference_null_space_matches_reference - AttributeError: 'GAM' object has no attribute 'Z'
FAILED tests/test_mgcv_snapshot_parity.py::TestMgcvParity::test_gaussian_fs_reml_matches_mgcv - ValueError: _flapack._flapack.dormqr: failed to create array from the 4th argument `tau` -- 0-th dimension must be fixed to 19 but got 18
FAILED tests/test_mgcv_snapshot_parity.py::TestMgcvParity::test_gaussian_sz_reml_matches_mgcv - ValueError: _flapack._flapack.dormqr: failed to create array from the 4th argument `tau` -- 0-th dimension must be fixed to 13 but got 8
FAILED tests/test_mgcv_snapshot_parity.py::TestMgcvParity::test_gaussian_reml_sig2_matches_mgcv_joint_outer_tensor_smooth - AssertionError: 
FAILED tests/test_mgcv_snapshot_parity.py::TestMgcvParity::test_gaussian_reml_sig2_matches_mgcv_joint_outer_mrf_exact - AssertionError: 
FAILED tests/test_mgcv_snapshot_parity.py::TestMgcvParity::test_optimized_tensor_t2_snapshot_matches_mgcv[poisson-<lambda>-y ~ t2(x0, x1, bs=["cr", "cr"], k=[6, 6])-0.0001-5e-08-3e-08-1e-10-1e-10] - AssertionError: 
FAILED tests/test_mgcv_snapshot_parity.py::TestMgcvParity::test_optimized_tensor_t2_snapshot_matches_mgcv[binomial-<lambda>-y ~ t2(x0, x1, bs=["cr", "cr"], k=[6, 6])-0.0001-5e-08-3e-08-1e-10-1e-10] - AssertionError: 
FAILED tests/test_mgcv_snapshot_parity.py::TestMgcvParity::test_optimized_tensor_t2_snapshot_matches_mgcv[family3-<lambda>-y ~ t2(x0, x1, bs=["cr", "cr"], k=[6, 6])-3e-06-2e-08-2e-08-0.13-1e-10] - AssertionError: 
FAILED tests/test_mgcv_trace_parity.py::TestMgcvTraceParity::test_gaussian_reml_trace_matches_mgcv_endpoint - assert 1 >= 3
FAILED tests/test_mgcv_trace_parity.py::TestMgcvTraceParity::test_gaussian_reml_newton_score_hist_matches_exactly - AssertionError: 
================================================================================ 35 failed, 312 passed, 5 skipped, 5 warnings in 500.70s (0:08:20) 