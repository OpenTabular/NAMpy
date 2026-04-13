Group by bug family, not by file. Clear split:

  1. gamma division-by-zero family

  - tests/test_mgcv_additional_scenarios.py::test_gamma_identity_link_reml_matches_mgcv
  - tests/test_mgcv_snapshot_parity.py::test_gamma_reml_matches_mgcv
  - tests/test_mgcv_snapshot_parity.py::test_gamma_reml_optimizes_without_abnormal_warning
  - tests/test_mgcv_snapshot_parity.py::test_optimized_tensor_t2_snapshot_matches_mgcv[gamma-...]
  - tests/test_mgcv_score_gamma_parity.py::test_gamma_pirls_reml_fixed_sp_value_gradient_hessian_match_mgcv

  2. non-gaussian IRLS / link / gradient-hessian family

  - tests/test_mgcv_trace_parity.py::test_non_gaussian_final_gradient_matches_mgcv[...]
  - tests/test_mgcv_trace_parity.py::test_non_gaussian_final_hessian_matches_mgcv[...]
  - tests/test_mgcv_score_gamma_parity.py::test_poisson_pirls_reml_fixed_sp_value_gradient_hessian_match_mgcv
  - tests/test_mgcv_snapshot_parity.py::test_binomial_reml_matches_mgcv
  - tests/test_mgcv_snapshot_parity.py::test_poisson_reml_matches_mgcv
  - tests/test_mgcv_snapshot_parity.py::test_poisson_reml_with_formula_offset_matches_mgcv
  - tests/test_mgcv_output_parity.py::test_output_parity_newdata_standard_errors[poisson_cr_uni_reml-poisson-response]

  3. factor-smooth / fs / sz family

  - tests/parity/test_mgcv_parity.py::... [gaussian_fs_by_factor]
  - tests/test_mgcv_additional_scenarios.py::test_gaussian_fs_ps_marginal_reml_matches_mgcv
  - tests/test_mgcv_additional_scenarios.py::test_gaussian_fs_ps_marginal_select_reml_matches_mgcv
  - tests/test_mgcv_additional_scenarios.py::test_gaussian_fs_4levels_reml_matches_mgcv
  - tests/test_mgcv_additional_scenarios.py::test_gaussian_sz_3x3_reml_matches_mgcv
  - tests/test_mgcv_k_check_parity.py::test_fs_k_prime_and_edf_match
  - tests/test_mgcv_output_parity.py::test_output_parity_terms_all_smooth_types[fs]
  - tests/test_mgcv_output_parity.py::test_output_parity_terms_all_smooth_types[sz]
  - tests/test_mgcv_output_parity.py::test_output_parity_terms_standard_errors[sz]
  - tests/test_mgcv_parity_failing_and_warnings.py::test_requested_mgcv_parity_models_failing_or_warning[factor_smooth_sz]
  - tests/test_mgcv_parity_failing_and_warnings.py::test_gaussian_fs_select_reml_matches_mgcv
  - tests/test_mgcv_parity_failing_and_warnings.py::test_output_parity_terms_standard_errors_fs[fs]
  - tests/test_mgcv_snapshot_parity.py::test_gaussian_fs_reml_matches_mgcv
  - tests/test_mgcv_snapshot_parity.py::test_gaussian_sz_reml_matches_mgcv

  4. mrf / spatial family

  - tests/test_mgcv_parity_failing_and_warnings.py::test_requested_mgcv_parity_models_failing_or_warning[mrf_lattice]
  - tests/test_mgcv_snapshot_parity.py::test_gaussian_mrf_low_rank_reml_matches_mgcv
  - tests/test_mgcv_smoothcon_parity.py::test_mrf_smoothcon_basis_matches_mgcv
  - tests/test_mgcv_smoothcon_parity.py::test_mrf_smoothcon_penalty_matches_mgcv

  5. tensor / ti / t2 / k-check / pc-id family

  - tests/test_mgcv_parity_failing_and_warnings.py::test_requested_mgcv_parity_models_failing_or_warning[gaussian_ti_mc]
  - tests/test_mgcv_parity_failing_and_warnings.py::test_requested_mgcv_parity_models_failing_or_warning[gaussian_t2_full_false]
  - tests/test_mgcv_snapshot_parity.py::test_optimized_tensor_t2_snapshot_matches_mgcv[...]
  - tests/test_mgcv_snapshot_parity.py::test_gaussian_reml_sig2_matches_mgcv_joint_outer_tensor_smooth
  - tests/test_mgcv_k_check_parity.py::test_gaussian_gp_reml
  - tests/test_mgcv_k_check_parity.py::test_mixed_numeric_and_re_terms
  - tests/test_mgcv_pc_id_parity.py::TestPcParityREML::test_gp_numeric_by_pc_reml_matches_mgcv

  6. concurvity / output-parity / misc family

  - tests/test_mgcv_snapshot_parity.py::test_poisson_concurvity_full_matches_mgcv
  - tests/test_mgcv_snapshot_parity.py::test_poisson_concurvity_pairwise_matches_mgcv
  - tests/test_mgcv_parity_failing_and_warnings.py::test_binomial_separation

  - Fix gamma divide-by-zero first. Big blocker, many tests.
  - Fix non-gaussian IRLS/link parity second. Hits Poisson/binomial core.
  - Fix fs/sz third. Many downstream output and k-check failures likely same root.
  - Fix mrf and tensor families after. Likely separate basis/penalty bugs.