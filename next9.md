Big picture: 25 fails look like about 6 root bugs, not 25 separate bugs.

  - fs cluster is mostly one root problem: Gaussian REML for bs="fs" is not following mgcv control flow. Local fs builder plus local REML
    optimizer path drift together. Hot spots: nampy/gam/smooths/categorical/factor_smooth.py:523, nampy/gam/smoothing_selection/optimize/
    driver.py:615, and test note already says this is a known REML gap at tests/test_mgcv_additional_scenarios.py:265. This likely explains:
    gaussian_fs_*, fs_4levels, output_parity_terms_*[fs], fs_k_prime_and_edf_match, and the gaussian_fs_select_reml warning test.
  - sz is a direct code-parity miss. mgcv smooth.construct.sz.smooth.spec builds raw duplicated tensor basis first, then handles constraints
    later. Python applies an explicit contrast transform during construction with X_raw @ T and transforms penalties there too. That is not the
    same parameterization/order. Hot spot: nampy/gam/smooths/categorical/factor_smooth.py:782. This explains:
    gaussian_sz_reml_matches_mgcv and factor_smooth_sz.
  - t2 is also not exact mgcv. Python uses handwritten marginal split and custom tensor block assembly, while mgcv uses nat.param(..., type=3,
    unit.fnorm=TRUE) plus the exact t2.model.matrix block ordering. Hot spots: nampy/gam/basis/tensor.py:103 and nampy/gam/basis/tensor.py:220.
    This is the likely root for:
    gaussian_t2_full_false and the failing optimized t2 snapshot.
  - Non-Gaussian fixed-sp failures mostly point to PIRLS / family validity logic not matching mgcv. Local path clips links and uses fallback
    heuristics instead of mgcv validity hooks and step-halving behavior. Hot spots: nampy/gam/fit/solvers/irls_core.py:339, nampy/gam/fit/solvers/
    irls_core.py:424, nampy/gam/families/_function_maps.py:97, nampy/gam/families/_function_maps.py:195, nampy/gam/fit/penalized_system.py:19.
    This likely explains:
    weighted Poisson, fixed-sp negbin, binomial cloglog fixed-sp, gamma inverse fixed-sp, and strict Poisson residual parity. I did not find a
    separate residual-formula bug; residual failure looks downstream of fit mismatch.
  - Negative-binomial theta estimation is not implemented as Python mgcv parity at all. It shells out to R. Hot spot: nampy/gam/
    smoothing_selection/optimize/driver.py:37. That directly explains:
    negbin_theta_estimation, and it violates your “logic must live in Python and mirror mgcv” goal.
  - Gaussian REML score / outer optimization still diverges from mgcv in two ways:
    fixed-sp score algebra is local, not an exact port, at nampy/gam/smoothing_selection/criteria/gaussian.py:41;
    optimized REML uses custom joint L-BFGS-B plus polishing/Newton, and Gaussian derivatives are intentionally finite-diff in nampy/gam/
    smoothing_selection/criteria/dispatch.py:125 and nampy/gam/smoothing_selection/optimize/driver.py:671.
    This likely explains:
    gaussian_exact_reml_fixed_sp_matches_mgcv, gaussian_reml_sig2_matches_mgcv_joint_outer_tensor_smooth, and likely
    gp_numeric_by_pc_reml_matches_mgcv.
  - binomial_separation looks smaller than the others. Failure is only tiny SE drift, so this is probably covariance conditioning, not a model-
    mean mismatch. Hot spots: nampy/gam/fit/penalized_system.py:19 and nampy/gam/fit/covariance.py:20.
  - k.check is not exact mgcv by its own docstring at nampy/gam/diagnostics/k_check.py:79. But current failing fs k-check test looks mostly
    downstream of the fs EDF drift, not like a standalone k.check bug.

  Upstream refs used

  - mgcv/R/smooth.r: smooth.construct.fs.smooth.spec, smooth.construct.sz.smooth.spec, t2.model.matrix, smooth.construct.t2.smooth.spec
  - mgcv/R/gam.fit3.r: gam.fit3, gam.fit3.post.proc, fix.family.link.family, fix.family.link.extended.family
  - mgcv/R/gam.fit4.r: dDeta, gam.fit4, gam.fit5
  - mgcv/R/efam.r: estimate.theta, nb
  - mgcv/R/plots.r: k.check
  - mgcv/R/mgcv.r: predict.gam

  Bottom line

  Main blockers are structural parity misses, not tolerance noise:
  fs, sz, t2, non-Gaussian PIRLS/family control flow, Gaussian REML score/optimizer, and negbin theta shell-out.


tests/parity/test_mgcv_parity.py: 2 warnings
tests/test_mgcv_additional_scenarios.py: 5 warnings
tests/test_mgcv_k_check_parity.py: 4 warnings
tests/test_mgcv_output_parity.py: 7 warnings
tests/test_mgcv_parity_failing_and_warnings.py: 5 warnings
tests/test_mgcv_pc_id_parity.py: 3 warnings
tests/test_mgcv_smoothcon_parity.py: 1 warning
tests/test_mgcv_snapshot_parity.py: 10 warnings
  /home/ad32/projects/package/NAMpy/nampy/gam/model/gam_solve.py:552: UserWarning: Smoothing optimisation did not converge: step failed
    return optimize_smoothing_params(

tests/parity/test_mgcv_parity.py::test_requested_mgcv_parity_models[gaussian_random_intercept_re]
tests/test_mgcv_additional_scenarios.py::TestAdditionalScenarioParity::test_gaussian_fs_ps_marginal_reml_matches_mgcv
tests/test_mgcv_additional_scenarios.py::TestAdditionalScenarioParity::test_gaussian_fs_ps_marginal_select_reml_matches_mgcv
tests/test_mgcv_additional_scenarios.py::TestFsSzMoreFactors::test_gaussian_fs_4levels_reml_matches_mgcv
tests/test_mgcv_k_check_parity.py::TestKCheckParity::test_fs_k_prime_and_edf_match
tests/test_mgcv_output_parity.py::test_output_parity_terms_all_smooth_types[fs]
tests/test_mgcv_parity_failing_and_warnings.py::TestAdditionalScenarioParityFailingOrWarning::test_gaussian_fs_select_reml_matches_mgcv
tests/test_mgcv_parity_failing_and_warnings.py::test_output_parity_terms_standard_errors_fs[fs]
tests/test_mgcv_snapshot_parity.py::TestMgcvParity::test_gaussian_fs_reml_matches_mgcv
  /home/ad32/miniconda3/envs/nampy/lib/python3.11/site-packages/scipy/optimize/_optimize.py:2358: RuntimeWarning: invalid value encountered in scalar subtract
    p = (xf - fulc) * q - (xf - nfc) * r

tests/parity/test_mgcv_parity.py::test_requested_mgcv_parity_models[gaussian_random_intercept_re]
tests/test_mgcv_additional_scenarios.py::TestAdditionalScenarioParity::test_gaussian_fs_ps_marginal_reml_matches_mgcv
tests/test_mgcv_additional_scenarios.py::TestAdditionalScenarioParity::test_gaussian_fs_ps_marginal_select_reml_matches_mgcv
tests/test_mgcv_additional_scenarios.py::TestFsSzMoreFactors::test_gaussian_fs_4levels_reml_matches_mgcv
tests/test_mgcv_k_check_parity.py::TestKCheckParity::test_fs_k_prime_and_edf_match
tests/test_mgcv_output_parity.py::test_output_parity_terms_all_smooth_types[fs]
tests/test_mgcv_parity_failing_and_warnings.py::TestAdditionalScenarioParityFailingOrWarning::test_gaussian_fs_select_reml_matches_mgcv
tests/test_mgcv_parity_failing_and_warnings.py::test_output_parity_terms_standard_errors_fs[fs]
tests/test_mgcv_snapshot_parity.py::TestMgcvParity::test_gaussian_fs_reml_matches_mgcv
  /home/ad32/miniconda3/envs/nampy/lib/python3.11/site-packages/scipy/optimize/_optimize.py:2359: RuntimeWarning: invalid value encountered in scalar subtract
    q = 2.0 * (q - r)

tests/test_mgcv_additional_scenarios.py::TestAdditionalScenarioParity::test_gamma_identity_link_reml_matches_mgcv
  /home/ad32/projects/package/NAMpy/nampy/gam/model/gam_solve.py:552: UserWarning: Smoothing optimisation did not converge: iteration limit reached
    return optimize_smoothing_params(
  ============================================================ short test summary info =============================================================
FAILED tests/test_mgcv_additional_scenarios.py::TestAdditionalScenarioParity::test_weighted_poisson_fixed_sp_matches_mgcv - AssertionError: 
FAILED tests/test_mgcv_additional_scenarios.py::TestAdditionalScenarioParity::test_gaussian_fs_ps_marginal_reml_matches_mgcv - AssertionError
FAILED tests/test_mgcv_additional_scenarios.py::TestAdditionalScenarioParity::test_gaussian_fs_ps_marginal_select_reml_matches_mgcv - AssertionError
FAILED tests/test_mgcv_additional_scenarios.py::TestAdditionalScenarioParity::test_negbin_theta_0p5_fixed_sp_matches_mgcv - AssertionError: 
FAILED tests/test_mgcv_additional_scenarios.py::TestAdditionalScenarioParity::test_negbin_theta_2p0_fixed_sp_matches_mgcv - AssertionError: 
FAILED tests/test_mgcv_additional_scenarios.py::TestAdditionalScenarioParity::test_binomial_cloglog_fixed_sp_matches_mgcv - AssertionError: 
FAILED tests/test_mgcv_additional_scenarios.py::TestAdditionalScenarioParity::test_gamma_inverse_link_fixed_sp_matches_mgcv - AssertionError: 
FAILED tests/test_mgcv_additional_scenarios.py::TestAdditionalScenarioParity::test_gamma_inverse_link_reml_matches_mgcv - AssertionError: 
FAILED tests/test_mgcv_additional_scenarios.py::TestAdditionalScenarioParity::test_gamma_identity_link_reml_matches_mgcv - AssertionError: 
FAILED tests/test_mgcv_additional_scenarios.py::TestFsSzMoreFactors::test_gaussian_fs_4levels_reml_matches_mgcv - AssertionError
FAILED tests/test_mgcv_k_check_parity.py::TestKCheckParity::test_fs_k_prime_and_edf_match - AssertionError: 
FAILED tests/test_mgcv_known_gaps.py::test_strict_poisson_reml_residual_parity - AssertionError: 
FAILED tests/test_mgcv_output_parity.py::test_output_parity_terms_all_smooth_types[fs] - AssertionError: 
FAILED tests/test_mgcv_parity_failing_and_warnings.py::test_requested_mgcv_parity_models_failing_or_warning[gaussian_t2_full_false] - AssertionError: gaussian_t2_full_false: |beta-beta_mgcv| exceeded tolerance; max_err=1.214e+00, max_tol=1.633e-06
FAILED tests/test_mgcv_parity_failing_and_warnings.py::test_requested_mgcv_parity_models_failing_or_warning[binomial_separation] - AssertionError: binomial_separation: |se-se_mgcv| exceeded tolerance; max_err=2.578e-06, max_tol=2.091e-06
FAILED tests/test_mgcv_parity_failing_and_warnings.py::test_requested_mgcv_parity_models_failing_or_warning[factor_smooth_sz] - AssertionError: factor_smooth_sz: |beta-beta_mgcv| exceeded tolerance; max_err=1.070e+00, max_tol=2.687e-06
FAILED tests/test_mgcv_parity_failing_and_warnings.py::test_requested_mgcv_parity_models_failing_or_warning[negbin_theta_estimation] - AssertionError: negbin_theta_estimation: |beta-beta_mgcv| exceeded tolerance; max_err=2.604e-06, max_tol=2.613e-06
FAILED tests/test_mgcv_parity_failing_and_warnings.py::TestAdditionalScenarioParityFailingOrWarning::test_gaussian_fs_select_reml_matches_mgcv - AssertionError
FAILED tests/test_mgcv_parity_failing_and_warnings.py::test_output_parity_terms_standard_errors_fs[fs] - AssertionError: 
FAILED tests/test_mgcv_pc_id_parity.py::TestPcParityREML::test_gp_numeric_by_pc_reml_matches_mgcv - AssertionError: 
FAILED tests/test_mgcv_score_gamma_parity.py::TestMgcvScoreGammaParity::test_gaussian_exact_reml_fixed_sp_matches_mgcv - AssertionError: 
FAILED tests/test_mgcv_snapshot_parity.py::TestMgcvParity::test_gaussian_fs_reml_matches_mgcv - AssertionError
FAILED tests/test_mgcv_snapshot_parity.py::TestMgcvParity::test_gaussian_sz_reml_matches_mgcv - AssertionError: 
FAILED tests/test_mgcv_snapshot_parity.py::TestMgcvParity::test_gaussian_reml_sig2_matches_mgcv_joint_outer_tensor_smooth - AssertionError: 
FAILED tests/test_mgcv_snapshot_parity.py::TestMgcvParity::test_optimized_tensor_t2_snapshot_matches_mgcv[gamma-<lambda>-y ~ t2(x0, x1, bs=["cr", "cr"], k=[6, 6])-1e-06-0.0007-0.0007-0.1-1e-05] - AssertionError: 