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