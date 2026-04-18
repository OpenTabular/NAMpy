Across the focused mgcv parity files I reran today, the current state is:

  - 272 passed
  - 21 failed
  - 4 skipped

  What Matches
  Green today against upstream mgcv:

  - Smooth construction and penalty parity: tests/test_mgcv_smoothcon_parity.py
    48/48 passed. This covers te, ti, t2, fs, sz, mrf, re, cyclic cubic, P-
    splines, GP, and predict-matrix consistency.
  - pc= and linked id= behavior: tests/test_mgcv_pc_id_parity.py 35/35 passed.
  - Output surfaces: tests/test_mgcv_output_parity.py 28/28 passed for
    prediction, anova, terms, lpmatrix, offsets, and SE-related output.
  - Optimizer traces and Newton parity: tests/test_mgcv_trace_parity.py 7/7,
    tests/test_mgcv_newton_parity.py 3/3, tests/
    test_mgcv_newton_exact_parity.py 3/3.
  - k.check() parity: tests/test_mgcv_k_check_parity.py 8 passed, 1 skipped.
    The skip is the already-noted fs whole-surface case under triage.
  - General-family / GAMLSS surfaces: tests/test_general_family_mgcv_parity.py
    23/23 and tests/test_mgcv_gamlss_core.py 15/15 passed. gaulss, gammals,
    gevlss, shashlss, ziplss derivative/vcov/predict/anova paths are in good
    shape where implemented.
  - Lower-level derivative/backend machinery: tests/
    test_mgcv_score_gamma_parity.py 4/4, tests/
    test_gam_gaussian_smoothness_postprocess_parity.py 5/5, tests/
    test_mgcv_gaussian_backend_selection.py 3/3, tests/
    test_gam_mgcv_patch_regressions.py 15/15.
  - Core snapshot matrix is partly green: tests/test_mgcv_snapshot_parity.py
    33/44 passed.
  - Additional scenarios are mostly green: tests/
    test_mgcv_additional_scenarios.py 32 passed, 3 failed, 2 skipped.

  What Remains Next
  Highest-value remaining work, in order:

  - Gaussian bs="re" exact backend is the main blocker. Multiple snapshot
    failures die in nampy/gam/fit/solvers/gaussian_exact.py:123 on
    np.linalg.solve(A, eye) with a singular matrix. That currently breaks the
    RE REML / near-singular row-space-null-space parity cluster.
  - Joint negative-binomial estimate_theta=True parity is still not restored.
    It fails in tests/test_mgcv_known_gaps.py, tests/
    test_mgcv_additional_scenarios.py, and tests/
    test_mgcv_parity_failing_and_warnings.py. One tracked test now also misses
    expected endpoint metadata (joint_negbin_efs_outer), so this looks like
    both a parity gap and a regression in diagnostics.
  - Cyclic-cubic tensor ti(..., bs=["cc","cc"]) prediction mapping is broken:
    nampy/gam/compiler/structures.py:92 hits a 25 vs 36 matrix-shape mismatch
    in the additional-scenarios file.
  - Remaining endpoint drifts are smaller but still real:
      - Gaussian MRF REML sig2 / smoothing endpoint mismatch in the snapshot
        matrix.
      - Optimized non-Gaussian t2(cr, cr) REML cases for Poisson, Binomial, and
        NegBin.
      - mrf_lattice REML criterion gap.
      - factor_smooth_sz coefficient parity gap.
      - gaussian_fs_select_reml missing the expected flat_ridge_suspected
        endpoint flag.
      - Weighted Poisson fixed-sp is extremely close but still misses a very
        tight 1e-10 tolerance.
  - Skipped/triaged items still open:
      - fs whole-surface k_check parity.
      - gaussian_ti_mc.
      - Two slow fs_ps_marginal scenario tests.

  Practical Read
  The repo already mirrors upstream well on basis construction, penalty
  assembly, pc/linked-id, output surfaces, trace/Newton behavior, and most
  general-family derivative machinery. The remaining work is concentrated in a
  smaller set of endpoint-sensitive fit paths: Gaussian random effects, joint
  negbin theta estimation, one cyclic-cubic tensor prediction path, and a
  handful of MRF/tensor/factor-smooth edge cases.