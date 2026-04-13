Static review only. No project code run. No tests run. No files changed.

  Findings

  - Biggest gap: negbin REML theta path not Python port at all. nampy/gam/
    smoothing_selection/optimize/driver.py:36 shells out to Rscript; nampy/gam/
    parity/mgcv_negbin_reml_opt.R:29 runs real mgcv::gam(...) and hands back sp
    and theta. Upstream logic you should be mirroring lives in mgcv/R/
    gam.fit4.r:255 and mgcv/R/efam.r:5 (estimate.theta). This not parity
    implementation. This is delegation.
  - fs, sz, t2 REML explicitly avoid exact Gaussian backend. nampy/gam/
    smoothing_selection/criteria/ml_reml.py:25 forces factor_smooth_fs,
    factor_smooth_sz, tensor_anova onto gaussian_dynamic, and comment says
    these terms are “not yet an exact mixed-model port”. Upstream constructors
    are mgcv/R/smooth.r:1996 and mgcv/R/smooth.r:2187. This matches tracked
    gaps in tests/test_mgcv_additional_scenarios.py:265 and failing
    factor_smooth_sz in tests/test_mgcv_parity_failing_and_warnings.py:98.
  - Gaussian REML outer optimization still diverges from mgcv control flow.
    Active path does joint L-BFGS-B warm start/polish nampy/gam/
    smoothing_selection/optimize/driver.py:675, random-effect boundary snap
    nampy/gam/smoothing_selection/optimize/driver.py:763, and 1D nested scalar
    refinement nampy/gam/smoothing_selection/optimize/driver.py:796. Even exact
    Gaussian only reaches mgcv::newton() after warm start nampy/gam/
    smoothing_selection/optimize/driver.py:878. mgcv path not do these extra
    searches. Good candidate for REML endpoint drift.
  - Tensor marginals not built from same raw state as mgcv for many bases.
    Upstream smooth.construct.tensor.smooth.spec uses smoothCon(...
    absorb.cons=TRUE) for constrained margins, else raw smooth.construct(...),
    then NP/re-scaling later mgcv/R/smooth.r:764 and mgcv/R/smooth.r:824. Local
    nampy/gam/smooths/tensor/marginals.py:132 special-cases only cr/cs;
    everything else falls through to already-processed basis_train and
    penalties[0] nampy/gam/smooths/tensor/marginals.py:153. That bakes in
    scaling/constraint state too early. Test file already says ps/ts/cc scaling
    differs from mgcv tests/test_mgcv_pc_id_parity.py:15.
  - t2 natural-parameter reparameterization still not exact. Upstream t2 margin
    setup uses nat.param(..., type=3, unit.fnorm=TRUE) mgcv/R/smooth.r:1064.
    Local replacement is custom _eigen_split(..., mode="t2") /
    t2_marginal_reparameterization() nampy/gam/basis/tensor.py:103 and nampy/
    gam/basis/tensor.py:180, then custom t2.model.matrix rebuild nampy/gam/
    basis/tensor.py:220. Repo already documents EDF drift from this port in
    tests/test_mgcv_k_check_parity.py:386. Likely root of
    gaussian_t2_full_false structural miss, not mere optimizer noise.
  - One false lead: binomial_separation not clean NAMpy bug signal. Failure log
    shows upstream mgcv subprocess itself dies there FAILING_TESTS.md:184 and
    FAILING_TESTS.md:731.

  Lower confidence:

  - gaussian_ti_mc looks like smaller tensor-construction mismatch, but I did
    not isolate one branch with same confidence as items above. Best suspects
    still tensor interaction setup around marginal raw/centered/NP handling,
    not broad solver math.