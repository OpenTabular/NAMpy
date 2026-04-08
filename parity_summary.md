• Verified Match
  The current mgcv parity harness is real: it shells out to bundled mgcv through tests/
  mgcv_parity_utils.py, tests/parity/mgcv_snapshot.R, and tests/parity/mgcv_anova.R. So the covered
  comparisons are against actual mgcv, not hand-built snapshots.

  What is already matched and explicitly covered:

  - Core end-to-end snapshot parity for Gaussian, Binomial, Poisson, Gamma, and NegBin cases in
    tests/test_mgcv_snapshot_parity.py and tests/_mgcv_snapshot_parity_shared.py.
  - Smooth construction / penalty parity for te, ti, t2 natparam, fs, fs(..., xt=list(bs="ps")), sz,
    mrf, re, cc, ps, and gp in tests/test_mgcv_smoothcon_parity.py.
  - pc= and linked id= parity for the currently supported cases in tests/test_mgcv_pc_id_parity.py.
  - Output parity for predict(type="response"|"link"|"terms"|"lpmatrix"), standard errors, offsets,
    and model-comparison anova() in tests/test_mgcv_output_parity.py.
  - Outer-optimizer trace, final gradient, and Hessian parity in tests/test_mgcv_trace_parity.py.
  - Additional covered scenarios such as select=True, weighted fits, non-default links, and theta !=
    1 cases in tests/test_mgcv_additional_scenarios.py.

  The important caveat is that “matched” does not always mean machine-precision. Some scenarios are
  only asserted with looser prediction tolerances, especially where the repo already documents
  penalty-scale differences from mgcv, e.g. cs, some pc= paths for ps/tp/ts/cc, fs with ps
  marginals, and some non-Gaussian REML/link cases. The comments in tests/test_mgcv_pc_id_parity.py
  and tests/_mgcv_snapshot_parity_shared.py are the current source of truth there.

  Remaining Work
  What still remains before this is an exact mgcv surface replica:

  - Factor-by replicated cc and ps smooths are still not implemented in nampy/gam/smooths/
    univariate/cubic_regression.py and nampy/gam/smooths/univariate/pspline.py.
  - pc= is still blocked for multivariate tp/ts/gp and some factor-by replicated cases in nampy/gam/
    smooths/univariate/thin_plate.py, nampy/gam/smooths/univariate/gp.py, and nampy/gam/smooths/
    univariate/cubic_regression.py.
  - Only one active linear predictor / offset path is supported in nampy/gam/formula/extract.py, so
    this is not yet full mgcv formula surface.
  - Extended/general families are still unimplemented in nampy/gam/fit/solvers/penalized_irls.py.
  - score_gamma != 1 is still not implemented across the outer objective / derivative code in nampy/
    gam/smoothing_selection/criteria/gaussian.py, nampy/gam/smoothing_selection/criteria/
    gaussian_dyn.py, and related PIRLS criteria modules.

  So the current state is: the covered parity matrix is broad and real, but the package is not yet
  an exact mgcv reimplementation across the full feature surface. The main remaining work is to
  eliminate the explicit NotImplementedError paths and tighten the remaining “close-to-mgcv”
  tolerances into true exact parity.