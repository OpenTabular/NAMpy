- High. nampy/gam/fit/solvers/irls_core.py:460: ordinary PIRLS defaults to stabilized_cholesky_solve, not upstream C_pls_fit1 in mgcv/R/gam.fit3.r
    and mgcv/R/gam.fit4.r. Changes coef/eta update path, signed-weight handling, step-halving.
  - High. nampy/gam/fit/solvers/irls_core.py:783: post-convergence Vp/Vf/H_coef rebuilt from dense XtWX + S; upstream uses gdi1() +
    gam.fit3.post.proc(). EDF/covariance parity breaks on aliased or near-singular fits.
  - High. nampy/gam/fit/state.py:388: assign_fit_solution() does second non-upstream Fisher rebuild for ordinary families. Upstream estimate.gam()
    does not. Mutates parity-critical covariance/EDF after solver returns.
  - High. nampy/gam/smoothing_selection/optimize/driver.py:541: exact-Gaussian unknown-scale path only joined for REML, and excluded for BFGS.
    Upstream appends log(scale) for Gaussian ML and REML in both newton() and bfgs().
  - High. nampy/gam/constraints/identifiability.py:264: side.constrain=FALSE terms still enter dependence graph. Upstream mgcv::gam.side excludes
    them. Can wrongly drop later smooth columns.
  - High. nampy/gam/smooths/categorical/re.py:193: numeric by scaling applied before penalty rescale for bs="re". Upstream smoothCon rescales raw X
    before by. Wrong S.scale / variance-component parity.
  - High. nampy/gam/compiler/compile_predictors.py:141: duplicate penalty ids split into distinct smoothing ids. Breaks upstream fs shared-smoothing-
    parameter structure.
  - High. nampy/gam/compiler/compile_model.py:83: non-upstream np.allclose(...) short-circuit skips building nontrivial G$P; rank-deficient branch
    also uses non-pivoted QR. Upstream gam.setup always builds QR-based P.
  - High. nampy/gam/families/gamlss/ziplss.py:703: response-scale se.fit formula differs from mgcv/R/gamlss.r::ziplss()$predict. predict(...,
    type="response", se.fit=TRUE) will not match.
  - Medium-high. nampy/gam/predict/general.py:153: multi-predictor general-family predict hard-fails on type="iterms"; upstream warns and downgrades
    to terms.
  - Medium-high. nampy/gam/predict/general.py:166: type="terms" assembled blockwise, not by upstream assign/formula-term grouping with Xoff. Term-
    level predictions/SEs can differ.
  - Medium. nampy/gam/inference/anova.py:651: accepts test="Cp" and treats all general families like extended.family. Upstream anova.gam only allows
    Chisq/LRT/F and only rewrites extended-family path.
  - Medium. nampy/gam/fit/solvers/gaussian_exact.py:162: n < p falls back to lstsq; upstream mgcv/src/gdi.c::pls_fit1 handles underdetermined case in
    same QR/backsolve path.
  - Medium. nampy/gam/linalg/cholesky.py:54: unconditional eigenvalue-floor / identity fallback after Cholesky failure. Upstream mgcv/R/gam.fit4.r
    keeps ridge-inflation loop unless eigen.fix=TRUE; no identity fallback.