1. gaussian_ti_mc
     nampy/gam/smooths/tensor/ti.py:145 not true port of mgcv tensor-
     interaction path.
     Upstream mgcv/R/smooth.r:760 does smoothCon(..., absorb.cons=TRUE) on
     mc=TRUE margins, then optional XP SVD reparameterization using PredictMat.
     NAMpy rebuilds margins through nampy/gam/smooths/tensor/marginals.py:118,
     normalizes penalties itself, then row-Kroneckers. Same idea, not same
     control flow. Result: small but systematic coef drift.
  2. gaussian_t2_full_false
     Biggest bug. nampy/gam/basis/tensor.py:223 custom
     build_t2_basis_and_penalties() not exact mgcv::t2.model.matrix.
     Upstream mgcv/R/smooth.r:911 builds block order very specifically for
     full=FALSE: penalized sub-blocks first, single final null block last.
     NAMpy own grouping/ordering metadata differs. Failure shape says same
     coefficients mostly present but moved/mixed, plus null-block/intercept
     handling shifted. This one parity bug, not optimizer noise.
  3. binomial_separation
     Most likely covariance bug, not mean-fit bug.
     Final PIRLS covariance in nampy/gam/fit/solvers/irls_core.py:712 comes
     from stabilized_cholesky_solve(), which adds diagonal jitter from nampy/
     gam/fit/penalized_system.py:18 when X'WX+S near singular.
     Near separation makes weights collapse, so this path easy trigger. mgcv
     handles ill-conditioned PIRLS systems through its own QR/Cholesky path,
     not this jitter ladder. SE mismatch fits that exactly.
  4. mrf_lattice
     Coefficients already match in failure log. EDF also matches. Only REML
     score off.
     That points at Gaussian REML score path, not MRF constructor.
     Suspect path: nampy/gam/fit/postprocess/
     gaussian_smoothness_postprocess.py:62 plus penalty-logdet/reparam in
     nampy/gam/smoothing_selection/reparam.py:653.
     For low-rank mrf, NAMpy mixed-model reparam / log|S| bookkeeping not exact
     mgcv, even though fitted coefficients are.
  5. factor_smooth_sz
     nampy/gam/smooths/categorical/factor_smooth.py:730 not same as
     mgcv::smooth.construct.sz.smooth.spec.
     Upstream mgcv/R/smooth.r:2187 builds raw tensor-product factor blocks,
     leaves object$C <- c(0,nf), and lets later smoothCon/PredictMat apply
     XZKr.
     NAMpy pre-applies explicit contrast transform _xzkr_contrast_transform()
     and transformed penalties in constructor. Same target space, different
     basis/penalty parameterization. Sign-flipped / reordered coefficients in
     failure log come from this.
  6. gaussian_fs_select_reml_matches_mgcv and standard_errors_fs[fs]
     Same root family: fs port partial.
     Constructor in nampy/gam/smooths/categorical/factor_smooth.py:472 mirrors
     broad structure, but outer optimization then rides NAMpy heuristics for
     flat ridges in nampy/gam/smoothing_selection/optimize/heuristics/
     stabilize.py:459, not upstream gam.fit3/magic.
     Warning "step failed" and criterion gap both fit flat-ridge/shared-penalty
     mismatch. SE failure downstream likely from same off-target fit/
     covariance, not separate predict bug.
  7. strict_factor_by_link_parity
     This one looks like machine-precision mismatch in factor-by rewrite path.
     NAMpy rewrites factor by smooths in nampy/gam/formula/preprocess.py:176
     into hidden numeric indicators plus constraint_mode="always".
     Upstream does factor-by through smoothCon / PredictMat wrapper behavior in
     mgcv/R/smooth.r:4300 and mgcv/R/mgcv.r:2692.
     Broad parity can pass, but this rewrite is not machine-identical, so
     strict 5e-10 link check can fail.