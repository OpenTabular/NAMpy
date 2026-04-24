Collected findings from completed reviewers only. No tests run. No more agents.

  Big Themes

  - ML path often uses REML determinant math. mgcv uses MLpenalty1() for ML null-space handling.
  - Several rank/tolerance choices differ from mgcv.
  - Prediction missing unconditional, terms, exclude, NA/new factor level, and some iterms parity.
  - Tensor defaults use k=10; mgcv te/t2 defaults to k=5 per 1D margin.
  - Categorical smooths lose factor level semantics: unused levels, level order, NA/new levels.
  - Result objects cannot fully carry mgcv fields: Vc, coefficient EDF vector, edf1, scale flags.

  Inference

  - chi_square_mixtures.py
      - psum_chisq truncates df; mgcv::psum.chisq rounds.
      - Davies fallback passes lower_tail into liu2; mgcv fallback always uses upper-tail default.
      - ln1 uses log1p(x)-x; mgcv uses Rmath log1pmx, more stable.
      - Davies trace counter missing increments.
      - Davies ifault==2 / fallback warnings absent.
  - anova.py
      - Supplied dispersion not used to rescale Vp/Ve; mgcv::summary.gam does.
      - Fixed smoothing parameter treated like fx=TRUE; mgcv separates fixed sp from fixed smooth.
      - Random-effect tests ignore reTest()/recov() multi-random-effect partitioning.
      - Parametric Wald uses NumPy default pinv rank; mgcv uses .Machine$double.eps^.5.
  - inference/__init__.py
      - Does not export psum_chisq, while mgcv exports psum.chisq.

  Smoothing Criteria

  - gaussian_reml_algebra.py
      - ML determinant documented/used like REML; mgcv ML uses MLpenalty1().
      - Profiled variance ignores gamma, positive-weight count, ML vs REML denominator.
      - Pearson scale adds pearson_extra; mgcv ordinary Pearson/Fletcher uses dev.extra.
      - NaN Fletcher correction can become finite via Python max.
  - gaussian_dyn.py
      - ML derivatives use REML determinant derivatives.
      - Joint/profiled dynamic path rejects ML; mgcv supports unknown-scale ML.
      - Log-scale derivatives hard-code REML -Mp.
      - Rejects finite joint eval when n/gamma - Mp <= 0; mgcv evaluates fixed-scale score.
  - gaussian.py / pirls.py
      - GCV returns inf when denominator nonpositive; mgcv squares denominator and can return finite score.
      - Tensor ML uses REML-style full coefficient determinant instead of MLpenalty1().
      - Saturated likelihood uses unit weights; mgcv uses prior weights and n.true/nobs.
  - pirls_reml_derivative_blocks.py
      - Fisher/canonical branch ignores working weights including prior weights.
      - Clips signed mu_eta / weights; mgcv preserves signs and negative Newton weights.
      - Pearson derivatives hard-code prior weights to one.
  - pirls_deriv.py
      - Rank tolerance hard-coded 1e-10; mgcv uses eps*100 or eps^.75.
      - ML determinant uses leading range block, ignoring pivoted null/range columns.
      - Zero-weight derivative path clips to tiny value; mgcv uses separate zero-weight path.
      - Gamma joint scale profiles where mgcv gam.fit4 uses supplied scale parameter.
      - NegBin nonfinite lsth2 skipped; mgcv subtracts unconditionally.
  - ncv.py
      - _infer_index_base() misdetects 1-based nei if final row absent.
      - Requires all d/md/a/ma; mgcv allows partial/default nei.
      - General-family NCV nonfinite fold derivative replaced by zeros; mgcv propagates derivative surface.
  - laplace.py
      - Penalty derivative blocks placed with one global intercept offset; mgcv uses per-penalty offsets.
      - Can mix incompatible UrS/Sl reparameterization states.

  Reparameterization / Postfit

  - reparam.py
      - Groups UrS by smoothing index and passes log(sp) directly; mgcv keeps one root per penalty and maps via L %*% lsp + lsp0.
      - gam_reparam returns early when M==0, missing fixed penalty roots.
      - Range/null eigen cutoff floors max eigenvalue at 1; mgcv does not.
      - Legacy build_penalty_reparameterized_system not faithful to Sl.setup.
  - postfit.py
      - gam.vcomp uses recomputed smoothing-only Hessian, misses scale row / family-param handling.
      - Edge correction adds scalar reg to whole matrix; mgcv adds diag(p)*reg.
      - Recomputes Hessian when mgcv would use stored outer.info$hess or return unavailable.
      - Eigen cutoff uses eps^.75; vendored mgcv code uses .Machine$double.eps^75.
      - Default rescale=False; mgcv gam.vcomp default TRUE.

  Optimize

  - driver.py
      - Default optimizer lbfgsb; mgcv default outer/newton.
      - All-fixed-SP branch returns before optimizing unknown scale/family params.
      - optim stores scaled objective, while mgcv stores recomputed unscaled objective.
      - NegBin joint vector order [log_sp..., log_theta]; mgcv uses family params first.
      - efs with non-REML raises; mgcv forces method to REML.
  - objectives.py
      - Gaussian ML joint objective reaches criterion that rejects ML, despite mgcv support.
      - NegBin joint objective order differs from mgcv.
      - NegBin accepted trace can duplicate rows.
      - LAML normalization inconsistent.
  - basics.py
      - Extended families use ordinary GLM starting weight path; mgcv initial.spg has extended-family path.
      - gaulss initializer ignores non-identity mean link and offsets.
      - Design-balance initial SP fallback is heuristic, not mgcv.
      - Hard clipping bounds differs from mgcv transform/cap logic.

  Results / Engine

  - solution.py
      - edf stored scalar; mgcv object has per-coefficient edf vector.
      - edf2 stored but no edf1.
      - scale collapses scale, sig2, scale.estimated, reml.scale.
  - fit_result.py
      - No cov_unconditional / Vc.
      - Missing coefficient-level edf, edf1, edf2, hat.
      - Scalar family/link/intercept does not fit multi-LP general families.
      - to_dict() drops Vp/Ve by default.
  - engine/__init__.py
      - Does not export solve_general_family_fit.

  Prediction

  - predictions.py
      - No unconditional; cannot select Vc.
      - Default type="response"; mgcv predict.gam default is "link".
      - iterms mean uncertainty uses heuristic, not nCons.
      - Missing terms / exclude.
      - Smooth-specific prediction offsets not handled.
  - linear_predictor_matrix.py
      - Rejects numeric NA/NaN newdata; mgcv uses na.pass + napredict.
      - Intercept inclusion ignores family$drop.intercept.
      - General-family branch bypasses common fitted/design/newdata validation.
  - general.py
      - Single-predictor general families do not handle type="iterms".
      - Term SE mapping can shift for later linear predictors.
      - Response shape normalization differs for families like ziplss.

  Smooth Core / Registry

  - smooth_base.py
      - by prediction can use positional column, not name.
      - Numeric by constancy test differs from mgcv.
      - Penalty rank/null metadata accepted but not forwarded to PenaltySpec.
      - pc dict fallback can silently constrain wrong feature.
      - Selection penalty sums penalties once; mgcv appears to double first penalty in null-space selection block.
  - smooths/__init__.py / univariate/__init__.py
      - cs, cc, ts aliases instantiate default cr/tp unless caller passes basis.
      - Registry omits supported categorical fs, sz, re, mrf.
  - registry.py
      - fs/sz/re/mrf not registered.
      - Case-insensitive names accepted; mgcv constructor dispatch is case-sensitive.
      - Unsupported upstream smooths cp/bs/ad/sos/ds absent; explicit rejection okay but not full registry parity.

  Univariate Smooths

  - tp.py
      - Applies pc with linked id, broadening unsupported mgcv surface.
      - select=True with fixed=True errors; mgcv treats as no-op/omitted penalties.
      - ts rank metadata can remain pre-constraint.
  - cr.py
      - cr/cs point-constraint branch forgets cached by on training basis.
      - Too-small/negative k raises; mgcv defaults/bumps with warning.
  - ps.py
      - Negative/default k handling differs from mgcv max(10,m[1]+1).
      - cp cyclic P-spline surface absent.
      - Derivative / monotone ps branches absent.
  - gp.py
      - Penalty rank recorded but not passed into penalty block.
      - Numeric by centering differs via constancy heuristic.
      - Prediction builds full kernel; mgcv chunks large prediction.

  Tensor Smooths

  - marginals.py
      - Negative scalar m repeated without clamping to 0.
      - Repeated variables not rejected in runtime path.
      - fx=np.nan coerces to True; mgcv treats NA as missing/default false.
  - algebra.py
      - t2_marginal_reparameterization does not accept/pass known rank.
      - Hard-coded ps null basis/sign flips not upstream.
      - Range/null ordering can be ascending vs mgcv eigen() decreasing.
  - te.py
      - Default k=10, not mgcv k=5 per 1D margin.
      - Constraint skipped for all nonconstant by; factor-by should keep constraints.
      - No np option; NP applied unconditionally.
      - No pc support.
      - Rejects one-margin tensor terms; mgcv te/ti can use constructor path.
  - ti.py
      - No pc.
      - Scalar mc repeated across margins; mgcv expects per-margin vector.
      - No np option.
  - t2_basis.py
      - After all-null constraint, final penalized block slice replaced by constrained all-null slice.
      - All-null 1D constraint uses mean row; mgcv special-cases C <- ncol(X).
  - t2.py
      - transform_new() applies cached training by, not newdata by.
      - Marginal rank inferred, not passed from marginal constructor.
      - Default k=10, not mgcv 5.

  Categorical Smooths

  - categorical_utils.py
      - Levels derived from observed values, not factor categories; loses unused/order levels.
      - Missing values can become levels.
      - Bool/numeric-with-NaN can be treated factor-like; mgcv requires factor.
  - mrf.py
      - Unseen prediction levels become zero rows; mgcv warns/propagates NA/new-level behavior.
      - Supplied xt["penalty"] still forces neighbor/polygon validation; mgcv ignores nb then.
      - Supplied penalty symmetrized; mgcv does not average with transpose.
      - Numeric nb accepts 0-based; mgcv uses 1-based.

  No Issue Found

  - artifacts.py
  - results/__init__.py
  - predict/__init__.py
  - tensor/__init__.py
  - categorical/__init__.py

  Unfinished Because Interrupted
  Some running reviews had not reported before stop/no-more-agents: likely fs.py, re.py, engine/state.py, newton_mgcv.py, newton.py. I did not
  include unreported findings.


› q


⚠ Falling back from WebSockets to HTTPS transport. timeout waiting for child process to exit

• I read q as no new actionable request. No commands run.

• Added late findings to record:

  re.py

  - High: numeric-looking factor labels become numeric slopes, unlike mgcv model.matrix(~ term - 1).
  - High: observed-only levels lose declared factor order/unused levels.
  - High: select=True adds selection penalty per supplied xt["S"]; mgcv builds one joint null-space selection penalty after all penalties.
  - Low: unseen prediction levels zeroed without mgcv warning.

  newton_mgcv.py

  - High: steepest-descent fallback sign is opposite mgcv.
  - Medium: edge-correction target fixed once; mgcv resets target after each flat-parameter move.
  - Medium: indefinite-Hessian branch keeps previous accepted state on failed trial; mgcv updates to selected trial state before returning.

  No commands or agents run.

• Added fs.py late findings:

  - High: all fs penalties share same smoothing_id; mgcv creates distinct smoothing parameters per penalty unless linked by id.
  - High: missing/unseen factor levels map to zero rows; mgcv errors or propagates NA through model-frame/xlevel handling.
  - Medium: levels derived from observed values; mgcv uses declared factor levels, preserving order and unused levels.
  - Medium: rejects xt="cs" / xt="ts" for bs="fs" even though mgcv accepts these single-penalty shrinkage bases.
  - Medium: prediction numeric by missing values zeroed; mgcv propagates NA.

  No commands or agents run.