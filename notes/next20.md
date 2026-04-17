mgcv has 4 coarse stages in upstream gam() itself, but for exact parity you
  should split it more finely.

  See upstream summary in /home/ad32/projects/package/mgcv/R/mgcv.r:2240: formula
  split, gam.setup, estimation, final object assembly. NAMpy’s repo-level 7-stage
  view is in CLAUDE.md:45.

  Recommended Parity Stages

  1. Formula interpretation.
     Compare interpret.gam output, fake formula/model frame, parametric terms,
     response, weights, offsets, factor levels.
  2. Per-smooth constructor.
     Compare raw smooth.construct.* outputs before wrapper logic: basis X, raw
     penalties S, rank, null.space.dim, knots and constructor-specific fields.
  3. smoothCon wrapper.
     This is big. Compare both absorb.cons=False/True and scale.penalty=False/
     True, plus by, select, pc, linked id, C/Cp, qrc, S.scale, prediction-vs-fit
     parameterizations.
  4. gam.setup assembly.
     Compare full G object: G$X, G$S, G$off, G$rank, G$L, G$lsp0, G$sp, G$smooth,
     G$P, G$cmX, assign, xlevels, offset.
  5. Side conditions / identifiability.
     Compare gam.side effects exactly: deleted columns, constraint ordering,
     nested smooth handling, centering, penalty/block order.
  6. Preoptimization / reparameterization.
     For ordinary families: compare gam.reparam, Eb, UrS, U1, E.
     For general families / GAMLSS: compare Sl.setup objects, St, rS, E, S,
     lambda.
  7. Fixed-smoothing inner fit.
     At fixed sp, compare coefficient solve and PIRLS state from gam.fit3 /
     gam.fit5 / magic: coefficients, linear predictor, deviance/REML, weights,
     working responses, derivative blocks.
  8. Outer smoothing optimization.
     Compare gam.outer / newton / bfgs / optim / efs traces: lsp path, score
     history, gradients, Hessians, step halving, convergence flags, edge
     correction.
  9. Post-processing / final fit object.
     Compare gam.fit3.post.proc / gam.fit5.post.proc / magic.post.proc: Vp, Ve,
     Vc, EDF, hat/trA, scale, AIC, outer.info, warnings.
  10. Prediction / inference / diagnostics.
     Compare predict.gam for link, response, terms, iterms, lpmatrix, newdata,
     se.fit, unconditional, plus anova.gam, k.check, residuals.

  If you want exact parity, these are the key object boundaries

  - Raw smooth object
  - smoothCon object
  - G from gam.setup
  - Reparam object: Eb/UrS/U1 or Sl
  - Fixed-sp fit object
  - Outer optimizer trace
  - Final fitted object
  - Prediction/diagnostic outputs

  Current Test Map

  - Smooth constructor / smoothCon: tests/smooths/test_mgcv_smoothcon_parity.py:1
  - Preoptimization ordinary families: tests/optimization/
    test_mgcv_preoptimization_blocks_parity.py:1
  - Preoptimization general families: tests/optimization/
    test_mgcv_general_family_preoptimization_parity.py:1
  - Outer optimizer trace: tests/optimization/test_mgcv_trace_parity.py:1
  - Final snapshot parity: tests/parity/test_mgcv_snapshot_parity.py:1
  - Output/predict/anova parity: tests/parity/test_mgcv_output_parity.py:1
  - pc= / linked id=: tests/smooths/test_mgcv_pc_id_parity.py:1
  - General family / GAMLSS outputs: tests/families/
    test_general_family_mgcv_parity.py:1
  - Diagnostics: tests/diagnostics/test_mgcv_k_check_parity.py:1
