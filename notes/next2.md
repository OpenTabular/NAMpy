High — blocks common models

  1. ML/REML reparam rejects overlapping multi-penalty terms — reparam.py:368,
  gam_solve.py:435. mgcv ref: gam.fit3.r gam.reparam().
  2. t2 by= handling missing — t2.py:77. mgcv ref:
  smooth.construct.t2.smooth.spec in smooth.r.

  Medium — partial smooth constructors

  4. fs/sz — factor_smooth.py:80,112,204 restricts xt and multivariate bases.
  mgcv ref: smooth.construct.fs.smooth.spec.
  5. cc smooth — no shared_basis_setup, no factor-by cyclic cubic —
  cubic_regression.py:298.
  6. re smooth id= blocked — random_effect.py:131.

  Lower — surface/API completeness

  7. NB Gap 2 (theta-space vs log-theta-space Newton in estimate_theta_mle) —
  cosmetic, converges to same point.
  8. sp_vcov ignores edge_correct — postfit.py:49.
  9. gam_vcomp(rescale=True) not implemented — postfit.py:79.
  10. Formula layer — parser.py:92 rejects - terms, . formulas, richer
  c()/list() shapes.