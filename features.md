 High-Value Deferrals

  1. General-family / GAMLSS families
      - gaulss, gammals, gevlss
      - These pull in multi-predictor formulas, gam.fit5, special residuals, special prediction layout, post-processing, covariance quirks, and
        optimizer behavior.
      - I would keep them out of v1 entirely, or keep only one smoke-tested experimental family behind an explicit “unsupported/experimental”
        boundary.
  2. Multi-predictor formula lists
      - Formulas like ["y ~ s(x)", "~ 1"]
      - Mostly needed for general families. If general families are deferred, this can be deferred too.
  3. Advanced smooth families
      - Tensor products: te, ti, t2
      - Factor smooths: fs, sz
      - MRF smooths
      - Gaussian process smooths
      - Random effects with tricky id= behavior
      - For v1, I’d keep only core univariate smooths: cr, maybe tp, maybe ps.
  4. Linked id= smooths
      - Shared smoothing parameters / shared bases across terms are very mgcv-specific and easy to get subtly wrong.
      - Defer unless there is a concrete user need.
  5. Multiple smoothing optimizers
      - outer_newton, bfgs, efs, optim, edge correction, exact trace parity.
      - For v1, pick one robust optimizer path, probably REML with a simple/scipy optimizer, and make everything else unsupported.
  6. Exact optimizer trace parity
      - Tests around score histories, outer traces, Hessian rows, initial.spg, lifecycle metadata.
      - Useful for a parity project, but too much for a product-facing v1.
  7. Full mgcv post-processing parity
      - Vp, Ve, Vc, edf2, unconditional covariance, sandwich covariance, sp.vcov, one-standard-error rules.
      - V1 probably only needs coefficients, fitted values, predictions, basic covariance/SE if reliable.
  8. Full diagnostic parity
      - anova.gam, summary.gam, k.check, concurvity, gam.vcomp, exact residual flavors.
      - Keep a small summary and maybe basic residuals. Defer the rest.
  9. Representation-level parity tests
  - Raw basis constructor parity, eigenspace orientation, exact side-condition deletion, QR/LAPACK quirks.
  - These are excellent if the goal is “Python mgcv clone”; they are expensive if the goal is “usable GAM v1”.

  What I Would Keep For V1

  A sane first version would be:

  - Families: gaussian, binomial, poisson; maybe gamma later.
  - Smooths: s(x, bs="cr"), maybe tp or ps.
  - Formula: single-response, single-linear-predictor formulas only.
  - Fitting: fixed smoothing + one REML path.
  - Prediction: link, response, maybe terms.
  - Basic SEs only if stable.
  - Clear NotImplementedError for everything else.

  The current codebase is closer to “attempted mgcv clone with strict parity” than “small stable GAM library.” The largest simplification would be
  drawing a hard boundary around single-predictor ordinary GAMs first, and moving everything general-family, tensor-heavy, optimizer-trace-heavy, and
  diagnostic-parity-heavy out of the v1 support contract.
