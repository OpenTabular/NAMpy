# Changelog

## 0.3.0 (unreleased)

- Use one complete installation, `pip install nampy`, for statistical and
  neural models. Remove the `gam`, `neural`, `all`, and `igann-sparse` extras.
- Add a first-class `GAMLSS` estimator with named multi-parameter formulas,
  array-mode smooth construction, natural-parameter predictions, likelihood
  scoring, standard errors, persistence, and additive components.
- Restrict `GAMRegressor` to single-predictor regression families and
  `GAMClassifier` to binary binomial and quasibinomial families. Raw `nampy.gam.GAM` remains
  unrestricted.
- Add quasipoisson, quasibinomial and general quasi families with nine named
  links, five general variance functions and numeric power links; support
  grouped fractional responses, dispersion estimation, extended quasi-likelihood
  selection, prediction, uncertainty and persistence. Public log likelihood,
  AIC and BIC remain unavailable (`NaN`) for quasi models.
- Match R's log, cloglog, logit and cauchit link behavior at extreme predictors,
  including machine-epsilon floors and source-defined tail thresholds; verify
  these shared GLM primitives with static boundary fixtures.
- Extend matrix-valued linear functionals across numeric and categorical
  smooths, supported adaptive terms and soap films, with upstream repeated-point
  packing, penalty scaling and row-sum constraints. Add single-penalty soap
  margins for `te`, `ti` and `t2`, plus supported ordinary-input soap `fs`/`sz`
  nesting, with fresh fit and uncertainty references.
- Add numeric/factor random-effect and full/reduced-rank MRF functionals,
  single-penalty random-effect tensor margins, and `fs` matrix terms with
  explicit `cr`/`ps` bases; support `sz` matrix functionals with `tp`/`cr`/`ps`
  and scalar factor columns. Preserve explicit
  restrictions for packed repeated-location `cs`, matrix `fs` with `tp`,
  weighted 2D adaptive terms and known-boundary soap functionals.
- Standardize `predict_point()` across GAM-backed and neural LSS estimators.
- Add committed parametric semantic parity references from mgcv 1.9-4,
  gamlss 5.5-0, and gamlss.dist 6.1-1.
