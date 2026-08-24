# Changelog

## 0.3.0 (unreleased)

- Add a first-class `GAMLSS` estimator with named multi-parameter formulas,
  array-mode smooth construction, natural-parameter predictions, likelihood
  scoring, standard errors, persistence, and additive components.
- Restrict `GAMRegressor` to single-predictor regression families and
  `GAMClassifier` to binary binomial families. Raw `nampy.gam.GAM` remains
  unrestricted.
- Standardize `predict_point()` across GAM-backed and neural LSS estimators.
- Add committed parametric semantic parity references from mgcv 1.9-4,
  gamlss 5.5-0, and gamlss.dist 6.1-1.
