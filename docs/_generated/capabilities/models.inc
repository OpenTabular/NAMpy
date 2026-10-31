.. list-table::
   :header-rows: 1

   * - Model
     - Observation type
     - Fitting route
     - Smoothing selection
   * - GAMRegressor
     - Exact scalar response
     - Statistical penalized fitting
     - Configured statistical criterion
   * - GAMClassifier
     - Binary or grouped binomial
     - Statistical penalized fitting
     - Configured statistical criterion
   * - GAMLSS
     - Family-dependent
     - Joint statistical likelihood
     - Family-declared criteria/optimizers
   * - ExpectileGAM
     - Exact scalar response; optional observation weights
     - Asymmetric least squares (LAWS)
     - Supplied penalties; no automatic selection
   * - QuantileGAM
     - Exact scalar response
     - Quantile loss
     - Supplied penalties
   * - CalibratedQuantileGAM
     - Exact scalar response
     - ELF and scalar calFast calibration
     - Supplied penalties or REML at each scale
   * - SurvivalGAM
     - Right-censored; delayed entry
     - Piecewise-exponential full likelihood
     - Poisson statistical engine
   * - SurvivalGAM
     - Right-censored; delayed entry
     - Parametric full likelihood (Weibull shown)
     - Fixed or eligible joint REML
   * - CoxGAM
     - Static right-censored; optional strata
     - Breslow partial likelihood
     - Fixed or eligible REML; stratified automatic requires BFGS
   * - NAMSurvival
     - Right-censored; full-likelihood route
     - Neural survival objective
     - Gradient training
   * - NAMSurvival
     - Static right-censored, one stratum
     - Neural Breslow partial likelihood
     - Gradient training on full cohort
   * - SemiStructuredRegressor
     - Exact scalar response
     - Identified spline and neural composition
     - Configured penalties and gradient training
