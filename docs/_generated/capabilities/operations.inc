.. list-table::
   :header-rows: 1

   * - Model / route
     - Prediction and probability operations
     - Uncertainty methods
     - Conditions
   * - GAMRegressor / Statistical penalized fitting
     - predict_components (conditional); predict_link (conditional)
     - fisher.exact (conditional); fisher.negative_hessian (conditional); fisher.score_outer_product (conditional); uncertainty.coefficients (conditional)
     - available_covariance, fitted_model
   * - GAMClassifier / Statistical penalized fitting
     - predict_components (conditional); predict_link (conditional)
     - fisher.exact (conditional); fisher.negative_hessian (conditional); fisher.score_outer_product (conditional); uncertainty.coefficients (conditional)
     - available_covariance, fitted_model
   * - GAMLSS / Joint statistical likelihood
     - distribution.cdf (conditional); distribution.cdf_left (conditional); distribution.log_cdf (conditional); distribution.log_prob (conditional); distribution.log_survival (conditional); distribution.mean (conditional); distribution.quantile (conditional); distribution.sample (conditional); distribution.survival (conditional); distribution.variance (conditional); predict_components (conditional); predict_distribution (conditional); predict_link (conditional); predict_parameters (conditional); predict_quantiles (conditional)
     - fisher.exact (conditional); fisher.negative_hessian (conditional); fisher.score_outer_product (conditional); uncertainty.coefficients (conditional)
     - available_covariance, fitted_model
   * - ExpectileGAM / Asymmetric least squares (LAWS)
     - predict_components (conditional); predict_link (conditional)
     - None declared
     - fitted_model
   * - QuantileGAM / Quantile loss
     - predict_components (conditional); predict_link (conditional)
     - None declared
     - fitted_model
   * - CalibratedQuantileGAM / ELF and scalar calFast calibration
     - predict_components (conditional); predict_link (conditional)
     - uncertainty.coefficients (conditional); uncertainty.quantile_function (conditional)
     - calibrated_covariance, fitted_model
   * - SurvivalGAM / Piecewise-exponential full likelihood
     - predict_components (conditional); predict_distribution (conditional); predict_link (conditional); survival.bootstrap (conditional); survival.curves (conditional); survival.hazard (conditional); survival.intervals (conditional)
     - uncertainty.coefficients (conditional)
     - available_covariance, fitted_model, valid_subject_resampling
   * - SurvivalGAM / Parametric full likelihood (Weibull shown)
     - predict_components (conditional); predict_distribution (conditional); predict_link (conditional); survival.bootstrap (conditional); survival.curves (conditional); survival.hazard (conditional); survival.intervals (conditional)
     - uncertainty.coefficients (conditional)
     - available_covariance, fitted_model, valid_subject_resampling
   * - CoxGAM / Breslow partial likelihood
     - predict_components (conditional); predict_distribution (conditional); predict_link (conditional); survival.bootstrap (conditional); survival.curves (conditional); survival.intervals (conditional)
     - uncertainty.coefficients (conditional)
     - available_covariance, fitted_model, valid_subject_resampling
   * - NAMSurvival / Neural survival objective
     - predict_components (conditional); predict_distribution (conditional); predict_link (conditional); survival.bootstrap (conditional); survival.curves (conditional); survival.hazard (conditional)
     - None declared
     - fitted_model, valid_subject_resampling
   * - NAMSurvival / Neural Breslow partial likelihood
     - predict_components (conditional); predict_distribution (conditional); predict_link (conditional); survival.bootstrap (conditional); survival.curves (conditional)
     - None declared
     - fitted_model, valid_subject_resampling
   * - SemiStructuredRegressor / Identified spline and neural composition
     - predict_components (conditional); predict_link (conditional)
     - fisher.exact (conditional); fisher.score_outer_product (conditional)
     - fitted_model, regression_scale
