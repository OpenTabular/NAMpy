.. list-table::
   :header-rows: 1

   * - Backend / family
     - Observation type
     - Declared smoothing optimizers
     - Probability operations
     - Prediction context
   * - statistical / bccg_lss
     - continuous
     - No declaration at family level
     - cdf, cdf_left, log_cdf, log_prob, log_survival, mean, quantile, sample, survival, variance
     - None
   * - statistical / bcpe_lss
     - continuous
     - No declaration at family level
     - cdf, cdf_left, log_cdf, log_prob, log_survival, mean, quantile, sample, survival, variance
     - None
   * - statistical / bct_lss
     - continuous
     - No declaration at family level
     - cdf, cdf_left, log_cdf, log_prob, log_survival, mean, quantile, sample, survival, variance
     - None
   * - statistical / beinf_lss
     - mixed
     - No declaration at family level
     - cdf, cdf_left, log_cdf, log_prob, log_survival, mean, quantile, sample, survival, variance
     - None
   * - statistical / beta_lss
     - continuous
     - No declaration at family level
     - cdf, cdf_left, log_cdf, log_prob, log_survival, mean, quantile, sample, survival, variance
     - None
   * - statistical / betabinomial_lss
     - discrete
     - No declaration at family level
     - cdf, cdf_left, log_cdf, log_prob, log_survival, mean, quantile, sample, survival, variance
     - trials
   * - statistical / gammals
     - continuous
     - No declaration at family level
     - cdf, cdf_left, log_cdf, log_prob, log_survival, mean, quantile, sample, survival, variance
     - None
   * - statistical / gaulss
     - continuous
     - No declaration at family level
     - cdf, cdf_left, log_cdf, log_prob, log_survival, mean, quantile, sample, survival, variance
     - None
   * - statistical / gevlss
     - Family-specific; no shared probability law declared
     - No declaration at family level
     - No shared probability adapter
     - None
   * - statistical / gumbls
     - Family-specific; no shared probability law declared
     - No declaration at family level
     - No shared probability adapter
     - None
   * - statistical / lognormal_lss
     - continuous
     - No declaration at family level
     - cdf, cdf_left, log_cdf, log_prob, log_survival, mean, quantile, sample, survival, variance
     - None
   * - statistical / lognormal_observation_lss
     - Exact/censored observations with configured truncation
     - bfgs, outer_newton
     - cdf, cdf_left, log_cdf, log_prob, log_survival, quantile, sample, survival
     - None
   * - statistical / multinom
     - Family-specific; no shared probability law declared
     - No declaration at family level
     - No shared probability adapter
     - None
   * - statistical / nb_lss
     - discrete
     - No declaration at family level
     - cdf, cdf_left, log_cdf, log_prob, log_survival, mean, quantile, sample, survival, variance
     - None
   * - statistical / normal_observation_lss
     - Exact/censored observations with configured truncation
     - bfgs, outer_newton
     - cdf, cdf_left, log_cdf, log_prob, log_survival, mean, quantile, sample, survival, variance
     - None
   * - statistical / shash
     - Family-specific; no shared probability law declared
     - No declaration at family level
     - No shared probability adapter
     - None
   * - statistical / student_lss
     - continuous
     - No declaration at family level
     - cdf, cdf_left, log_cdf, log_prob, log_survival, mean, quantile, sample, survival, variance
     - None
   * - statistical / twlss
     - Family-specific; no shared probability law declared
     - No declaration at family level
     - No shared probability adapter
     - None
   * - statistical / weibull_lss
     - continuous
     - No declaration at family level
     - cdf, cdf_left, log_cdf, log_prob, log_survival, mean, quantile, sample, survival, variance
     - None
   * - statistical / weibull_observation_lss
     - Exact/censored observations with configured truncation
     - bfgs, outer_newton
     - cdf, cdf_left, log_cdf, log_prob, log_survival, quantile, sample, survival
     - None
   * - statistical / zaga_lss
     - mixed
     - No declaration at family level
     - cdf, cdf_left, log_cdf, log_prob, log_survival, mean, quantile, sample, survival, variance
     - None
   * - statistical / ziplss
     - Family-specific; no shared probability law declared
     - No declaration at family level
     - No shared probability adapter
     - None
   * - neural / bccg
     - continuous
     - No declaration at family level
     - cdf, cdf_left, log_cdf, log_prob, log_survival, mean, quantile, sample, survival, variance
     - None
   * - neural / bcpe
     - continuous
     - No declaration at family level
     - cdf, cdf_left, log_cdf, log_prob, log_survival, mean, quantile, sample, survival, variance
     - None
   * - neural / bct
     - continuous
     - No declaration at family level
     - cdf, cdf_left, log_cdf, log_prob, log_survival, mean, quantile, sample, survival, variance
     - None
   * - neural / beinf
     - mixed
     - No declaration at family level
     - cdf, cdf_left, log_cdf, log_prob, log_survival, mean, quantile, sample, survival, variance
     - None
   * - neural / beta
     - continuous
     - No declaration at family level
     - cdf, cdf_left, log_cdf, log_prob, log_survival, mean, quantile, sample, survival, variance
     - None
   * - neural / betabinomial
     - discrete
     - No declaration at family level
     - cdf, cdf_left, log_cdf, log_prob, log_survival, mean, quantile, sample, survival, variance
     - trials
   * - neural / categorical
     - Family-specific; no shared probability law declared
     - No declaration at family level
     - No shared probability adapter
     - None
   * - neural / dirichlet
     - Family-specific; no shared probability law declared
     - No declaration at family level
     - No shared probability adapter
     - None
   * - neural / gamma
     - continuous
     - No declaration at family level
     - cdf, cdf_left, log_cdf, log_prob, log_survival, mean, quantile, sample, survival, variance
     - None
   * - neural / hurdlenegativebinom
     - Family-specific; no shared probability law declared
     - No declaration at family level
     - No shared probability adapter
     - None
   * - neural / hurdlepoisson
     - Family-specific; no shared probability law declared
     - No declaration at family level
     - No shared probability adapter
     - None
   * - neural / inversegamma
     - Family-specific; no shared probability law declared
     - No declaration at family level
     - No shared probability adapter
     - None
   * - neural / loglogistic
     - Family-specific; no shared probability law declared
     - No declaration at family level
     - No shared probability adapter
     - None
   * - neural / lognormal
     - continuous
     - No declaration at family level
     - cdf, cdf_left, log_cdf, log_prob, log_survival, mean, quantile, sample, survival, variance
     - None
   * - neural / lognormal_observation
     - Exact/censored observations with configured truncation
     - No declaration at family level
     - cdf, cdf_left, log_cdf, log_prob, log_survival, quantile, sample, survival
     - None
   * - neural / mvnormdiag
     - Family-specific; no shared probability law declared
     - No declaration at family level
     - No shared probability adapter
     - None
   * - neural / negativebinom
     - discrete
     - No declaration at family level
     - cdf, cdf_left, log_cdf, log_prob, log_survival, mean, quantile, sample, survival, variance
     - None
   * - neural / normal
     - continuous
     - No declaration at family level
     - cdf, cdf_left, log_cdf, log_prob, log_survival, mean, quantile, sample, survival, variance
     - None
   * - neural / normal_observation
     - Exact/censored observations with configured truncation
     - No declaration at family level
     - cdf, cdf_left, log_cdf, log_prob, log_survival, mean, quantile, sample, survival, variance
     - None
   * - neural / ordinal
     - Family-specific; no shared probability law declared
     - No declaration at family level
     - No shared probability adapter
     - None
   * - neural / poisson
     - Family-specific; no shared probability law declared
     - No declaration at family level
     - No shared probability adapter
     - None
   * - neural / quantile
     - Family-specific; no shared probability law declared
     - No declaration at family level
     - No shared probability adapter
     - None
   * - neural / robustnormal
     - Family-specific; no shared probability law declared
     - No declaration at family level
     - No shared probability adapter
     - None
   * - neural / studentt
     - continuous
     - No declaration at family level
     - cdf, cdf_left, log_cdf, log_prob, log_survival, mean, quantile, sample, survival, variance
     - None
   * - neural / tweedie
     - Family-specific; no shared probability law declared
     - No declaration at family level
     - No shared probability adapter
     - None
   * - neural / weibull
     - continuous
     - No declaration at family level
     - cdf, cdf_left, log_cdf, log_prob, log_survival, mean, quantile, sample, survival, variance
     - None
   * - neural / weibull_observation
     - Exact/censored observations with configured truncation
     - No declaration at family level
     - cdf, cdf_left, log_cdf, log_prob, log_survival, quantile, sample, survival
     - None
   * - neural / zaga
     - mixed
     - No declaration at family level
     - cdf, cdf_left, log_cdf, log_prob, log_survival, mean, quantile, sample, survival, variance
     - None
   * - neural / zinb
     - Family-specific; no shared probability law declared
     - No declaration at family level
     - No shared probability adapter
     - None
   * - neural / zip
     - Family-specific; no shared probability law declared
     - No declaration at family level
     - No shared probability adapter
     - None
