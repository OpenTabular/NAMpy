Neural Distributions
====================

Distribution classes transform raw network outputs into valid parameters and
provide the likelihood used by LSS estimators.

Distribution families
---------------------

.. list-table::
   :header-rows: 1
   :widths: 39 61

   * - Public symbol
     - Response
   * - ``NormalDistribution``, ``RobustNormalDistribution``
     - Gaussian location and scale; the robust form constrains scale more
       defensively.
   * - ``StudentTDistribution``
     - Location, scale, and degrees of freedom for heavy-tailed data.
   * - ``LogNormalDistribution``, ``LogLogisticDistribution``
     - Positive continuous responses.
   * - ``GammaDistribution``, ``InverseGammaDistribution``
     - Positive continuous responses with gamma-family tails.
   * - ``BetaDistribution``
     - Continuous responses in the unit interval.
   * - ``PoissonDistribution``, ``NegativeBinomialDistribution``
     - Count responses, with overdispersion in the negative-binomial model.
   * - ``HurdlePoissonDistribution``,
       ``HurdleNegativeBinomialDistribution``
     - Count responses with a separate zero-versus-positive process.
   * - ``ZeroInflatedPoissonDistribution``,
       ``ZeroInflatedNegativeBinomialDistribution``
     - Count responses with an additional structural-zero component.
   * - ``TweedieDistribution``
     - Non-negative responses with a Tweedie mean-variance relationship.
   * - ``WeibullDistribution``
     - Positive duration or survival-type responses.
   * - ``CategoricalDistribution``
     - Unordered categorical responses.
   * - ``OrdinalCumulativeLogitDistribution``
     - Ordered categorical responses through cumulative logits.
   * - ``DirichletDistribution``
     - Compositional responses on a simplex.
   * - ``MultivariateNormalDiagDistribution``
     - Multivariate Gaussian responses with diagonal covariance.
   * - ``Quantile``
     - Multiple conditional quantiles.
   * - ``BaseDistribution``
     - Base class for defining a custom neural distribution family.

Metrics
-------

.. list-table::
   :header-rows: 1
   :widths: 39 61

   * - Public symbol
     - Metric
   * - ``poisson_deviance``
     - Poisson deviance.
   * - ``negative_binomial_deviance``
     - Negative-binomial deviance.
   * - ``gamma_deviance``
     - Gamma deviance.
   * - ``beta_mean_mse``
     - Mean-squared error for the beta mean.
   * - ``student_t_loss``
     - Student-t negative log likelihood.
   * - ``inverse_gamma_loss``
     - Inverse-gamma loss.
   * - ``dirichlet_error``
     - Error metric for compositional responses.

.. code-block:: python

   from nampy.models import NAMLSS

   model = NAMLSS(family="studentt").fit(X_train, y_train)
   parameters = model.predict(X_test)
   point_prediction = model.predict_point(X_test)

Distributional plots are documented in :doc:`plotting`.
