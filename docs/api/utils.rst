Neural Distributions
====================

Distribution classes transform raw network outputs into valid parameters and
provide the likelihood used by LSS estimators.

.. include:: /_generated/reference/neural_families.inc

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
