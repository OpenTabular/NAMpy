Utilities
=========

Utility functions and distribution classes.

.. currentmodule:: nampy.utils

The utils module contains interpretability helpers, summary utilities,
distributional metrics, and probability distributions used in LSS models.

Interpretability
----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   predict_terms
   term_contributions
   feature_importance
   plot_terms
   plot_interactions

Modules
-------

.. autosummary::
   :toctree: generated/

   distributional_metrics
   distributions
   interpretability

Available Distributions
-----------------------

NAMpy supports the following distribution families for LSS models:

* **normal** - Normal/Gaussian distribution
* **poisson** - Poisson distribution (count data)
* **gamma** - Gamma distribution (positive continuous)
* **beta** - Beta distribution (values between 0 and 1)
* **studentt** - Student's t-distribution (heavy tails)
* **negativebinom** - Negative binomial (overdispersed counts)
* **inversegamma** - Inverse gamma distribution
* **categorical** - Categorical distribution
* **dirichlet** - Dirichlet distribution (multivariate)

Example Usage
-------------

.. code-block:: python

   from nampy.models import NAMLSS
   
   # Use a specific distribution family
   model = NAMLSS()
   model.fit(X, y, family="normal", max_epochs=150)
   
   # Predict distribution parameters
   params = model.predict(X_test)
