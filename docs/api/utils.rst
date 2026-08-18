Distributions
=============

Torch distribution families and metrics for LSS models.

.. currentmodule:: nampy.neural.distributions

Modules
-------

.. autosummary::
   :toctree: generated/

   distributions
   metrics

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
