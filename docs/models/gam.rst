:orphan:

:orphan:

Generalized Additive Models
===========================

When to use them
----------------

Use a statistical GAM when smoothness selection, model-based uncertainty,
effective degrees of freedom, term tests, and residual diagnostics are part of
the analysis.

Model structure
---------------

A generalized additive model represents a transformed conditional mean as a
sum of parametric and smooth effects,

.. math::

   \eta(x) \equiv g\!\left(\mathbb{E}[Y \mid x]\right)
   = \beta_0 + X\beta + \sum_{j=1}^{p} f_j(x_j)
     + \sum_{j<k} f_{jk}(x_j, x_k).

Each smooth is represented by basis coefficients and a roughness penalty.
Smoothing parameters control the trade-off between fit and wiggliness and can
be supplied directly or estimated with GCV, UBRE, ML, REML, or the supported
LAML routes. Unlike the neural models, the statistical backend also provides
model-based covariance matrices, effective degrees of freedom, term tests,
and residual diagnostics.

Fit and inspect
---------------

.. code-block:: python

   from nampy.gam import GAM

   model = GAM(
       formula="y ~ s(x, bs='tp', k=15) + s(z, bs='cr', k=10)",
       family="gaussian",
       optimize_smoothing=True,
       smoothing_method="reml",
   ).fit(data=df)

   prediction, se = model.predict(df, return_se=True)
   terms = model.predict_terms(df)
   model.summary()
   model.plot(residuals=True)

For distributional regression, provide one formula per parameter:

.. code-block:: python

   from nampy.models import GAMLSS

   model = GAMLSS(
       family="normal",
       formula={"mu": "y ~ s(x)", "sigma": "~ s(z)"},
   ).fit(df)
   parameters = model.predict(df)

Interpretation and limits
-------------------------

Use a formula when the basis, tensor-product structure, offsets, or shape
constraints should be explicit. Both ``GAM`` and the estimator adapters accept
formulas. ``GAMRegressor`` and ``GAMClassifier`` add estimator conventions and
automatic REML smoothing by default; raw ``GAM`` starts with fixed smoothing
unless selection is requested.

Available interfaces
--------------------

* :class:`nampy.gam.GAM`
* :class:`nampy.models.GAMRegressor`
* :class:`nampy.models.GAMClassifier`
* :class:`nampy.models.GAMLSS`

See :doc:`../user_guide/statistical_gams` for the complete formula and
analysis surface.

References
----------

* Hastie and Tibshirani (1986), `Generalized Additive Models
  <https://doi.org/10.1214/ss/1177013604>`_.
* Wood (2017), *Generalized Additive Models: An Introduction with R*, second
  edition.

