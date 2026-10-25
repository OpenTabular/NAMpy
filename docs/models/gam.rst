Generalized Additive Models
===========================

When to use them
----------------

Use a statistical GAM when smoothness selection, model-based uncertainty,
effective degrees of freedom, term tests, and residual diagnostics are part of
the analysis.

Model structure
---------------

Statistical GAMs use penalized spline bases for smooth effects, alongside
parametric terms and explicit interactions. See
:ref:`additive-predictor-interpretation` for the decomposition and
:ref:`smoothness-and-basis-dimension` for the basis and penalty equations.

Smoothing parameters can be supplied directly or estimated with GCV, UBRE, ML,
REML, or the supported LAML routes. The statistical backend provides
model-based covariance matrices, effective degrees of freedom, and term tests.

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

