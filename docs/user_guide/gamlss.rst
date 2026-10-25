Distributional GAMs and GAMLSS
==============================

GAMLSS extends additive regression beyond the mean (Rigby and Stasinopoulos, 2005).

This guide covers ``GAMLSS`` formulas, parameter ordering, predictions, and
scoring. For a fitted example, see :doc:`../notebooks/04_distributional_models`.

The lower-level :class:`nampy.gam.GAM` also accepts supported multi-parameter
families and exposes their statistical parameterization directly.

.. _distributional-model-concepts:

Decide which aspects of the response to model
---------------------------------------------

A mean model assumes that the conditional expectation is the main quantity
that changes with predictors. Distributional regression is useful when
variability, skewness, tail behavior, or another parameter also changes.

For parameter :math:`\theta_k`, NAMpy fits an additive predictor
:math:`\eta_k` and applies the parameter's inverse link,

.. math::

   \eta_k(x)=\beta_{0k}+\sum_j f_{jk}(x_j)
   +\sum_{S\in\mathcal I_k}f_{Sk}(x_S)+o_k(x),\qquad
   \theta_k(x)=h_k^{-1}\{\eta_k(x)\}.

Together these parameters specify the response distribution,
:math:`Y\mid x\sim\mathcal D\{\theta_1(x),\ldots,\theta_K(x)\}`.
The term definitions and identifiability considerations follow
:ref:`additive-predictor-interpretation`, separately for each parameter.

Terms add on the :math:`\eta_k` scale. Natural parameters generally do not add
after a nonlinear inverse link. Conditional standard deviation describes
observation-level predictive spread. It is different from a standard error
for an estimated mean and from variation across repeated fits.

Quantile models instead estimate selected conditional quantiles using pinball
loss. These quantiles do not by themselves specify a complete probability
density; their spacing and empirical coverage describe different properties.

Public parameter contract
-------------------------

``GAMLSS.predict`` always returns natural distribution parameters. Columns
follow ``parameter_names_``. ``predict(raw=True)`` and ``predict_link`` return
the additive linear predictors instead.

.. list-table:: Built-in families
   :header-rows: 1

   * - Public aliases
     - ``parameter_names_``
     - Meaning
   * - ``normal``, ``gaussian``, ``gaulss``
     - ``("mu", "sigma")``
     - Normal conditional mean and standard deviation
   * - ``gamma``, ``gammals``
     - ``("mu", "sigma")``
     - Gamma conditional mean and coefficient of variation, so
       ``Var(Y | X) = mu**2 * sigma**2``

``predict_point`` returns the conditional mean. ``score`` returns mean log
likelihood (higher is better), while the default ``evaluate`` result reports
mean negative log likelihood.

Formula and array modes
-----------------------

A named mapping is the preferred formula form. It prevents accidental changes
to parameter order and requires secondary formulas to be one-sided:

.. code-block:: python

   from nampy.models import GAMLSS

   model = GAMLSS(
       family="normal",
       formula={
           "mu": "y ~ s(age, bs='cr', k=10) + group",
           "sigma": "~ s(age, bs='cr', k=8)",
       },
       smoothing_method="reml",
   ).fit(data)

   parameters = model.predict(data)
   standard_errors = model.standard_errors(data)
   components = model.predict_components(data)
   components.validate_additive_reconstruction()

An ordered formula list remains supported for compatibility. Without a
formula, each distribution parameter receives its own independently compiled
copy of the configured main-effect smooths:

.. code-block:: python

   model = GAMLSS(family="gamma", k=10, basis="tp").fit(X, y)

Offsets must be a list or tuple with one entry per parameter. Contributions in
``predict_components`` are zero-padded matrices, so every term has the same
``(n_samples, n_parameters)`` shape as the link prediction.

For neural LSS effect plots, centering references, and uncertainty envelopes,
see :doc:`interpretability`.

Statistical smoothness selection
--------------------------------

Supported multi-parameter fitting routes use the general smooth-model
framework described by Wood, Pya, and Säfken (2016), including Laplace
approximate marginal likelihood where available. See :doc:`../api/gam_advanced`
for compatible family and fitting combinations.

References
----------

Rigby and Stasinopoulos (2005), `Generalized additive models for location,
scale and shape <https://doi.org/10.1111/j.1467-9876.2005.00510.x>`_. Journal
of the Royal Statistical Society: Series C 54:507–554.

Wood, Pya, and Säfken (2016), `Smoothing Parameter and Model Selection for
General Smooth Models <https://doi.org/10.1080/01621459.2016.1180986>`_. Journal
of the American Statistical Association 111:1548–1563.
