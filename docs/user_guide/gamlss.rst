Distributional GAMs and GAMLSS
==============================

NAMpy exposes three statistical GAM estimators with distinct prediction
contracts:

* :class:`nampy.models.GAMRegressor` fits one response predictor and returns a
  one-dimensional numeric response.
* :class:`nampy.models.GAMClassifier` fits a binary binomial predictor and
  returns labels or two-column probabilities.
* :class:`nampy.models.GAMLSS` jointly fits two or more distribution parameters
  and returns a parameter matrix.

The low-level :class:`nampy.gam.GAM` remains unrestricted. It deliberately
keeps mgcv-shaped family names and response transformations for parity work.

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

Migration
---------

Code that previously placed ``gaulss`` or ``gammals`` inside
``GAMRegressor`` must move to ``GAMLSS``:

.. code-block:: python

   # Before
   GAMRegressor(family="gaulss", formula=["y ~ s(x)", "~ s(z)"])

   # Now
   GAMLSS(
       family="normal",
       formula={"mu": "y ~ s(x)", "sigma": "~ s(z)"},
   )

This is an intentional role guard: a multi-parameter prediction is neither a
single regression target nor a classification label. Direct ``GAM`` code does
not need to migrate, but should remember that raw ``type="response"`` follows
mgcv's internal parameterization rather than this natural-parameter contract.
