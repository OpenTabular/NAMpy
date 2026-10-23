Distributional GAMs and GAMLSS
==============================

This guide covers ``GAMLSS`` formulas, parameter ordering, predictions, and
scoring. For the underlying model, see
:doc:`../concepts/distributional_regression`; for a fitted example, see
:doc:`../tutorials/changing_variability`.

The lower-level :class:`nampy.gam.GAM` also accepts supported multi-parameter
families and exposes their statistical parameterization directly.

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
