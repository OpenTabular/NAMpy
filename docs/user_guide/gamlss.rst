Distributional GAMs and GAMLSS
==============================

Two groups can have similar average outcomes and very different variability.
A mean-only model cannot represent that distinction. Distributional
regression assigns an additive predictor to location, spread, or another
family parameter so each can change with the covariates.

Start with the :doc:`travel-time variability example
<../auto_examples/distributional/plot_changing_variability>` for a visual
analysis in which every variable and unit is defined.
This page defines the statistical formula and prediction contracts in detail.

Choose the estimator by the quantity being modeled:

* :class:`nampy.models.GAMRegressor` fits one response predictor and returns a
  one-dimensional numeric response.
* :class:`nampy.models.GAMClassifier` fits a binary binomial predictor and
  returns labels or two-column probabilities.
* :class:`nampy.models.GAMLSS` jointly fits two or more distribution parameters
  and returns a parameter matrix.

The lower-level :class:`nampy.gam.GAM` also accepts supported
multi-parameter families and exposes their statistical parameterization
directly.

Public parameter contract
-------------------------

For distribution parameter :math:`k`, NAMpy uses an additive predictor and a
parameter-specific inverse link,

.. math::

   \eta_k(x)
   = \beta_{0k} + \sum_{j=1}^{p} f_{jk}(x_j)
     + \sum_{S \in \mathcal I_k} f_{Sk}(x_S) + o_k(x),
   \qquad
   \theta_k(x) = h_k^{-1}\!\left(\eta_k(x)\right),

with :math:`Y \mid x \sim
\mathcal{D}\{\theta_1(x),\ldots,\theta_K(x)\}`. Contributions returned by
``predict_components`` are additive on the :math:`\eta_k` scale; natural
parameters returned by ``predict`` need not be additive after applying
:math:`h_k^{-1}`.

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

   # travel_time_min: trip duration in minutes
   # distance_km: commute distance in kilometres
   # peak_hour: 1 during peak commuting hours, otherwise 0
   model = GAMLSS(
       family="normal",
       formula={
           "mu": (
               "travel_time_min ~ s(distance_km, bs='cr', k=10) "
               "+ peak_hour"
           ),
           "sigma": "~ s(distance_km, bs='cr', k=8) + peak_hour",
       },
       smoothing_method="reml",
   ).fit(trips)

   parameters = model.predict(trips)
   standard_errors = model.standard_errors(trips)
   components = model.predict_components(trips)
   components.validate_additive_reconstruction()

An ordered formula list remains supported for compatibility. Without a
formula, each distribution parameter receives its own independently compiled
copy of the configured main-effect smooths:

.. code-block:: python

   model = GAMLSS(family="gamma", k=10, basis="tp").fit(X, y)

Offsets must be a list or tuple with one entry per parameter. Contributions in
``predict_components`` are zero-padded matrices, so every term has the same
``(n_samples, n_parameters)`` shape as the link prediction.

Interpreting distributional effects
-----------------------------------

For neural LSS estimators, ``plot_distributional_effects`` obtains the
conditional point prediction and standard deviation from the fitted family.
The central curve is the point effect relative to the centered reference
configuration. The black envelope is point effect plus or minus one
conditional standard deviation; it is predictive spread, not a confidence
interval for the central curve.

``reference_X`` defines the empirical distribution used for centering. The
zero-centered reference can be useful even when no observed row has every
term at its reference value. ``response_scale`` applies a linear conversion
back to original target units after response standardization. Nonlinear
parameter links are always evaluated by the family before plotting.

For repeated fits, the optional blue quantile ribbon describes between-fit
variation. Keep that separate from the black conditional-spread envelope when
reporting results.
