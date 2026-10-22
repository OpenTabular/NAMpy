Statistical GAMs
================

The modeling problem
--------------------

A generalized additive model is useful when the response may depend
nonlinearly on several predictors, but you still want named effects,
statistical uncertainty, and model diagnostics. NAMpy fits these models in
Python with formula terms such as ``s(temperature)`` and
``te(latitude, longitude)``.

Consider electricity demand measured in megawatts. Cold and hot weather can
both raise demand, so a straight line for outdoor temperature is implausible.
A smooth term estimates that relationship without requiring its shape to be
specified in advance.

.. image:: ../_static/first_gam_effects.png
   :alt: Smooth temperature and cyclic hour effects for electricity demand
   :class: scientific-figure
   :align: center

Minimal fit
-----------

``temperature`` is measured in degrees Celsius, ``hour`` is time on a 24-hour
clock, and ``demand`` is measured in megawatts. A cubic regression spline is
appropriate for temperature; a cyclic spline encodes that hour 24 returns to
hour 0.

.. code-block:: python

   from nampy.models import GAMRegressor

   model = GAMRegressor(
       formula=(
           'demand ~ s(temperature, bs="cr", k=12) '
           '+ s(hour, bs="cc", k=12)'
       ),
       family="gaussian",
   ).fit(train)

   predictions = model.predict(test)
   model.plot(scheme=1, shade_col="#f3b4c2", scale=0)

Important choices
-----------------

**Family and link.** Choose these from the response, not from the predictor.
A Gaussian family describes continuous measurements; binomial and Poisson
families address probabilities and counts. The link determines the scale on
which additive contributions combine.

**Smooth basis.** The basis expresses what shapes are possible. Use cyclic
bases for periodic predictors, tensor products when scales or units differ,
and shape-constrained bases only when the scientific assumption is defensible.

**Basis dimension.** ``k`` supplies capacity; it does not directly set the
curve's wiggliness. The smoothing penalty determines how much capacity is
used. Choose a plausible ``k`` and check it after fitting rather than treating
it as a tuning knob for visual smoothness.

**Diagnostics.** Examine residual patterns, basis-dimension checks,
concurvity, and sensitivity to modeling choices. A good held-out score does
not establish that individual effects are stable or scientifically adequate.

Interfaces
----------

:class:`nampy.models.GAMRegressor` provides estimator conventions and enables
automatic REML smoothness selection by default. :class:`nampy.gam.GAM`
provides direct access to formula compilation and statistical fitting controls;
automatic selection is requested explicitly there. Both accept formulas.

Continue
--------

* Run the complete :doc:`first GAM example
  <../auto_examples/statistical/plot_first_gam>`.
* Use the :ref:`statistical-gam-tools` inventory to look up every supported
  formula term, basis, family, fitting route, diagnostic, and public symbol.
* Consult the :doc:`statistical GAM API <../api/gam>` for exact signatures.
* Read :doc:`../coming_from_mgcv` for task equivalents and default differences.
