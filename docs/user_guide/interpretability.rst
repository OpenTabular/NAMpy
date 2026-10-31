Interpretability
================

This guide covers component extraction, effect tables, and plot controls.

.. _additive-predictor-interpretation:

Understand the decomposition before plotting
---------------------------------------------

For a scalar conditional-mean model, NAMpy represents a link-scale prediction as named
contributions:

.. math::

   \eta(x) \equiv g\!\left(\mathbb{E}[Y \mid x]\right)
   =\beta_0+\sum_j f_j(x_j)+\sum_{S\in\mathcal I}f_S(x_S)+o(x).

Here :math:`g` is the link, :math:`\beta_0` the intercept, :math:`f_j` a
main effect, :math:`f_S` an interaction over the named feature subset
:math:`S`, and :math:`o` an optional known offset. Omit the corresponding
terms when there are no interactions or offset.

The conditional mean is :math:`g^{-1}\{\eta(x)\}`. Contributions add on the
link scale; applying a nonlinear inverse link to each term separately and
summing the results does not reconstruct the response prediction.

The :class:`nampy.contracts.AdditivePrediction` object stores ``link``,
``terms``, ``intercept``, ``offset`` and predictor/term metadata. It describes
additive predictors. Obtain response values or probabilities through the
estimator's task-specific prediction methods; distributional models expose
``predict_point`` and ``predict_distribution`` for those queries.
``validate_additive_reconstruction()`` checks the accounting identity above.
It does not establish that each term is uniquely determined or reliably
estimated.

Centering removes constant-shift ambiguity relative to a stated reference
dataset. Main effects and interactions can still exchange structure when they
overlap, and correlated predictors can make recovered curves unstable. Report
these questions separately:

* Do the returned components reconstruct the prediction?
* Which data distribution defines centering?
* Which inputs may each term depend on?
* Are the estimated effects stable across samples or reasonable model choices?

Extract term contributions
--------------------------

Every estimator that exposes additive components uses the same result
contract:

.. code-block:: python


   from dataclasses import replace
   from nampy.neural.configuration import GradientTrainingConfig
   from nampy.models import NAMRegressor

   model = NAMRegressor()
   model.set_params(training=replace(model.training or GradientTrainingConfig(), max_epochs=100)).fit(X_train, y_train)

   components = model.predict_components(X_test, center=True, reference_X=X_train)
   table = model.explain_terms(X_test, center=True, reference_X=X_train)
   importance = model.term_importance(X_test, center=True, reference_X=X_train)
   figures = model.plot_terms(X_test, center=True, reference_X=X_train)

Choose a representation from the :doc:`model guide <../models/index>` based
on term dependence, interaction structure, objective support, and inferential
needs.

Decompose one prediction
------------------------

This Poisson example includes two main effects, an interaction, and a known
exposure offset. The waterfall adds them on the log-mean scale. Its final bar
is the total predictor; exponentiating that total gives the expected count.
For logistic classification the corresponding sum would be log-odds, never
additive probability contributions.

.. literalinclude:: ../scripts/generate_howto_assets.py
   :language: python
   :start-after: # INTERPRETATION START
   :end-before: # INTERPRETATION PLOTTING
   :dedent: 4

.. dropdown:: Plotting code

   .. literalinclude:: ../scripts/generate_howto_assets.py
      :language: python
      :start-after: # INTERPRETATION PLOTTING
      :end-before: # INTERPRETATION END
      :dedent: 4

.. image:: /_static/howto_waterfall.png
   :alt: Link-scale waterfall of intercept, two main effects, interaction, and exposure offset for one count prediction
   :class: scientific-figure

Change the centering reference
------------------------------

The same fitted terms can be represented relative to all training rows or
the subset with positive x. Subtract each reference mean from its term and
add those constants to the intercept. This is an explicit re-expression of
the fitted statistical components, not a refit or a change to the exposure.
The neural estimator API performs the analogous shift through ``center=True``
and ``reference_X``.

For either backend, the shared helper also accepts a separate reference cohort
and weights defining that cohort's centering distribution:

.. code-block:: python

   from nampy.diagnostics import center_additive_prediction

   query_components = model.predict_components(X_test)
   reference_components = model.predict_components(X_reference)
   centered = center_additive_prediction(
       query_components,
       reference=reference_components,
       sample_weight=reference_weights,
   )

``reference_weights`` has one finite nonnegative entry per reference row and a
positive total. Each term's weighted reference mean is subtracted from that
term on every query row and added to the intercept. Link predictions and fitted
coefficients are preserved. Supply any required offsets separately for each
component query.

.. literalinclude:: ../scripts/generate_howto_assets.py
   :language: python
   :start-after: # CENTERING START
   :end-before: # CENTERING PLOTTING
   :dedent: 4

.. dropdown:: Plotting code

   .. literalinclude:: ../scripts/generate_howto_assets.py
      :language: python
      :start-after: # CENTERING PLOTTING
      :end-before: # CENTERING END
      :dedent: 4

.. image:: /_static/howto_centering.png
   :alt: Shifted effect curves and compensating intercepts under two reference distributions, with unchanged predictions
   :class: scientific-figure

The effect curve moves vertically and the intercept compensates. Both
prediction curves overlap exactly; the assertions verify reconstruction on
the link scale. Centering does not remove dependence between overlapping
main effects and interactions.

Visualization
-------------

``plot_terms`` renders one-dimensional contribution curves, while
``plot_interactions`` handles interaction heatmaps. ``explain_terms`` and
``term_importance`` return reusable tables for custom plotting. By default,
``plot_terms`` shades the empirical predictor density in red behind each
curve. Pass ``density=False`` to remove the shading or ``rug=True`` to add
individual observation marks.

Neural LSS estimators additionally provide ``plot_distributional_effects``.
It shows the family-defined point effect and point effect plus/minus one
conditional standard deviation. This envelope describes predictive spread;
it is not a confidence interval for the central curve.

``reference_X`` defines the centering distribution; the centered reference
configuration need not correspond to an observed row. ``response_scale`` converts
standardized targets back to their original units. The family evaluates its
parameter links before plotting, through ``predict_point`` and
``predict_spread``; the plot does not assume a particular parameter order.

.. code-block:: python


   from nampy.neural.configuration import ObjectiveConfig
   from nampy.models import NAMLSS

   model = NAMLSS(objective=ObjectiveConfig(family='normal')).fit(X_train, y_train)
   figures = model.plot_distributional_effects(
       X_test,
       reference_X=X_train,
       feature_names=["age", "income"],
   )

For repeated fits, use ``prepare_distributional_effects`` on each fitted model,
combine the results with ``combine_distributional_effects``, and pass the
result to ``render_distributional_effects``. The renderer then adds a
between-fit quantile ribbon, which describes variation across fitted models
rather than conditional predictive spread.

For prediction shapes and parameter scales, see :doc:`../howto/prediction_outputs`.

For worked examples, compare :doc:`GAM and NAM feature effects <../tutorials/gam_nam>`,
or examine :doc:`distributions, quantiles, and ensemble spread <../notebooks/04_distributional_models>`.
