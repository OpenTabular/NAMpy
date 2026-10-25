Interpretability
================

This guide covers component extraction, effect tables, and plot controls.

.. _additive-predictor-interpretation:

Understand the decomposition before plotting
---------------------------------------------

For a scalar response, NAMpy represents a link-scale prediction as named
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
``response``, ``terms``, ``intercept``, and ``offset``.
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

   from nampy.models import NAMRegressor
   
   model = NAMRegressor()
   model.fit(X_train, y_train, max_epochs=100)
   
   components = model.predict_components(X_test, center=True)
   table = model.explain_terms(X_test)
   importance = model.term_importance(X_test)
   figures = model.plot_terms(X_test)

Choose a representation from the :doc:`model guide <../models/index>` based
on term dependence, interaction structure, objective support, and inferential
needs.

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

   model = NAMLSS(family="normal").fit(X_train, y_train)
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

Continue the analysis
---------------------

* :doc:`../models/index` - Model comparison
* :doc:`../howto/index` - Practical examples
* GitHub: https://github.com/Ananyapam7/NAMpy
