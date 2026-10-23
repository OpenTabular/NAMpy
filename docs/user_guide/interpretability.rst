Interpretability
================

This guide covers component extraction, effect tables, and plot controls.
For the link-scale decomposition, centering, and identifiability, see
:doc:`../concepts/additive_predictors`.

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
* :doc:`../examples/index` - Practical examples
* GitHub: https://github.com/Ananyapam7/NAMpy
