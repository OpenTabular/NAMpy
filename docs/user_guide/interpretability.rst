Interpretability
================

NAMpy exposes the terms that form an additive prediction.

What a term contribution means
------------------------------

For a scalar response, the central model is

.. math::

   \eta(x) \equiv g\!\left(\mathbb{E}[Y \mid x]\right)
   = \beta_0 + \sum_{j=1}^{p} f_j(x_j)
     + \sum_{S \in \mathcal I} f_S(x_S) + o(x).

Here :math:`\eta` is the additive predictor, :math:`g` is the link,
:math:`\beta_0` is the intercept, :math:`f_j` is a main effect,
:math:`f_S` is an explicitly included interaction, and :math:`o` is an
optional known offset. The conditional mean is
:math:`g^{-1}\{\eta(x)\}`. With no interactions or offset, the corresponding
terms are omitted.

``predict_components`` reconstructs :math:`\eta` from the reported intercept,
terms, and offset. This accounting identity does not make the individual terms
uniquely identifiable. Centering fixes constant shifts against a stated
reference dataset; correlated inputs and overlapping main effects and
interactions can still affect term stability.

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

Distributional additive estimators additionally provide
``plot_distributional_effects``. It plots the family-defined conditional point
effect together with point effect plus/minus one conditional standard deviation
and predictor-density shading. The implementation calls the distribution
family's ``predict_point`` and ``predict_spread`` methods; it does not assume a
particular parameter order or distribution.

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
between-fit quantile ribbon.

Continue the analysis
---------------------

* :doc:`../models/index` - Model comparison
* :doc:`../examples/index` - Practical examples
* GitHub: https://github.com/Ananyapam7/NAMpy
