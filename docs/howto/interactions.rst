Add an Interaction
==================

Add an interaction when the effect of one input is allowed to depend on another. In a statistical GAM, use a joint smooth or combine main effects with a tensor interaction. In a neural additive model, specify the feature combination supported by the architecture.

Use ``s(x) + s(z) + ti(x, z)`` to represent separate main effects and an additional interaction. Use ``te(x, z)`` for a full joint surface. ``t2(x, z)`` provides an alternative tensor construction with a different penalty decomposition.

.. code-block:: python

   from nampy.gam import GAM

   model = GAM(
       formula="y ~ s(x1) + s(x2) + ti(x1, x2, k=c(8, 8))",
       family="gaussian",
       optimize_smoothing=True,
       smoothing_method="reml",
   ).fit(data)

   model.plot(scheme=2)

See :ref:`formula-visual-reference` for side-by-side tensor constructions.

For a neural estimator, name the interaction explicitly:

.. code-block:: python

   from nampy.models import NAMRegressor

   model = NAMRegressor(interactions=[("x1", "x2")])
   model.fit(X_train, y_train)
   model.plot_interactions(X_test)

An interaction contribution depends on every variable in its label. When
main effects and interactions overlap, exact reconstruction does not imply a
unique separation between them.

For train/validation interaction discovery followed by final test evaluation,
continue with :doc:`../tutorials/interaction_selection`.
