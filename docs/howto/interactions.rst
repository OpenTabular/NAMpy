Add an Interaction
==================

Use a tensor-product term when a statistical GAM needs a joint smooth whose
covariates have different scales. Choose the constructor according to the
decomposition you need:

.. code-block:: python

   model = GAM(
       formula="y ~ s(x1) + s(x2) + ti(x1, x2, k=c(8, 8))",
       family="gaussian",
       optimize_smoothing=True,
       smoothing_method="reml",
   ).fit(data)

   model.plot(scheme=2)

``te(x1, x2)`` represents the full surface in one term. ``ti(x1, x2)``
removes the marginal main-effect spaces, so it belongs beside ``s(x1)`` and
``s(x2)`` when you want a main-effects-plus-interaction decomposition.
``t2(x1, x2)`` represents a full surface using SS-ANOVA-style range/null-space
penalty blocks. See :ref:`formula-visual-reference` for side-by-side fitted
examples.

For a neural estimator, name the interaction explicitly:

.. code-block:: python

   model = NAMRegressor(interactions=[("x1", "x2")])
   model.fit(X_train, y_train)
   model.plot_interactions(X_test)

An interaction contribution depends on every variable in its label. When
main effects and interactions overlap, exact reconstruction does not imply a
unique separation between them.
