Add an Interaction
==================

Use a tensor-product term when a statistical GAM needs a joint smooth whose
covariates have different scales:

.. code-block:: python

   model = GAM(
       formula="y ~ s(x1) + s(x2) + te(x1, x2, k=c(8, 8))",
       family="gaussian",
       optimize_smoothing=True,
       smoothing_method="reml",
   ).fit(data)

   model.plot(scheme=2)

For a neural estimator, name the interaction explicitly:

.. code-block:: python

   model = NAMRegressor(interactions=[("x1", "x2")])
   model.fit(X_train, y_train)
   model.plot_interactions(X_test)

An interaction contribution depends on every variable in its label. When
main effects and interactions overlap, exact reconstruction does not imply a
unique separation between them.
