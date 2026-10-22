Add an interaction
==================

Suppose ``latitude`` and ``longitude`` locate an observation and ``response``
is a measured environmental outcome. A spatial effect cannot generally be
described by adding separate latitude and longitude curves: it depends on the
coordinates jointly.

For a statistical GAM, use a tensor-product term:

.. code-block:: python

   model = GAM(
       formula=(
           "response ~ s(latitude) + s(longitude) "
           "+ te(latitude, longitude, k=c(8, 8))"
       ),
       family="gaussian",
       optimize_smoothing=True,
       smoothing_method="reml",
   ).fit(observations)

   model.plot(scheme=2)

For a neural estimator, name the interaction explicitly:

.. code-block:: python

   model = NAMRegressor(interactions=[("latitude", "longitude")])
   model.fit(X_train, y_train)
   model.plot_interactions(X_test)

An interaction contribution depends on every variable in its label. When main
effects and interactions overlap, exact reconstruction does not imply a unique
separation between them. The :doc:`interaction tutorial
<../tutorials/interaction_surface>` shows the fitted surface.
