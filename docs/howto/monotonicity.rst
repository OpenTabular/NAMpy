Impose Monotonicity
===================

Replace the smooth basis code with ``mpi`` for an increasing effect or ``mpd``
for a decreasing effect:

.. code-block:: python

   increasing = GAM(
       formula="y ~ s(dose, bs='mpi', k=12)",
       family="gaussian",
       optimize_smoothing=True,
       smoothing_method="gcv",
       smoothing_optimizer="bfgs",
   ).fit(data)

   derivative = increasing.derivative(smooth_number=1, deriv=1)

The constraint applies during fitting. Check the derivative and the fitted
range, and state why the direction is justified. The
:doc:`../user_guide/shape_constrained_gams` reference lists curvature,
positivity, local, and bivariate constraints.
