Encode a Monotone Relationship
==============================

Suppose prior knowledge says that a dose effect may level off but cannot
decrease over the studied range. The formula below makes that assumption part
of estimation by replacing an ordinary cubic basis with ``bs="mpi"``.

.. code-block:: python

   from nampy.gam import GAM

   ordinary = GAM(
       formula='response ~ s(dose, bs="cr", k=12)',
       family="gaussian",
       optimize_smoothing=True,
       smoothing_method="reml",
   ).fit(data)

   increasing = GAM(
       formula='response ~ s(dose, bs="mpi", k=12)',
       family="gaussian",
       optimize_smoothing=True,
       smoothing_method="gcv",
       smoothing_optimizer="bfgs",
   ).fit(data)

   derivative = increasing.derivative(smooth_number=1, deriv=1)

.. image:: ../_static/shape_constraint_comparison.png
   :alt: Ordinary and monotone-increasing fits to noisy observations
   :class: scientific-figure
   :align: center

The ordinary smooth follows local noise and briefly turns downward. The
constrained fit does not. Its first derivative is nonnegative over the fitted
range. This restriction is useful only when the monotonicity assumption is
defensible; it should not be selected merely because the resulting curve
looks simpler. See :doc:`../user_guide/shape_constrained_gams` for the full
basis and fitting reference.
