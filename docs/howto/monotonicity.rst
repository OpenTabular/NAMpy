Impose monotonicity
===================

In a crop study, let ``fertilizer_kg_ha`` denote fertilizer dose in kilograms
per hectare and ``yield_t_ha`` denote crop yield in tonnes per hectare. If the
expected yield is known not to decrease within the studied dose range, replace
the ordinary smooth basis with ``mpi``:

.. code-block:: python

   increasing = GAM(
       formula='yield_t_ha ~ s(fertilizer_kg_ha, bs="mpi", k=12)',
       family="gaussian",
       optimize_smoothing=True,
       smoothing_method="gcv",
       smoothing_optimizer="bfgs",
   ).fit(data)

   derivative = increasing.derivative(smooth_number=1, deriv=1)

The constraint applies during fitting. Check the derivative and fitted range,
and state why the direction is justified. Run the :doc:`complete comparison
<../auto_examples/statistical/plot_monotone_effect>` or consult the
:doc:`shape-constraint guide <../user_guide/shape_constrained_gams>` for
curvature, positivity, local, and bivariate constraints.
