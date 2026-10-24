GAM Diagnostics and Plots
=========================

Diagnostics are available as methods on a fitted :class:`nampy.gam.GAM`, which
is the usual interface:

.. code-block:: python

   model.summary()
   model.gam_check(type="deviance", seed=7)
   model.k_check(seed=7)
   model.concurvity(full=False)
   derivative = model.derivative(smooth_number=1, deriv=1)
   figures = model.plot(residuals=True, pages=1)

Use ``k_check`` to investigate whether residual structure remains near the
smooth's basis limit; a low index or small permutation p-value motivates a
larger-k refit, not automatic acceptance or rejection. ``concurvity`` concerns
overlap among fitted design spaces and the stability of effect allocations.
``smooth_derivative`` reports predictor-scale slope/curvature and conditional
pointwise Bayesian uncertainty. Under a nonlinear link, a derivative of the
response mean also needs the chain rule.

The functions below expose the same operations for extension and reporting
code; their references describe outputs, interpretation and limits.

Model checks
------------

.. currentmodule:: nampy.gam.diagnostics

.. autosummary::
   :toctree: generated/
   :nosignatures:

   residuals_gam
   k_check
   gam_check
   concurvity
   smooth_derivative
   SmoothDerivativeResult

Term plots
----------

Plotting is split into preparation and rendering so applications can reuse the
computed effects with a custom visual style.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   plot_gam
   prepare_plot_gam_data
   render_plot_gam

Summary formatting and optimizer traces
---------------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   summary_text
   print_summary
   build_optimizer_trace
   save_optimizer_trace
   load_optimizer_trace

