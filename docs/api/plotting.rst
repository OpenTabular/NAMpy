Plotting
========

Plotting is separated into data preparation and rendering so prepared effects
can be inspected, combined, or rendered with custom presentation code.

Term effects
------------

.. currentmodule:: nampy.plotting

.. autosummary::
   :toctree: generated/
   :nosignatures:

   prepared_from_contributions
   prepared_from_interactions
   render_term_plots
   add_density_shading

Distributional effects
----------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   plot_distributional_effects
   prepare_distributional_effects
   combine_distributional_effects
   render_distributional_effects
   DistributionalEffects


Diagnostic and surface renderers
--------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   render_qq
   render_model_check
   render_prediction_surface

.. include:: /_includes/statistical_plot_notes.inc
