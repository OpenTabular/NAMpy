Fit, diagnose, and extend a statistical GAM
===========================================

Start after :doc:`../notebooks/01_getting_started` or with basic Python and pandas knowledge.
Allow a few minutes on CPU for fitting, diagnostics, and the extensions below.

.. include:: ../_includes/first_gam_walkthrough.inc

Extend the model structure
--------------------------

The energy-demand example already uses a cyclic smooth for hour. The next
examples use new datasets to explore two other modeling choices: a monotone
effect when prior knowledge constrains the shape, and a joint smooth when two
features act together.

.. include:: ../_includes/tutorial_monotone_effect.inc

.. include:: ../_includes/tutorial_interaction_surface.inc

Continue learning
-----------------

Continue with :doc:`../notebooks/02_statistical_gams` for grouped effects,
offsets, response families, smoothing selection, and inference, then
:doc:`gam_nam` to compare statistical and neural representations.
