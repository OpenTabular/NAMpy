Fit, diagnose, and extend a statistical GAM
===========================================

.. include:: ../quickstart.rst
   :start-line: 3
   :end-before: Choose the next analysis

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
