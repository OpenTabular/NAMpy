:html_theme.sidebar_secondary.remove:

NAMpy: interpretable additive modeling in Python
================================================

.. container:: homepage-hero

   NAMpy combines a Python-native statistical GAM engine with neural additive
   models. Use formulas or estimator classes to fit nonlinear effects, then
   inspect the contribution each feature or interaction makes to a prediction.

   .. container:: homepage-actions

      .. button-ref:: auto_examples/statistical/plot_first_gam
         :color: primary

         Fit your first GAM

      .. button-ref:: auto_examples/neural/plot_first_nam
         :outline:

         Fit your first neural additive model

   .. code-block:: bash
      :class: install-command

      pip install nampy

   .. container:: version-note

      Documentation for NAMpy |version|. See :doc:`getting_started` for
      installation and version guidance.

See an additive model
---------------------

The figure comes from a synthetic energy-demand study. ``temperature`` is
outdoor temperature in degrees Celsius, ``hour`` is time on a 24-hour clock,
and ``demand`` is electricity demand in megawatts. Each curve shows how one
variable changes predicted demand after accounting for the other; the shaded
bands show pointwise uncertainty.

.. image:: _static/first_gam_effects.png
   :alt: Temperature and hour effects for synthetic electricity demand
   :class: homepage-demo scientific-figure
   :align: center

The U-shaped temperature effect says that demand is lowest in mild weather
and rises in colder or hotter conditions. The cyclic hour effect joins
midnight to the end of the day. Follow the complete, downloadable
:doc:`first GAM example <auto_examples/statistical/plot_first_gam>` to see how
the data, model, evaluation, effects, and diagnostics fit together.

Choose an approach
------------------

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Statistical GAMs
      :link: user_guide/statistical_models
      :link-type: doc

      Fit penalized smooths with formulas, automatic smoothness selection,
      statistical inference, and model diagnostics—all in Python.

   .. grid-item-card:: Neural additive models
      :link: user_guide/neural_models
      :link-type: doc

      Learn feature functions with neural networks through one shared training,
      prediction, and effect-inspection workflow.

Explore capabilities
--------------------

.. grid:: 1 1 3 3
   :gutter: 2

   .. grid-item-card:: Shape constraints
      :link: auto_examples/statistical/plot_monotone_effect
      :link-type: doc

      Encode knowledge that an effect should increase, decrease, or follow a
      specified curvature.

   .. grid-item-card:: Interactions
      :link: howto/interactions
      :link-type: doc

      Model and visualize effects that depend jointly on two or more inputs.

   .. grid-item-card:: Changing variability
      :link: auto_examples/distributional/plot_changing_variability
      :link-type: doc

      Predict scale and other distribution parameters as well as location.

Learn more
----------

.. grid:: 1 1 3 3
   :gutter: 2

   .. grid-item-card:: Browse examples
      :link: examples/index
      :link-type: doc

      Learn from complete analyses with generated plots and downloadable code.

   .. grid-item-card:: Look up the API
      :link: api/index
      :link-type: doc

      Find exact constructors, parameters, methods, return values, and symbols.

   .. grid-item-card:: Check compatibility
      :link: compatibility
      :link-type: doc

      See what is implemented, compared with ``mgcv`` or ``scam``, intentionally
      different, and unsupported.

.. toctree::
   :maxdepth: 1
   :hidden:

   getting_started
   user_guide
   examples/index
   api/index
   about
