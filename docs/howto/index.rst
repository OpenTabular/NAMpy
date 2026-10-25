:orphan:

How-to Guides
=============

Use these guides to complete a specific task with NAMpy. For a worked analysis with data, fitted results, and interpretation, see the :doc:`tutorials <../tutorials/index>`.

Prepare and fit
---------------

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Prepare feature data
      :link: /user_guide/preprocessing
      :link-type: doc

      Configure numerical and categorical transformations and keep training and prediction inputs consistent.

   .. grid-item-card:: Train and tune neural models
      :link: /user_guide/training
      :link-type: doc

      Choose fitting controls, validation strategy, and hyperparameters.

   .. grid-item-card:: Statistical GAM guide
      :link: /user_guide/statistical_gams
      :link-type: doc

      Choose bases and smoothing penalties, then inspect residuals, inference, and diagnostics.

   .. grid-item-card:: Distributional GAM guide
      :link: /user_guide/gamlss
      :link-type: doc

      Choose distribution parameters, their links, formulas, and prediction scales.

Control the model structure
---------------------------

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Add an interaction
      :link: interactions
      :link-type: doc

      Choose a joint smooth or an explicit neural interaction.

   .. grid-item-card:: Constrain the shape of an effect
      :link: /user_guide/shape_constrained_gams
      :link-type: doc

      Impose justified monotonicity, curvature, or positivity and inspect the fitted constraint.

   .. grid-item-card:: Predict with an offset
      :link: offsets
      :link-type: doc

      Include a known predictor contribution during fitting and prediction.

Inspect predictions and reuse a fit
-----------------------------------

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Interpret and plot effects
      :link: /user_guide/interpretability
      :link-type: doc

      Understand decomposition, centering, term dependence, and the meaning of effect displays.

   .. grid-item-card:: Choose a prediction output
      :link: prediction_outputs
      :link-type: doc

      Match labels, probabilities, parameters, quantiles, and contribution arrays to the right display.

   .. grid-item-card:: Save and reload a model
      :link: persistence
      :link-type: doc

      Preserve the estimator and external preprocessing for new prediction rows.

Quick fit recipes
-----------------

:doc:`Fit a regression, classification, or distributional model </examples/fit_models>`.

Extensions and compatibility
----------------------------

* :doc:`Build a custom neural model </user_guide/custom_models>`

Architecture-specific features, including sparse selection, quantile models,
and tree ensembles, are described with their assumptions and examples in
:doc:`../models/index`. For held-out quantile coverage and between-fit
uncertainty, see :doc:`../notebooks/04_distributional_models`.
