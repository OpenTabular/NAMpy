:orphan:

How-to Guides
=============

Choose the task you need to complete. Each guide explains the relevant ideas
alongside the code. For a complete analysis from data to results, start with
:doc:`../tutorials/index`.

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

   .. grid-item-card:: Specify and diagnose a statistical GAM
      :link: /user_guide/statistical_gams
      :link-type: doc

      Choose bases and smoothing penalties, then inspect residuals, inference, and diagnostics.

   .. grid-item-card:: Model location and spread
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

   .. grid-item-card:: Select interactions
      :link: interaction_selection
      :link-type: doc

      Use validation evidence and interaction discovery before the final test evaluation.

   .. grid-item-card:: Impose monotonicity
      :link: monotonicity
      :link-type: doc

      Constrain a smooth to increase or decrease and check its derivative.

   .. grid-item-card:: Choose other shape constraints
      :link: /user_guide/shape_constrained_gams
      :link-type: doc

      Work with curvature, positivity, local constraints, and constrained surfaces.

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

For a short API example rather than a full tutorial:

* :doc:`Fit a neural regressor </examples/basic_regression>`
* :doc:`Fit a binary classifier </examples/basic_classification>`
* :doc:`Fit a distributional model </examples/distributional_regression>`

Extensions and compatibility
----------------------------

* :doc:`Build a custom neural model </user_guide/custom_models>`
* :doc:`Check pretab compatibility </user_guide/pretab_compatibility>`

Architecture-specific features, including sparse selection, quantile models,
and tree ensembles, are described with their assumptions and examples in
:doc:`../models/index`. For held-out quantile coverage and between-fit
uncertainty, see :doc:`../notebooks/04_distributional_models`.
