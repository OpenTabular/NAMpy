Interpretable deep learning
===========================

This track is for machine-learning researchers and developers who want additive structure with neural or tree representations. Prediction accuracy is reported, and so is the fitted effect. Similar scores are not treated as evidence that two models learned the same curves.

The first tutorial is the comparison to keep in mind: a GAM and a NAM can both predict well while their feature curves differ.

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Do a GAM and a NAM recover the same effects?
      :link: gam_nam
      :link-type: doc

      Compare spline and neural curves with four known effects, including saturation and diminishing returns.

      :notebook-download:`gam_nam`

   .. grid-item-card:: Compare neural additive architectures
      :link: ../notebooks/03_neural_models
      :link-type: doc

      Compare NAM and NBM feature curves and parameter counts, then inspect a NAMformer global contribution.

      :download:`Download notebook <../notebooks/03_neural_models.ipynb>`

   .. grid-item-card:: Select interactions from validation data
      :link: interaction_selection
      :link-type: doc

      Fit a main-effects NAM, then use SIAN to add interactions chosen on validation rows.

      :notebook-download:`interaction_selection`

   .. grid-item-card:: Recover step-shaped effects with additive trees
      :link: ../in_depth/tree_effects
      :link-type: doc

      Compare one additive tree with a jointly trained ensemble on thresholds and categorical shifts.

      :notebook-download:`../in_depth/tree_effects`

   .. grid-item-card:: Compare distributions, quantiles, and ensemble spread
      :link: ../notebooks/04_distributional_models
      :link-type: doc

      Compare a normal location-scale NAM with a quantile model, and separate that spread from disagreement among fitted NAMs.

      :download:`Download notebook <../notebooks/04_distributional_models.ipynb>`

   .. grid-item-card:: Fit and assess conditional quantiles
      :link: ../in_depth/quantile_calibration
      :link-type: doc

      Fit ordered QNAM quantiles and compare their coverage with the known conditional distribution.

      :notebook-download:`../in_depth/quantile_calibration`

   .. grid-item-card:: Combine structured effects with a neural remainder
      :link: ../notebooks/semistructured_remainder
      :link-type: doc

      Project the neural contribution out of the structured training design and check the decomposition on new rows.

      :download:`Download notebook <../notebooks/semistructured_remainder.ipynb>`

Architectures without a separate tutorial
-----------------------------------------

:doc:`NodeGAM </models/nodegam>` represents effects with differentiable oblivious trees and optional pairs. The tree tutorial above is TreeNAM and EnsembleTreeNAM, which are a different construction. :ref:`Custom models <howto-user-guide-custom-models>` shows how to register an architecture whose global branch is not a marginal effect. The :doc:`model guide </models/index>` lists the remaining estimators, including SNAM, SPAM, and NBM-SPAM.

The :doc:`breast-cancer classification </notebooks/05_real_data_classification>` in the first track is the measured-data NAM comparison with logistic regression.

.. toctree::
   :maxdepth: 1
   :hidden:

   Do a GAM and a NAM recover the same effects? <gam_nam>
   Compare neural additive architectures <../notebooks/03_neural_models>
   Select interactions from validation data <interaction_selection>
   Recover step-shaped effects with additive trees <../in_depth/tree_effects>
   Compare distributions, quantiles, and ensemble spread <../notebooks/04_distributional_models>
   Fit and assess conditional quantiles <../in_depth/quantile_calibration>
   Combine structured effects with a neural remainder <../notebooks/semistructured_remainder>
