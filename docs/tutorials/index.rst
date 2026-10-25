Tutorials
=========

These tutorials follow additive models from fitting and interpretation to interactions, classification, and distributional prediction. Each example includes the code, fitted results, and an explanation of what the model has learned.

If you are new to NAMpy, start with Your first additive models. Otherwise, choose the problem closest to your own data.

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Your first additive models
      :link: ../notebooks/01_getting_started
      :link-type: doc

      Fit a GAM and a NAM to the same data, compare their predictions, and inspect the contribution of each feature.

      :download:`Download notebook <../notebooks/01_getting_started.ipynb>`

   .. grid-item-card:: Classification with missing values and categorical features
      :link: ../notebooks/05_real_data_classification
      :link-type: doc

      Prepare a real dataset, compare a NAM with logistic regression, examine predicted probabilities, and save the complete prediction workflow.

      :download:`Download notebook <../notebooks/05_real_data_classification.ipynb>`

   .. grid-item-card:: Fit and diagnose a GAM
      :link: first_gam
      :link-type: doc

      Model energy demand with temperature and a daily cycle, then inspect residuals and the flexibility of the fitted smooths.

      :notebook-download:`first_gam`

   .. grid-item-card:: GAMs with groups, offsets, and changing variance
      :link: ../notebooks/02_statistical_gams
      :link-type: doc

      Choose a basis dimension, account for exposure in count data, and fit separate predictors for a response's mean and standard deviation.

      :download:`Download notebook <../notebooks/02_statistical_gams.ipynb>`

   .. grid-item-card:: Compare GAM and NAM feature effects
      :link: gam_nam
      :link-type: doc

      Compare spline and neural representations on a synthetic dataset with known feature effects.

      :notebook-download:`gam_nam`

   .. grid-item-card:: Compare neural additive architectures
      :link: ../notebooks/03_neural_models
      :link-type: doc

      Compare independent feature networks with shared bases, then examine a model that also includes a global transformer contribution.

      :download:`Download notebook <../notebooks/03_neural_models.ipynb>`

   .. grid-item-card:: Model a changing response distribution
      :link: changing_variability
      :link-type: doc

      Fit a normal distribution whose mean and standard deviation both depend on the input.

      :notebook-download:`changing_variability`

   .. grid-item-card:: Compare distributions, quantiles, and ensemble spread
      :link: ../notebooks/04_distributional_models
      :link-type: doc

      Compare prediction intervals from a normal model and a quantile model, and distinguish them from disagreement between fitted models.

      :download:`Download notebook <../notebooks/04_distributional_models.ipynb>`

   .. grid-item-card:: Fit and assess conditional quantiles
      :link: ../in_depth/quantile_calibration
      :link-type: doc

      Fit the 10th, 50th, and 90th percentiles and examine their coverage and feature effects.

      :notebook-download:`../in_depth/quantile_calibration`

   .. grid-item-card:: Recover step-shaped effects with additive trees
      :link: ../in_depth/tree_effects
      :link-type: doc

      Compare a single additive tree with a jointly trained ensemble on numerical thresholds and categorical shifts.

      :notebook-download:`../in_depth/tree_effects`

   .. grid-item-card:: Four modeling examples
      :link: modeling_workflows
      :link-type: doc

      Run compact GAM, GAMLSS, NAM, and neural distributional examples and inspect their outputs.

      :notebook-download:`modeling_workflows`

.. toctree::
   :maxdepth: 1
   :hidden:

   Your first additive models <../notebooks/01_getting_started>
   Classification with missing values and categorical features <../notebooks/05_real_data_classification>
   Fit and diagnose a GAM <first_gam>
   GAMs with groups, offsets, and changing variance <../notebooks/02_statistical_gams>
   Compare GAM and NAM feature effects <gam_nam>
   Compare neural additive architectures <../notebooks/03_neural_models>
   Model a changing response distribution <changing_variability>
   Compare distributions, quantiles, and ensemble spread <../notebooks/04_distributional_models>
   Fit and assess conditional quantiles <../in_depth/quantile_calibration>
   Recover step-shaped effects with additive trees <../in_depth/tree_effects>
   Four modeling examples <modeling_workflows>
