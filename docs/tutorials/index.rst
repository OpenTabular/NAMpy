Tutorials
=========

The tutorials work through complete modeling examples, from fitting a first additive model to interpreting interactions, response distributions, and uncertainty. Start with your first additive models, or choose an example that matches your data.

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

   .. grid-item-card:: Fit group-specific curves with partial pooling
      :link: ../notebooks/group_smooths
      :link-type: doc

      Compare group shifts, separate curves, and partial pooling when groups have unequal sample sizes.

      :download:`Download notebook <../notebooks/group_smooths.ipynb>`

   .. grid-item-card:: Choose smoothness and shrink unnecessary terms
      :link: ../notebooks/smoothness_selection
      :link-type: doc

      Separate basis capacity from roughness penalties and compare two ways to shrink irrelevant effects.

      :download:`Download notebook <../notebooks/smoothness_selection.ipynb>`

   .. grid-item-card:: Estimate rates of change and differences between predictions
      :link: ../notebooks/derivatives_contrasts
      :link-type: doc

      Calculate derivatives and prediction contrasts with pointwise covariance-based uncertainty.

      :download:`Download notebook <../notebooks/derivatives_contrasts.ipynb>`

   .. grid-item-card:: Use GAM diagnostics to improve a model
      :link: ../notebooks/gam_diagnostics
      :link-type: doc

      Refit after capacity and interaction checks, and examine ambiguity among correlated effects.

      :download:`Download notebook <../notebooks/gam_diagnostics.ipynb>`

   .. grid-item-card:: Model interactions with tensor-product smooths
      :link: ../notebooks/tensor_interactions
      :link-type: doc

      Distinguish isolated joint contributions from total surfaces and conditional predictions.

      :download:`Download notebook <../notebooks/tensor_interactions.ipynb>`

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

   .. grid-item-card:: Select interactions from validation data
      :link: interaction_selection
      :link-type: doc

      Inspect validation residuals, select joint effects, and compare predictions on held-out test rows.

      :notebook-download:`interaction_selection`

   .. grid-item-card:: Model counts and changing variance
      :link: ../notebooks/02_statistical_gams
      :link-type: doc

      Account for exposure in count data, and fit separate predictors for a response's mean and standard deviation.

      :download:`Download notebook <../notebooks/02_statistical_gams.ipynb>`

   .. grid-item-card:: Model overdispersed counts
      :link: ../notebooks/overdispersed_counts
      :link-type: doc

      Compare Poisson and estimated negative-binomial fits with a shared exposure-adjusted mean structure.

      :download:`Download notebook <../notebooks/overdispersed_counts.ipynb>`

   .. grid-item-card:: Model positive right-skewed responses
      :link: ../notebooks/gamma_response
      :link-type: doc

      Fit a Gamma mean model to positive outcomes and inspect variance-scaled residuals.

      :download:`Download notebook <../notebooks/gamma_response.ipynb>`

   .. grid-item-card:: Model proportions strictly between zero and one
      :link: ../notebooks/beta_response
      :link-type: doc

      Use a beta response distribution for continuous proportions and assess held-out predictions.

      :download:`Download notebook <../notebooks/beta_response.ipynb>`

   .. grid-item-card:: Model zero and positive continuous responses
      :link: ../notebooks/tweedie_response
      :link-type: doc

      Fit compound Poisson–Gamma outcomes without replacing zeros.

      :download:`Download notebook <../notebooks/tweedie_response.ipynb>`

   .. grid-item-card:: Predict ordered class probabilities
      :link: ../notebooks/ordered_response
      :link-type: doc

      Inspect probabilities for ranked response categories with an ordered GAM.

      :download:`Download notebook <../notebooks/ordered_response.ipynb>`

   .. grid-item-card:: Model changing mean and dispersion in positive responses
      :link: ../notebooks/gamma_location_scale
      :link-type: doc

      Separate the Gamma mean, coefficient of variation, and conditional standard deviation.

      :download:`Download notebook <../notebooks/gamma_location_scale.ipynb>`

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

   .. grid-item-card:: Smooth responses over neighboring regions
      :link: ../notebooks/regional_smoothing
      :link-type: doc

      Compare graph-based MRF smoothing with independent regional effects.

      :download:`Download notebook <../notebooks/regional_smoothing.ipynb>`

   .. grid-item-card:: Fit a GAM with correlated Gaussian errors
      :link: ../notebooks/correlated_errors
      :link-type: doc

      Supply known AR(1) correlation and inspect residuals within independent series.

      :download:`Download notebook <../notebooks/correlated_errors.ipynb>`

   .. grid-item-card:: Choose smoothness with neighborhood cross-validation
      :link: ../notebooks/neighborhood_cv
      :link-type: doc

      Define excluded neighborhoods and compare predictions on separate test rows.

      :download:`Download notebook <../notebooks/neighborhood_cv.ipynb>`

   .. grid-item-card:: Allow smoothness to vary across a predictor
      :link: ../notebooks/adaptive_smoothing
      :link-type: doc

      Compare a single roughness penalty with an adaptive fit in smooth and irregular regions.

      :download:`Download notebook <../notebooks/adaptive_smoothing.ipynb>`

   .. grid-item-card:: Fit a smooth on the sphere
      :link: ../notebooks/spherical_smoothing
      :link-type: doc

      Fit globally covered locations and examine longitude seam and pole behavior.

      :download:`Download notebook <../notebooks/spherical_smoothing.ipynb>`

   .. grid-item-card:: Model weighted functional inputs
      :link: ../notebooks/functional_inputs
      :link-type: doc

      Learn a coefficient function for weighted trajectories and predict held-out responses.

      :download:`Download notebook <../notebooks/functional_inputs.ipynb>`

   .. grid-item-card:: Fit a convex effect and inspect its curvature
      :link: ../notebooks/curvature_constraints
      :link-type: doc

      Compare ordinary and convex P-splines and inspect the fitted curvature.

      :download:`Download notebook <../notebooks/curvature_constraints.ipynb>`

.. toctree::
   :maxdepth: 1
   :hidden:

   Your first additive models <../notebooks/01_getting_started>
   Classification with missing values and categorical features <../notebooks/05_real_data_classification>
   Fit and diagnose a GAM <first_gam>
   Fit group-specific curves with partial pooling <../notebooks/group_smooths>
   Choose smoothness and shrink unnecessary terms <../notebooks/smoothness_selection>
   Estimate rates of change and differences between predictions <../notebooks/derivatives_contrasts>
   Use GAM diagnostics to improve a model <../notebooks/gam_diagnostics>
   Model interactions with tensor-product smooths <../notebooks/tensor_interactions>
   Compare GAM and NAM feature effects <gam_nam>
   Compare neural additive architectures <../notebooks/03_neural_models>
   Select interactions from validation data <interaction_selection>
   Model counts and changing variance <../notebooks/02_statistical_gams>
   Model overdispersed counts <../notebooks/overdispersed_counts>
   Model positive right-skewed responses <../notebooks/gamma_response>
   Model proportions strictly between zero and one <../notebooks/beta_response>
   Model zero and positive continuous responses <../notebooks/tweedie_response>
   Predict ordered class probabilities <../notebooks/ordered_response>
   Model changing mean and dispersion in positive responses <../notebooks/gamma_location_scale>
   Compare distributions, quantiles, and ensemble spread <../notebooks/04_distributional_models>
   Fit and assess conditional quantiles <../in_depth/quantile_calibration>
   Recover step-shaped effects with additive trees <../in_depth/tree_effects>
   Smooth responses over neighboring regions <../notebooks/regional_smoothing>
   Fit a GAM with correlated Gaussian errors <../notebooks/correlated_errors>
   Choose smoothness with neighborhood cross-validation <../notebooks/neighborhood_cv>
   Allow smoothness to vary across a predictor <../notebooks/adaptive_smoothing>
   Fit a smooth on the sphere <../notebooks/spherical_smoothing>
   Model weighted functional inputs <../notebooks/functional_inputs>
   Fit a convex effect and inspect its curvature <../notebooks/curvature_constraints>
