Statistical modeling with GAMs
==============================

This track is for statisticians, R users, and researchers who want the penalized-spline workflow: basis dimension, roughness penalties, REML and other selection criteria, response families, uncertainty, shape constraints, and distributional regression.

:doc:`Coming from R's mgcv </from_mgcv>` puts a familiar ``gam`` call beside ``GAMRegressor`` and compares a committed mgcv snapshot with a Python refit. Use that page whenever a formula, EDF, or smoothing criterion needs to be checked against R. The tutorials below are the modeling sequence around that translation.

The first tutorial separates basis capacity from the penalty, including a predictor that should shrink away.

Smoothness, penalties, and selection
-------------------------------------

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Which parts of a curve should the penalty keep?
      :link: ../notebooks/smoothness_selection
      :link-type: doc

      Compare a small basis, stronger smoothing, and penalties that can shrink an irrelevant term.

      :download:`Download notebook <../notebooks/smoothness_selection.ipynb>`

   .. grid-item-card:: Choose smoothness with neighborhood cross-validation
      :link: ../notebooks/neighborhood_cv
      :link-type: doc

      Exclude neighboring rows together when nearby observations share information.

      :download:`Download notebook <../notebooks/neighborhood_cv.ipynb>`

   .. grid-item-card:: Allow smoothness to vary across a predictor
      :link: ../notebooks/adaptive_smoothing
      :link-type: doc

      Compare one roughness penalty with an adaptive fit in smooth and irregular regions.

      :download:`Download notebook <../notebooks/adaptive_smoothing.ipynb>`

Response families
-----------------

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Model counts and changing variance
      :link: ../notebooks/02_statistical_gams
      :link-type: doc

      Enter exposure as an offset, then fit a distributional model whose standard deviation depends on a predictor.

      :download:`Download notebook <../notebooks/02_statistical_gams.ipynb>`

   .. grid-item-card:: Model overdispersed counts
      :link: ../notebooks/overdispersed_counts
      :link-type: doc

      Compare Poisson and negative-binomial fits that share a mean structure and a log-exposure offset.

      :download:`Download notebook <../notebooks/overdispersed_counts.ipynb>`

   .. grid-item-card:: Model positive right-skewed responses
      :link: ../notebooks/gamma_response
      :link-type: doc

      Fit a Gamma mean model and inspect variance-scaled residuals.

      :download:`Download notebook <../notebooks/gamma_response.ipynb>`

   .. grid-item-card:: Model proportions strictly between zero and one
      :link: ../notebooks/beta_response
      :link-type: doc

      Use a beta response for continuous proportions and score held-out predictions.

      :download:`Download notebook <../notebooks/beta_response.ipynb>`

   .. grid-item-card:: Model zero and positive continuous responses
      :link: ../notebooks/tweedie_response
      :link-type: doc

      Fit a compound Poisson–Gamma mean without replacing exact zeros.

      :download:`Download notebook <../notebooks/tweedie_response.ipynb>`

   .. grid-item-card:: Predict ordered class probabilities
      :link: ../notebooks/ordered_response
      :link-type: doc

      Fit an ordered categorical GAM and read the probability of each ranked class.

      :download:`Download notebook <../notebooks/ordered_response.ipynb>`

Uncertainty and distributional regression
------------------------------------------

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Estimate rates of change and differences between predictions
      :link: ../notebooks/derivatives_contrasts
      :link-type: doc

      Calculate derivatives and contrasts with pointwise covariance-based uncertainty.

      :download:`Download notebook <../notebooks/derivatives_contrasts.ipynb>`

   .. grid-item-card:: Estimate a conditional quantile and quantify its uncertainty
      :link: ../notebooks/quantile_uncertainty
      :link-type: doc

      Separate a QuantileGAM median, a confidence band for that median, and a conformal interval for a future response.

      :download:`Download notebook <../notebooks/quantile_uncertainty.ipynb>`

   .. grid-item-card:: Model changing mean and dispersion in positive responses
      :link: ../notebooks/gamma_location_scale
      :link-type: doc

      Give a Gamma mean and coefficient of variation their own predictors.

      :download:`Download notebook <../notebooks/gamma_location_scale.ipynb>`

   .. grid-item-card:: Model time to an event with censoring
      :link: ../notebooks/survival_censoring
      :link-type: doc

      Fit one Weibull survival model and score censored follow-up on held-out subjects.

      :download:`Download notebook <../notebooks/survival_censoring.ipynb>`

The :ref:`GAMLSS guide <howto-user-guide-gamlss>` is the reference for multi-parameter families. :ref:`Shape-constrained GAMs <howto-user-guide-shape-constrained-gams>` shows how a monotone basis is imposed during fitting. The :doc:`diagnostic tutorial <../notebooks/gam_diagnostics>` in the first track is the worked revision loop: basis capacity, a missing interaction, and concurvity.

Structure that is still a GAM
-----------------------------

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Model interactions with tensor-product smooths
      :link: ../notebooks/tensor_interactions
      :link-type: doc

      Separate an isolated interaction from the full predicted surface.

      :download:`Download notebook <../notebooks/tensor_interactions.ipynb>`

   .. grid-item-card:: Fit group-specific curves with partial pooling
      :link: ../notebooks/group_smooths
      :link-type: doc

      Compare group shifts, separate curves, and partial pooling when sample sizes differ.

      :download:`Download notebook <../notebooks/group_smooths.ipynb>`

   .. grid-item-card:: Fit a convex effect and inspect its curvature
      :link: ../notebooks/curvature_constraints
      :link-type: doc

      Compare an ordinary P-spline with a convex smooth and inspect the second derivative.

      :download:`Download notebook <../notebooks/curvature_constraints.ipynb>`

   .. grid-item-card:: Fit a GAM with correlated Gaussian errors
      :link: ../notebooks/correlated_errors
      :link-type: doc

      Supply a known AR(1) correlation and mark the starts of independent series.

      :download:`Download notebook <../notebooks/correlated_errors.ipynb>`

   .. grid-item-card:: Smooth responses over neighboring regions
      :link: ../notebooks/regional_smoothing
      :link-type: doc

      Compare a Markov random field with independent regional effects.

      :download:`Download notebook <../notebooks/regional_smoothing.ipynb>`

   .. grid-item-card:: Fit a smooth on the sphere
      :link: ../notebooks/spherical_smoothing
      :link-type: doc

      Fit locations on a globe and check the longitude seam and the poles.

      :download:`Download notebook <../notebooks/spherical_smoothing.ipynb>`

   .. grid-item-card:: Model weighted functional inputs
      :link: ../notebooks/functional_inputs
      :link-type: doc

      Learn a coefficient function for weighted trajectories.

      :download:`Download notebook <../notebooks/functional_inputs.ipynb>`

.. toctree::
   :maxdepth: 1
   :hidden:

   Which parts of a curve should the penalty keep? <../notebooks/smoothness_selection>
   Choose smoothness with neighborhood cross-validation <../notebooks/neighborhood_cv>
   Allow smoothness to vary across a predictor <../notebooks/adaptive_smoothing>
   Model counts and changing variance <../notebooks/02_statistical_gams>
   Model overdispersed counts <../notebooks/overdispersed_counts>
   Model positive right-skewed responses <../notebooks/gamma_response>
   Model proportions strictly between zero and one <../notebooks/beta_response>
   Model zero and positive continuous responses <../notebooks/tweedie_response>
   Predict ordered class probabilities <../notebooks/ordered_response>
   Estimate rates of change and differences between predictions <../notebooks/derivatives_contrasts>
   Estimate a conditional quantile and quantify its uncertainty <../notebooks/quantile_uncertainty>
   Model changing mean and dispersion in positive responses <../notebooks/gamma_location_scale>
   Model time to an event with censoring <../notebooks/survival_censoring>
   Model interactions with tensor-product smooths <../notebooks/tensor_interactions>
   Fit group-specific curves with partial pooling <../notebooks/group_smooths>
   Fit a convex effect and inspect its curvature <../notebooks/curvature_constraints>
   Fit a GAM with correlated Gaussian errors <../notebooks/correlated_errors>
   Smooth responses over neighboring regions <../notebooks/regional_smoothing>
   Fit a smooth on the sphere <../notebooks/spherical_smoothing>
   Model weighted functional inputs <../notebooks/functional_inputs>
