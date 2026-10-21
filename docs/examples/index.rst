Examples
========

This gallery keeps every worked example in one place. **Tutorials** develop a
complete analysis, **recipes** solve one focused task, and **extended
notebooks** survey a broader part of the package.

Tutorials
---------

.. grid:: 1 2 3 3
   :gutter: 2

   .. grid-item-card:: Tutorial — Fit and diagnose a GAM
      :link: ../tutorials/first_gam
      :link-type: doc

      Estimate nonlinear effects, evaluate held-out predictions, and check
      basis dimensions.

      .. image:: ../_static/first_gam_effects.png
         :alt: Smooth temperature and cyclic hour effects

   .. grid-item-card:: Tutorial — Encode monotonicity
      :link: ../tutorials/monotone_effect
      :link-type: doc

      Compare an unconstrained smooth with a justified monotone effect.

      .. image:: ../_static/shape_constraint_comparison.png
         :alt: Unconstrained and monotone smooths

   .. grid-item-card:: Tutorial — Model a periodic effect
      :link: ../tutorials/periodic_effect
      :link-type: doc

      Use a cyclic basis when both ends represent the same boundary.

   .. grid-item-card:: Tutorial — Estimate an interaction surface
      :link: ../tutorials/interaction_surface
      :link-type: doc

      Recover a joint spatial pattern that separate main effects miss.

      .. image:: ../_static/interaction_surface.png
         :alt: Tensor-product interaction surface

   .. grid-item-card:: Tutorial — Compare NAM and NBM
      :link: ../tutorials/nam_nbm
      :link-type: doc

      Compare two neural representations on the same data and scale.

      .. image:: ../_static/nam_nbm_comparison.png
         :alt: NAM and NBM feature effects

   .. grid-item-card:: Tutorial — Model changing variability
      :link: ../tutorials/changing_variability
      :link-type: doc

      Estimate conditional location and spread together.

      .. image:: ../_static/changing_variability.png
         :alt: Distributional feature effect and conditional spread

Recipes
-------

.. grid:: 1 2 3 3
   :gutter: 2

   .. grid-item-card:: Recipe — Regression
      :link: basic_regression
      :link-type: doc

      Fit an additive regressor and generate continuous predictions.

   .. grid-item-card:: Recipe — Classification
      :link: basic_classification
      :link-type: doc

      Fit a classifier and inspect labels and class probabilities.

   .. grid-item-card:: Recipe — Distributional regression
      :link: distributional_regression
      :link-type: doc

      Predict distribution parameters rather than one conditional mean.

   .. grid-item-card:: Recipe — Add an interaction
      :link: ../howto/interactions
      :link-type: doc

      Add a tensor-product smooth or an explicit neural interaction.

   .. grid-item-card:: Recipe — Impose monotonicity
      :link: ../howto/monotonicity
      :link-type: doc

      Replace an unconstrained smooth with a constrained basis.

   .. grid-item-card:: Recipe — Predict with an offset
      :link: ../howto/offsets
      :link-type: doc

      Preserve a known link-scale contribution at fit and prediction time.

   .. grid-item-card:: Recipe — Save and reload a model
      :link: ../howto/persistence
      :link-type: doc

      Store a fitted estimator and reproduce its predictions.

Extended notebooks
------------------

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Extended notebook — Getting started
      :link: ../notebooks/01_getting_started
      :link-type: doc

      Fit models, inspect components, and save fitted estimators.

   .. grid-item-card:: Extended notebook — Statistical GAMs
      :link: ../notebooks/02_statistical_gams
      :link-type: doc

      Survey formulas, bases, inference, diagnostics, and constraints.

   .. grid-item-card:: Extended notebook — Neural models
      :link: ../notebooks/03_neural_models
      :link-type: doc

      Compare feature functions and interaction structures.

   .. grid-item-card:: Extended notebook — Distributional models
      :link: ../notebooks/04_distributional_models
      :link-type: doc

      Survey classification, distributional, quantile, and ensemble APIs.

.. toctree::
   :maxdepth: 1
   :hidden:

   ../tutorials/first_gam
   ../tutorials/monotone_effect
   ../tutorials/periodic_effect
   ../tutorials/interaction_surface
   ../tutorials/nam_nbm
   ../tutorials/changing_variability
   basic_regression
   basic_classification
   distributional_regression
   ../howto/interactions
   ../howto/monotonicity
   ../howto/offsets
   ../howto/persistence
   ../notebooks/01_getting_started
   ../notebooks/02_statistical_gams
   ../notebooks/03_neural_models
   ../notebooks/04_distributional_models
