User guide
==========

The user guide explains modeling decisions: what an additive effect means,
which approach fits a scientific question, how to assess a fitted model, and
which limitations matter. Complete analyses live in :doc:`examples/index`;
exact signatures and return types live in :doc:`api/index`.

Foundations
-----------

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Additive predictors and interpretation
      :link: concepts/additive_predictors
      :link-type: doc

      Understand term contributions, centering, interactions, and what exact
      prediction reconstruction does—and does not—guarantee.

   .. grid-item-card:: Smoothness and basis dimension
      :link: concepts/smoothness
      :link-type: doc

      Learn how basis capacity and penalties jointly determine a fitted curve.

Modeling approaches
-------------------

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Statistical GAMs
      :link: user_guide/statistical_models
      :link-type: doc

      Choose families, smooth bases, fitting controls, inference, and
      diagnostics.

   .. grid-item-card:: Neural additive models
      :link: user_guide/neural_models
      :link-type: doc

      Learn the shared workflow, then compare specialized neural
      representations.

   .. grid-item-card:: Shape-constrained effects
      :link: user_guide/shape_constrained_gams
      :link-type: doc

      Include defensible monotonicity, curvature, positivity, or bivariate
      constraints during fitting.

   .. grid-item-card:: Distributional modeling
      :link: user_guide/gamlss
      :link-type: doc

      Model location, scale, and shape parameters with separate additive
      predictors.

Working with fitted models
--------------------------

Use :doc:`user_guide/interpretability` to inspect and reconstruct named
contributions. Data preparation, training controls, and custom extensions are
covered after the modeling guides so that implementation details do not block
a first analysis.

.. toctree::
   :maxdepth: 2
   :hidden:

   concepts/additive_predictors
   concepts/smoothness
   user_guide/statistical_models
   user_guide/statistical_gams
   user_guide/neural_models
   models/index
   user_guide/shape_constrained_gams
   user_guide/gamlss
   concepts/distributional_regression
   user_guide/interpretability
   user_guide/preprocessing
   user_guide/training
   user_guide/custom_models
   user_guide/pretab_compatibility
