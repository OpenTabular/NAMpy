User Guide
==========

Additive models describe a transformed prediction as a sum of named effects,

.. math::

   \eta(x) = \beta_0 + \sum_j f_j(x_j)
   + \sum_{(j,k) \in \mathcal I} f_{jk}(x_j, x_k).

NAMpy offers two ways to estimate these effects. Statistical GAMs represent
them with penalized smooth bases and support inference and diagnostics. Neural
additive models learn them with neural networks. Shape constraints extend the
statistical smooths, while distributional regression lets either modeling
approach describe more than a conditional mean.

Choose the topic that matches the analysis you are building.

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Statistical GAMs
      :link: user_guide/statistical_gams
      :link-type: doc

      Specify formulas and smooth bases, select smoothness, quantify
      uncertainty, and diagnose a fitted model.

   .. grid-item-card:: Neural additive models
      :link: user_guide/neural_models
      :link-type: doc

      Understand neural feature functions, preprocessing, training, and the
      available interaction structures.

   .. grid-item-card:: Shape constraints
      :link: user_guide/shape_constrained_gams
      :link-type: doc

      Impose defensible monotonicity, curvature, positivity, or bivariate
      shape assumptions.

   .. grid-item-card:: Distributional regression
      :link: user_guide/gamlss
      :link-type: doc

      Give location, scale, or shape parameters their own additive predictors.

   .. grid-item-card:: Interpreting effects
      :link: user_guide/interpretability
      :link-type: doc

      Read centered contributions, uncertainty bands, density shading, and
      interaction surfaces without confusing reconstruction with
      identifiability.

.. toctree::
   :maxdepth: 1
   :hidden:

   user_guide/statistical_gams
   user_guide/neural_models
   user_guide/shape_constrained_gams
   user_guide/gamlss
   user_guide/interpretability

Answers to shorter setup and modeling questions are collected in the
:doc:`frequently asked questions <faq>`.
