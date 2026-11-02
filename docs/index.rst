NAMpy: Interpretable Additive Models
====================================

.. image:: _static/logo.png
   :alt: NAMpy — Interpretable Additive Modeling
   :class: landing-logo no-scaled-link
   :align: center
   :width: 100%

NAMpy brings serious statistical additive modeling and deep learning
interpretability into a unified scikit-learn workflow. Fit complex
non-linear curves, model full response distributions, and run rigorous GAM
diagnostics, all in Python.

:doc:`Coming from R's mgcv? <from_mgcv>`

Quick installation
------------------

.. code-block:: bash

   pip install nampy

For Python compatibility, installation checks, and GPU setup, see the :doc:`FAQ <faq>`.

.. _download-the-manual:

`Download the reference manual (PDF) <https://github.com/OpenTabular/NAMpy/releases/download/docs-latest/NAMpy-reference.pdf>`__,
`the full PDF manual <https://github.com/OpenTabular/NAMpy/releases/download/docs-latest/NAMpy.pdf>`__,
or
`the EPUB manual <https://github.com/OpenTabular/NAMpy/releases/download/docs-latest/NAMpy.epub>`__.

Tutorials
---------

The tutorials are grouped into three tracks. Each track starts with a question and a fitted result, then moves into the tools that question needs.

.. grid:: 1 1 3 3
   :gutter: 2

   .. grid-item-card:: New to additive models
      :link: tutorials/track_new
      :link-type: doc
      :img-top: _static/track_intro.png
      :img-alt: Points from a sine-shaped mean, the least-squares line through those points, and the mean itself
      :class-img-top: track-card-figure

      Fit a first model, read the feature curves, diagnose a poor fit, and check predictions on unseen rows.

   .. grid-item-card:: Statistical modeling with GAMs
      :link: tutorials/track_statistical
      :link-type: doc
      :img-top: _static/track_statistical.png
      :img-alt: Equirectangular map of a latitude-longitude surface, the display used for a spherical smooth
      :class-img-top: track-card-figure

      Spline bases, penalties, REML, families, uncertainty, shape constraints, and distributional regression.

   .. grid-item-card:: Interpretable deep learning
      :link: tutorials/track_neural
      :link-type: doc
      :img-top: _static/track_neural.png
      :img-alt: Interaction heatmap of two features, with three slices of that surface at fixed values of the second feature
      :class-img-top: track-card-figure

      Compare NAM, NBM, trees, interaction selection, distributional neural regression, and custom architectures.

Additive models
---------------

The main-effects additive model builds its predictor from one effect per
feature. Interactions and contextual/global components are extensions of this
structure; their dependence must be interpreted separately:

.. math::

   \eta(x)=\beta_0+\sum_{j=1}^{p}f_j(x_j).

Here :math:`\beta_0` is the intercept and :math:`f_j` describes how feature
:math:`x_j` contributes. GAMs represent these effects with penalized splines;
neural additive models learn them with feature networks or other learned
functions. The inverse link maps the additive predictor to the conditional mean:

.. math::

   \mu(x)=\mathbb{E}[Y\mid x]=g^{-1}\!\left(\eta(x)\right).

For distributional regression, each distribution parameter gets its own
additive predictor:

.. math::

   \eta_k(x)=\beta_{0k}+\sum_{j=1}^{p}f_{jk}(x_j),
   \qquad \theta_k(x)=h_k^{-1}\!\left(\eta_k(x)\right).

.. math::

   Y\mid x\sim\mathcal D\!\left(\theta_1(x),\ldots,\theta_K(x)\right).

This lets quantities such as the conditional mean and spread vary with the
inputs. Contributions add on each predictor's link scale.

.. include:: /_includes/model_showcases.inc

More modeling examples
----------------------

For censored time-to-event outcomes, :doc:`model survival with censoring
<notebooks/survival_censoring>` and interpret fitted survival curves alongside
observed follow-up.

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Constrain a feature effect
      :link: tutorial-monotone
      :link-type: ref

      .. image:: _static/shape_constraint_comparison.png
         :alt: Comparison of unconstrained and monotone feature effects

      Encode a defensible monotonicity assumption during fitting.

   .. grid-item-card:: Model interacting features
      :link: tutorial-interaction
      :link-type: ref

      .. image:: _static/interaction_surface.png
         :alt: Joint spatial effect estimated by a tensor-product smooth

      Learn a joint surface that separate main effects cannot represent.

   .. grid-item-card:: Predict conditional quantiles
      :link: in_depth/quantile_calibration
      :link-type: doc

      .. image:: _static/detailed_quantile_median.png
         :alt: Learned and known conditional median in the QNAM example

      Compare quantile recovery, interval width, and held-out coverage.

   .. grid-item-card:: Recover step-shaped effects
      :link: in_depth/tree_effects
      :link-type: doc

      .. image:: _static/detailed_tree_effects.png
         :alt: Additive tree predictions and numerical and categorical effects

      Compare a single additive tree with a jointly trained tree ensemble.

Explore the documentation
-------------------------

If you are new to additive models, start with
:doc:`Discover nonlinear effects and check whether you can trust them <notebooks/01_getting_started>`.
Use the guides below to choose a model, develop your analysis, or
look up a specific method.

.. grid:: 1 1 2 2
   :gutter: 2
   :class-container: documentation-navigation

   .. grid-item-card:: Browse the tutorials
      :link: tutorials/index
      :link-type: doc

      Choose the track that matches your background, then work through its tutorials in order.

   .. grid-item-card:: Choose a model
      :link: models/index
      :link-type: doc

      Compare modeling approaches and find one suited to your response and scientific question.

   .. grid-item-card:: Specify smooths and interactions
      :link: user_guide/formula_visual_reference
      :link-type: doc

      Build formulas with smooth terms, group effects, interactions, and constraints.

   .. grid-item-card:: Understand fitted effects
      :link: user_guide/interpretability
      :link-type: doc

      Read additive contributions, distinguish prediction scales, and interpret uncertainty.

   .. grid-item-card:: Complete a specific task
      :link: howto/index
      :link-type: doc

      Find focused instructions for fitting, calibration, prediction, and persistence.

   .. grid-item-card:: Look up the API
      :link: api/index
      :link-type: doc

      Check parameters, return values, supported methods, and examples.

.. toctree::
   :caption: Getting Started
   :maxdepth: 3
   :hidden:

   Tutorials <tutorials/index>
   Model guide <models/index>
   faq
   Coming from R <from_mgcv>

.. toctree::
   :caption: How-to Guides
   :maxdepth: 2
   :hidden:

   Fit a first model <howto/fit_first_model>
   Fit and diagnose a GAM <howto/fit_and_diagnose>
   Read the fit <howto/read_the_fit>
   Calibrate predictions <howto/calibrate>
   Change the predictor <howto/change_predictor>
   Model location and spread <howto/location_and_spread>
   Fit survival models <howto/survival>
   Add a neural estimator <howto/neural>
   Save and reload a model <howto/persistence>

.. toctree::
   :caption: Reference
   :maxdepth: 3
   :hidden:

   statistical_reference/index
   neural_reference/index
   Predictions and explanations <api/contracts>
   Probability laws <api/distributions>
   Plotting <api/plotting>
   Calibration <api/calibration>
   Extension APIs <extension_reference/index>
   Bibliography <references>

.. toctree::
   :caption: Development
   :maxdepth: 1
   :hidden:

   contributing
   architecture
   Browse source code <development/source_code>
   license
