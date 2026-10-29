NAMpy: Interpretable Additive Models
====================================

.. image:: _static/logo.png
   :alt: NAMpy — Interpretable Additive Modeling
   :class: landing-logo no-scaled-link
   :align: center
   :width: 100%

NAMpy combines statistical and neural additive models in one Python workflow.
Fit nonlinear feature effects, inspect their contributions, and model changing
response distributions through a scikit-learn-style estimator interface.

Its statistical engine reimplements a broad range of mgcv methods in Python,
including smoothness selection, inference, and diagnostics, alongside
SCAM-style shape constraints. Fitting and prediction do not require R.

:doc:`Coming from R’s mgcv or scam? See compatibility and interface differences. <compatibility>`

Quick installation
------------------

.. code-block:: bash

   pip install nampy

For Python compatibility, installation checks, and GPU setup, see the :doc:`FAQ <faq>`.

For offline reading, :doc:`download the PDF or EPUB manual <downloads>`.

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

If you are new to NAMpy, start with
:doc:`your first additive models <notebooks/01_getting_started>`.
Use the guides below to choose a model, develop your analysis, or
look up a specific method.

.. grid:: 1 1 2 2
   :gutter: 2
   :class-container: documentation-navigation

   .. grid-item-card:: Learn with examples
      :link: tutorials/index
      :link-type: doc

      Follow complete analyses with runnable code, fitted plots,
      and explanations of the results.

   .. grid-item-card:: Choose a model
      :link: models/index
      :link-type: doc

      Compare statistical and neural architectures by their supported
      tasks, interactions, and interpretation.

   .. grid-item-card:: Build a GAM formula
      :link: user_guide/formula_visual_reference
      :link-type: doc

      See how smooths, group effects, interactions, and offsets
      translate modeling assumptions into formulas.

   .. grid-item-card:: Interpret fitted effects
      :link: user_guide/interpretability
      :link-type: doc

      Extract and plot contributions, choose a centering reference,
      and understand what each display represents.

   .. grid-item-card:: Solve a specific task
      :link: howto/index
      :link-type: doc

      Find focused instructions for preprocessing, training,
      constraints, prediction, and saving models.

   .. grid-item-card:: Browse the API
      :link: api/index
      :link-type: doc

      Look up estimator classes, method arguments, return values,
      and statistical and neural interfaces.

.. toctree::
   :caption: Getting Started
   :maxdepth: 2
   :hidden:

   Tutorials <tutorials/index>
   Model guide <models/index>
   faq
   Download documentation <downloads>

.. toctree::
   :caption: How-to Guides
   :maxdepth: 2
   :hidden:

   Fit, predict and diagnose <user_guide/model_workflow>
   Prepare feature data <user_guide/preprocessing>
   Train and tune neural models <user_guide/training>
   Build a GAM formula <user_guide/formula_visual_reference>
   Specify and diagnose a statistical GAM <user_guide/statistical_gams>
   Choose an API and response family <user_guide/api_and_families>
   Model location and spread <user_guide/gamlss>
   Fit survival models <user_guide/survival>
   Calibrate prediction intervals <user_guide/calibration>
   Calibrate quantile-function inference <user_guide/quantile_inference>
   Compose spline and neural effects <user_guide/semistructured>
   Add an interaction <howto/interactions>
   Constrain the shape of an effect <user_guide/shape_constrained_gams>
   Predict with an offset <howto/offsets>
   Interpret and plot effects <user_guide/interpretability>
   Inspect Fisher information <user_guide/fisher_diagnostics>
   Choose a prediction output <howto/prediction_outputs>
   Fit a regression, classification, or distributional model <examples/fit_models>
   Save and reload a model <howto/persistence>
   Build a custom neural model <user_guide/custom_models>

.. toctree::
   :caption: Reference
   :maxdepth: 3
   :hidden:

   statistical_reference/index
   neural_reference/index
   Predictions and explanations <api/contracts>
   Plotting <api/plotting>
   Extension APIs <extension_reference/index>
   Research references <references>
   Compatibility and limitations <compatibility>

.. toctree::
   :caption: Development
   :maxdepth: 1
   :hidden:

   contributing
   architecture
   Validation reports <development/index>
   license
