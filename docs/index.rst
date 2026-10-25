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

Quick installation
------------------

.. code-block:: bash

   pip install nampy

For Python compatibility, installation checks, and GPU setup, see the :doc:`FAQ <faq>`.

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

Start with :doc:`your first additive models <notebooks/01_getting_started>`,
then :doc:`a real-data classification workflow <notebooks/05_real_data_classification>`.
Choose the statistical, neural, or distributional branch in :doc:`tutorials/index`.

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Fit your first GAM
      :link: tutorials/first_gam
      :link-type: doc

      Follow one nonlinear dataset through fitting, held-out evaluation,
      effect interpretation, and diagnostics.

   .. grid-item-card:: Read formulas visually
      :link: user_guide/formula_visual_reference
      :link-type: doc

      Match smooths, cycles, group curves, random effects, tensor products,
      and offsets to fitted examples.

   .. grid-item-card:: Interpret additive effects
      :link: user_guide/interpretability
      :link-type: doc

      Understand centering and term dependence, then inspect components
      and plot feature effects.

   .. grid-item-card:: Work through tutorials
      :link: tutorials/index
      :link-type: doc

      Explore question-led analyses with executable code and fitted plots.

   .. grid-item-card:: Look up an exact symbol
      :link: api/models
      :link-type: doc

      Find constructors, methods, defaults, return values, and extension
      interfaces.

   .. grid-item-card:: Explore detailed examples
      :link: in_depth/index
      :link-type: doc

      Work through longer statistical, neural, and distributional analyses.

   .. grid-item-card:: Choose a model
      :link: models/index
      :link-type: doc

      Compare model capabilities and find the estimator for your analysis.

   .. grid-item-card:: Find a practical recipe
      :link: howto/index
      :link-type: doc

      Find focused guides for preprocessing, fitting, interpretation, and prediction.

Contributors
------------

.. raw:: html

   <div class="contributors-grid" id="github-contributors">
   <a class="contributor-profile" href="https://github.com/Ananyapam7"><img src="https://avatars.githubusercontent.com/u/39508435?v=4&amp;s=96" alt="" width="64" height="64" loading="lazy"><span>Ananyapam7</span></a>
   <a class="contributor-profile" href="https://github.com/AnFreTh"><img src="https://avatars.githubusercontent.com/u/51477933?v=4&amp;s=96" alt="" width="64" height="64" loading="lazy"><span>AnFreTh</span></a>
   <a class="contributor-profile" href="https://github.com/22GC22"><img src="https://avatars.githubusercontent.com/u/80445708?v=4&amp;s=96" alt="" width="64" height="64" loading="lazy"><span>22GC22</span></a>
   <a class="contributor-profile" href="https://github.com/eltociear"><img src="https://avatars.githubusercontent.com/u/22633385?v=4&amp;s=96" alt="" width="64" height="64" loading="lazy"><span>eltociear</span></a>
   </div>

`All contributors on GitHub <https://github.com/OpenTabular/NAMpy/graphs/contributors>`_.

.. toctree::
   :caption: Getting Started
   :maxdepth: 2
   :hidden:

   Tutorials <tutorials/index>
   Advanced examples <in_depth/index>
   faq

.. toctree::
   :caption: How-to Guides
   :maxdepth: 2
   :hidden:

   Prepare feature data <user_guide/preprocessing>
   Train and tune neural models <user_guide/training>
   Specify and diagnose a statistical GAM <user_guide/statistical_gams>
   Model location and spread <user_guide/gamlss>
   Add an interaction <howto/interactions>
   Select interactions <howto/interaction_selection>
   Impose monotonicity <howto/monotonicity>
   Choose other shape constraints <user_guide/shape_constrained_gams>
   Predict with an offset <howto/offsets>
   Interpret and plot effects <user_guide/interpretability>
   Choose a prediction output <howto/prediction_outputs>
   Save and reload a model <howto/persistence>
   Quick regression recipe <examples/basic_regression>
   Quick classification recipe <examples/basic_classification>
   Quick distributional recipe <examples/distributional_regression>
   Build a custom neural model <user_guide/custom_models>
   Check pretab compatibility <user_guide/pretab_compatibility>

.. toctree::
   :caption: Reference
   :maxdepth: 2
   :hidden:

   models/index
   user_guide/formula_visual_reference
   api/models
   api/gam
   api/gam_formula
   api/gam_smooths
   api/gam_families
   api/gam_diagnostics
   api/gam_inference
   api/gam_results
   api/gam_advanced
   api/basemodels
   api/configs
   api/utils
   api/neural
   api/contracts
   api/plotting

.. toctree::
   :caption: Development
   :maxdepth: 1
   :hidden:

   compatibility
   contributing
   development/validation_reports
   architecture
   development/reference_sources
   development/mgcv_optimizer_boundaries
   license
