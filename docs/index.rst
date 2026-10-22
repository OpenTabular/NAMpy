NAMpy: Additive Modeling in Python
==================================

.. image:: _static/logo-mark.svg
   :alt: NAMpy logo
   :class: brand-mark
   :align: center
   :width: 108px

NAMpy brings statistical GAMs, shape-constrained smooths, neural additive
models, and distributional regression into one Python library. Fit models with
formulas or estimator classes, then inspect the named effects that make up
each prediction.

.. button-ref:: quickstart
   :color: primary

   Get started

.. button-ref:: compatibility
   :outline:

   Coming from mgcv or scam?

.. code-block:: bash

   pip install nampy

What the model learned
----------------------

This fitted statistical GAM describes demand in a synthetic energy dataset.
The curves are estimated feature effects; the shaded regions show uncertainty.
Temperature has a U-shaped association with demand, while the cyclic smooth
joins the beginning and end of the day.

.. image:: _static/first_gam_effects.png
   :alt: Estimated temperature and hour effects from a statistical GAM
   :class: homepage-demo scientific-figure
   :align: center

The vertical axis is the contribution of each term to the additive predictor,
not the response itself. Together with the intercept, the displayed terms
reconstruct the model's prediction. The :doc:`quickstart` develops this
analysis from data generation through held-out evaluation and diagnostics.

Four ways to use NAMpy
----------------------

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Statistical GAMs
      :link: quickstart
      :link-type: doc

      Fit penalized smooths, select their complexity, quantify uncertainty,
      and diagnose the fitted model entirely in Python.

   .. grid-item-card:: Shape-constrained GAMs
      :link: user_guide/shape_constrained_gams
      :link-type: doc

      Encode increasing, decreasing, convex, concave, positive, local, and
      bivariate shape assumptions during estimation.

   .. grid-item-card:: Neural additive models
      :link: models/index
      :link-type: doc

      Choose among neural representations while retaining named feature and
      interaction effects.

   .. grid-item-card:: Distributional regression
      :link: user_guide/gamlss
      :link-type: doc

      Model how location, spread, and shape change with predictors rather than
      reducing every conditional distribution to one mean.

The additive structure
----------------------

An additive model represents a transformed prediction as a sum of named
effects,

.. math::

   \eta(x) = \beta_0 + \sum_j f_j(x_j)
   + \sum_{(j,k) \in \mathcal{I}} f_{jk}(x_j, x_k).

Statistical GAMs represent each :math:`f_j` with a penalized smooth basis and
estimate its effective complexity. Neural additive models learn the feature
functions with neural networks. NAMpy preserves the different estimation and
uncertainty semantics of these approaches while exposing their contributions
through a common result contract.

The returned :class:`nampy.contracts.AdditivePrediction` contains response and
link predictions, named terms, the intercept, and any offset. Its
``validate_additive_reconstruction()`` method checks that those pieces recover
the linear predictor.

Statistical modeling with formulas
----------------------------------

Use :class:`nampy.gam.GAM` for direct control of the statistical engine, or
:class:`nampy.models.GAMRegressor` for estimator conventions such as
``score`` and automatic REML smoothing by default. Both accept formulas.

.. code-block:: python

   from nampy.models import GAMRegressor

   gam = GAMRegressor(
       formula='demand ~ s(temperature, bs="cr", k=12) '
               '+ s(hour, bs="cc", k=12)',
       family="gaussian",
   ).fit(train)

   predictions = gam.predict(test)
   components = gam.predict_components(test)
   effects = gam.plot(scheme=1, shade_col="#f3b4c2", scale=0)

The Python-native backend covers formula compilation, smooth construction,
penalized fitting, automatic smoothness selection, prediction matrices,
standard errors, summaries, ANOVA, residual diagnostics, derivatives,
concurvity, basis-dimension checks, and term plots. The
:ref:`statistical-gam-tools` guide is the full capability inventory.

Neural feature functions
------------------------

Neural estimators use the same familiar fit-and-predict pattern. Their model
pages explain what each architecture assumes and which variables a reported
effect may depend on.

.. code-block:: python

   from nampy.models import NAMRegressor

   nam = NAMRegressor(layer_sizes=[48, 24])
   nam.fit(X_train, y_train, max_epochs=100, patience=12)

   predictions = nam.predict(X_test)
   components = nam.predict_components(X_test)
   figures = nam.plot_terms(X_test, center=True, reference_X=X_train)

Use the :doc:`model chooser <models/index>` to start with NAM, NBM, SNAM,
SIAN, GPNAM, or another architecture according to the structure required by
the analysis.

Beyond conditional means
------------------------

Shape constraints and distributional predictors answer different modeling
questions. A shape constraint encodes a defensible direction or curvature
during estimation. A distributional model lets conditional spread or shape
change with predictors as well as location.

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: Compare constrained and unconstrained effects
      :link: tutorials/monotone_effect
      :link-type: doc

      .. image:: _static/shape_constraint_comparison.png
         :alt: Ordinary and monotone-increasing smooths

   .. grid-item-card:: Inspect changing conditional variability
      :link: tutorials/changing_variability
      :link-type: doc

      .. image:: _static/changing_variability.png
         :alt: Conditional location effect and spread

Compatibility you can inspect
-----------------------------

``mgcv`` is the methodological and compatibility reference for the
statistical GAM engine, and ``scam`` is the reference for shape-constrained
smooths. NAMpy compares representative design matrices, penalties,
coefficients, fitted values, uncertainty quantities, diagnostics, and
predictions against versioned upstream results.

The :doc:`compatibility and validation <compatibility>` page separates what is
implemented, what is compared against upstream, what intentionally differs,
and what remains unsupported. Unsupported combinations fail explicitly rather
than silently selecting a different method.

Explore the documentation
-------------------------

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Fit your first GAM
      :link: quickstart
      :link-type: doc

      Follow one nonlinear dataset through fitting, held-out evaluation,
      effect interpretation, and diagnostics.

   .. grid-item-card:: Understand additive models
      :link: overview
      :link-type: doc

      Learn the statistical, neural, distributional, and interpretation
      foundations.

   .. grid-item-card:: Work through tutorials
      :link: tutorials/index
      :link-type: doc

      Explore question-led analyses with executable code and fitted plots.

   .. grid-item-card:: Look up an exact symbol
      :link: api/index
      :link-type: doc

      Find constructors, methods, defaults, return values, and extension
      interfaces.

Citing NAMpy
------------

If NAMpy contributes to published work, cite the software as:

   Ananyapam De, Anton Thielmann, and OpenTabular contributors (2026).
   *NAMpy: Statistical and Neural Additive Models for Python*.
   https://github.com/OpenTabular/NAMpy

.. toctree::
   :caption: Getting Started
   :maxdepth: 1
   :hidden:

   overview
   installation
   quickstart
   faq

.. toctree::
   :caption: Tutorials
   :maxdepth: 2
   :hidden:

   tutorials/index
   notebook_tutorials

.. toctree::
   :caption: How-to Guides
   :maxdepth: 2
   :hidden:

   howto/index
   examples/index

.. toctree::
   :caption: Concepts
   :maxdepth: 2
   :hidden:

   concepts/index
   user_guide

.. toctree::
   :caption: Models
   :maxdepth: 1
   :hidden:

   models/index

.. toctree::
   :caption: API Reference
   :maxdepth: 2
   :hidden:

   api/index

.. toctree::
   :caption: Validation and Development
   :maxdepth: 1
   :hidden:

   compatibility
   contributing
   architecture
   development/reference_sources
   development/mgcv_optimizer_boundaries
   license
