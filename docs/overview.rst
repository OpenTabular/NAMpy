Additive Models in NAMpy
========================

NAMpy offers statistical and neural estimators with a shared interface for
fitting, prediction, and inspecting named effects. This page helps you choose
between them. For the mathematical definition, see
:doc:`concepts/additive_predictors`.

Statistical generalized additive models
---------------------------------------

Choose a statistical GAM when smoothness selection, standard errors, term
tests, and residual diagnostics are part of the analysis. Effects use penalized
spline bases, with flexibility estimated from the data or supplied directly.

Start with :doc:`quickstart` for a fitted example. The
:ref:`statistical-gam-tools` reference lists supported formulas, smooths,
families, fitting controls, and diagnostics.

Neural additive models
----------------------

Choose a neural model when you want feature networks, shared learned bases,
differentiable trees, or transformer-based representations. Architectures
differ in which features an effect can depend on and how interactions are
specified or discovered.

Use :doc:`models/index` to compare architectures, or follow
:doc:`tutorials/nam_nbm` for a fitted comparison of NAM and NBM.

Distributional additive models
------------------------------

Choose a distributional estimator when conditional spread or shape should vary
alongside location. ``GAMLSS`` uses statistical fitting; neural ``*LSS``
estimators use their architecture's training procedure.

See :doc:`concepts/distributional_regression` for the parameterization,
:doc:`tutorials/changing_variability` for a visual example, and
:doc:`user_guide/gamlss` for formulas and prediction methods.

Interpreting additive effects
-----------------------------

Use ``predict_components`` to inspect term contributions and
``plot_terms`` or the statistical GAM's ``plot`` method to visualize effects.
The :doc:`user_guide/interpretability` guide covers centering, effect tables,
and plot options.

Continue with NAMpy
-------------------

* :doc:`quickstart` — fit and diagnose a statistical GAM.
* :doc:`tutorials/index` — work through modeling questions with fitted examples.
* :doc:`models/index` — compare model architectures.
* :doc:`api/index` — look up exact signatures and return values.
