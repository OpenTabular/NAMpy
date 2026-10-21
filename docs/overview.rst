Overview
========

NAMpy brings statistical and neural additive models into one Python workflow.
The models share familiar estimator operations, but they keep their own
estimation methods, uncertainty statements, and interpretation rules.

What NAMpy provides
-------------------

.. list-table::
   :header-rows: 1
   :widths: 24 40 36

   * - Model family
     - Use it when
     - Main outputs
   * - Statistical GAM
     - You need smooth effects, automatic smoothness selection, or classical
       inference.
     - Predictions, smooth terms, standard errors, and diagnostics.
   * - Neural additive model
     - You want flexible learned feature functions or explicit higher-order
       terms.
     - Predictions, learned term contributions, and feature importance.
   * - Distributional model
     - The conditional spread or shape matters in addition to the mean.
     - Natural distribution parameters, point predictions, and parameter-wise
       effects.

The shared workflow
-------------------

High-level estimators use ``fit``, ``predict``, ``score``, and persistence
methods. Additive models also expose their component functions so a prediction
can be examined as a sum of terms rather than only as a final value.

.. code-block:: python

   from nampy.models import NAMRegressor

   model = NAMRegressor()
   model.fit(X_train, y_train, max_epochs=50)

   predictions = model.predict(X_test)
   components = model.predict_components(X_test)
   importance = model.term_importance(X_test)

Statistical GAMs additionally provide a formula interface:

.. code-block:: python

   from nampy.gam import GAM

   model = GAM(
       formula='y ~ s(income, bs="cr", k=10) + s(age, bs="tp", k=10)',
       family="gaussian",
       smoothing_method="REML",
   ).fit(data)

   predictions = model.predict(data, type="response")

Choosing a model
----------------

Start with the simplest structure that answers the analytical question. Use a
statistical GAM when the smooth specification and inferential output are central.
Use a neural model when learned feature functions or the published architecture
are the object of interest. Use a distributional estimator when a single
conditional mean is insufficient.

The :doc:`models/index` compares the available architectures. The
:doc:`quickstart` shows the two primary backends on small, complete examples.

Interpretation
--------------

An additive prediction reconstructs as an intercept plus model terms. This is
an accounting identity, not a guarantee that every term is uniquely identified
or stable under correlated inputs. Main effects, interactions, and nonlinear
distribution links therefore need to be interpreted on the scale documented by
the fitted estimator. See :doc:`user_guide/interpretability` for the precise
contracts.

Next steps
----------

* :doc:`installation` — install and verify NAMpy.
* :doc:`quickstart` — fit a statistical GAM and a neural model.
* :doc:`notebook_tutorials` — follow complete executable analyses.
* :doc:`api/index` — look up classes, methods, and plotting functions.
