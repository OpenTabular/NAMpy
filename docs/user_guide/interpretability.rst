Interpretability
================

One of NAMpy's key strengths is model interpretability, especially for additive
models like NAM, NodeGAM, SplineNAM, and SparseNAM.

Understanding Additive Models
------------------------------

Additive models decompose predictions into feature-level contributions:

.. math::

   f(x) = \\beta_0 + \\sum_{i=1}^{n} f_i(x_i)

Where:
* :math:`f(x)` is the final prediction
* :math:`\\beta_0` is a bias term
* :math:`f_i(x_i)` is the contribution of feature :math:`i`

Feature-Level Predictions
--------------------------

For interpretable models, you can extract per-term contributions on the raw
additive scale:

.. code-block:: python

   from nampy.models import NAMRegressor
   
   model = NAMRegressor()
   model.fit(X_train, y_train, max_epochs=100)
   
   terms = model.predict_terms(X_test)
   terms_frame = model.predict_terms(X_test, as_frame=True)
   raw = model.predict_feature_vals(X_test)
   prediction = raw["prediction"]
   regularization = raw["regularization"]

The same method works for regression, classification, and LSS wrappers. For
classification, contributions are logits. For LSS models, contributions are raw
distribution-parameter outputs.

The raw model output uses a nested dictionary: ``"prediction"`` stores the final
tensor, ``"terms"`` stores per-term contributions, ``"intercept"`` stores the
optional intercept, and ``"regularization"`` stores loss penalties.

Feature Importance
------------------

Feature importance can be computed from the variation in term contributions:

.. code-block:: python

   importance = model.feature_importance(X_test, method="variance")

Supported methods are ``"variance"``, ``"range"``, ``"mean_abs"``, and
``"max_abs"``.

Visualization
-------------

Use the generic plotting helpers for main effects and pairwise interactions:

.. code-block:: python

   model.plot_terms(X_test)
   model.plot_interactions(X_test)

The helper functions are also available from :mod:`nampy.utils`:

.. code-block:: python

   from nampy.utils import predict_terms, feature_importance

   terms = predict_terms(model, X_test)
   importance = feature_importance(model, X_test)

Summaries and Diagnostics
-------------------------

All sklearn-style estimators expose a compact summary:

.. code-block:: python

   info = model.summary()

For model-specific diagnostics, use:

.. code-block:: python

   diagnostics = model.diagnostics()

SplineNAM diagnostics include knot locations and regularization penalties.
SparseNAM diagnostics include group norms and selected groups.

Model Comparison
----------------

Compare interpretable vs. black-box models:

.. list-table::
   :header-rows: 1
   :widths: 20 30 30 20

   * - Model
     - Interpretability
     - Performance
     - Best For
   * - NAM
     - High (feature-level)
     - Good
     - Explanation needed
   * - GPNAM
     - High (feature-level)
     - Good
     - With uncertainty
   * - TreeNAM
     - High (feature-level)
     - Good
     - Categorical data
   * - NAMformer
     - Medium (attention)
     - Better
     - Performance + some interpretation
   * - NATT
     - Low
     - Best
     - Pure performance

For More Information
--------------------

* :doc:`../models/index` - Model comparison
* :doc:`../examples/index` - Practical examples
* GitHub: https://github.com/Ananyapam7/NAMpy
