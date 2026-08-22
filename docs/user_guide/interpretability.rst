Interpretability
================

One of NAMpy's key strengths is model interpretability, especially for additive models like NAM.

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

For interpretable models (NAM, GPNAM, etc.), you can extract feature
contributions:

.. code-block:: python

   from nampy.models import NAMRegressor
   
   model = NAMRegressor()
   model.fit(X_train, y_train, max_epochs=100)
   
   components = model.predict_components(X_test, center=True)
   table = model.explain_terms(X_test)
   importance = model.term_importance(X_test)
   figures = model.plot_terms(X_test)

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
     - Smooth, parameter-efficient effects
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

Visualization
-------------

``plot_terms`` renders one-dimensional contribution curves, while
``plot_interactions`` handles interaction heatmaps. ``explain_terms`` and
``term_importance`` return reusable tables for custom plotting.

Contributing
------------

Interpretability tools are an active area of development. Contributions
are welcome! See :doc:`../contributing` for details.

For More Information
--------------------

* :doc:`../models/index` - Model comparison
* :doc:`../examples/index` - Practical examples
* GitHub: https://github.com/OpenTabular/NAMpy
