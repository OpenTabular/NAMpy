Interpretability
================

One of NAMpy's key strengths is model interpretability, especially for additive models like NAM.

.. note::
   This feature is under active development. More comprehensive interpretability
   tools and visualizations will be added in future releases.

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
   
   # Get feature-level predictions
   # (Implementation depends on model internals)
   # This is an area for future development

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

Visualization (Future)
----------------------

Future releases will include:

* Feature importance plots
* Shape functions for individual features
* Partial dependence plots
* Individual prediction explanations
* Interactive visualization tools

Contributing
------------

Interpretability tools are an active area of development. Contributions
are welcome! See :doc:`../contributing` for details.

For More Information
--------------------

* :doc:`../models/index` - Model comparison
* :doc:`../examples/index` - Practical examples
* GitHub: https://github.com/OpenTabular/NAMpy

