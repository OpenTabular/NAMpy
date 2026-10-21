Neural Additive Models
======================

A neural additive model retains an additive predictor but learns each feature
effect with a neural network,

.. math::

   \eta(x) = \beta_0 + \sum_j h_j(x_j; w_j).

Each function :math:`h_j` receives the feature named by its term. This makes
its contribution directly inspectable while allowing a nonlinear response to
be learned from data. Architectures in NAMpy can change how these functions
share parameters, select terms, or include explicit interactions; their model
pages state which variables each reported contribution may depend on.

Start with NAM
--------------

For a first neural additive analysis, use :class:`nampy.models.NAMRegressor`.
It learns one feature network per input and exposes predictions and effects
through the standard estimator interface.

.. code-block:: python

   from nampy.models import NAMRegressor

   model = NAMRegressor(layer_sizes=[32, 16], dropout=0.0)
   model.fit(X_train, y_train, max_epochs=100, patience=12)

   predicted = model.predict(X_test)
   components = model.predict_components(X_test, center=True)
   figures = model.plot_terms(X_test, reference_X=X_train, center=True)

The :doc:`first neural analysis <../neural_quickstart>` develops this workflow
from data generation through held-out evaluation and effect interpretation.

Choose another architecture when its structure is useful
---------------------------------------------------------

After fitting a NAM, use the :doc:`model guide <../models/index>` to compare
shared bases, sparsity, interaction discovery, Gaussian-process
approximations, polynomial structure, trees, and transformer-based feature
functions. These are structural choices rather than performance rankings.

Training and interpretation
---------------------------

* :doc:`preprocessing` explains how numeric and categorical inputs become
  model terms.
* :doc:`training` covers optimization, validation, callbacks, and
  reproducibility.
* :doc:`interpretability` defines centering, reconstruction, plotting, and
  interaction semantics.
* :doc:`custom_models` shows how to register a custom feature architecture.

.. toctree::
   :maxdepth: 1
   :hidden:

   preprocessing
   pretab_compatibility
   training
   custom_models
