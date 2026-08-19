API Reference
=============

This section contains the complete API documentation for NAMpy.

.. toctree::
   :maxdepth: 2

   models
   gam
   basemodels
   configs
   utils

Package Overview
----------------

The NAMpy package is organized into several main modules:

* :mod:`nampy.models` - High-level estimator classes (scikit-learn style),
  including the ``GAMRegressor``/``GAMClassifier`` adapters
* :mod:`nampy.gam` - mgcv-parity GAM backend (``GAM`` and the fit core)
* :mod:`nampy.contracts` - Shared backend-neutral contracts (feature schema,
  additive predictions, capabilities, persistence)
* :mod:`nampy.neural` - Torch backend internals (modules, task, contracts,
  data, distributions, configs)

Quick API Reference
-------------------

Models
~~~~~~

All models follow the scikit-learn API:

.. code-block:: python

   from nampy.models import NAMRegressor, NAMClassifier, NAMLSS
   
   # Initialize
   model = NAMRegressor()
   
   # Fit
   model.fit(X_train, y_train, max_epochs=100)
   
   # Predict
   predictions = model.predict(X_test)
   
   # Score
   score = model.score(X_test, y_test)

Base Models
~~~~~~~~~~~

For advanced users who want direct access to PyTorch models:

.. code-block:: python

   from nampy.neural.modules import NAM
   from nampy.neural.configs import DefaultNAMConfig
   
   config = DefaultNAMConfig()
   model = NAM(
       cat_feature_info={},
       num_feature_info={'feature1': 1, 'feature2': 1},
       num_classes=1,
       config=config
   )

Preprocessing (PreTab)
~~~~~~~~~~~~~~~~~~~~~~

NAMpy uses the PreTab library for preprocessing. Pass a PreTab preprocessor into the data module or sklearn-style models:

.. code-block:: python

   from pretab.preprocessor import Preprocessor
   
   preprocessor = Preprocessor(task="regression", n_bins=50)
   # Use with NAMpyDataModule or model.fit(X, y, ...)
   
   X_processed = preprocessor.fit_transform(X, y)

Version Information
-------------------

.. autodata:: nampy.__version__
   :annotation:
