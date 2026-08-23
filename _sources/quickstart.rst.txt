Quick Start
===========

This page is intentionally short. It establishes the shared estimator workflow;
the :doc:`notebook_tutorials` contain model theory and broader demonstrations.

Install
-------

Install both backends:

.. code-block:: bash

   pip install "nampy[all]"

Use ``nampy[gam]`` for only the statistical GAM backend or ``nampy[neural]``
for only the neural backend. See :doc:`installation` for development installs
and optional dependencies.

Statistical GAM
---------------

Formula mode exposes the ``mgcv``-aligned backend directly:

.. code-block:: python

   import numpy as np
   import pandas as pd

   from nampy.gam import GAM

   x = np.linspace(0.0, 1.0, 120)
   data = pd.DataFrame({"x": x, "y": np.sin(2 * np.pi * x)})

   model = GAM(
       formula='y ~ s(x, bs="cr", k=10)',
       family="gaussian",
       smoothing_method="REML",
   ).fit(data=data)
   prediction = model.predict(data, type="response")

The GAM notebook develops this interface into one coherent case study covering
smooth bases, smoothing selection, response families, multiple linear
predictors, shape constraints, inference, and diagnostics.

Neural additive model
---------------------

Neural estimators use a scikit-learn-style ``fit``/``predict``/``score``
contract:

.. code-block:: python

   from sklearn.datasets import make_regression
   from sklearn.model_selection import train_test_split

   from nampy.models import NAMRegressor

   X, y = make_regression(
       n_samples=500,
       n_features=6,
       noise=0.2,
       random_state=7,
   )
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=7
   )

   model = NAMRegressor(numerical_preprocessing="standardization")
   model.fit(X_train, y_train, max_epochs=20, batch_size=64)
   prediction = model.predict(X_test)
   score = model.score(X_test, y_test)

Where to continue
-----------------

* Choose an architecture in :doc:`models/index`.
* Open :doc:`notebook_tutorials` for theory and complete model demonstrations.
* Read :doc:`user_guide/preprocessing` and :doc:`user_guide/training` for
  cross-model controls.
* Use :doc:`examples/index` for compact classification and distributional
  recipes.
* Consult :doc:`api/index` when you need exact signatures.
