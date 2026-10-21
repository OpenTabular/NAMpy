Quickstart
==========

This guide fits one statistical GAM and one neural additive model. Both use a
small regression dataset and the familiar ``fit`` and ``predict`` workflow.

Install NAMpy
-------------

.. code-block:: bash

   pip install nampy

Fit a statistical GAM
---------------------

The formula interface specifies which smooth function is fitted for each
feature.

.. code-block:: python

   import numpy as np
   import pandas as pd

   from nampy.gam import GAM

   x = np.linspace(0.0, 1.0, 240)
   data = pd.DataFrame(
       {
           "x": x,
           "y": np.sin(2 * np.pi * x) + 0.15 * np.cos(6 * np.pi * x),
       }
   )

   gam = GAM(
       formula='y ~ s(x, bs="cr", k=12)',
       family="gaussian",
       smoothing_method="REML",
   ).fit(data)

   predictions = gam.predict(data, type="response")
   gam.plot()

The model estimates a penalized cubic regression spline, selects its
smoothness with REML, and returns predictions on the response scale.

Fit a neural additive model
---------------------------

The high-level neural estimators follow the scikit-learn convention.

.. code-block:: python

   from sklearn.datasets import make_regression
   from sklearn.model_selection import train_test_split

   from nampy.models import NAMRegressor

   X, y = make_regression(
       n_samples=1000,
       n_features=8,
       n_informative=6,
       noise=0.2,
       random_state=7,
   )
   X_train, X_test, y_train, y_test = train_test_split(
       X,
       y,
       test_size=0.2,
       random_state=7,
   )

   nam = NAMRegressor(numerical_preprocessing="standardization")
   nam.fit(X_train, y_train, max_epochs=50, batch_size=64)

   predictions = nam.predict(X_test)
   score = nam.score(X_test, y_test)

The estimator fits one learned contribution function for each input feature.
Preprocessing, batching, optimization, and prediction use the same fitted
estimator object.

Inspect the fitted terms
------------------------

Additive components are available separately from the final prediction:

.. code-block:: python

   components = nam.predict_components(X_test)
   importance = nam.term_importance(X_test)
   figures = nam.plot_terms(X_test)

``predict_components`` returns the terms used to reconstruct the additive
predictor. ``term_importance`` summarizes their magnitude, and ``plot_terms``
draws the learned feature functions.

Where to go next
----------------

* :doc:`notebooks/01_getting_started` introduces fitting, prediction, and
  interpretation in NAMpy.
* :doc:`notebooks/02_statistical_gams` covers smooths, inference, diagnostics,
  and shape constraints.
* :doc:`notebooks/03_neural_models` compares neural model
  architectures.
* :doc:`notebooks/04_distributional_models` covers
  distributional predictions and ensembles.
* :doc:`api/index` lists exact constructor and method signatures.
