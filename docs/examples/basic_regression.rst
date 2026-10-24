Fit a Neural Regressor
======================

Use named pandas columns for feature-effect plots. This first example fits
a single response.

.. code-block:: python

   import pandas as pd
   from sklearn.datasets import make_regression
   from sklearn.model_selection import train_test_split

   from nampy.models import NAMRegressor

   X, y = make_regression(
       n_samples=400,
       n_features=6,
       n_targets=1,
       noise=0.2,
       random_state=7,
   )
   X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=7
   )

   model = NAMRegressor(numerical_preprocessing="standardization")
   model.fit(X_train, y_train, max_epochs=20, batch_size=64)
   predictions = model.predict(X_test)
   effects = model.plot_terms(X_test, center=True, reference_X=X_train)

``predictions`` contains one response per test row. ``effects`` contains
feature-contribution figures on the additive-predictor scale.

Inspect multiple outputs
------------------------

A two-dimensional target produces one prediction column per output. Inspect
its component arrays directly: the current ``plot_terms`` renderer skips
multi-column contributions.

.. code-block:: python

   import numpy as np

   y_multi = np.column_stack([y_train, 0.5 * y_train])
   multi_model = NAMRegressor(numerical_preprocessing="standardization")
   multi_model.fit(X_train, y_multi, max_epochs=20, batch_size=64)
   multi_predictions = multi_model.predict(X_test)
   components = multi_model.predict_components(X_test)
   for name, contribution in components.terms.items():
       print(name, np.asarray(contribution).shape)  # test rows, two outputs
