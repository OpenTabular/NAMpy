Basic Regression Examples
==========================

The regressors accept NumPy arrays or pandas data frames. A two-dimensional
target produces one prediction column per output.

.. code-block:: python

   from sklearn.datasets import make_regression
   from sklearn.model_selection import train_test_split

   from nampy.models import NAMRegressor

   X, y = make_regression(
       n_samples=400,
       n_features=6,
       n_targets=2,
       noise=0.2,
       random_state=7,
   )
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=7
   )

   model = NAMRegressor(numerical_preprocessing="standardization")
   model.fit(X_train, y_train, max_epochs=20, batch_size=64)
   predictions = model.predict(X_test)

   assert predictions.shape == y_test.shape
