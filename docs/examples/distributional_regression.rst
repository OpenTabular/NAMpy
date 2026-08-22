Distributional Regression Examples
===================================

LSS estimators predict the parameters of a chosen response distribution.

.. code-block:: python

   from sklearn.datasets import make_regression
   from sklearn.model_selection import train_test_split

   from nampy.models import TreeNAMLSS

   X, y = make_regression(
       n_samples=400, n_features=6, noise=2.0, random_state=13
   )
   X_train, X_test, y_train, _ = train_test_split(
       X, y, test_size=0.2, random_state=13
   )

   model = TreeNAMLSS(
       family="normal", numerical_preprocessing="standardization"
   )
   model.fit(
       X_train,
       y_train,
       max_epochs=20,
       batch_size=64,
   )
   distribution_parameters = model.predict(X_test)
   assert distribution_parameters.shape[0] == X_test.shape[0]
