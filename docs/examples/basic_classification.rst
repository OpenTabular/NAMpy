Basic Classification Examples
==============================

Classifiers expose both class labels and class probabilities.

.. code-block:: python

   from sklearn.datasets import make_classification
   from sklearn.model_selection import train_test_split

   from nampy.models import SNAMClassifier

   X, y = make_classification(
       n_samples=400,
       n_features=8,
       n_informative=5,
       random_state=11,
   )
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=11, stratify=y
   )

   model = SNAMClassifier(numerical_method="standardization")
   model.fit(X_train, y_train, max_epochs=20, batch_size=64)

   labels = model.predict(X_test)
   probabilities = model.predict_proba(X_test)
   assert labels.shape == y_test.shape
   assert probabilities.shape[0] == y_test.shape[0]
