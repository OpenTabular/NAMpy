Fit a binary classifier
=======================

Suppose ``customers`` records ``contract_months`` (months under contract),
``support_tickets`` (tickets opened in the previous year), and ``churned``
(1 if the customer left, otherwise 0). This recipe fits a sparse additive
classifier and obtains both labels and probabilities.

.. code-block:: python

   from sklearn.model_selection import train_test_split
   from nampy.models import SNAMClassifier

   X = customers[["contract_months", "support_tickets"]]
   y = customers["churned"]
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=11, stratify=y
   )

   model = SNAMClassifier(numerical_preprocessing="standardization")
   model.fit(X_train, y_train, max_epochs=40, batch_size=64)
   labels = model.predict(X_test)
   probabilities = model.predict_proba(X_test)

``labels`` contains the selected class for each customer. ``probabilities``
has one column per class and is the appropriate output for threshold selection,
calibration, and probability-based scoring.
