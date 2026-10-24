Save and Reload a Model
=======================

Neural estimators expose a versioned persistence interface:

.. code-block:: python

   model.save_model("nam.pkl")
   restored = NAMRegressor.load_model("nam.pkl")
   prediction = restored.predict(X_test)

For statistical models, serialize the fitted estimator with ``joblib``:

.. code-block:: python

   from joblib import dump, load

   dump(model, "gam.joblib")
   restored = load("gam.joblib")

Store the package version and the model specification with the artifact so a
future analysis can reproduce the same environment.

External preprocessing is part of the prediction workflow. Persist training
medians, category rules, expected column names/order, and any target transforms
alongside the estimator. Recompute neither medians nor category boundaries on
new prediction rows. The :doc:`real-data classification workflow
<../notebooks/05_real_data_classification>` saves these objects, verifies
reloaded probabilities against the originals, and predicts on fresh rows.
