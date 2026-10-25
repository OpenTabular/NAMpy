Save and Reload a Model
=======================

Save the fitted estimator together with any preprocessing performed outside it. After loading, check that predictions agree on the same input rows before applying the model to new data.

Neural estimators expose a versioned persistence interface:

.. code-block:: python

   import numpy as np

   model.save_model("nam.pkl")
   restored = NAMRegressor.load_model("nam.pkl")
   np.testing.assert_allclose(restored.predict(X_test), model.predict(X_test))

For statistical models, serialize the fitted estimator with ``joblib``:

.. code-block:: python

   from joblib import dump, load

   dump(model, "gam.joblib")
   restored = load("gam.joblib")
   np.testing.assert_allclose(restored.predict(X_test), model.predict(X_test))

Store the package version and the model specification with the artifact so a
future analysis can reproduce the same environment.

External preprocessing is part of the prediction workflow. Persist training
medians, category rules, expected column names/order, and any target transforms
alongside the estimator. Recompute neither medians nor category boundaries on
new prediction rows. The :doc:`real-data classification workflow
<../notebooks/05_real_data_classification>` saves these objects, verifies
reloaded probabilities against the originals, and predicts on fresh rows.
