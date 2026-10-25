Save and Reload a Model
=======================

Save the fitted estimator together with any preprocessing performed outside it. After loading, check that predictions agree on the same input rows before applying the model to new data.

Neural estimators and statistical adapters (``GAMRegressor``,
``GAMClassifier``, and ``GAMLSS``) expose a versioned persistence interface.
For an already fitted ``NAMRegressor``:

.. code-block:: python

   import numpy as np
   from nampy.models import NAMRegressor

   model.save_model("nam.pkl")
   restored = NAMRegressor.load_model("nam.pkl")
   np.testing.assert_allclose(restored.predict(X_test), model.predict(X_test))

For an already fitted statistical adapter, use its own class to load it:

.. code-block:: python

   from nampy.models import GAMRegressor

   gam.save_model("gam.pkl")
   restored_gam = GAMRegressor.load_model("gam.pkl")
   np.testing.assert_allclose(restored_gam.predict(X_test), gam.predict(X_test))

Alternatively, ``joblib`` can serialize a fitted estimator, including the
low-level ``nampy.gam.GAM`` interface:

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
