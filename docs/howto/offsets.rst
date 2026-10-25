Predict with an Offset
======================

An offset is a known contribution to the linear predictor with coefficient fixed at one. It must be included consistently during fitting and prediction.

Supply it for new rows when it cannot be recovered from the formula:

.. code-block:: python

   model = GAM(
       formula="count ~ s(age)",
       family="poisson",
       optimize_smoothing=True,
       smoothing_method="reml",
   ).fit(data, offset=np.log(data["exposure"]))

   prediction = model.predict(
       new_data,
       type="response",
       offset=np.log(new_data["exposure"]),
   )

``predict_components`` records the offset separately so the reported terms
still reconstruct the link-scale prediction.

Neural binary-classification offsets at inference
-------------------------------------------------

The known contribution below is a **log-odds** offset. It enters the training
loss with coefficient one. Neural prediction does not remember or reapply
per-row offsets: new rows require new offsets, composed before the sigmoid.

.. code-block:: python

   import tempfile
   import numpy as np
   import pandas as pd
   from scipy.special import expit
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import log_loss
   from nampy.models import NAMClassifier

   rng = np.random.default_rng(31)
   X = pd.DataFrame({"x": rng.uniform(-1, 1, 600)})
   known_log_odds = rng.normal(0, .7, len(X))
   probability = expit(np.sin(3*X.x.to_numpy()) + known_log_odds)
   y = rng.binomial(1, probability)
   fit_rows, test_rows = train_test_split(np.arange(len(X)), test_size=.2, random_state=31)
   train_rows, val_rows = train_test_split(fit_rows, test_size=.25, random_state=32)
   neural = NAMClassifier(layer_sizes=[16, 8], dropout=0., numerical_preprocessing="minmax")
   neural.fit(
       X.iloc[train_rows], y[train_rows], offset=known_log_odds[train_rows],
       X_val=X.iloc[val_rows], y_val=y[val_rows], offset_val=known_log_odds[val_rows],
       max_epochs=80, lr=.01, patience=12, random_state=31,
       logger=False, enable_progress_bar=False, enable_model_summary=False,
       checkpoint_path=tempfile.mkdtemp(),
   )
   parts = neural.predict_components(X.iloc[test_rows])
   parts.validate_additive_reconstruction(atol=1e-5)
   model_contribution = np.asarray(parts.link)
   known_offset = known_log_odds[test_rows]
   full_linear_predictor = model_contribution + known_offset
   probability_with_offset = expit(full_linear_predictor)
   np.testing.assert_allclose(neural.predict_proba(X.iloc[test_rows])[:, 1], expit(model_contribution), atol=1e-6)
   offset_results = pd.DataFrame([{
       "Test n": len(test_rows),
       "Test log loss with supplied offset": log_loss(y[test_rows], probability_with_offset),
       "Test log loss without offset": log_loss(y[test_rows], expit(model_contribution)),
   }])

.. include:: /_generated/neural_offset_results.inc

The table measures the supplied known log-odds contribution on untouched test
rows. Do not add it directly to a probability, and do not add it a second time
if you have already composed the full logit. Other neural objectives require
their own link-scale composition; the statistical example above uses a
log-mean offset instead.
