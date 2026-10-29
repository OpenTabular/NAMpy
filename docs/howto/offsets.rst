Predict with an Offset
======================

An offset is a known contribution to the linear predictor with coefficient fixed at one. It must be included consistently during fitting and prediction.

Compare two supplied exposures
------------------------------

Here the same fitted rows receive exposure one or two at prediction:
log-mean increases by log(2), while the expected response doubles. The fitted
age smooth is unchanged.

.. literalinclude:: ../scripts/generate_howto_assets.py
   :language: python
   :start-after: # OFFSETS START
   :end-before: # OFFSETS END
   :dedent: 4

.. include:: /_generated/howto_offset_comparison.inc

Neural binary-classification offsets at inference
-------------------------------------------------

The known contribution below is a **log-odds** offset. It enters the training
loss with coefficient one. Neural prediction does not remember or reapply
per-row offsets: pass new offsets through ``offset=`` for prediction. They are
added to the logit before the sigmoid, separately from learned contributions.

.. code-block:: python


   from dataclasses import replace
   from nampy.neural.configs.nam_config import DefaultNAMConfig
   from nampy.neural.configuration import GradientTrainingConfig
   from nampy.neural.configuration import PreprocessingConfig
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
   neural = NAMClassifier(architecture_config=DefaultNAMConfig(layer_sizes=[16, 8], dropout=0.0), preprocessing=PreprocessingConfig(numerical_preprocessing='minmax'))
   neural.set_params(training=replace(neural.training or GradientTrainingConfig(), max_epochs=80, lr=0.01, patience=12, random_state=31, checkpoint_path=tempfile.mkdtemp(), trainer_kwargs={'logger': False, 'enable_progress_bar': False, 'enable_model_summary': False})).fit(X.iloc[train_rows], y[train_rows], offset=known_log_odds[train_rows], X_val=X.iloc[val_rows], y_val=y[val_rows], offset_val=known_log_odds[val_rows])
   parts = neural.predict_components(X.iloc[test_rows])
   parts.validate_additive_reconstruction(atol=1e-5)
   model_contribution = np.asarray(parts.link)[:, 0]
   known_offset = known_log_odds[test_rows]
   full_linear_predictor = model_contribution + known_offset
   probability_with_offset = neural.predict_proba(
       X.iloc[test_rows], offset=known_offset
   )[:, 1]
   np.testing.assert_allclose(probability_with_offset, expit(full_linear_predictor), atol=1e-6)
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
their own predictor coordinates and exact output width; the statistical example above uses a
log-mean offset instead.
