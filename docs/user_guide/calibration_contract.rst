Conformal numerical and snapshot contract
=========================================

This reference records the finite-sample order statistic and fitted-snapshot
lifecycle used by :doc:`calibration`.

Numerical contract
------------------

For calibration sample size ``n`` and coverage ``level``, correction uses the
``ceil((n+1)*level)``-th smallest score (one-based), without interpolation.
If this rank exceeds ``n``, the correction is positive infinity and prediction
sets are unbounded. The implementation never clamps the rank to produce a
finite interval unsupported by the finite-sample construction.

Absolute scores are ``abs(y - prediction)``. CQR scores are
``max(lower - y, y - upper)`` and may be negative. Corrected bounds are
``[lower - correction, upper + correction]``. A negative correction may shrink
or empty a prediction set. ``ConformalInterval.empty`` explicitly marks
``lower > upper``; bounds are neither sorted nor clipped to conceal empty sets.

Every calibration and prediction row must be finite and aligned. Crossed input
quantiles, missing rows, vector responses and silent row dropping are rejected.
Scores, method, level, unweighted rank/correction and calibration size are inspectable.
Weighted corrections require each query's density ratio, as described below.
Returned arrays are read-only. Metadata explicitly identifies
``future_observation_conformal`` uncertainty and
``marginal_under_exchangeability`` coverage.

For existing prediction pipelines or unsupported estimator adapters:

.. code-block:: python

   from nampy.calibration import calibrate_conformal

   state = calibrate_conformal(y_cal, prediction_cal, level=.9)
   interval = state.interval(prediction_test)
   # CQR: prediction arrays are (n,2) lower/upper columns.
   quantile_state = calibrate_conformal(
       y_cal, quantile_prediction_cal, method="quantile", level=.9)

Array-level state does not retain an estimator. Its caller must use the same
unchanged predictor for calibration and new predictions. Shared
``DistributionPrediction`` inputs are also supported.

Snapshot lifecycle, offsets and persistence
-------------------------------------------

``calibrate_regressor`` takes an isolated fitted-state snapshot **before**
scoring. Later refits or parameter changes on the source estimator cannot alter
the calibrated predictor. It has no training or parameter-update methods:
recalibrate explicitly to use another model or coverage level. Private snapshot
internals are not a supported mutation interface. Calibration failures publish
no partial result and leave source estimators unchanged.

Prediction arguments belong to each dataset. For example, pass calibration
``predict_kwargs={"offset": offset_cal}``, then
``predict_interval(X_test, offset=offset_test)``. Calibration-row offsets are
never carried over to test rows. Batch size and supported distribution context
arguments remain owned by the model's prediction method.

``calibrated.save_model(path)`` and ``ConformalPredictor.load_model(path)`` use
NAMpy's trusted-file artifact format (currently version 7). They store fitted snapshots and
scores, not calibration features/targets, trainers, callbacks or optimizer
execution state. Backend artifact owners handle fitted-state exclusions. Previous artifacts are rejected before payload loading; there is no legacy
reader or artifact migration route.

API
---

.. automodule:: nampy.calibration
   :members:

For methodological citations and paper links, see :doc:`/references`.
