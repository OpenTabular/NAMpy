Calibrate scalar prediction intervals
=====================================

``nampy.calibration`` adds post-fit calibration shared by the statistical and
neural backends. It supports absolute residuals, conformalized quantile
regression (CQR), explicit covariate-shift density ratios, fixed strata,
exchangeable complete groups and censor-aware survival lower bounds. It does not fit or tune a model, alter
its coefficients, or require a density when explicit quantile predictions exist.

Keep data roles separate
------------------------

Fit preprocessing and the model on training data. Use a distinct validation
cohort for tuning or early stopping where needed. Once all fitting and selection
are complete, use a separate calibration cohort. Evaluate on a fourth held-out
cohort. The calibration rows must not influence fitting, early stopping,
selection or the choice of calibration method/coverage level.

For the unweighted scalar methods described first, the finite-sample guarantee
is marginal over exchangeable calibration and new
observations, conditional on the independently trained predictor. It is not
coverage at every covariate value, joint coverage across new observations,
coefficient uncertainty, or confidence intervals for conditional quantiles.
NAMpy cannot infer whether supplied datasets overlap: callers own their split
and exchangeability assumptions. Snapshotting protects the fitted model, not
those assumptions.

Absolute-residual intervals
---------------------------

.. code-block:: python

   from nampy import GAMRegressor
   from nampy.calibration import calibrate_regressor

   model = GAMRegressor(formula="~ s(age, k=6)").fit(X_train, y_train)
   calibrated = calibrate_regressor(model, X_cal, y_cal, level=.9)
   interval = calibrated.predict_interval(X_test)
   lower, upper = interval.lower, interval.upper

The default ``prediction="response"`` requires a scalar regressor. Use
``prediction="point"`` for an LSS model's documented ``predict_point`` summary,
or ``prediction="distribution"`` for the mean of a supported shared
``DistributionPrediction``. An unavailable summary raises explicitly. A natural
parameter column is never chosen implicitly as a response prediction.

Neural regression uses the same operation after its ordinary fit. Calibration
never reruns preprocessing or accesses a trainer. Keep its early-stopping data
separate from calibration data.

Quantile calibration
--------------------

For a statistical or neural distributional model with quantile support:

.. code-block:: python

   calibrated = calibrate_regressor(
       lss_model, X_cal, y_cal, method="quantile",
       prediction="distribution", level=.9)
   interval = calibrated.predict_interval(X_test)

This evaluates central quantiles at ``(1-level)/2`` and ``(1+level)/2`` through
the existing law. Neural quantile-loss models need no CDF or density: use
``prediction="quantiles"`` instead. Their public
``predict_quantiles(X, probabilities)`` selects explicitly fitted levels, in
requested order, without interpolation or extrapolation. Other supported neural
LSS families implement that operation through their shared probability law.

Alternatively, calibrate two already fitted scalar quantile regressors:

.. code-block:: python

   calibrated = calibrate_regressor(
       lower_quantile_model, X_cal, y_cal, upper_estimator=upper_quantile_model,
       method="quantile", level=.9)

Central lower/upper quantile fits are recommended; their configured levels
remain explicit in the source models. Both models receive the same supplied
prediction kwargs. No sorting, crossing repair, selection of output columns, or
assumption that a multi-output parameter array contains quantiles is performed.
For different per-model prediction contexts, use the array-level interface.

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

Current scope
-------------

Supported: scalar regression, absolute-residual split conformal, signed-score
CQR, supported shared distributions, explicit quantile levels, and lower/upper
scalar regressor pairs. Fitting routes and probability capabilities do not
expand through calibration. The additional density-ratio, stratum/cluster and
survival lower-bound routes below each have their own assumptions.

Unsupported: likelihood/frequency/survey weighting, combined weighted/grouped
calibration, time-series calibration, adaptive coverage selection, classification,
multivariate responses, CV+/jackknife+/full conformal and general conditional
coverage guarantees. The supported extensions and their distinct assumptions
are specified below. Numeric arrays
alone cannot identify whether a caller has encoded censored or categorical
responses; they must represent exact scalar regression observations.

The implemented equations follow Romano, Patterson and Candès (2019),
`Conformalized Quantile Regression
<https://papers.neurips.cc/paper_files/paper/2019/file/5103c3584b063c431bd1268e9b5e76fb-Paper.pdf>`_,
sections 3--4. No upstream implementation code is copied.

Runnable example
----------------

.. literalinclude:: ../../examples/example_conformal_calibration.py
   :language: python

API
---

.. automodule:: nampy.calibration
   :members:

Covariate-shift weighting
-------------------------

Supply ``density_ratio`` values representing
``dP_target(X)/dP_calibration(X)``, on a common arbitrary scale, while the
conditional response distribution remains unchanged. These are not case,
frequency, survey or optimizer loss weights. A density-ratio function must be
known or fitted separately; NAMpy does not estimate one during calibration.

.. code-block:: python

   weighted = calibrate_regressor(
       model, X_cal, y_cal, density_ratio=ratio_cal, level=.9,
       weight_source="known")
   interval = weighted.predict_interval(X_test, density_ratio=ratio_test)

The weighted empirical measure includes each new observation's weight as an
atom at positive infinity. Consequently corrections depend on the supplied
new-point ratios; there is no single global rank/correction. Explicit positive
finite test ratios are required. Calibration ratios can be zero. If calibration
mass is insufficient, including all-zero mass, the result is unbounded.
No mean-weight substitution, clipping or percentile interpolation occurs.
``ConformalCalibration.corrections(ratio_test)`` exposes the per-row corrections.

``weight_source="known"`` labels marginal coverage under known covariate shift.
``"estimated"`` labels approximate coverage with estimated ratios; accuracy and
coverage are not guaranteed merely because estimates were supplied. Ratio
estimation or method selection using calibration responses would invalidate the
stated split procedure. Array-level ``calibrate_conformal`` accepts these same
arguments. Positive rescaling leaves the mathematical quantile unchanged.

The method follows Tibshirani, Barber, Candès and Ramdas (2019),
`Conformal Prediction Under Covariate Shift
<https://www.stat.berkeley.edu/~ryantibs/papers/weightedcp.pdf>`_.

Fixed strata and complete groups
--------------------------------

``calibrate_grouped`` and ``calibrate_regressor(..., groups=...)`` distinguish
two targets; group IDs are never treated as frequency weights:

* ``group_mode="stratum"`` (array-level ``mode="stratum"``): apply separate
  quantiles within an externally fixed covariate partition. Coverage is within
  a stratum under within-stratum exchangeability. Small strata can give
  unbounded sets; unseen strata raise without pooled fallback. Outcome-defined
  or calibration-tuned partitions are unsupported.
* ``group_mode="cluster"`` (default): take the maximum nonconformity score over
  all members of each complete calibration group, then calibrate across those
  group scores. Each group contributes once, irrespective of its size. Coverage
  is simultaneous over all members of one new exchangeable complete group,
  marginal over groups. It is not simultaneous across all future groups and
  does not describe an additional member of a previously seen group.

Pass ``groups=group_ids_test`` to ``predict_interval``. Cluster-mode query IDs
must be disjoint from calibration group IDs. The caller must also keep training,
validation and calibration groups separate and supply complete groups. Group
sizes and their observation mechanisms are part of the exchangeability
assumption. The software cannot certify that a supplied trajectory is complete.
Stratum membership must use the same fixed rule at calibration and prediction.
``calibration_counts`` records the relevant calibration count per query row.
Combining density-ratio weighting with grouping, hierarchical random-member
methods, partial groups and temporal dependence are not implemented.

Survival lower prediction bounds
--------------------------------

``calibrate_survival`` consumes ``SurvivalTarget`` plus base lower predictions
or ``SurvivalPrediction``. The estimator wrapper
``calibrate_survival_estimator`` snapshots a fitted model from either backend:

.. code-block:: python

   from nampy.calibration import calibrate_survival_estimator

   calibrated = calibrate_survival_estimator(
       survival_model, X_cal, target_cal, level=.9)
   lower = calibrated.predict_interval(X_test).lower

Default calibration uses all observed ``min(T,C)`` values in the one-sided score
``q_(1-level)(X) - min(T,C)``. This is explicitly a conservative lower bound for
latent event time because ``min(T,C) <= T``. It never treats censored times as
exact latent events for two-sided calibration. Exchangeability of the observed
records and the independent fitted model is required; independence of censoring
and event times is not required for this conservative construction.

A sharper administrative-cutoff route calibrates ``min(T,cutoff)`` on the subset
whose **known potential censoring time** satisfies ``C >= cutoff``:

.. code-block:: python

   calibrated = calibrate_survival_estimator(
       survival_model, X_cal, target_cal, level=.9, cutoff=3.,
       censoring_time=potential_censor_time_cal,
       selection_probability=probability_C_ge_3_cal,
       weight_source="known")
   interval = calibrated.predict_interval(
       X_test, selection_probability=probability_C_ge_3_test)

Potential censoring times must be observed even for subjects who had an event.
Ordinary ``(time,event)`` records cannot reconstruct them. Censored records must
agree with their supplied censoring time. No event-only selection or censor-time
imputation is performed. The cutoff must be chosen independently of calibration
outcomes. Calibration uses the capped base prediction and exact restricted time,
weighted by ``1/P(C>=cutoff | X)``. Conditional independence of T and C given X,
positivity and a common mechanism between calibration and target populations are
required. Known probabilities have a finite-sample marginal target; estimated
probabilities receive an explicitly approximate label, without a promise of
double robustness. Mechanism estimation is caller-owned and must be independent
of calibration outcomes. Zero probabilities and numerically unrepresentable
inverse weights are rejected without clipping.

Bounds are one-sided with upper limit positive infinity. Lower bounds respect
nonnegative event-time support, and cutoff bounds cannot exceed the cutoff.
Insufficient calibration mass yields the trivial lower bound zero. A capped
piecewise quantile can be evaluated within its fitted support even if the full
quantile lies beyond that support; no tail extrapolation is introduced. Results
carry ``future_survival_time_conformal`` uncertainty and explicit conservative,
known-selection or estimated-selection coverage labels. Source observations and
selected subject IDs remain auditable; calibration stores scores and IDs, not
raw target/covariate data. Survival wrappers support the same version-7 trusted
persistence and isolated-fit lifecycle as scalar wrappers.

These operations do not calibrate an entire survival curve or confidence band.
Delayed entry, landmark conditioning, latent-time upper bounds, random-censoring
imputation, adaptive cutoffs and joint group/weighted survival methods remain
unsupported. This cutoff construction follows Candès, Lei and Ren,
`Conformalized Survival Analysis, Algorithm 1
<https://candes.su.domains/publications/downloads/ConformalSurvivalAnalysis.pdf>`_.

Runnable extension example
--------------------------

.. literalinclude:: ../../examples/example_calibration_extensions.py
   :language: python
