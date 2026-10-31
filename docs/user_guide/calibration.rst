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

Current scope
-------------

Supported: scalar regression, absolute-residual split conformal, signed-score
CQR, supported shared distributions, explicit quantile levels, and lower/upper
scalar regressor pairs. Fitting routes and probability capabilities do not
expand through calibration. The additional density-ratio, stratum/cluster and
survival lower-bound routes in the focused guides each have their own assumptions.

Unsupported: likelihood/frequency/survey weighting, combined weighted/grouped
calibration, time-series calibration, adaptive coverage selection, classification,
multivariate responses, CV+/jackknife+/full conformal and general conditional
coverage guarantees. The supported extensions and their distinct assumptions
are specified in those guides. Numeric arrays
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


Choose a focused task
---------------------

.. toctree::
   :maxdepth: 1

   calibration_shift
   calibration_groups
   calibration_survival
   calibration_contract

Measure the width cost of calibration
-------------------------------------

.. figure:: /_static/features/conformal_intervals.png
   :alt: Before and after response interval calibration with held-out coverage and mean width shown on separate axes.
   :width: 100%

   An uncalibrated plus/minus 0.5 response band covers about 55% of 150
   independent test responses; the 90% conformal interval covers 88% in
   this sample. Mean width increases from 1.0 to about 2.08, making its
   coverage and width cost explicit. The initial band has no coverage
   claim, and this finite test result is not coverage at every x.

References
----------

Split-conformal regression and CQR follow [Lei2018]_ and [Romano2019]_.
Density-ratio weighting is based on [Tibshirani2019]_; group-conditional
constructions are discussed by [Vovk2012]_. The implemented survival cutoff
procedure follows [Candes2023]_. See :doc:`/references` for paper links and the
separate methodological scope of each route.
