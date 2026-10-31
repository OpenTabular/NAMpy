Conformal lower bounds for censored event times
===============================================

Use one-sided lower bounds for future event times; do not treat censored
observations as exact responses for two-sided scalar calibration.

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

For methodological citations and paper links, see :doc:`/references`.
