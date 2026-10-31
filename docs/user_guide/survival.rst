Survival models
===============

NAMpy supports piecewise-exponential, censored Weibull, and censored lognormal
models through its statistical engine and gradient-trained LinReg and NAM
architectures. Both consume
``nampy.survival.SurvivalTarget`` and produce the same survival prediction view.
Static records use one terminal event or right censoring per subject.
Piecewise models additionally accept complete changing-covariate histories
through ``LongitudinalSurvivalTarget``. A separate statistical ``CoxGAM``
supports static partial likelihood with fixed penalties or eligible automatic
REML smoothing. Neural LinReg/NAM Cox training is also available; see
:doc:`survival_cox`.
Conditionally independent censoring and entry are assumed.


Choose a route
--------------

* Use ``SurvivalGAM`` for piecewise hazards or censored Weibull/lognormal
  regression, including supported delayed entry.
* Use ``CoxGAM`` for a statistical relative-risk model with a Breslow baseline;
  fixed penalties and eligible automatic smoothing have distinct controls.
* Use ``LinRegSurvival`` or ``NAMSurvival`` for gradient-trained piecewise,
  parametric or static unstratified Cox predictors.
* Use complete subject histories only with the piecewise route; choose the
  :doc:`survival_longitudinal` workflow before splitting those records.

Observations and fitting
------------------------

Create a target explicitly from time, event (0/1), and optional entry and subject
IDs. Times must satisfy ``0 <= entry < time``. Keep these fields out of the
feature matrix. Sample weights are subject likelihood weights, not censoring
indicators. Integer weights have frequency-replication semantics; survey-design
or clustered robust inference is not implied.

.. code-block:: python

   from nampy import SurvivalGAM
   from nampy.survival import SurvivalTarget

   target = SurvivalTarget(time, event, entry=entry)
   model = SurvivalGAM(cuts=[0, 1, 3], formula="~ s(age, k=6)")
   model.fit(X, target, sample_weight=weights)
   curves = model.predict_survival(X_new, times=[0, .5, 1, 2, 3])

The statistical model defaults to an interval-factor baseline and delegates
fitting and smoothing to the existing Poisson GAM. The expanded
count mean is exposure times hazard; survival predictions remove the bookkeeping
log-exposure offset. Genuine user log-rate offsets must be supplied again for
new subjects. ``gam_`` exposes the unchanged statistical fit.
``native_poisson_log_likelihood_`` is the expanded count likelihood;
``survival_log_likelihood_`` removes the event log-exposure constant.
``score`` returns weighted observed-event log likelihood divided by total subject
weight, with larger values better. It is not concordance or prediction accuracy.

Follow-up and fitted profiles
-----------------------------

.. figure:: /_static/features/survival_profiles.png
   :alt: Observed follow-up with event circles and censoring crosses, beside fitted Weibull survival for three covariate profiles.
   :width: 100%

   Follow-up uses arbitrary time units. The separate test cohort contains
   33% right-censored records; a cross
   records surviving to the end of follow-up, not an event at that time.
   At time 2, the fitted survival probabilities for x=-0.8, 0 and 0.8 are
   approximately 0.65, 0.31 and 0.04. See the complete
   :doc:`censoring tutorial </notebooks/survival_censoring>` for subject splits
   and training-only censoring estimation.

Curve semantics
---------------

Curves have shape ``(n_subjects, n_times)``. Times are finite, nonnegative, and
strictly increasing. Estimator cuts start at zero; unconditional survival starts
at one. ``conditional_on=landmark`` computes survival conditional on reaching
that landmark. Cells before a subject's landmark are invalid and returned as
NaN. This prediction setting is independent of training entry times.

Hazards are constant on ``(left, right]`` intervals; an event exactly at a cut is
assigned to the preceding interval. Exposure integrates partial first and last
intervals exactly. ``predict_distribution(X)`` returns a snapshot of hazard
rates and saved cuts. Its quantiles invert the event-time CDF; unreached
quantiles and medians are NaN. ``rmst(horizon)`` integrates survival from zero to
the requested horizon exactly. It is not an unrestricted mean lifetime.
``rmst(horizon, conditional_on=landmark)`` integrates conditional survival from
the landmark to the horizon, returning remaining lifetime after that landmark.
Landmarks can be scalar or one per subject and must not exceed the horizon.
Conditional quantiles are not yet exposed.

For piecewise models, statistical ``predict_components`` returns a ``SurvivalComponents`` record with
an additive prediction on subject-by-interval rows and explicit row mappings.
Neural components have shape ``(n_subjects, n_intervals)``. In both cases they
describe log-hazard predictors, with exposure offsets excluded. They are not
additive effects on survival probabilities or confidence intervals.

For Weibull/lognormal, both backends return ``AdditivePrediction`` with shape
``(n_subjects, 2)`` for the parameter predictors. Centering changes only the
decomposition, preserving predictors and known offsets.

Existing Weibull and lognormal ``DistributionPrediction`` objects can be wrapped
with ``SurvivalPrediction(distribution=...)`` for probability curves. This
provides latent event-time probability operations. Fitted parametric estimators
return the same view; ``distribution`` exposes their existing scalar law.
Parametric RMST integrates the shared event-time law. Both unconditional and landmark-conditioned RMST are available
without dividing underflowed survival probabilities or requiring a finite
unrestricted mean. Parametric predictions beyond
observed follow-up use the assumed event-time law, so they are model-based tail
extrapolations.

Evaluate on held-out subjects
-----------------------------

For right-censored evaluation without delayed entry, fit censoring survival
from training outcomes and evaluate the frozen model on test subjects:

.. code-block:: python

   from nampy.survival import CensoringDistribution, brier_score

   prediction = model.predict_distribution(X_test)
   censoring = CensoringDistribution.fit(target_train)
   errors = brier_score(
       prediction, target_test, times=[.5, 1., 2.], censoring=censoring,
   )
   print(errors.integrated())

Brier error measures probability prediction error; smaller values are better.
It assumes marginal independent censoring and positive training censoring
survival on the requested support. Choose times within model, training and test
follow-up; do not extrapolate the censoring distribution. See
:doc:`survival_evaluation` for the full contract, descriptive curves, residuals
and concordance.


Choose a focused task
---------------------

.. toctree::
   :maxdepth: 1

   survival_likelihoods
   survival_cox
   survival_longitudinal
   survival_evaluation
   survival_uncertainty
   survival_calibration

Limitations
-----------

Every fitted interval needs positive-weight exposure. The default saturated
factor baseline also needs events in every interval; its zero-event boundary
fits are explicitly unsupported. Smooth baselines use coupled coefficients and
retain native finite/converged fitting requirements. Cuts must cover all
observed times; the implementation never merges intervals, adds pseudo-events,
or extrapolates the final rate beyond the last cut automatically.

Neural survival currently supports LinReg and NAM gradient training. Native
boosting, fixed-design least squares, custom sampling, dropping final batches,
distributed execution, gradient accumulation, and external warm-start
checkpoints remain unavailable. Failed refits preserve the published model;
ordinary in-memory warm starts require unchanged task and cut semantics.
Switching a fitted model's survival likelihood is incompatible with a warm start.

Longitudinal/delayed-entry Cox, censored EFS and separate
ML/LAML/GCV/UBRE/NCV smoothing criteria, statistical censored likelihood
derivatives above order four, parametric time-varying covariates,
competing risks, recurrent events, left/interval censoring, delayed-entry
IPCW/concordance, conditional censoring models, dynamic AUC, Uno concordance,
randomized censor-aware PIT, continuous-time bands, single-fit neural
coefficient/effect/hazard/RMST intervals and simultaneous inference from replicated fits
are not implemented.
Do not pass censored times to ordinary uncensored residual or calibration
functions as exact events. The piecewise model is not mgcv's event-risk-set
Poisson representation of Cox partial likelihood.

References
----------

Cox partial likelihood and Breslow baseline/tie handling are grounded in
[Cox1972]_ and [Breslow1974]_. The piecewise-exponential/Poisson representation
is described by [Laird1981]_; censored and truncated likelihood conventions
are covered by [Lawless2003]_. [Katzman2018]_ supplies neural Cox background,
while NAMpy retains its own registered architectures and supported cohort rules.

Descriptive curves, evaluation and restricted means refer to [Kaplan1958]_,
[Aalen1978]_, [Gerds2006]_, [Harrell1982]_ and [Royston2013]_. Subject resampling
uses the bootstrap principle of [Efron1979]_. The cutoff conformal route refers
to [Candes2023]_. Full citations and paper links are in :doc:`/references`.
