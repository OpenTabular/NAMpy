Survival models
===============

NAMpy supports piecewise-exponential, censored Weibull, and censored lognormal
models through its statistical engine and gradient-trained LinReg and NAM
architectures. Both consume
``nampy.survival.SurvivalTarget`` and produce the same survival prediction view.
Static records use one terminal event or right censoring per subject.
Piecewise models additionally accept complete changing-covariate histories
through ``LongitudinalSurvivalTarget``. A separate statistical ``CoxGAM``
supports static fixed-smoothing partial likelihood.
Conditionally independent censoring and entry are assumed.

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

Smooth piecewise baseline
-------------------------

``SurvivalGAM(baseline="smooth", k=6, ...)`` replaces the independent interval
intercepts with a penalized ``s(survival_time,k=6)`` term evaluated at each
interval's right endpoint. This follows the piecewise additive model convention:
hazards remain constant inside intervals and integrate using actual exposure.
It is neither continuous spline-hazard quadrature nor Cox partial likelihood.
``baseline="factor"`` retains the existing default behavior.

At least ``k`` distinct interval endpoints are needed. Spline bases, penalties,
centering, fixed smoothing and supported automatic smoothing remain owned by
the existing Poisson GAM. Fitted basis transformations are reused at prediction.
All intervals need positive-weight exposure. A coupled smooth baseline may
support intervals without observed events, subject to the ordinary native
finite/converged endpoint checks; no pseudo-events or optimizer rescue are added.
The saturated factor baseline still requires events in every fitted interval.

The same shared prediction, components, curve/effect inference, bootstrap and
conformal operations apply. Baseline terms remain temporal context in effect
preparation rather than becoming source covariates. Explicit time-dependent covariate effects use the temporal_formula contract
described below. Parametric time-varying covariates remain
rejected in this new route. Weibull/lognormal models do not accept a smooth
piecewise baseline. Neural survival retains its interval-specific heads; an
identical penalized spline-baseline representation is not implemented there.

Censored Weibull and lognormal models
-------------------------------------

The parametric routes reuse the existing latent event-time laws. An exact event
contributes ``log f(time)`` and a right-censored record contributes
``log S(time)``. Every record also subtracts ``log S(entry)`` for delayed entry;
entry zero contributes zero. This conditions on survival to entry and excludes
the censoring-process density.

.. code-block:: python

   weibull = SurvivalGAM(
       likelihood="weibull", smoothing_optimizer="outer_newton",
       formula={"scale": "~ s(age, k=6)", "shape": "~ 1"},
   ).fit(X, target, sample_weight=weights)

   lognormal = SurvivalGAM(
       likelihood="lognormal", smoothing_optimizer="bfgs",
       formula={"loc": "~ s(age, k=6)", "scale": "~ 1"},
   ).fit(X, target)

Omit cuts for these models. A single RHS formula models the first parameter
with an intercept-only second parameter; a mapping can model both. Weibull
parameters are positive scale and shape, both log linked. Lognormal has real
log-time location and positive log-time standard deviation, with identity/log
links. The lognormal density retains its response Jacobian. An AFT or
proportional-hazards interpretation requires the appropriate constant
scale/shape restrictions; variable-parameter models are distributional survival.

Statistical censor-aware adapters provide analytic mixed predictor derivatives
through order four using the generic joint fitting engine. Automatic joint REML
supports ``smoothing_optimizer="outer_newton"`` and ``"bfgs"``. Both distribution
parameters can have smooth terms. For fixed smoothing, use
``optimize_smoothing=False`` with explicit ``smoothing_params``. The existing
general-family method coercion, numerical kernels, and stopping rules are
unchanged; the survival facade rejects separate ML/LAML/GCV/UBRE/NCV criteria.

Censored EFS remains unavailable because its fitted endpoints did not pass the
independent exact-REML profile check. It is rejected by the backend family route
declaration as well as the survival facade. Existing uncensored EFS behavior is
preserved. The optim/magic/nlm.fd routes are also unavailable for these censored
adapters.

``gam_`` exposes coefficients and their covariance. Where the backend computes
smoothing-corrected covariance, it is available through
``gam_.vcov(unconditional=True)``; conditional covariance remains distinct. A
pointwise survival interval API is described below. Neither covariance
describes event-time variation or variability between neural fits.

For either backend, parametric offsets are a finite ``(n_subjects, 2)`` matrix,
one known offset per parameter predictor. Supply new offsets for new subjects
when the fit used them. Fitting initially requires at least two distinct
positive-weight exact event times; all-censored and degenerate/boundary fits are
explicitly unsupported. These guards do not guarantee arbitrary formulas are
identifiable.

Neural configuration
--------------------

.. code-block:: python

   from nampy import GradientTrainingConfig, LinRegSurvival, SurvivalObjectiveConfig

   model = LinRegSurvival(
       objective=SurvivalObjectiveConfig(cuts=(0, 1, 3)),
       training=GradientTrainingConfig(max_epochs=100, batch_size=128),
   )
   model.fit(X, target, X_val=X_validation, y_val=validation_target)

To fit a neural parametric model, replace the objective and omit cuts:

.. code-block:: python

   model = LinRegSurvival(
       objective=SurvivalObjectiveConfig(likelihood="weibull"),
       training=GradientTrainingConfig(max_epochs=500, batch_size=128),
   )
   model.fit(X, target, X_val=X_validation, y_val=validation_target)

``likelihood="lognormal"`` uses the same interface. Parametric neural outputs
are the two parameter predictors, using the same links as the statistical
adapters. Their observation likelihood is differentiable, includes delayed
entry, and uses the same subject-weight normalization. Times and entry retain
double precision in this route; parameters are not silently clipped to make
invalid likelihoods finite.

``NAMSurvival`` accepts the same objective and training configuration, with its
usual ``architecture_config``. Each interval has its own log-hazard output and
can have its own covariate effects. This permits nonproportional hazards; it is
not equivalent to a shared covariate effect across the statistical model's
intervals unless the fitted predictors satisfy that restriction.

Training batches contain whole subjects. Their likelihood sums event and
exposure contributions across intervals and uses a global subject-weight
normalizer, so adding interval rows does not dilute the likelihood or change
the declared regularization scale. Validation loss uses its own global subject
normalizer. Preprocessing is fitted on training subjects only, with no outcome
labels supplied: initially numerical none/standardization and categorical
one-hot/int transformations are supported. Explicit subject IDs must not overlap
between training and validation sets. A random validation split may leave an
interval without events; choose an explicit supported split or appropriate cuts.

Changing the number of intervals also changes the neural output parameterization.
Parameter penalties can therefore depend on that model choice. Subject-level
loss normalization prevents exposure-row dilution; it does not assert identical
penalties for two different output parameterizations.

Curve semantics
---------------

.. autoclass:: nampy.survival.SurvivalTarget
   :members: from_frame, take

.. autoclass:: nampy.survival.SurvivalPrediction
   :members: survival, log_survival, cumulative_hazard, cumulative_hazard_at, hazard, quantile, median, rmst, take

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
Weibull/lognormal RMST uses adaptive quadrature of the shared log-survival law,
with a logarithmic time change and a median break point. Absolute error control
is ``1e-10`` in time units and relative control is ``1e-9``; integration failures
raise explicitly. Both unconditional and landmark-conditioned RMST are available
without dividing underflowed survival probabilities or requiring a finite
unrestricted mean. Parametric predictions beyond
observed follow-up use the assumed event-time law, so they are model-based tail
extrapolations.

Shared diagnostics and evaluation
---------------------------------

Functions in ``nampy.survival`` consume the same ``SurvivalPrediction`` and
``SurvivalTarget`` for either backend. They do not refit a model or change its
``score`` definition. Residual rows retain subject order and IDs.

.. code-block:: python

   from nampy.survival import survival_residuals, survival_curve

   prediction = model.predict_distribution(X_test)
   residuals = survival_residuals(prediction, y_test)
   observed = survival_curve(y_test, sample_weight=test_weights)
   remaining_lifetime = prediction.rmst(3, conditional_on=y_test.entry)

``cox_snell`` is the integrated hazard from entry to exit. It remains right
censored when ``residuals.event`` is false. ``martingale`` is event minus this
integrated hazard; ``deviance`` is its counting-process transformation. These
are unweighted row diagnostics, not uncensored normal residuals, coefficient
uncertainty, or independent fitted errors. A zero-hazard exact event retains
infinite deviance. Censor-aware randomized PIT and worm plots remain unavailable.

``survival_curve`` provides weighted Kaplan–Meier survival and Nelson–Aalen
cumulative hazard with entry-aware ``entry < t <= exit`` risk sets. Its
``at_risk``, ``events`` and ``censored`` arrays expose tied-time handling.
Events precede censoring at tied exits. The two curve estimates remain distinct:
KM survival is not replaced by ``exp(-Nelson–Aalen)``. Descriptive curves do not
provide confidence bands or identify population survival before delayed entry.

For right-censored evaluation **without delayed entry**, fit a censoring
distribution on training outcomes explicitly and pass it to the metric:

.. code-block:: python

   from nampy.survival import CensoringDistribution, brier_score, concordance_index

   censoring = CensoringDistribution.fit(y_train, sample_weight=train_weights)
   errors = brier_score(
       prediction, y_test, times=[.5, 1, 2], censoring=censoring,
       sample_weight=test_weights,
   )
   integrated_error = errors.integrated()
   concordance = concordance_index(
       y_test, -prediction.rmst(2), higher_risk=True,
       sample_weight=test_weights,
   )

IPCW Brier errors use reverse-KM censoring survival from training, never an
estimate fitted implicitly on test outcomes. Marginal independent censoring
is assumed; covariate-dependent censoring adjustment is unavailable. At tied
training times, observed events are removed before a censoring jump. Evaluation
uses right-continuous ``G(t)``, including event exit times. Horizons must be
within model and training support and precede final positive-weight test
follow-up. Required censoring probabilities must be positive; no clipping or
tail extrapolation is performed. Earlier censored rows contribute zero error
but retain their case weights in the score denominator. ``integrated()`` is a
trapezoidal average over the supplied grid's first to last time, requiring at
least two grid points; it is not an integral from zero implicitly. These follow
the `scikit-survival Brier definition <https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.metrics.brier_score.html>`_
with explicit support checks.

Harrell concordance requires an explicit risk direction. Here negative RMST
defines a horizon-specific ordering; a distribution does not imply a universal
risk ordering. An earlier event compares with later exits and same-time censored
exits; event/event time ties are excluded and risk ties receive half credit.
The result reports weighted concordant, discordant, tied-risk and comparable
pair totals. Pair weights are products of case weights. No comparable pairs is
an error. This is Harrell concordance, not Uno's IPCW concordance, and its value
can depend on censoring. Its tie convention follows
`scikit-survival's concordance contract <https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.metrics.concordance_index_censored.html>`_.

.. automodule:: nampy.survival.evaluation
   :members:

.. automodule:: nampy.survival.diagnostics
   :members:

Survival effects and curve uncertainty
--------------------------------------

Shared effect preparation consumes additive predictor components from either
backend and evaluates hypothetical predictors through ``survival_from_predictors``.
It returns arrays and labels, without rendering or accessing backend training
objects.

.. code-block:: python

   from nampy.survival import prepare_survival_effects

   effects = prepare_survival_effects(
       model, X_test, times=[0, .5, 1, 2], reference_X=X_reference,
       reference_weight=reference_weights, conditional_on=.2,
   )

For each selected term, its reference mean is moved to the predictor intercept;
the reference predictor plus that centered term is then transformed to survival.
``reference_survival`` and each ``survival[term]`` have shape
``(n_subjects, n_times)``. ``effects[term]`` is their difference. These contrasts
do not sum to a decomposition of survival probabilities and are not partial
dependence or causal effects. Piecewise interval baselines retain their own
reference values. Known offsets stay at their supplied subject values; use
``offset`` and ``reference_offset`` explicitly when the fit requires them.
Reference weights center contributions, never event times or censoring flags.

Term metadata selects known main effects, categorical effects, and interactions.
``features`` names all coordinates and ``feature_values`` retains them, including
categorical labels. ``terms=[...]`` selects explicit term labels. Unsupported or
contextual terms are rejected when requested; they are not silently treated as
unary effects. Conditional cells before a landmark remain NaN. This preparation
operation returns point contrasts; their separate interval operation is below.

Statistical curve intervals reuse the established joint coefficient sampler,
including full covariance across predictors and terms:

.. code-block:: python

   from nampy.survival import prepare_survival_curve_interval

   interval = prepare_survival_curve_interval(
       statistical_model, X_test, times=[0, .5, 1, 2],
       conditional_on=.2, covariance="bayes", n_draws=1000, seed=42,
   )

``estimate``, ``lower`` and ``upper`` have shape ``(n_subjects, n_times)``.
The estimate is the fitted survival curve; bounds are central simulation
quantiles. ``uncertainty_kind="coefficient_conditional"`` identifies Bayesian
coefficient covariance conditional on smoothing. ``covariance="unconditional"``
requires available smoothing-corrected covariance and labels it
``"coefficient_smoothing_corrected"``. Absence of that covariance raises, without
substituting conditional covariance. Offsets stay fixed in each joint draw.
The default ``interval="pointwise"`` is approximate and subject to Monte Carlo
error; ``level`` alone does not imply simultaneous coverage.
``standard_draws`` can supply common standard normals for reproducible checks.
Transformed coefficients and wider/nonidentity prediction-coordinate maps are
not validated for this survival simulation route and raise explicitly.

To request a simultaneous survival band, set ``interval="simultaneous"``:

.. code-block:: python

   band = prepare_survival_curve_interval(
       statistical_model, X_test, times=[0, .5, 1, 2],
       conditional_on=.2, covariance="bayes", n_draws=2000, seed=42,
       interval="simultaneous",
   )

The coverage set is **all valid cells across all supplied subjects and times**,
not a separate band per subject. ``scope="all_subjects_and_times"`` records this
choice. The calculation uses joint draws of log cumulative hazard, their sample
standard deviations (``ddof=1``), and a simulated quantile of the maximum absolute
standardized deviation from the fitted log cumulative hazard. Critical-value
quantiles use R type 8, as in the existing joint functional calibration. Bounds
are transformed back through ``exp(-exp(value))`` and lie in [0, 1]. Reported
``standard_error`` is on the ``transformation="log_cumulative_hazard"`` scale,
and ``critical_value`` is shared over the coverage set.

Landmark/time-zero cells with zero cumulative hazard in every draw have bounds
exactly one. Pre-landmark cells remain NaN and are excluded from calibration.
An entirely invalid grid raises. Nonfinite hazards, inconsistent zero boundaries,
or zero-variance draws differing from the fitted estimate raise explicitly.
Bounds need not themselves be monotone survival laws; no monotonicity projection
is performed. Coverage is approximate under the joint coefficient Gaussian
simulation, on the **supplied finite grid only**. It is not a continuous-time,
new-subject population, or event-time prediction guarantee.

Effect-contrast intervals retain the existing reference-plus-term definition:

.. code-block:: python

   from nampy.survival import prepare_survival_effect_interval

   effect_intervals = prepare_survival_effect_interval(
       statistical_model, X_test, times=[0, .5, 1, 2],
       reference_X=X_reference, reference_weight=reference_weights,
       conditional_on=.2, interval="simultaneous", n_draws=2000, seed=42,
   )
   # effect_intervals.effects retains feature labels, coordinates and point curves.
   # effect_intervals.intervals maps each selected term to its interval record.

Every coefficient draw produces both the term curve and its reference curve;
the contrast is their paired difference. Reference centering is recomputed from
the same coefficient draw, with fixed reference rows and weights. This retains
reference, intercept, cross-term and cross-predictor covariance. Subtracting two
independently generated interval endpoints would discard that dependence and
does not implement this calculation. Known offsets remain fixed. These intervals
condition on the reference population supplied; they do not resample it.

``interval="pointwise"`` returns central quantiles of the paired differences.
``interval="simultaneous"`` calibrates a **single critical value jointly across
all selected terms, subjects and valid times**, using maximum standardized
deviations on the survival-difference scale. Its metadata includes
``scope="all_selected_terms_subjects_and_times"`` and
``estimand="reference_plus_term_survival_difference"``. Bounds intersect the
known difference support [-1, 1]. Use ``terms=[...]`` to declare the set to cover;
changing that set or the evaluation grid changes the simultaneous coverage set.
Bayesian and available smoothing-corrected covariance remain distinct choices.
Grouped categorical/interaction columns use the existing statistical prediction
term groups and retain original covariate metadata.

Hazard and RMST intervals use the same shared prediction snapshots and joint
coefficient sampler:

.. code-block:: python

   from nampy.survival import (
       prepare_survival_hazard_interval, prepare_survival_rmst_interval,
   )

   hazard_interval = prepare_survival_hazard_interval(
       statistical_model, X_test, times=[.1, .5, 1, 2], conditional_on=.2,
       n_draws=1000, seed=42, interval="simultaneous",
   )
   rmst_interval = prepare_survival_rmst_interval(
       statistical_model, X_test, horizons=[0, .5, 1, 2], conditional_on=.2,
       n_draws=1000, seed=42, interval="simultaneous",
   )

Both default to ``interval="pointwise"`` (central draw quantiles) and accept the
same explicit offsets, covariance choice and common ``standard_draws`` as curve
intervals. Conditional and available smoothing-corrected covariance remain
separate; unavailable correction raises. The estimate is the fitted functional,
not a posterior mean. Values and bounds have shape ``(n_subjects, n_times)``
for hazard and ``(n_subjects, n_horizons)`` for RMST. In the latter record,
``time`` stores the explicitly supplied restriction horizons.

Hazards retain inverse-time units and the existing ``(left, right]`` piecewise
cut convention. A conditioning landmark masks earlier times; it does not change
the hazard at or after that landmark. Parametric hazard limits at time zero
remain unsupported, even if that cell would be masked. Simultaneous hazard
bands use log hazard, with ``scope="all_subjects_and_times"`` and
``transformation="log_hazard"``. Reported standard errors are on that scale.
Zero hazards shared by all draws have zero limits; inconsistent zero boundaries
or nonfinite numerical bounds raise.

RMST means remaining lifetime integrated from each landmark (default zero) to
each requested horizon. A horizon earlier than its subject's landmark is NaN;
an equal horizon gives zero. **Each joint coefficient draw is integrated** using
the existing exact piecewise or controlled Weibull/lognormal operation. Integrating
survival confidence-band endpoints is not this calculation. Subject selection
uses ``SurvivalPrediction.take`` without reordering the resulting subject rows.
Parametric integration cost grows with draw count, subjects and horizons; use
an explicit grid and Monte Carlo budget appropriate to the requested precision.
Quadrature failures propagate without substituting approximations.

Simultaneous RMST intervals use the same maximum-standardized-deviation
calibration in time units, with ``scope="all_subjects_and_horizons"`` and
``estimand="restricted_mean_remaining_lifetime"``. Both pointwise and simultaneous
limits intersect [0, horizon - landmark], preserving ordered bounds even at a
quadrature upper boundary. The fitted estimate retains its numerical integration
precision. These are finite-horizon functionals, not unrestricted mean lifetime.
Hazard and RMST simultaneous coverage each concern their own supplied finite
grid; separate calls do not claim joint coverage across functionals. Piecewise
tails beyond cuts, continuous-time bands, and single-fit neural coefficient
inference remain unsupported.

A single neural gradient fit has no supported coefficient covariance or
smoothing interval. Both engines can instead summarize snapshots from
independently replicated fits on the **same subject rows in the same order**:

.. code-block:: python

   from nampy.survival import combine_survival_predictions

   variability = combine_survival_predictions(
       [fit.predict_distribution(X_test) for fit in replicated_fits],
       times=[0, .5, 1, 2], level=.95,
   )

This averages survival probabilities, retaining nonlinear response semantics;
it does not average hazards or natural parameters before transformation.
``uncertainty_kind="between_fits"`` and ``coverage="empirical_range"`` identify
central empirical quantiles, not confidence intervals. Replication and subject
alignment are caller responsibilities; the helper does not bootstrap, refit,
or reorder predictions. Neither coefficient intervals nor between-fit ranges
describe event-time spread. Single-fit neural simultaneous/effect coefficient
intervals (including hazard/RMST), simultaneous calibration from replicated fits,
and continuous-time bands remain unsupported. Existing ``mgcv`` fitting,
prediction and covariance behavior is unchanged.

.. automodule:: nampy.survival.interpretation
   :members:

.. automodule:: nampy.survival.uncertainty
   :members:

Subject-level bootstrap uncertainty
-----------------------------------

``bootstrap_survival`` refits fresh statistical or neural estimators on ordinary
uniform subject-pairs samples. Supply an already fitted estimator, its original
training data/configuration, and a fixed prediction cohort. Each selected row
carries its covariates, time, event, delayed entry, case weight and offset
together. Duplicate occurrences receive distinct local IDs for fitting; the
result retains their original source IDs and full resampling indices for audit.
Data are physically duplicated and recompiled, so statistical basis construction,
centering and configured automatic smoothing run through their native routes.
Multiplicity weights on the original design are not substituted for row refits.

.. code-block:: python

   from nampy.survival import bootstrap_survival

   bootstrap = bootstrap_survival(
       fitted_model, X_train, y_train, X_test,
       sample_weight=train_weights, offset=train_offsets,
       prediction_offset=test_offsets, n_resamples=200, seed=42,
   )
   sampling_interval = bootstrap.percentile_interval(
       grid=[0, .5, 1, 2], conditional_on=.2,
   )
   # estimand="hazard" uses positive times for parametric models.
   # estimand="rmst" interprets grid as restriction horizons and integrates each refit.

The point estimate is the original fitted functional; bounds are pointwise
percentiles of refitted predictions using R type-7 quantiles. This is distinct
from a coefficient Gaussian posterior approximation and from generic between-fit
empirical ranges. ``uncertainty_kind="subject_bootstrap"`` records subject
sampling, with ``coverage="pointwise_percentile"``. Support, conditioning and
numerical integration rules remain those of the shared prediction contracts.
More refits are needed for a useful Monte Carlo approximation than the tiny
budgets used in runnable examples. No repeated-data coverage guarantee is claimed.

Neural bootstrap requires **an explicit fixed external validation cohort**, with
globally meaningful, disjoint subject IDs on both original targets. Fit the
original neural estimator with that same cohort, then pass it to bootstrap:

.. code-block:: python

   # y_train and y_validation contain explicit, disjoint subject_id values.
   neural.fit(X_train, y_train, X_val=X_validation, y_val=y_validation)
   bootstrap = bootstrap_survival(
       neural, X_train, y_train, X_test,
       X_val=X_validation, y_val=y_validation,
       n_resamples=200, seed=42, training_randomness="fixed",
   )

This conditions on the supplied validation cohort rather than resampling it.
Automatic row splitting after duplication is rejected, since copies of the same
subject could otherwise cross training and validation. Fitted preprocessing is
never reused: each clone learns it from its own resampled training rows only.
Validation rows remain fixed and prediction rows are not used for training.
Statistical bootstrap does not accept validation data.

``training_randomness="fixed"`` preserves the original neural fit seed, so
sampling variation is conditional on the configured training algorithm/random
stream. Seed control does not eliminate GPU or numerical nondeterminism.
``training_randomness="independent"`` is a separate neural route with independent
recorded seeds, labeled ``subject_bootstrap_with_training_randomness``. It mixes
subject sampling and training variation and does not estimate separate variance
components. Statistical training randomness is not applicable and its independent
neural-seed mode is rejected. Resampling and training use separate RNG streams;
changing the neural seed policy does not change the subject-index plan.

``resample_indices`` optionally supplies an integer matrix of shape
``(n_resamples, n_original_subjects)`` for replay or external reference checks;
its row count replaces ``n_resamples``. Uniform sampling is over original rows.
Case weights are carried with those rows, not used as sampling probabilities or
expanded into virtual subjects. This does not implement survey, cluster,
stratified, censoring-stratified or parametric bootstrap semantics.

Each refit must have eligible native fit status and finite observed-event
training likelihood. Neural training completion remains distinct from optimizer
convergence. The default failure policy raises ``SurvivalBootstrapError`` with
replicate/stage provenance. ``on_failure="record"`` attempts the preselected
samples without retries and returns ``failures`` and ``successful_replicates``;
Every interval operation refuses any failed replicate rather than silently
conditioning inference on successful fits. Samples with absent interval events,
lost feature support or unsupported native endpoints may therefore make inference
unavailable. No cut merging, pseudo-events, optimizer rescue or replacement
sampling is performed.

The source estimator and caller-owned configuration are not refitted or changed.
Neural warm starts and caller-owned trainer execution objects are rejected;
temporary checkpoints are isolated per replicate and cleaned up. Results retain
only shared prediction snapshots and resampling/failure/seed metadata, not
trainers or fitted replicas. Existing model save/load formats are unchanged.
Simultaneous bands and paired effect contrasts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``simultaneous_interval`` supports survival, hazard and RMST on a declared finite
grid. Calibration takes the maximum absolute standardized deviation from the
original fitted estimate over **all valid subjects and grid cells**, using
empirical bootstrap standard deviations (ddof=1) and an R type-8 critical
quantile. Survival uses log cumulative hazard, hazard uses log hazard, and RMST
uses its original scale with physical lifetime bounds. This is a centered
maximum-deviation bootstrap band; it does not use nested studentization or imply
continuous-time coverage. Separate functional calls have separate coverage.

.. code-block:: python

   band = bootstrap.simultaneous_interval(
       grid=[.5, 1, 2], estimand="survival", conditional_on=.2)
   hazard_band = bootstrap.simultaneous_interval(
       grid=[.5, 1, 2], estimand="hazard", conditional_on=.2)
   rmst_band = bootstrap.simultaneous_interval(
       grid=[.5, 1, 2], estimand="rmst", conditional_on=.2)

Declare effect preparation **before refitting**, because fitted replicas are not
retained. ``effects`` accepts the shared ``prepare_survival_effects`` arguments;
use ``prediction_offset`` for prediction offsets rather than an ``offset`` key
inside this mapping. A fixed reference cohort, reference weights and known
reference offsets are supported.

.. code-block:: python

   bootstrap = bootstrap_survival(
       model, X_train, y_train, X_predict, n_resamples=200, seed=42,
       effects={"times": [0, .5, 1, 2], "reference_X": X_reference,
                "reference_weight": reference_weight, "conditional_on": .2})
   pointwise_effects = bootstrap.effect_interval()
   simultaneous_effects = bootstrap.effect_interval(interval="simultaneous")

For neural models pass the same fixed external validation arguments shown above.
Each refit recenters its own contributions against the fixed reference cohort,
then computes both sides of the reference-plus-selected-term survival contrast
from that same fitted model. Contrasts are captured as shared ``SurvivalEffects``
arrays, retaining term and predictor metadata. Their simultaneous calibration is
joint over **all selected terms, subjects and valid times**, on the survival
difference scale with bounds [-1,1]. Missing or incompatible term layouts count
as failed replicates, rather than becoming zero effects or a smaller target.
The reference population is not resampled.

These operations return the existing shared interval records, with coverage
``simultaneous_bootstrap`` or ``pointwise_percentile`` and the original bootstrap
uncertainty source. Structural boundaries and pre-landmark NaNs are preserved.
Zero empirical variance is allowed only when every draw agrees with the fitted
estimate; disagreement raises explicitly. Tiny bootstrap budgets illustrate the
API but do not provide useful calibrated inference.

BCa, basic and nested-studentized intervals, bootstrap validation/reference
population uncertainty and automatic parallel refits remain unsupported. Single-fit neural coefficient intervals remain unsupported even
though subject-level neural bootstrap is available.

.. automodule:: nampy.survival.bootstrap
   :members:

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

Conformal event-time lower bounds
---------------------------------

See :doc:`calibration` for conservative observed-time and administrative-cutoff
survival conformal methods shared across both backends. They return one-sided
future-event-time bounds. They do not calibrate survival curves, confidence
bands or event-time upper limits. Delayed-entry conformal and automatic
censoring-mechanism estimation remain unsupported.

Longitudinal piecewise models
-----------------------------

``LongitudinalSurvivalTarget(start, stop, subject_id, event)`` describes complete
contiguous subject histories. Covariates in X are aligned with those risk records
and must be known/predictable over each ``(start,stop]`` interval. Group blocks
cannot be interleaved; gaps, overlaps and nonterminal events are rejected. Event
0 on an intermediate row means the subject continues into the next row. The
final row supplies one event or right censoring. Delayed entry is the first
start time; it is allowed during piecewise fitting.

``SurvivalGAM``, ``LinRegSurvival`` and ``NAMSurvival`` accept this contract for
piecewise-exponential fitting. Exposure splitting includes both model cuts and
record boundaries. Case weights are supplied per record but must agree across
all records of each subject. Offsets may vary across records. The likelihood
is the sum of conditional at-risk contributions; scores divide by total subject
case weight. It does not treat record count as independent-subject count.

Neural longitudinal fitting requires an explicit disjoint whole-subject
validation cohort with the same observation contract. No automatic record
split is performed. ``target.subset(subject_ids)`` selects whole subjects only.
The objective packs subject-normalization allocations alongside exposure/events;
minibatches retain the fixed full-cohort normalization. Preprocessing sees only
training records. Neural preprocessing/optimization can depend on record
frequency and minibatch order: likelihood invariance when an identical record
is subdivided does not imply identical neural training trajectories.
Longitudinal warm starts, native neural fitting, parametric time-varying
Weibull/lognormal paths, recurrent events and partial-history bootstrap are
unsupported. Whole-trajectory bootstrap is supported through the subject
resampling contract described below.

Use ``SurvivalTrajectory`` and shared ``predict_trajectory`` for future covariate
paths. Every prediction subject must supply a complete path from zero to the
same finite horizon within fitted support. The helper combines model cuts and
all change times, then uses the correct covariate state/model interval in each
cell. It returns the existing ``SurvivalPrediction``; row order follows
``trajectory.subject_ids``. Missing future histories are not forecast, filled
or extrapolated. Supplied paths describe conditional scenarios; uncertainty in
future covariates is not modeled. Path-specific effect preparation and bootstrap
contrasts/bands are supported below. Single-fit path coefficient intervals and
longitudinal conformal calibration remain unavailable.

Ordinary ``predict_link``/``predict_distribution`` retain their existing
constant-covariate interpretation for each supplied row; they do not silently
join record rows into subject paths. Existing constant-covariate inference
therefore remains distinct from inference about a changing trajectory.

.. literalinclude:: ../../examples/example_longitudinal_survival.py
   :language: python

Native fixed-smoothing Cox models
---------------------------------

``CoxGAM`` fits static-covariate penalized partial likelihood in the existing
identified spline designs. Risk sets and tied event/censor handling follow the
actual ``coxlpl``/``coxpp`` calculations in upstream mgcv, rather than substituting
an Efron tie approximation or a coarse Poisson model. Independent mgcv fixtures
cover linear, smooth, categorical and stratified endpoints and coefficient
covariance. This validates fixed-smoothing endpoints, not all mgcv Cox routes.

.. code-block:: python

   from nampy import CoxGAM

   cox = CoxGAM(formula="~s(age,k=6)+treatment",
                smoothing_params=[.5]).fit(X_train, target_train)
   prediction = cox.predict_distribution(X_test)
   survival = prediction.survival([.5, 1, 2])
   restricted_mean = prediction.rmst(2)

Use explicit ``strata=...`` for a stratified fit and prediction; new strata have
no baseline and are rejected. Explicit training offsets must be supplied anew
for prediction. The global intercept is removed after conventional contrast
coding because it cancels from partial likelihood. Fixed spline penalties,
compiler bases and term ownership are reused. The dedicated statistical solver
uses exact risk-set coefficient gradients/Hessians with Newton-CG; it does not
claim to reproduce mgcv's outer smoothing iterations. Nonconvergence/nonfinite
curvature and rank failures raise without covariance jitter or solver rescue.

``predict_components`` reconstructs log relative hazard. ``score`` is partial
log likelihood per subject on the supplied risk-set cohort, not a normalized
survival density or an observation-independent scoring rule. Do not rank it
against full event-time likelihoods from piecewise or parametric models.

Absolute survival predictions use a step cumulative baseline hazard. The shared
prediction record supports right-continuous survival/cumulative hazard, slicing,
quantiles reached on observed support, conditional survival and exact step RMST.
No continuous hazard density is manufactured from cumulative jumps. Prediction
uses common follow-up support across requested strata and rejects extrapolation.
Baseline/coefficient-to-curve intervals need their own validated propagation;
coefficient covariance alone is not used to claim survival confidence coverage.

Alternative ties, delayed-entry or longitudinal Cox,
case weights, analytic simultaneous/conditional Cox intervals and Cox effect
intervals remain unsupported. The
existing probability comparison helpers do not treat partial likelihood as a
full density. Save/load retains the version-6 trusted fitted-state contract.

Neural Cox training
-------------------

``LinRegSurvival`` and ``NAMSurvival`` accept
``SurvivalObjectiveConfig(likelihood='cox')``. They fit one unstratified Cox
predictor with Breslow ties, including subjects censored at an event time in
that risk set. Each training and validation cohort is evaluated as one complete
batch regardless of ``training.batch_size``. Custom samplers, partial-cohort
trainer limits, distributed training and case weights are rejected. Each cohort
must contain an event; delayed entry and longitudinal histories are unsupported.

The fitted baseline uses only training subjects and the selected model weights.
Validation subjects never enter baseline estimation. Prediction uses the shared
right-continuous step survival contract, supports explicit offsets and retains
baseline state on save/load. No continuous hazard density or neural coefficient
uncertainty is inferred from the baseline jumps. ``score`` is partial log
likelihood per subject on the supplied cohort; comparisons require matching
risk-set cohorts and should not mix it with full event-time likelihood scores.


Longitudinal workflow completion
--------------------------------

``target.split_subjects(validation_fraction=..., seed=...)`` returns complete
training/validation targets. Apply ``target.record_indices(split.subject_ids)``
to features, weights and offsets. ``resample_subjects(indices)`` selects whole
subject occurrences, returns aligned source-record indices and gives duplicates
fresh occurrence IDs. Case weights remain constant within each history.

``survival_curve``, training ``CensoringDistribution.fit``, ``brier_score`` and
``concordance_index`` accept terminal-event longitudinal targets and count each
subject once. Supply predictions/risk in ``target.subject_ids`` order. Metrics
retain their censoring assumptions and delayed-entry restrictions. Observed
histories ending at events/censoring cannot supply unknown future covariates;
future curve evaluation needs independently specified complete prediction paths.
``survival_residuals`` consumes record-level predictions and sums their integrated
hazard increments per subject, retaining event/censor flags.

``bootstrap_survival(..., prediction_trajectory=path)`` samples complete histories
on both engines and retains the existing strict failure policy. Neural validation
remains an explicit fixed disjoint subject cohort. Curves, RMST, finite-grid bands
and paired effect contrasts reuse shared calculations. Request contrasts through
``effects={'times': grid}``; an explicit reference cohort also requires its
``reference_trajectory``. Preparation through ``prepare_survival_effects`` accepts
``trajectory`` and uses the same centered reference-predictor-plus-term operation
before integration. Path feature arrays have shape (subjects, path intervals,
source features); the predictor names identify interval endpoints. No future
covariate forecasting, neural coefficient intervals or longitudinal conformal
coverage is implied.

Temporal smooth effects
-----------------------

``SurvivalGAM(baseline='smooth', temporal_formula='~ti(survival_time,x,k=c(4,4))', ...)``
adds identified time/covariate tensor interactions using the existing compiler.
Keep static effects in ``formula``. The initial contract accepts ``ti`` terms
containing ``survival_time`` and source covariates; offsets and arbitrary baseline
rewrites are rejected. Right-endpoint evaluation and exact constant-hazard
exposure integration remain unchanged. Fixed and automatic Poisson smoothing
endpoints are checked against independent mgcv captures. Existing neural
interval-specific heads already permit effects to vary by interval; they do not
acquire statistical spline smoothing by analogy.

Cox smoothing, diagnostics and uncertainty
------------------------------------------

``CoxGAM(optimize_smoothing=True, smoothing_optimizer='bfgs')`` uses the existing
statistical general-family REML machinery with exact third/fourth risk-set
derivatives. One-stratum models also support ``outer_newton``. Stratified Newton
is explicitly unavailable until strict coefficient convergence passes its
reference gate; there is no tolerance relaxation or optimizer retry.

``predict_standard_error(X, times=..., strata=..., offset=...)`` follows mgcv's
baseline-plus-coefficient variance calculation. ``predict_interval`` gives
approximate pointwise Wald survival bounds with a distinct
``cox_baseline_and_coefficient`` source. These condition on selected smoothing;
they are not coefficient-only bands, simultaneous bands or response intervals.
``diagnostics`` supplies source Schoenfeld and unscaled drift-corrected cumulative
score matrices. It also computes baseline-consistent martingale/deviance
residuals. Source mgcv 1.9-4 stratified fitted-value indexing and varying-offset
prediction ordering quirks are isolated in reference captures; the new residuals
use each subject's actual stratum baseline and offset. No PH-test p-values are
inferred from the diagnostic matrices.

Cox subject bootstrap re-estimates the baseline and eligible smoothing with every
coefficient fit. Stratified calls require ``fit_context={'strata': training_strata}``
and ``prediction_context={'strata': query_strata}``; these aligned labels travel
with sampled rows. Shared bootstrap survival/RMST intervals and finite-grid bands
include this refitting uncertainty. Unsupported or failed replicates are never
replaced or silently omitted. Continuous hazard density remains unavailable.
