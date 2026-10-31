Interpret survival effects and quantify uncertainty
===================================================

Hazard and survival answer different questions
----------------------------------------------

.. figure:: /_static/features/survival_hazard.png
   :alt: Rising Weibull hazards beside decreasing survival probabilities for three feature profiles.
   :width: 100%

   Hazard is an instantaneous event rate among subjects still at risk;
   survival accumulates those rates into event-free probability. The fitted
   Weibull shape exceeds one, so hazard rises while survival falls, and the
   x=0.8 profile retains only about 4% survival at time 2. A hazard is a rate,
   rather than a probability restricted to zero through one.

Separate changes in hazard, changes in survival, parameter uncertainty and
future event-time variation. The calculations below retain the fitted
backend's coordinate and covariance restrictions.

Survival effects and curve uncertainty
--------------------------------------

Shared effect preparation consumes additive predictor components from either
backend and evaluates hypothetical predictors through ``survival_from_predictors``.
It returns arrays and labels, without rendering or accessing backend training
objects.

.. code-block:: python

   from nampy.diagnostics import prepare_survival_effects

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

For methodological citations and paper links, see :doc:`/references`.
