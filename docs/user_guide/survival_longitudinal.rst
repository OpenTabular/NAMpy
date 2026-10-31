Fit changing-covariate survival histories
=========================================

Use complete subject histories for piecewise-exponential models. On the current
Cox routes, delayed entry and longitudinal risk records remain unsupported.

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
resampling contract described in this guide.

Use ``SurvivalTrajectory`` and shared ``predict_trajectory`` for future covariate
paths. Every prediction subject must supply a complete path from zero to the
same finite horizon within fitted support. The helper combines model cuts and
all change times, then uses the correct covariate state/model interval in each
cell. It returns the existing ``SurvivalPrediction``; row order follows
``trajectory.subject_ids``. Missing future histories are not forecast, filled
or extrapolated. Supplied paths describe conditional scenarios; uncertainty in
future covariates is not modeled. Path-specific effect preparation and bootstrap
contrasts/bands are supported in :doc:`survival_uncertainty`. Single-fit path coefficient intervals and
longitudinal conformal calibration remain unavailable.

Ordinary ``predict_link``/``predict_distribution`` retain their existing
constant-covariate interpretation for each supplied row; they do not silently
join record rows into subject paths. Existing constant-covariate inference
therefore remains distinct from inference about a changing trajectory.

.. literalinclude:: ../../examples/example_longitudinal_survival.py
   :language: python

Split, evaluate and resample whole histories
--------------------------------------------

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

For methodological citations and paper links, see :doc:`/references`.
