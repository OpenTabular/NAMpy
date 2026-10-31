Compare conditional response distributions
==========================================

Compare explicit candidates on a held-out cohort using their normalized
probability laws. Complete the first fit before using this model-selection
workflow.

Compare explicit distribution candidates
----------------------------------------

``compare_distributions`` fits estimator templates and compares normalized
held-out log probabilities. Each candidate receives an isolated copy of the
same training inputs and the same validation rows. Formulas, smoothing options
and neural training options remain owned by each candidate. Input templates
are cloned rather than fitted in place.

.. code-block:: python

   from nampy import GAMLSS
   from nampy.models import DistributionCandidate, compare_distributions

   candidates = {
       "lognormal": GAMLSS(
           family="lognormal_lss",
           formula={"loc": "y ~ s(x, k=6)", "scale": "~ 1"},
       ),
       "weibull": GAMLSS(
           family="weibull_lss",
           formula={"scale": "y ~ s(x, k=6)", "shape": "~ 1"},
       ),
   }
   comparison = compare_distributions(
       candidates, X_train, y_train, X_valid, y_valid,
       sample_weight=training_weights, validation_weight=validation_weights,
       response_units="original response units",
   )
   table = comparison.to_frame()
   ordered_names = comparison.ranking
   fitted_models = comparison.models

For a neural candidate, wrap the template with explicit training options:

.. code-block:: python


   from nampy.neural.configuration import GradientTrainingConfig, ObjectiveConfig
   from nampy.models import LinRegLSS

   candidate = DistributionCandidate(
       LinRegLSS(
           objective=ObjectiveConfig(family="lognormal"),
           training=GradientTrainingConfig(max_epochs=100, random_state=7),
       ),
       predict_kwargs={"batch_size": 256},
   )

Training and comparison validation data must be supplied explicitly. The
comparison holdout is not passed to ``fit`` or used for early stopping.
Backend-internal validation splits remain configured by each estimator; any
``X_val``/``y_val`` in ``fit_kwargs`` must be an inner training-validation split,
not the comparison holdout. After choosing a family, report performance on a
separate untouched test set or use nested selection. No automatic family search,
formula search, ensemble construction or refit is performed.

Weights and observation context
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Common ``sample_weight`` is validated once and passed to every fit. It cannot
be overridden in per-candidate arguments. Its numerical fitting semantics remain
backend-owned. Common ``validation_weight`` defines the scoring weights: zero
weights are excluded and positive weights are normalized to sum to one. The
reported mean log probability is higher-is-better and uses the probability law,
not ``estimator.score`` or neural regularization losses. ``effective_n`` is
``1/sum(normalized_weight**2)``, a weight-concentration descriptor, not an
inferential sample-size guarantee.

Responses must be finite scalar vectors; there is no per-model deletion of
observations. Predictions must preserve the complete validation-row order and
count. A missing prediction on a positive-weight row makes that candidate
ineligible; it does not shrink anyone else's scoring set. Zero-weight prediction
rows may be missing. The predictor adapter owns row-order preservation; comparison
checks row count but cannot detect an external adapter silently permuting rows.

Pass family-specific trial context and offsets explicitly through each
``DistributionCandidate``'s ``fit_kwargs`` and ``predict_kwargs``. For example,
beta-binomial needs ``{"context": {"trials": training_trials}}`` at fit and the
corresponding validation-trial vector at prediction. Common feature/response
inputs cannot be replaced through these mappings. Trial counts, offsets and
response units must describe the same observations across candidates. Response
transformations and their density Jacobians belong to the probability law;
comparison does not infer or repair them. ``response_units`` is a user-supplied
label, not automatic unit conversion.

Eligibility and comparable measures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every result retains the backend fit status, Python warnings, failure stage/reason and
fit/evaluation elapsed time. Fit time includes cloning and fit-status collection.
Exceptions, nonconverged fits and unknown status remain visible and unranked.
Coefficient and any reported outer-optimizer convergence are checked for GAMLSS.
The adapter preserves the statistical backend's convergence flags, including its
iteration-budget semantics; it does not introduce another stopping rule.

Neural ``completed`` means the Lightning routine finished, performed an optimizer
step and did not report nonfinite tracked losses. It is not statistical
convergence. Interrupted or zero-step training is ineligible. Legacy artifacts
and training routes without completion metadata remain ``unknown`` until an
appropriate status adapter is supplied. External estimators can provide
``distribution_fit_status()`` or a candidate ``status_reader`` returning
``nampy.contracts.DistributionFitStatus``. Returning from ``fit`` alone is not
sufficient eligibility evidence.

Continuous log densities, discrete log masses and mixed log probabilities
cannot be ranked together without an explicit common-reference-measure
conversion, which is currently unsupported. Mixed candidates additionally need
the same declared ``spec.atom_locations``. The shared atom composition declares
these automatically. If successful candidates have incompatible measures, all
of those candidates remain unranked; their raw scores stay available for
inspection. No arbitrary first candidate defines the ranking cohort.

A normalized zero probability gives log score ``-inf`` and ranks last, with its
count reported. It is distinct from a failed fit. NaN and positive-infinite log
scores are ineligible. Exact score ties share their minimum rank and preserve
candidate order. The result never chooses a tie winner or automatically refits.

Calibration and remaining limits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When CDF and quantile operations exist, results include the existing centile
calibration on positive-weight validation rows. This diagnostic is **unweighted**
even when scoring weights differ, and is labelled accordingly. It reports actual
row counts and attainable CDF probabilities for discrete or mixed centiles.
Missing operations or diagnostic failures are recorded separately without
invalidating a normalized likelihood score. Set ``calibration_probs=None`` to
skip calibration. Rowwise log probabilities are retained as read-only snapshots.

Statistical EDF/AIC and neural training/regularization quantities are not pooled
into a common criterion. Vector responses, censoring records, transformed-response
measure conversion, parallel execution and validation uncertainty intervals remain
unsupported. No parity with GAMLSS RS/backfitting family searches is claimed;
``fitDistPred`` is a semantic reference for explicit held-out likelihood scoring.

Runnable held-out comparison
----------------------------

.. include:: /_generated/reference/comparison.inc

For methodological citations and paper links, see :doc:`/references`.
