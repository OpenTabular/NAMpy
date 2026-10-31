Censoring and truncation
========================

Use observation targets when a response is known through bounds rather than an
exact value. Keep censoring bounds, truncation limits, offsets and weights
separate.

Censoring and fixed truncation
------------------------------

``ObservationTarget(lower, upper)`` records exact values (equal finite
endpoints), left/right censoring (one infinite endpoint), intervals (ordered
finite endpoints), or uninformative full intervals. ``subset`` preserves the
metadata. Observation bounds are separate from offsets and case weights.
``observation_log_prob(prediction, target)`` evaluates densities for exact
values and stable interval log masses otherwise. It currently requires a
scalar continuous law.

``TruncatedLaw(base, lower, upper)`` conditions an existing continuous law on
fixed bounds and derives available probability capabilities from that base.
Moments currently require a normal base. Normal quantiles retain far-tail
precision; generic inverse-CDF ranges that lose probability precision raise.

Statistical ``normal_observation_lss`` uses formula keys ``mean, scale``
(identity/log), ``ObservationTarget`` directly in ``fit``, and fixed truncation
bounds through its family options. Neural ``normal_observation`` takes the
same target in fitting/scoring and bounds through ``distributional_kwargs``.
The latent location/SD parameters are distinct from the truncated response
mean/SD. New prediction rows do not take training censoring metadata. Both
routes share the observation contract and probability composition; the
statistical adapter has analytic derivatives through order four and validated
Newton/BFGS REML. Ordinary response/Pearson/deviance residuals, observation-
specific truncation limits and discrete/mixed censoring compositions remain
unsupported. Existing mgcv ``cnorm`` behavior is unchanged.

Lognormal and Weibull observation families reuse the same composition and
target contract. Statistical names are ``lognormal_observation`` and
``weibull_observation`` (the low-level names add ``_lss``); neural objective
names are identical. Lognormal formula keys are ``loc, scale`` with identity/log
links. Weibull keys are ``scale, shape`` with log/log links. Bounds use ``lower``
and ``upper`` family options, intersected with positive response support.
Both support fixed smoothing and analytic Newton/BFGS REML. These positive
families return the conditional median from ``predict_point``; truncated moments
and SD summaries remain unavailable. Their prediction laws condition on
truncation bounds, independently of the training censoring intervals.

For methodological citations and paper links, see :doc:`/references`.

Score new observations
----------------------

Pass the new target explicitly when evaluating a held-out cohort:

.. code-block:: python

   from nampy.distributions import observation_log_prob

   prediction = model.predict_distribution(X_test)
   contributions = observation_log_prob(prediction, target_test)
   mean_log_likelihood = model.score(
       X_test, target_test, sample_weight=test_weights,
   )
   negative_log_likelihood = model.evaluate(
       X_test, target_test, sample_weight=test_weights,
   )["Negative Log-Likelihood"]

The score is the weighted average of the per-observation contributions; the
default evaluation metric is its negative. Exact-response arrays instead use
ordinary density evaluation. Prediction rows never inherit the training
censoring intervals. Supply any required new-row distribution context and
offsets explicitly.
