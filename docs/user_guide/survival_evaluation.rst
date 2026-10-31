Evaluate censored survival predictions
======================================

Use censor-aware metrics on an untouched subject cohort. Fit the censoring
distribution on training outcomes and retain the metric's support conditions.

Shared diagnostics and evaluation
---------------------------------

Functions in ``nampy.survival`` consume the same ``SurvivalPrediction`` and
``SurvivalTarget`` for either backend. They do not refit a model or change its
``score`` definition. Residual rows retain subject order and IDs.

.. code-block:: python

   from nampy.diagnostics import survival_residuals
   from nampy.survival import survival_curve

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

For methodological citations and paper links, see :doc:`/references`.
