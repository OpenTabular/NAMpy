Fit, predict and diagnose
=========================

Choose the response task first: regression, classification, distributional
regression or survival. Statistical and neural estimators own their fitting
algorithms; shared prediction records and diagnostics support both engines.
GAMLSS describes distributional regression, rather than a third fitting engine.

Predict the quantity you need
-----------------------------

* ``predict(X)`` returns a response point prediction. Classification returns
  labels. Full-distribution GAMLSS/LSS models use the family's declared summary,
  which can be a mean, median or most probable class.
* ``predict_parameters(X)`` returns natural distribution parameters in
  ``parameter_names_`` order. Parameter widths can differ from predictor widths,
  for example when an ordinal predictor generates several class probabilities.
* ``predict_link(X)`` returns additive predictors with shape ``(n, K)`` in
  ``predictor_names_`` order. ``predict_components(X)`` separates these into
  intercept, terms and known offsets using ``AdditivePrediction``.
* ``predict_distribution(X)`` returns a shared probability prediction. Query its
  density, CDF, quantiles or moments only where the family declares support.
* Survival ``predict(X, times=...)`` returns survival probabilities at explicit
  times. Its distribution object provides supported survival, hazard and
  restricted-mean queries within the fitted time domain.

Quantile-loss models return their explicitly fitted quantile or quantile vector.
They do not imply a complete density, CDF, natural parameter matrix or Fisher
sampling experiment. The advanced ``nampy.gam.GAM`` interface retains its
mgcv-style ``predict(type=...)`` contract.

Pass explicit evaluation rows. Task-facing statistical and neural prediction
methods use ``batch_size`` where batching is supported. Known offsets use
``(n, K)`` row-by-predictor arrays; a vector is also accepted for one predictor.
Do not transpose a list of predictor vectors implicitly. Supply new-row offsets
again for prediction, scoring and postfit Fisher diagnostics. Statistical
missing-row handling remains governed by ``na_action``; fitting and Fisher
reference inputs require valid aligned offsets.

Discover supported operations
-----------------------------

``model.capabilities()`` returns an immutable mapping of operation names to
``CapabilityStatus`` records. Each record is ``supported``, ``conditional`` or
``unsupported``, with an explanation and any remaining conditions. Queries use
family declarations, architecture validators and fitted metadata. They do not
fit a model, evaluate a network, draw outcomes or compute derivatives.

.. code-block:: python

   from nampy import GAMRegressor

   model = GAMRegressor(family="quasipoisson")
   status = model.capabilities()
   assert status["predict_link"].state == "conditional"  # fitting required
   assert status["fisher.exact"].state == "unsupported"

Conditional support can depend on fitting, explicit likelihood context, a
regression observation scale, offsets or covariance availability. Numerical
methods validate the actual arguments when called. Capability discovery does
not replace convergence checks or data validation.

Analyze the fitted model
------------------------

Shared analysis lives in ``nampy.diagnostics``: contribution centering, effects,
importance, distribution residuals, survival residuals and Fisher information.
Plotting consumes prepared diagnostic data. Backend-specific checks retain
their backend owner.

``model.fisher_information(X, offset=..., config=..., batch_size=...)`` evaluates
likelihood information at the fitted parameters. It does not refit the model.
Both engines report mean information per reference row and exclude penalties.
``FisherDiagnostics(record_diagonal=...)`` controls diagonal retention. Sampled
estimators and input-sampling designs carry their own estimation uncertainty;
this is separate from coefficient uncertainty and predictive intervals.
Quasi-likelihood and other undeclared response experiments are rejected.
See :doc:`fisher_diagnostics` for supported estimators and assumptions.

``nampy.models.compare_distributions`` fits explicit candidate templates and
compares held-out normalized likelihoods. ``DistributionFitStatus`` is a neutral
record in ``nampy.contracts``. Probability laws themselves do not orchestrate
fitting or diagnostics.

Continue with :doc:`gamlss`, :doc:`survival`, :doc:`interpretability` and
:doc:`calibration` for complete task workflows. Calibration uses separate data
and owns its calibrated predictor state; it is distinct from Fisher estimation
and from conditional response spread.
