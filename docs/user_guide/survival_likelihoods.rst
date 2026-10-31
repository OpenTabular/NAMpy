Choose a piecewise or parametric survival likelihood
====================================================

The common workflow is in :doc:`survival`. Use these details to choose a
baseline representation, event-time law and backend training route.

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
described in this guide. Parametric time-varying covariates remain
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
pointwise survival interval API is described in this guide. Neither covariance
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

For methodological citations and paper links, see :doc:`/references`.
