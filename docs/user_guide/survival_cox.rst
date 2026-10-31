Fit and diagnose Cox models
===========================

Use Cox partial likelihood for relative-risk effects with an unspecified
baseline. Statistical and neural routes share survival predictions, while their
fitting controls and inferential support differ.

Statistical Cox fitting
-----------------------

``CoxGAM`` fits static-covariate penalized partial likelihood in the existing
identified spline designs. Risk sets and tied event/censor handling follow the
actual ``coxlpl``/``coxpp`` calculations in upstream mgcv, rather than substituting
an Efron tie approximation or a coarse Poisson model. Independent mgcv fixtures
cover linear, smooth, categorical and stratified endpoints and coefficient
covariance. Fixed and eligible automatic routes have separate validation evidence; this
does not establish all mgcv Cox routes.

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
Baseline hazards and their uncertainty are stored at a fitted predictor reference
within each stratum (``baseline_reference_``). Survival predictions and standard
errors use that same reference, so adding a common predictor constant within a
stratum leaves the resulting curves unchanged.
Baseline-aware survival intervals and bootstrap propagation are described
below; coefficient covariance alone does not establish survival coverage.

Alternative ties, delayed-entry or longitudinal Cox,
case weights, analytic simultaneous/conditional Cox intervals and Cox effect
intervals remain unsupported. The
existing probability comparison helpers do not treat partial likelihood as a
full density. Save/load retains the version-7 trusted fitted-state contract.

Smoothing selection and inference
---------------------------------

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

For methodological citations and paper links, see :doc:`/references`.
