Discrete, mixed and positive outcomes
=====================================

Choose a response law from its observation support and parameter meanings. Start
with :doc:`gamlss` for the shared fitting and prediction workflow.

Beta-binomial counts with explicit trials
-----------------------------------------

``betabinomial_lss`` models integer successes with natural columns ``mu, kappa``
and logit/log predictors. ``mu`` is a success probability and ``kappa`` is beta
precision. The count mean is ``trials*mu``; its variance is
``trials*mu*(1-mu)*(trials+kappa)/(1+kappa)``. The R ``BB`` mapping is
``sigma=1/kappa`` and ``bd=trials``. Trials are separate from sample weights.

.. code-block:: python

   model = GAMLSS(
       family="betabinomial_lss",
       formula={"mu": "successes ~ s(x, k=6)", "kappa": "~ 1"},
       optimize_smoothing=False,
       smoothing_params=[0.3],
   ).fit(data, context={"trials": data["trials"].to_numpy()})
   distribution = model.predict_distribution(
       new_data, context={"trials": new_trials}
   )
   expected_counts = distribution.mean()
   count_centiles = distribution.quantile([0.05, 0.5, 0.95])

Supply trial context again for every explicit prediction dataset, including a
dataset with the same row count as training. Only omitted ``X`` reuses fitted
context. ``predict_parameters`` and ``standard_errors`` return the two natural parameters
and their errors, independently of new trials. ``predict_point``, ``score`` and
``evaluate`` accept the same ``context`` argument. Context predictions support
``na_action="pass"``; omission is rejected because its row mapping is not yet
implemented. Missing prediction rows remain missing in probability operations.

Trials must be finite nonnegative integers. Zero-trial observations have a point
mass at zero and supply no information. Fitting requires at least one
positive-weight row with two or more trials, plus positive-weight successes and
failures. These necessary guards do not guarantee identifiability of arbitrary
formulas. Boundary and separated fits may still fail to converge; inspect fit
status. No exact binomial boundary fit is substituted for infinite precision.

Fixed smoothing is supported and checked against weighted duplication, an
independent likelihood optimizer and saved R unpenalized fits. Exact joint
likelihood derivatives through fourth order and outer REML gradient/Hessian
checks with weights and offsets validate automatic ``outer_newton``, ``bfgs``
and ``efs`` routes. Other method labels retain general-family coercion to REML;
this does not enable separate ML, GCV/UBRE or NCV criteria. ``optim`` requires a
family-defined null deviance and remains unsupported, as do ``magic`` and
``nlm.fd``. Neural
beta-binomial training, alternate links, censoring/truncation and deviance
residuals remain unsupported. The shared ``BetaBinomialLaw`` is independent of
both training backends and supports shared quantile residuals and centiles.
R's small-sigma binomial substitutions and out-of-support log-mass quirks are
not copied: probabilities use the declared beta-binomial law, and log mass is
``-inf`` outside integer support. Very large trial counts or precision retain
SciPy's numerical and performance limits; extreme-tail stability is not claimed.

Zero-adjusted Gamma and reusable atom mixtures
----------------------------------------------

``zaga_lss`` has three predictors: positive Gamma mean ``mu``, positive-component
coefficient of variation ``sigma`` and zero probability ``pi0``. They use
log/log/logit links. This is the R ``ZAGA(mu, sigma, nu=pi0)`` parameterization.
The whole-response mean is ``(1-pi0)*mu``, and the variance is
``(1-pi0)*mu**2*(sigma**2+pi0)``.

.. code-block:: python

   model = GAMLSS(
       family="zaga_lss",
       formula={"mu": "y ~ s(x, k=6)", "sigma": "~ 1", "pi0": "~ s(x, k=5)"},
       smoothing_optimizer="outer_newton",
   ).fit(data)
   distribution = model.predict_distribution(new_data)
   zero_probabilities = distribution.cdf(0)
   expected_outcomes = distribution.mean()
   from nampy.diagnostics import quantile_residuals
   residuals = quantile_residuals(distribution, new_y, random_state=7)

Fixed smoothing and automatic general-family REML with ``outer_newton``,
``bfgs`` and ``efs`` are validated. Existing method coercion to REML is retained;
no separate ML/GCV/UBRE/NCV fit is implied. Fitting requires positive-weight zeros
and at least two distinct positive responses. Exact zero/one atom weights are
supported for probability queries, but have no finite-logit fit here. Necessary
component checks do not guarantee identifiability of arbitrary formulas. Neural
ZAGA training, alternate links, censoring/truncation, deviance residuals and
``optim``/``magic``/``nlm.fd`` smoothing routes remain unsupported.

The shared ``AtomMixtureLaw`` supplies the probability calculations and can also
combine a beta law with atoms at zero and one:

.. code-block:: python

   from nampy.distributions import AtomMixtureLaw, BetaLaw, DistributionPrediction

   law = AtomMixtureLaw(BetaLaw(), atoms=(0.0, 1.0), mass_names=("pi0", "pi1"))
   distribution = DistributionPrediction(law, {
       "alpha": [2.0], "beta": [4.0], "pi0": [0.2], "pi1": [0.1],
   })
   centiles = distribution.quantile([0.05, 0.5, 0.95])

This constructs an inflated-beta probability law directly; fitted models use
``beinf_lss`` as described below. By default the continuous weight is one minus
the sum of atom weights. Set ``continuous_mass_name`` to provide it explicitly
when a coupled transform produces a very small remaining weight. Explicit
weights must sum to one within eight float64 epsilons; they are not renormalized.
At any declared
atom, ``log_prob`` reports log mass, including ``-inf`` for zero mass; elsewhere
it reports log density. ``cdf_left`` excludes the atom at the query. Moments,
draws, centiles and residuals share the existing contracts. Purely atomic limits
ignore zero-weight components when finding endpoint quantiles.

The composition derives available operations from its continuous base; it does
not supply likelihood derivatives or fitting capabilities. Discrete/mixed bases,
vector events and automatically generated fit families are unsupported. Quantiles
evaluate the base law row by row. Log-tail operations use the base law's log
operations directly, inheriting their numerical limits (including SciPy Gamma
tail underflow), rather than promising stability for every base law.
Shared quantiles at probabilities zero and one use support limits; R's ZAGA
quantile function rejects probability one, so endpoint behavior is intentionally
documented separately from the saved interior-probability comparisons.

Inflated-beta fits with coupled probabilities
---------------------------------------------

``beinf_lss`` models continuous proportions together with exact zero and one
atoms. It has four formula keys, ``mu``, ``phi``, ``pi0`` and ``pi1``, and five
natural output columns: ``mu, phi, pi0, pi1, pi_interior``. The interior beta
shapes are ``mu*phi`` and ``(1-mu)*phi``. The R BEINF mapping is
``sigma=1/sqrt(1+phi)``, ``nu=pi0/pi_interior`` and ``tau=pi1/pi_interior``.

.. code-block:: python

   model = GAMLSS(
       family="beinf_lss",
       formula={
           "mu": "y ~ s(x, k=6)", "phi": "~ 1",
           "pi0": "~ s(x, k=5)", "pi1": "~ x",
       },
   ).fit(data)
   parameters = model.predict_parameters(new_data)                 # (n, 5)
   raw_predictors = model.predict_link(new_data)        # (n, 4)
   parameter_errors = model.standard_errors(new_data)   # (n, 5)
   distribution = model.predict_distribution(new_data)
   expected_proportion = distribution.mean()


Raw columns are ``logit(mu)``, ``log(phi)``, ``log(pi0/pi_interior)`` and
``log(pi1/pi_interior)``. The last two are category log ratios to a shared
interior reference. They are not independent probability logits: changing either
predictor changes all three mixture probabilities. Four offsets may be supplied,
one per predictor. Natural parameter errors use the full coefficient covariance,
including cross-predictor and shared-coefficient effects.

``predict_components`` preserves the shared ``AdditivePrediction`` shape
contract with four columns, one per actual predictor. The derived interior
probability has no independent additive terms or intercept. This does not mean that
``pi_interior`` is constant or unaffected by covariates: its probability is a
nonlinear function of both atom predictors. Interpret atom contributions on
their log-ratio scale; the reference log-ratio is not a probability effect.

Fixed smoothing and automatic joint REML with ``outer_newton``, ``bfgs`` and
``efs`` are validated, including all four predictors smoothed. Other method
labels preserve existing general-family REML coercion. Separate ML/GCV/UBRE/NCV,
``optim``/``magic``/``nlm.fd``, alternate links, neural training,
censoring/truncation and deviance residuals remain unsupported. Positive-weight
zeros, ones and at least two distinct interior responses are required for a
finite-predictor fit; these necessary checks do not guarantee identifiability
of arbitrary formulas. Boundary or absent-component fits are not substituted.

The shared composition keeps ``pi_interior`` explicitly, preserving small
weights that ``1-pi0-pi1`` can round to zero. Probability queries allow zero
component weights with otherwise valid beta parameters. Underflow/overflow in
the interior beta parameters still follows the existing numerical limits; no
endpoint clipping or beta approximation is introduced.
Finite-logit predictions whose softmax probabilities round to zero or one are
explicitly rejected, rather than silently returning a boundary distribution.
Boundary weights remain valid when constructing the shared probability law
directly.

The saved gamlss.dist 6.1.1 ``BEINF`` variance hook adds rather than subtracts
the squared response mean. NAMpy uses the normalized mixture variance and
records this discrepancy separately; density/mass, CDF, quantiles and the
unpenalized fit agree with the saved R reference. Smoothed RS/backfitting parity
is not claimed. Quantile residuals and centiles consume the shared mixed-law
operations without another diagnostic or plotting path.

Lognormal and Weibull statistical fits
--------------------------------------

Both positive-response families require finite, strictly positive responses.
They reject zero and negative observations rather than clipping them.
Lognormal ``loc`` may be negative because it describes log(Y). For lognormal
``predict_point`` returns ``exp(loc + scale**2/2)``, not its location or median.
Weibull uses the GAMLSS WEI convention: ``scale`` is the Weibull scale and
``shape`` the concentration, so the response mean is
``scale * Gamma(1 + 1/shape)``. WEI2, WEI3 and LOGNO2 are different
parameterizations and are not aliases.

.. code-block:: python

   lognormal_model = GAMLSS(
       family="lognormal_lss",
       formula={"loc": "y ~ s(x, bs='cr', k=7)", "scale": "~ 1"},
   ).fit(data)

   weibull_model = GAMLSS(
       family="weibull_lss",
       formula={"scale": "y ~ s(x, bs='cr', k=7)", "shape": "~ 1"},
   ).fit(data)

The lognormal density includes ``-log(y)``. This term matters for normalized
response-scale likelihoods and comparisons to other response laws, even though
it does not change the coefficient optimum of a model for log(Y).
The shared distribution object provides probability queries, simulation,
quantile residuals and centiles for both new families and their neural versions.
Lognormal CDF/log-density calculations work on log(y), avoiding overflow through
``exp(loc)``; response draws, moments and extreme Weibull operations retain
ordinary floating-point range limits.

Fixed smoothing and automatic REML with ``outer_newton`` (default), ``bfgs``
and ``efs`` are validated. The likelihood derivatives, including mixed terms,
are exact through fourth order. As for other mgcv general families, requested
method labels are coerced to REML by the existing fit engine: this does not
provide independent ML, GCV, UBRE or NCV fits. The ``optim`` route is explicitly
unavailable because its objective scaling requires a family-defined null
deviance, which these extensions do not define. ``magic`` and ``nlm.fd`` are
also unsupported; other optimizer/configuration combinations are not certified
by these tests. Existing transformed-coefficient, correlated-observation and
general-family NCV guards still apply.

Native response and Pearson residuals are available; native deviance residuals
and null deviance are unavailable. Use the shared distribution quantile
diagnostics below rather than expecting a default deviance QQ workflow.
The numeric fit record retains ``-2*loglik`` as its likelihood objective; it is
not a saturated-model deviance.
Alternate predictor links, censoring and truncation are not supplied by these
families. Saved R GAMLSS fixtures validate their probability parameterizations
and unpenalized fits, not parity with its smoother/backfitting algorithms.
Shared lognormal quantiles return the support boundary zero at probability zero,
rather than copying R ``qLOGNO``'s negative-infinity endpoint convention.
Out-of-support log densities are negative infinity rather than zero on the log
scale; R interior-probability comparisons do not certify those source quirks.

Neural distributional fitting
-----------------------------

Neural ``bccg``, ``zaga``, ``beinf`` and ``betabinomial`` objectives use
normalized likelihoods and the shared NumPy probability laws. They do not
claim GAMLSS RS/backfitting parity or statistical convergence. Neural
beta-binomial uses a fixed integer ``distributional_kwargs={'trials': m}``,
with m>=2; row-specific trials remain supported by the statistical route only.
BEINF has four raw predictors and five natural parameters in both backends.
The interior mass is retained explicitly through the coupled softmax.

For methodological citations and paper links, see :doc:`/references`.
Visualize atoms separately from density
--------------------------------------

.. figure:: /_static/features/inflated_components.png
   :alt: Fitted zero, one and interior probability bars beside weighted continuous beta densities at three feature values.
   :width: 100%

   From x=-0.8 to 0.8, fitted zero mass falls from about 0.48 to 0.075
   while one mass grows from about 0.024 to 0.295. These are endpoint
   probabilities, rather than spikes with arbitrary heights. Each
   continuous curve integrates to its interior probability, which can be
   less than one; its area complements the two endpoint masses to one.
