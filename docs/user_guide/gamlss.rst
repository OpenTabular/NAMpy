Distributional GAMs and GAMLSS
==============================

GAMLSS extends additive regression beyond the mean (Rigby and Stasinopoulos, 2005).

This guide covers ``GAMLSS`` formulas, parameter ordering, predictions, and
scoring. For fitted statistical examples, see :doc:`../notebooks/02_statistical_gams`
and :doc:`../notebooks/gamma_location_scale`. For a separate comparison of neural
distributional, quantile, and ensemble models, see
:doc:`../notebooks/04_distributional_models`.

The lower-level :class:`nampy.gam.GAM` also accepts supported multi-parameter
families and exposes their statistical parameterization directly.

.. _distributional-model-concepts:

Decide which aspects of the response to model
---------------------------------------------

A mean model assumes that the conditional expectation is the main quantity
that changes with predictors. Distributional regression is useful when
variability, skewness, tail behavior, or another parameter also changes.

For parameter :math:`\theta_k`, NAMpy fits an additive predictor
:math:`\eta_k` and applies the parameter's inverse link,

.. math::

   \eta_k(x)=\beta_{0k}+\sum_j f_{jk}(x_j)
   +\sum_{S\in\mathcal I_k}f_{Sk}(x_S)+o_k(x),\qquad
   \theta_k(x)=h_k^{-1}\{\eta_k(x)\}.

Together these parameters specify the response distribution,
:math:`Y\mid x\sim\mathcal D\{\theta_1(x),\ldots,\theta_K(x)\}`.
The term definitions and identifiability considerations follow
:ref:`additive-predictor-interpretation`, separately for each parameter.

Terms add on the :math:`\eta_k` scale. Natural parameters generally do not add
after a nonlinear inverse link. Conditional standard deviation describes
observation-level predictive spread. It is different from a standard error
for an estimated mean and from variation across repeated fits.

Quantile models instead estimate selected conditional quantiles using pinball
loss. These quantiles do not by themselves specify a complete probability
density; their spacing and empirical coverage describe different properties.

Public parameter contract
-------------------------

``GAMLSS.predict`` returns the family-defined point prediction.
``predict_parameters`` returns natural distribution parameters with columns
in ``parameter_names_`` order. ``predict_link`` returns additive predictors
with columns in ``predictor_names_`` order. See :doc:`model_workflow`.

.. list-table:: Built-in families
   :header-rows: 1

   * - Public aliases
     - ``parameter_names_``
     - Meaning
   * - ``normal``, ``gaussian``, ``gaulss``
     - ``("mu", "sigma")``
     - Normal conditional mean and standard deviation
   * - ``gamma``, ``gammals``
     - ``("mu", "sigma")``
     - Gamma conditional mean and coefficient of variation, so
       ``Var(Y | X) = mu**2 * sigma**2``

``predict_point`` returns the conditional mean. ``score`` returns mean log
likelihood (higher is better), while the default ``evaluate`` result reports
mean negative log likelihood.

Query the conditional distribution
----------------------------------

Both statistical ``GAMLSS`` and neural ``*LSS`` estimators provide
``predict_distribution`` for normal, Gamma, Student-t, beta, negative-binomial,
lognormal and Weibull
families. Statistical ``betabinomial_lss`` and ``zaga_lss`` additionally support
trial counts and mixed zero/positive outcomes, respectively. It returns a shared
``nampy.distributions.DistributionPrediction`` containing a snapshot of the
conditional probability law at each prediction row:

.. code-block:: python

   distribution = model.predict_distribution(X_test)
   log_density = distribution.log_prob(y_test)  # (n,), unweighted
   probabilities = distribution.cdf(y_test)    # (n,)
   intervals = distribution.quantile([0.05, 0.95])  # (n, 2)
   draws = distribution.sample(n_draws=500, random_state=7)  # (500, n)
   means = distribution.mean()                # (n,)
   variances = distribution.variance()        # (n,)
   available = distribution.spec.operations

``cdf_left`` returns ``P(Y < y)``; ``cdf`` returns ``P(Y <= y)``. They agree
for the supplied continuous laws; a custom discrete or mixed law must handle
jumps explicitly. ``log_cdf``, ``survival`` and ``log_survival`` call the
probability implementation's corresponding operations rather than subtracting
or taking logarithms of rounded CDF values. The supplied laws use SciPy kernels;
extreme Gamma tails can still underflow. No arbitrary-precision tail calculation
or parameter derivatives of CDFs are implied.

Evaluation accepts a scalar response value or exactly one value per prediction
row. Quantiles use a separate probability axis: even one probability returns
``(n, 1)``. Prediction rows retain their order, including all-NaN missing rows
under statistical ``na_action='pass'``. Parameter vectors are copied and made
read-only. Samples condition on fitted parameters; they do not include
coefficient uncertainty, bootstrap variation or an ensemble predictive mixture.
Scoring weights remain the estimator's responsibility.

The snapshot uses explicit probability-law parameter names, which can differ
from the estimator's existing prediction columns:

.. list-table:: Supported prediction adapters
   :header-rows: 1

   * - Backend family
     - Existing natural columns
     - Shared distribution parameters
   * - Statistical ``normal`` / ``gaulss``
     - ``mu``, ``sigma`` (mean and SD)
     - ``loc=mu``, ``scale=sigma``
   * - Statistical ``gamma`` / ``gammals``
     - ``mu``, ``sigma`` (mean and coefficient of variation)
     - ``shape=1/sigma**2``, ``rate=shape/mu``
   * - Neural ``normal``
     - ``mean``, ``scale``
     - ``loc=mean``, ``scale=scale``
   * - Neural ``gamma``
     - ``shape``, ``rate``
     - ``shape``, ``rate``
   * - Statistical ``student_lss``
     - ``mu``, ``sigma``; fixed ``df`` in the family
     - ``df``, ``loc=mu``, ``scale=sigma``
   * - Neural ``studentt``
     - ``df``, ``loc``, ``scale``
     - ``df``, ``loc``, ``scale``
   * - Statistical ``beta_lss``
     - ``mu``, ``phi`` (mean and precision)
     - ``alpha=mu*phi``, ``beta=(1-mu)*phi``
   * - Neural ``beta``
     - ``alpha``, ``beta``
     - ``alpha``, ``beta``
   * - Statistical ``nb_lss``
     - ``mu``, ``alpha`` (NB2 mean and dispersion)
     - ``mean=mu``, ``dispersion=alpha``
   * - Neural ``negativebinom``
     - ``mean``, ``dispersion``
     - ``mean``, ``dispersion``
   * - Statistical ``lognormal_lss`` / ``lognormal`` and neural ``lognormal``
     - ``loc``, ``scale`` (location and SD of log(Y))
     - ``loc``, ``scale``
   * - Statistical ``weibull_lss`` / ``weibull`` and neural ``weibull``
     - ``scale``, ``shape`` (WEI convention)
     - ``scale``, ``shape``
   * - Statistical ``betabinomial_lss``
     - ``mu``, ``kappa``; context ``trials``
     - ``mu``, ``kappa``; context ``trials``
   * - Statistical ``zaga_lss``
     - Positive-component ``mu``, ``sigma``; atom probability ``pi0``
     - ``shape=1/sigma**2``, ``rate=shape/mu``, ``pi0``

Statistical adapters use the fitted family's existing natural-parameter
conversion, including configured links and bounds. Neural adapters use the
fitted family's transformed parameters once, then detach to NumPy for inference.
Neural loss computation and gradients remain in Torch. Existing ``predict``,
``score``, statistical fitting and mgcv response conventions retain their
original behavior.

Student-t probability operations require positive degrees of freedom, but moments
need stronger conditions. ``mean()`` returns NaN for df <= 1;
``variance()`` returns NaN for df <= 1 and infinity for 1 < df <= 2.
Neural ``predict_point`` still returns the location in these regimes, while the
statistical ``student_lss`` family retains its existing fixed df > 2 requirement.

Probability queries evaluate the supplied response without training-time target
clipping. In particular, the existing neural beta loss clips targets near zero
and one; its training loss need not equal ``-log_prob`` at those targets.
The neural negative-binomial loss also clamps its parameter conversion. An
adapter request that activates those clamps raises ``NotImplementedError``
instead of silently returning a different law. Other negative-binomial
probability queries use the NB2 law, with variance
``mean + dispersion * mean**2``. Numerically unrepresentable size/probability
regimes are rejected; a Poisson approximation is not substituted. This differs
from small-dispersion Poisson branches in some R GAMLSS NBI probability functions.
Only the listed distributional families are adapted: ordinary mgcv ``scat``,
``betar`` and ``nb`` are not newly adapted by matching their names.

Other families and subclasses with potentially different semantics currently
raise ``NotImplementedError`` from ``predict_distribution``. Their existing
fitting, prediction and scoring remain available. Vector events, censoring,
truncation and inference bands for centile estimates are not added by this
interface. A quantile loss does not define a normalized probability law.

To add a law, implement the structural ``nampy.distributions.DistributionLaw``
protocol and construct ``DistributionPrediction(law, parameters)``. Its
``DistributionSpec`` declares named parameters, a parameterization version,
response kind, event shape and available operations. The law validates its
parameter domains and evaluates the declared operations. Rowwise fixed inputs,
such as trial counts, can be declared in ``spec.context_names`` and supplied
through ``DistributionPrediction(..., context=...)``. The object copies and
slices that context with the same row mask as the parameters. No inheritance from
a concrete estimator, statistical family or Torch module is required.
Unavailable operations raise explicit errors even when a method exists on the
shared prediction object. Declaring probability support does not enable any
fitting or smoothing route.

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

Writing a statistical extension family
--------------------------------------

The internal ``_DistributionalFamily`` adapter derives predictor count from
the declared predictor names and links. It assembles likelihood derivatives,
weights, offsets and initialization for any positive predictor count, reusing
the existing joint solver. Natural-parameter bounds are declared per named
column; no particular parameter position is implicitly positive.

The default natural transforms support independent identity, log and logit
links. Coupled transforms or a different number of natural parameters require
family-owned transforms and Jacobians. Full Jacobians retain covariance between
predictor blocks; treating coupled parameters as independent is not supported.
When natural and predictor counts differ, public additive components additionally
need a family-owned padded link representation and predictor-index mapping;
padding is not inferred. A custom family also needs an explicit probability
adapter before ``predict_distribution`` is available.
Initialization must return one target per predictor. New families supply their
response validation, normalized likelihood, initialization, response moments
and parameter meanings; they do not supply a new solver.

The default extension is fixed-smoothing only. Declare the available likelihood
derivative order and opt into automatic fitting capabilities separately, after
validation; automatic smoothing requires fourth-order derivatives here.
Three- and four-predictor normalized test families exercise derivative packing,
fitting and initialization, and a coupled three-predictor transform exercises
four natural outputs. These checks do not establish support for every new law
or formula composition. Existing source-specific mgcv general families retain
their current implementations and numerical paths.

Shared quantile residuals and worm plots
----------------------------------------

Use the fitted distribution on observed rows to obtain normalized quantile
residuals. Continuous observations use ``Phi^-1(F(y))``. At a probability jump,
the operation draws a uniform value between ``F(y-)`` and ``F(y)`` before applying
the normal quantile function. It works for discrete and mixed laws through
their left-limit CDF operations, without assuming every jump lies on an integer.

.. code-block:: python

   from nampy.diagnostics import prepare_worm_plot
   from nampy.plotting import render_worm_plot

   distribution = model.predict_distribution(X_test)
   from nampy.diagnostics import quantile_residuals
   residuals = quantile_residuals(distribution, y_test, random_state=7)
   worm_data = prepare_worm_plot(residuals, x=X_test["age"], n_bins=4)
   figure = render_worm_plot(worm_data)

The ``QuantileResiduals`` result contains ``residuals``, ``pit``, ``cdf_lower``,
``cdf_upper``, ``randomized`` and ``clipped`` arrays. Default clipping is disabled:
zero/one probabilities, including rounded tails, can produce infinite residuals.
To request clipping explicitly, pass ``clip=1e-12`` and inspect the ``clipped``
flags. The original CDF bounds remain available. Repeated randomizations can use
successive calls with a single NumPy Generator. Alternatively, supply an explicit
``uniforms`` vector, especially when comparing languages whose seeds use
different random-number streams. Observation weights are not expanded or used
to weight these diagnostics. Responses must respect the law's observation
support; probability-threshold queries are more permissive than observations.

Worm plots detrend normal QQ points and optionally fit a cubic trend.
``groups=...`` accepts category labels; ``x=...`` forms numeric bins. Explicit
``breaks`` define left-closed bins with the last right-closed. Tied quantile
edges collapse. Missing group labels/nonfinite x are counted separately; empty
explicit bins remain visible. Each panel records original row indices, total,
missing and infinite-residual counts. Cubic fitting requires four finite rows.
QQ positions follow R's default ``ppoints`` convention.

Shading is a pointwise reference envelope for independent standard normal
observations. It is not simultaneous coverage or an adjustment for fitting the
model. Prefer held-out or cross-fitted diagnostics for calibration. These shared
functions leave ``gam_.residuals`` and mgcv QQ behavior unchanged; family-specific
residual types can still be unsupported there. ``plot_worm`` is a convenience
that prepares data and renders it in one call.

Conditional centiles and calibration
------------------------------------

Choose explicit prediction rows for a centile curve. With several covariates,
vary the desired covariate on a grid while fixing the others to a chosen reference
row. Averaging conditional quantiles does not produce marginal quantiles.

.. code-block:: python

   import numpy as np
   import pandas as pd
   from nampy.diagnostics import prepare_centiles, centile_calibration
   from nampy.plotting import render_centiles

   grid = pd.concat([X_test.iloc[[0]]] * 100, ignore_index=True)
   grid["age"] = np.linspace(X_test["age"].min(), X_test["age"].max(), 100)
   grid_distribution = model.predict_distribution(grid)
   curves = prepare_centiles(grid_distribution, grid["age"], probs=[0.05, 0.5, 0.95])
   figure = render_centiles(curves, xlabel="Age")

   held_out = model.predict_distribution(X_test)
   calibration = centile_calibration(held_out, y_test, probs=[0.05, 0.5, 0.95])

``CentileCurves`` contains sorted x, probabilities, quantiles, original
``row_indices`` and the count of excluded nonfinite x. Missing prediction rows
remain gaps. Requested probabilities must be strictly increasing inside (0,1).
``plot_centiles`` combines preparation and rendering. These are conditional
outcome quantiles, with no confidence-band shading or parameter-uncertainty
calculation. Discrete curves can have steps and repeated quantile values.
They are rendered as step curves between supplied grid rows.

``CentileCalibration`` reports ``observed`` coverage, the mean attainable
``model_probability = F(q_p)`` and per-centile ``n_used``. For discrete laws,
``F(q_p)`` can exceed the nominal probability p even under a correctly specified
model. Missing/nonfinite observations or quantiles are excluded with counts;
there are no implicit weights or independence-based confidence intervals.

Read Gamma parameter meanings
-----------------------------

For Gamma responses, ``sigma`` is a coefficient of variation, so
:math:`\operatorname{SD}(Y\mid x)=\mu(x)\sigma(x)`. These three panels
use the same fitted model and prediction rows from the
:doc:`Gamma location–scale example <../notebooks/gamma_location_scale>`.
Absolute spread can change through either fitted parameter.

.. image:: /_static/howto_gamma_parameters.png
   :alt: Same Gamma fit showing conditional mean, coefficient of variation, and their product giving response standard deviation
   :class: scientific-figure

Formula and array modes
-----------------------

A named mapping is the preferred formula form. It prevents accidental changes
to parameter order and requires secondary formulas to be one-sided:

.. code-block:: python

   from nampy.models import GAMLSS

   model = GAMLSS(
       family="normal",
       formula={
           "mu": "y ~ s(age, bs='cr', k=10) + group",
           "sigma": "~ s(age, bs='cr', k=8)",
       },
       smoothing_method="reml",
   ).fit(data)

   parameters = model.predict_parameters(data)
   standard_errors = model.standard_errors(data)
   components = model.predict_components(data)
   components.validate_additive_reconstruction()


An ordered formula list remains supported for compatibility. Without a
formula, each distribution parameter receives its own independently compiled
copy of the configured main-effect smooths:

.. code-block:: python

   model = GAMLSS(family="gamma", k=10, basis="tp").fit(X, y)

Offsets must be a list or tuple with one entry per parameter. Contributions in
``predict_components`` are zero-padded matrices, so every term has the same
``(n_samples, n_parameters)`` shape as the link prediction.

For neural LSS effect plots, centering references, and uncertainty envelopes,
see :doc:`interpretability`.

Statistical smoothness selection
--------------------------------

Supported multi-parameter fitting routes use the general smooth-model
framework described by Wood, Pya, and Säfken (2016), including Laplace
approximate marginal likelihood where available. See :doc:`../api/gam_advanced`
for compatible family and fitting combinations.

References
----------

Rigby and Stasinopoulos (2005), `Generalized additive models for location,
scale and shape <https://doi.org/10.1111/j.1467-9876.2005.00510.x>`_. Journal
of the Royal Statistical Society: Series C 54:507–554.

Wood, Pya, and Säfken (2016), `Smoothing Parameter and Model Selection for
General Smooth Models <https://doi.org/10.1080/01621459.2016.1180986>`_. Journal
of the American Statistical Association 111:1548–1563.

Compare explicit distribution candidates
-----------------------------------------

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

BCCG probabilities, moments and joint REML fitting
--------------------------------------------------

``nampy.distributions.BCCGLaw`` supplies Box-Cox Cole-Green probabilities through
the same backend-neutral container and diagnostics as the other scalar laws::

    from nampy.distributions import BCCGLaw, DistributionPrediction
    from nampy.diagnostics import prepare_centiles

    prediction = DistributionPrediction(
        BCCGLaw(), {"mu": [1., 2., 3.], "sigma": [.4, .5, .6], "nu": [-1., 0., 1.]}
    )
    centiles = prepare_centiles(prediction, x=[0., 1., 2.])
    log_density = prediction.log_prob([.8, 2., 3.5])
    from nampy.diagnostics import quantile_residuals
    residuals = quantile_residuals(prediction, [.8, 2., 3.5])

The parameters require finite ``mu > 0``, ``sigma > 0`` and finite real ``nu``.
For positive responses, define
``z = ((y / mu)**nu - 1) / (nu * sigma)``. For positive ``nu`` the normal
variable is conditioned above ``-1 / (nu * sigma)``; for negative ``nu`` it is
conditioned below that boundary. Density includes the response Jacobian
``(y / mu)**nu / (sigma * y)`` and divides by
``Phi(1 / (sigma * abs(nu)))``. At exactly zero power the law is lognormal with
log-location ``log(mu)`` and log-standard-deviation ``sigma``.

``mu`` is generally neither the exact median nor the response mean. The declared
operations are log density, CDF/left CDF, survival, their logarithms, quantiles,
sampling, mean and variance. Shared quantile residuals, worm-plot preparation
and centiles use those capabilities directly. ``prediction.mean()`` and
``prediction.variance()`` evaluate response moments without substituting ``mu``.
For negative ``nu``, a positive-order moment of order ``k`` exists only when
``nu < -k``. In particular the mean diverges for ``-1 <= nu < 0``; the second
moment diverges for ``-2 <= nu < 0``. The mean returns positive infinity where
divergent. Variance returns NaN when the mean is infinite and positive infinity
for ``-2 <= nu < -1``, where the mean is finite but the second moment diverges.
These mathematical results are distinct from missing parameter rows, which
retain the shared all-NaN row policy.

Finite moments use checked adaptive quadrature of the normalized law. Positive
powers use log response centered at the moment integrand's mode; negative
powers use algebraic endpoint integration of the positive truncated-normal
variable ``U = (Y / mu)**nu``. Its distribution is normal with location one
and standard deviation ``abs(sigma * nu)``, conditioned on positivity. An
analytic endpoint correction retains accuracy just below the moment-existence
boundary. Small variances use centered integrals instead of subtracting nearly
equal raw moments. Exactly zero power uses the existing lognormal moment
implementation. Response units are applied in log space. Mathematically finite
moments may overflow or underflow float64; unsupported integration ranges or
failed error checks raise ``NotImplementedError``. Moments are numerical
inference operations, not Torch training objectives or differentiable losses.
Finite negative-power variance with ``sigma < sqrt(machine epsilon)`` is
explicitly unsupported: the current centered integral relies on a mean whose
precision has not been validated at that scale. This restriction does not
change the mathematical divergence results above or positive-power variance.

The statistical ``bccg_lss`` family supplies the normalized
log likelihood, exact score and observed Hessian in ``(log(mu), log(sigma), nu)``
predictors. It reuses the generic extension adapter's weighted coefficient
assembly, including offsets and shared coefficient columns. The removable
zero-power singularity is handled analytically, including mixed derivatives;
the positive-support normalizer and response Jacobian are retained. Independent
high-precision differentiation and fixed-penalty coefficient differences validate
these first/second derivatives. Saved R first-derivative hooks agree at ordinary
nonzero powers; R's second-derivative hooks supply working curvature and are
not substituted for the observed joint Hessian. The recorded R power score is
undefined at zero; the analytic kernel has its finite continuous limit.

Statistical fitting is available as ``GAMLSS(family='bccg_lss')``.
For a DataFrame ``data`` containing positive ``y`` and covariate ``x``::

    from nampy import GAMLSS

    model = GAMLSS(
        family="bccg_lss",
        formula={"mu": "y ~ s(x, bs='cr', k=5)",
                 "sigma": "~ s(x, bs='cr', k=4)",
                 "nu": "~ s(x, bs='cr', k=4)"},
        optimize_smoothing=True,
        smoothing_optimizer="outer_newton",
    ).fit(data)
    prediction = model.predict_distribution(data)
    parameters = model.predict(data)  # mu, sigma, nu
    errors = model.standard_errors(data)  # conditional natural-parameter errors

Automatic joint REML is validated with ``outer_newton`` (default), ``bfgs`` and
``efs``, including all three predictors smoothed. To hold smoothing strengths
fixed, set ``optimize_smoothing=False`` and ``smoothing_params=[.3, .5, .7]``.
Other method labels retain the existing mgcv general-family coercion to REML;
this is not separate ML/GCV/UBRE/NCV smoothing-selection support.

The likelihood has analytic mixed derivatives through fourth order, including
the zero-power limit and the truncated-normal normalizer. Independent
high-precision differentiation checks the tensors. Outer REML gradients and
Hessians are checked against finite differences with three penalties, offsets,
nonunit and zero weights. A derivative-free scalar REML profile independently
checks the selected penalty and fitted predictions for all three optimizers.
No finite-difference derivatives are used by fitting. Generic covariance
transport includes the existing approximate smoothing-uncertainty correction
when requested; this does not provide simulation/refit centile confidence bands.

Initialization uses upstream ``mu=(y+mean(y))/2``, ``sigma=.1``, ``nu=.5``;
the generic adapter projects their linked targets and subtracts predictor offsets.
At least three distinct positive-weight responses are required. This necessary
guard does not guarantee estimability or a unique optimum for arbitrary formulas.
Positive/negative-power unpenalized fits and nonunit/zero weights match saved
R GAMLSS endpoints. Three-predictor fixed-penalty fits are checked against an
independent normalized-likelihood optimum and observed covariance. Smoothed
GAMLSS RS/backfitting parity is not claimed. With frequency weights, hold physical
knots fixed when comparing to duplicated rows: automatic quantile knots otherwise
describe a different penalized model.

Natural-parameter errors, additive components, missing-row alignment, shared
diagnostics and persistence use the existing generic adapters. ``predict_point``
returns the response mean, including infinity when the fitted power implies
divergence. A power near zero is not rounded to zero to make moments finite.
Response/Pearson residuals require their moments to exist. Tight stopping rules
may report nonconvergence at a float64 objective plateau; the existing solver
status is retained and such fits are ineligible for comparison ranking.

Derivatives above order four, alternate links, BCCG censoring/truncation
and RS/backfitting remain unsupported. ``optim`` lacks the family-defined null
deviance it needs for objective scaling; ``magic`` and ``nlm.fd`` are also
unavailable. Existing transformed-fit and general-family NCV guards remain in
force. Method labels do not unlock another smoothing criterion.
This law is distinct from the
existing mgcv censored ``bcg`` family, whose fitting behavior is unchanged.
BCT/BCPE fitting and existence-checked moments, along with continuous truncation
and observation fitting, are described below.

Ordinary values are checked against saved ``gamlss.dist`` 6.1.1 ``d/p/qBCCG``
results. Recorded upstream out-of-support log-density and log-CDF quirks are
deliberately not reproduced: nonpositive responses have zero probability and
log density/CDF of negative infinity. Normal interval masses are evaluated in
log space, with controlled Gaussian quadrature near the truncation boundary;
boundary quantiles invert the same interval in log width. Independent
high-precision checks cover tails that upstream CDF subtraction rounds to zero.
Quantiles and samples retain float64 outcome limits: an interior probability
can produce zero by underflow or infinity by overflow. These values cannot
round-trip through the CDF. Unsupported numerical ranges raise explicit errors
instead of substituting another distribution.

Run ``examples/example_bccg_probabilities.py`` for probabilities, moments and
their mathematical existence boundaries.

BCT and BCPE fitting and probability laws
-----------------------------------------

``BCTLaw`` and ``BCPELaw`` use the shared ``DistributionPrediction`` interface.
The fitting adapters use the same natural parameters as these laws::

    import numpy as np
    from nampy.distributions import BCTLaw, BCPELaw, DistributionPrediction

    parameters = {"mu": np.array([1., 2., 3.]),
                  "sigma": np.array([.4, .5, .6]),
                  "nu": np.array([-1., 0., 1.]),
                  "tau": np.array([5., 8., 3.])}
    prediction = DistributionPrediction(BCTLaw(), parameters)
    centiles = prediction.quantile([.05, .5, .95])
    draws = prediction.sample(500, random_state=7)
    from nampy.diagnostics import quantile_residuals
    residuals = quantile_residuals(prediction, [.8, 2., 3.5])

Both laws require finite positive ``mu, sigma, tau`` and finite real ``nu``.
They share ``z = expm1(nu * log(y / mu)) / (nu * sigma)`` and its exact
zero-power limit. The normalized response density is
``f_Z(z) * (y / mu)**nu / (sigma * y * F_Z(a))``, where
``a = 1 / (sigma * abs(nu))``. Positive powers condition the symmetric latent
law above ``-a``; negative powers condition it below ``a``. At zero power there
is no truncation and ``log(Y / mu) = sigma * Z``.

.. list-table:: Latent parameter meanings
   :header-rows: 1

   * - Law
     - Symmetric latent distribution
     - Meaning of ``tau`` and ``sigma``
   * - BCT
     - Ordinary Student-t
     - ``tau`` is degrees of freedom; ``sigma`` scales the t variable and is
       not its standard deviation.
   * - BCPE
     - Variance-one power exponential
     - ``tau`` is generalized-normal shape; its scale is
       ``sqrt(Gamma(1/tau) / Gamma(3/tau))``. ``sigma`` scales that latent
       variable before truncation.

BCPE at ``tau=2`` is exactly BCCG. BCT approaches BCCG as its finite degrees
of freedom increase, while retaining its Student-t law. At ``nu=0`` the laws
are log-t and log-power-exponential, respectively; BCT does not become
lognormal simply because its Box-Cox power is zero. ``mu`` is generally neither
the response mean nor the exact median.

Declared operations are log density, CDF/left CDF, survival, logarithmic tails,
quantiles and sampling. These support existing shared quantile residuals,
worm plots, centiles, calibration and atom-mixture probability composition.
Mean and variance operations remain unavailable, including in parameter regions
where they exist mathematically. Tail parameters and powers can give divergent
moments; no location parameter is substituted for an expectation.

Ordinary density, CDF, survival and quantile values are checked against saved
``gamlss.dist`` 6.1.1 outputs. Separate discrepancy fixtures record:

* Nonpositive-response and log-tail quirks. The installed ``pBCPE`` source is
  captured because its boundary behavior differs from the vendored source.
  The shared laws return zero CDF and negative-infinite log density/CDF for
  nonpositive responses, with survival one.
* Upstream ``dBCT`` switches only its density to Gaussian when ``tau > 1e6``;
  its probability functions remain Student-t. ``BCTLaw`` keeps an internally
  consistent Student-t density, CDF, quantiles and samples for every supported
  finite ``tau``. This discrepancy is deliberately not copied.
* At large BCPE shape, the intermediate gamma probability or quantile can
  underflow although the final power-exponential value remains representable.
  Log-scale evaluation retains that value instead of creating a false median.

Probability calculations retain log latent magnitudes, including Student-t
values too large to represent directly. Stable special functions, an
incomplete-gamma continued fraction, and checked integration evaluate the same
law in extreme tails. Near a finite truncation boundary, interval probabilities
retain the explicit log width; boundary quantiles invert that interval.
Numerical failures raise ``NotImplementedError`` rather than substituting a
Gaussian law or silently clipping parameters. Extremely small BCPE shapes whose
standardized latent scale is outside float64 range are explicitly unsupported.
Outcome quantiles and draws can underflow to zero, overflow to infinity or
collapse to the same floating-point value; CDF round trips are not guaranteed
outside representable outcome resolution.

Use ``GAMLSS(family='bct', ...)`` or ``family='bcpe'`` with formula keys
``mu, sigma, nu, tau`` and fixed smoothing. Links are log/log/identity/log.
The joint fitting engine uses observed first and second likelihood derivatives,
including derivatives of the support normalizer. Checked adaptive quadrature
integrates analytic density derivatives; it never substitutes finite differences.
Automatic smoothing and alternate links remain unsupported. BCPE observed
derivatives at ``y=mu`` are explicitly guarded; BCT derivatives at ``nu=0``
require tail shape exceeding the derivative order.

Neural LSS accepts ``ObjectiveConfig(family='bct')`` or ``'bcpe'`` with the same
four predictor links. Its normalized objective provides first-order gradients;
the support normalizer uses vectorized checked CPU quadrature and returns
gradients to the input device. Higher autograd derivatives are unsupported.
This CPU calculation can limit accelerator throughput. Fitting does not assert
GAMLSS RS/backfitting parity or convergence for arbitrary data.

BCT/BCPE censoring/truncation fitting and untruncated-family aliases remain
unavailable. Run ``examples/example_boxcox_tail_probabilities.py`` for a standalone
probability and centile example.

Additional shared and neural routes
-----------------------------------

Neural ``bccg``, ``zaga``, ``beinf`` and ``betabinomial`` objectives use
normalized likelihoods and the shared NumPy probability laws. They do not
claim GAMLSS RS/backfitting parity or statistical convergence. Neural
beta-binomial uses a fixed integer ``distributional_kwargs={'trials': m}``,
with m>=2; row-specific trials remain supported by the statistical route only.
BEINF has four raw predictors and five natural parameters in both backends.
The interior mass is retained explicitly through the coupled softmax.

BCT/BCPE means and variances have analytic existence gates and checked
quadrature. A divergent mean is infinity and its variance is undefined (NaN);
a finite mean with a divergent second moment has infinite variance. BCT's
log-t case has no positive moments. BCPE's log-PE case has the usual
MGF boundary at tau=1. Finite integrals outside the certified numerical range
raise explicitly. The existence gates also apply to fitted predictions.

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

Runnable held-out comparison
----------------------------

.. include:: /_generated/reference/comparison.inc
