GAM Families
============

Family objects define links, variance functions, likelihoods, and response
transformations for statistical GAMs.

Family classes
--------------

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Public symbol
     - Family and link
   * - ``GaussianIdentityFamily``, ``GaussianLogFamily``,
       ``GaussianInverseFamily``
     - Gaussian responses with identity, log, or inverse links.
   * - ``BinomialLogitFamily``, ``BinomialProbitFamily``,
       ``BinomialCloglogFamily``, ``BinomialCauchitFamily``,
       ``BinomialLogFamily``
     - Binomial responses with the corresponding link.
   * - ``PoissonLogFamily``, ``PoissonIdentityFamily``,
       ``PoissonSqrtFamily``
     - Poisson responses with log, identity, or square-root links.
   * - ``GammaInverseFamily``, ``GammaLogFamily``,
       ``GammaIdentityFamily``
     - Gamma responses with inverse, log, or identity links.
   * - ``InverseGaussianFamily``
     - Positive inverse-Gaussian responses; ``1/mu^2``, inverse, log or identity.
   * - ``QuasiPoissonFamily``, ``QuasiBinomialFamily``, ``QuasiFamily``
     - Unknown-dispersion quasi families; named links and general variance/power options.
   * - ``NegativeBinomialLogFamily``
     - Negative-binomial response with fixed or estimated shape.
   * - ``ScaledTFamily``
     - Scaled Student t location model; identity, log, or inverse link.
   * - ``ZeroInflatedPoissonFamily``
     - Single-predictor hurdle Poisson model; identity link on log intensity.
   * - ``CensoredNormalFamily``
     - Censored normal location; identity, log, or square-root link.
   * - ``CensoredLogisticFamily``
     - Censored logistic location; see the acceptance boundaries below.
   * - ``CensoredPoissonFamily``
     - Censored Poisson mean; log, identity, or square-root link.
   * - ``CensoredBoxCoxGaussianFamily``
     - Censored Box-Cox Gaussian location with lambda and log-SD parameters.
   * - ``BetaRegressionFamily``
     - Beta regression for responses in the unit interval.
   * - ``OrderedCategoricalFamily``
     - Ordered categorical response with fitted cut points.
   * - ``TweedieTwFamily``
     - Tweedie response with an estimated power parameter.
   * - ``GaulssFamily``, ``gaulss``
     - Gaussian location-scale family and constructor.
   * - ``GammalsFamily``, ``gammals``
     - Gamma location-scale family and constructor.
   * - ``GumblsFamily``, ``gumbls``
     - Gumbel location and log scale.
   * - ``ZiplssFamily``, ``ziplss``
     - Independent hurdle-Poisson intensity and presence predictors.
   * - ``MultinomFamily``, ``multinom``
     - ``K`` log-odds predictors for ``K+1`` categories.
   * - ``GevlssFamily``, ``gevlss``
     - Generalized extreme value location, log scale, and shape.
   * - ``ShashFamily``, ``shash``
     - Sinh-arcsinh location, scale, skewness, and tail shape.
   * - ``TwlssFamily``, ``twlss``
     - Tweedie mean, power, and dispersion; EFS smoothing selection.

Base classes and constructors
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Public symbol
     - Purpose
   * - ``BaseFamily``
     - Shared family contract.
   * - ``GLMFamily``
     - Single-predictor GLM and quasi-family contract.
   * - ``ExtendedFamily``
     - Single-predictor family with additional fitted parameters.
   * - ``GeneralFamily``
     - Multi-predictor likelihood contract.
   * - ``GamlssFamily``
     - Base contract for distributional statistical GAMs.
   * - ``make_gam_family``
     - Resolve a family name, tuple, dictionary, or family instance.
   * - ``clone_gam_family``
     - Copy a family so fitted state is not shared between models.

See :doc:`../user_guide/statistical_gams` for accepted family names and short
examples.


Scaled Student t family
-----------------------

``ScaledTFamily(theta=None, link="identity", min_df=3)`` implements the
single-predictor ``mgcv::scat`` family. Import it from
``nampy.gam.families`` or use ``family="scat"`` in ``GAM``.

``theta`` contains the degrees of freedom and scale, in that order. Two
positive values fix both parameters. Any negative entry requests estimation
of both parameters, using their absolute values as constructor initial values.
``None`` or a zero entry selects the upstream default working parameters.
Before an estimated-parameter fit, mgcv's data-based initialization replaces
these constructor values with working parameters
``[1.5, log(0.8 * sample_sd(y))]``.

``getTheta()`` returns ``[log(df - min_df), log(scale)]``;
``getTheta(trans=True)`` returns the physical degrees of freedom and scale.
``putTheta()`` accepts the working parameter vector. If a supplied nonzero
degrees of freedom is at or below ``min_df``, the constructor warns and resets
``min_df`` to ``0.9 * abs(theta[0])``, following upstream.

The location predictor supports ``identity``, ``log`` and ``inverse`` links.
With degrees of freedom ``nu`` and scale ``sigma``, the conditional variance
is ``sigma**2 * nu / (nu - 2)`` when ``nu > 2``. The scale in ``theta`` is a
distribution parameter, distinct from the optional GAM dispersion argument.

The implementation follows mgcv 1.9-4 ``R/efam.r::scat``, including its
observed and Fisher derivatives. See the
`upstream scat documentation <https://stat.ethz.ch/R-manual/R-devel/library/mgcv/html/scat.html>`_.
Fixed smoothing and ``ML``/``REML`` selection support fixed or estimated
ancillary parameters. With ``optimize_smoothing=False``, an explicitly
selected ``ML`` or ``REML`` criterion still governs ancillary estimation;
``smoothing_method="fixed"`` uses REML. ``LAML`` is a REML alias. ``GCV.Cp``, ``P-ML`` and ``P-REML`` use the
upstream extended-family REML route. Unsupported criteria such as NCV
raise an explicit error, including with fixed smoothing parameters. Joint
estimation with the ``efs`` optimizer is unsupported and raises an explicit
error. ``ML`` with zero observation weights is also explicitly guarded:
actual mgcv 1.9-4 reference runs crash in this combination. Zero weights are
covered for fixed smoothing and REML.
Positive ``min_sp`` values are unsupported for SCAT with either fixed or
estimated ancillary parameters. The fixed-penalty offset parameterization
has not been ported for joint fitting and smoothing-uncertainty corrections;
zero ``min_sp`` retains the ordinary supported path.

Rank-deficient designs that drop coefficients are rejected when fitting
requires outer-parameter uncertainty corrections. The upstream correction
attempts Cholesky factorization of the full singular coefficient Hessian;
rounding can change whether that correction is included, affecting AIC and
unconditional covariance across platforms. Conditional fitting with both
ancillary parameters and smoothing fixed remains available.

The default external dispersion is one. A positive ``GAM(scale=...)``
override follows the upstream criterion and covariance scaling. For fixed
ancillary parameters and fixed smoothing, ``scale=2`` doubles the conditional
coefficient covariance without changing coefficients or distribution
parameters. When smoothing or ancillary parameters are estimated, the
external dispersion can change their optimum; it is not merely a multiplier
on the reported standard errors.

Outer ``bfgs``, ``optim`` and ``nlm`` routes have endpoint references for
identity-link, positive-weight models with one cubic smooth and fixed or
estimated ancillary parameters. ``optim`` comparisons use matching
``control={"epsilon": 1e-10, "optim": {"factr": 10}}`` to resolve a flat
ancillary-parameter optimum. BFGS retains its path-dependent approximate
Hessian; the final derivative-free ``nlm`` and ``optim`` fits do not supply
a smoothing-parameter uncertainty correction.

Conditional and unconditional coefficient covariance, effective degrees of
freedom, prediction standard errors, likelihood, AIC and BIC are available.
Unconditional corrections require an outer-parameter Hessian: they are
absent when both ancillary parameters and all smoothing parameters are
fixed, including models without smooths. Working residuals are undefined
on zero-weight rows. Near zero observed curvature, tiny differences between
converged fits can produce appreciable working-residual differences; parity
checks compare the working-score invariant and separately verify raw
residuals at the exact upstream fitted parameters.


Hurdle Poisson family
---------------------

``ZeroInflatedPoissonFamily(theta=None, link="identity", b=0)`` implements
mgcv's single-predictor ``ziP`` family. Public names ``"ziP"`` and ``"zip"``
resolve to the same family. Only the identity link is supported.

The fitted latent value ``gamma`` is the logarithm of a Poisson intensity
``lambda = exp(gamma)``. The probability of a positive count is
``p = 1 - exp(-exp(theta[0] + (b + exp(theta[1])) * gamma))``.
Conditional on presence, counts have a zero-truncated Poisson distribution.
``predict(type="response")`` returns the expected count
``p * lambda / (1 - exp(-lambda))``; ``predict(type="link")`` and stored
fitted values remain on the latent log-intensity scale. Response standard
errors describe the expected count, not an interval for a new observation.

A supplied ``theta`` pair fixes both working parameters, even when entries
are zero or negative. Omitting ``theta`` estimates both, starting from
``[0, 0]``. ``getTheta()`` returns the working intercept and log slope;
``getTheta(trans=True)`` returns the intercept and ``b + exp(theta[1])``.
``putTheta()`` accepts working parameters. Negative ``b`` is reset to zero,
following mgcv. Larger ``b`` enforces stronger dependence between intensity
and the probability of presence.

The response must contain nonnegative integer counts; binary data containing
both zero and one are rejected. The public model is a count regressor.
Deviance, working, and response residuals are supported. Pearson and quantile
residuals are explicitly unavailable. Reported deviance includes the
family-specific saturated-likelihood correction; adjusted R-squared is not
reported for this family.

The reference scope covers fixed and estimated ancillary parameters, fixed
smoothing, ML/REML (including the LAML alias), offsets, multiple smooths,
no-intercept and parametric-only formulas, and unit observation weights.
Positive external dispersion overrides are covered, including scale two. Nonunit or zero
prior weights are explicitly guarded: mgcv's ``ziP`` deviance ignores weights
while its derivatives and AIC use them, so the ordinary weighted-GLM contract
does not apply. Positive ``min_sp`` and joint estimation with ``efs`` are
also unsupported. Rank-deficient designs that drop coefficients are rejected when fitting
requires outer-parameter uncertainty corrections: the upstream Cholesky
branch for the full singular Hessian is sensitive to rounding, changing AIC
and unconditional covariance. Conditional fitting with both theta and
smoothing fixed remains available.

The ``bfgs``, ``optim`` and ``nlm`` outer optimizers have fixed- and
estimated-theta references for a unit-weight cubic smooth. BFGS and nlm use
default controls; optim comparisons use matching
``control={"epsilon": 1e-10, "optim": {"factr": 10}}``. BFGS retains its
approximate outer Hessian. The derivative-free final fits from optim and nlm
do not expose smoothing-parameter covariance or unconditional corrections.
``GCV.Cp``, ``P-ML`` and ``P-REML`` follow mgcv's extended-family REML route, rather than a separate
GCV objective. Default ``b=0`` and a near-boundary slope have direct references.

Source: mgcv 1.9-4 ``R/efam.r::ziP`` and ``R/gamlss.r::zipll``; see the
`upstream ziP documentation <https://stat.ethz.ch/R-manual/R-devel/library/mgcv/html/ziP.html>`_.


Censored normal family
----------------------

``CensoredNormalFamily(theta=None, link="identity")`` implements mgcv's
``cnorm`` location family. Import it from ``nampy.gam.families`` or use
``family="cnorm"``. Dictionaries such as
``{"name": "cnorm", "theta": 1, "link": "identity"}`` and tuples such as
``("cnorm", {"theta": 1})`` also resolve the family. It is a regression
family; the response's second column is a censor bound, not a class label.

The reference scope covers fixed or estimated response SD, fixed smoothing
or ML/REML smoothing selection, including joint objective gradients and
Hessians. Identity, log and square-root links have family-kernel references;
fixed-SD, fixed-smoothing fits also cover positive precision weights and
log/square-root links. References additionally cover two smooths, offsets,
no-intercept and parametric-only fits, conditional rank-deficient fits,
BFGS/optim/nlm endpoints, and fixed-SD EFS. These targeted checks do not
establish cross-platform CI acceptance.

For conditional fits with exactly aliased columns, the retained coefficient
can differ across numerical factorizations. Their parity checks compare
estimable coefficient effects and covariance kernels, including predictions
that preserve the training alias relation. Raw covariance coordinates and
predictions outside that relation are not uniquely identified; no parity
claim is made for such nonestimable newdata.

A vector response represents exact observations. A two-column response,
provided as ``cbind(value, bound)`` in a formula or an explicit ``(n, 2)``
array, encodes each observation as follows:

.. list-table:: Censor response encoding
   :header-rows: 1

   * - Second column
     - Observation
   * - Equal to the first column
     - Exact value
   * - A different finite value
     - Interval between the two columns, in either order
   * - ``-np.inf``
     - Left censored at the first column
   * - ``np.inf``
     - Right censored at the first column

Primary values must be finite and bounds cannot be NaN. Stored censor
bounds remain paired with their primary values through fitting and
serialization. Predictions describe the underlying normal location;
response standard errors describe the estimated location, not a new
observation or censoring probability.

A positive ``theta`` fixes the standard deviation and is stored on the log
scale. ``None`` or zero requests estimation starting at working value zero.
For a negative supplied value, mgcv 1.9-4 actually retains that value as the
initial *log standard deviation*, despite the manual's negative-SD wording.
``getTheta()`` returns working log SD and ``getTheta(trans=True)`` returns SD.

The source uses observation weights as precision: each row's SD is
``exp(theta_working) / sqrt(weight)``. They are not frequency weights.
Zero weights are rejected during fitting: upstream mixed-censor interval
observations can produce undefined derivatives and an error, even though
some uncensored zero-weight fits converge. They are not automatically
omitted rows. Positive ``min_sp`` is also unsupported, and estimated SD with
the ``efs`` optimizer is guarded. Estimated SD with nonunit external
dispersion is also guarded: a captured upstream failed outer fit returns
coefficients from a different parameter state. Fixed-SD fits accept positive
external dispersion. Rank-deficient outer uncertainty retains the shared
extended-family guard. The executable example uses unit weights.

The implementation preserves two mgcv 1.9-4 AIC quirks: its exact-observation
branch omits the log-SD Jacobian after standardizing, and its later
``bound <= 0`` branch also replaces contributions from nonpositive exact
or finite interval bounds. Consequently source-compatible AIC and derived
log likelihood need not equal the ordinarily normalized Gaussian likelihood.
The fitting criterion retains the separate source deviance and saturated
likelihood conventions. Do not use an ordinary Gaussian AIC calculation as
a substitute.

Source ``cnorm`` does not expose working simulation or quantile hooks;
simulation-based QQ envelopes and quantile residuals are unavailable.
See the executable ``examples/example_cnorm_gam.py`` and the
`upstream cnorm documentation <https://stat.ethz.ch/R-manual/R-devel/library/mgcv/html/cnorm.html>`_.
Source authority is mgcv 1.9-4 ``R/efam.r::cnorm``; the old ``cnorm0`` is not
used.

Censored logistic family
------------------------

``CensoredLogisticFamily(theta=None, link="identity")`` ports
``mgcv::clog``. String, dictionary and tuple family specifications accept
``clog``; for example ``{"name": "clog", "theta": 0.7}`` fixes the
logistic scale at 0.7. This scale is not the standard deviation; the
uncensored logistic standard deviation is ``scale * pi / sqrt(3)``.
The family describes a continuous logistic response, with one additive
location predictor. Supported links are identity, log and square root.

Responses use the same explicit paired encoding as ``cnorm`` above:
equal values are exact observations, distinct finite values define an
interval in either order, and infinite second bounds define one-sided
censoring. Predictions return the latent logistic location and need no
censor bounds. Observation weights are precision, giving each row a scale
``exp(theta_working) / sqrt(weight)``. A positive constructor ``theta`` fixes
scale; zero or ``None`` estimates it from working value zero; negative
``theta`` starts estimation at ``log(abs(theta))``. This last convention
differs from the source ``cnorm`` constructor.

Static R references cover all returned deviance derivative fields through
fourth order, saturation derivatives, tails, precision weights, mixed
censoring, fixed/estimated scale, fixed smoothing and ML/REML selection,
offsets, two smooths, parametric-only fits, covariance, EDF, likelihood
criteria, predictions, residuals, persistence and additive reconstruction.
Fixed- and estimated-scale optimizers have targeted references, including
weighted ancillary estimation, successful estimated-scale nlm fits, and
nonidentity REML/BFGS and ML/optim fits. EFS estimates scale inside each PIRLS
iteration using the source ``estimate.theta`` update, and retains the
accepted EFS fit rather than estimating theta again in a final solve.
EFS references cover exact and mixed-censored responses, all three links,
positive precision weights and positive external dispersion. EFS supplies
no smoothing-parameter covariance or unconditional uncertainty correction.
The example is ``examples/example_clog_gam.py``.

In captured fixed-scale nlm/optim cases the smooth reaches its linear null
space. Large raw smoothing parameters are not compared across endpoints;
fitted effects, covariance, criterion, and the limiting EDF retain their
original tolerances. Other accepted cases compare smoothing parameters
directly, including fixed parameters stored in upstream ``full.sp``.

Remaining restrictions have distinct causes:

* Estimated scale with log/square-root links under ML and Newton/BFGS is
  guarded after captured mgcv subprocess crashes. Use ``smoothing_optimizer="optim"``
  for this configuration; these ML fits have direct reference comparisons.
* A failed Newton selection raises ``RuntimeError`` and does not commit a
  new fitted state. Captured failed weighted and nonidentity REML fits prove
  that mgcv can report coefficients belonging to a different outer-parameter
  state. The former blanket weight and nonidentity restrictions are removed;
  successful weighted and nonidentity fits are supported. Select another
  optimizer when Newton fails instead of interpreting that failed source
  state as an optimum. A failed refit preserves the previous fitted model.
* Nonconverged BFGS/nlm results emit an explicit warning. The existing nlm
  compatibility backend can stop differently from R; failed nlm endpoints
  are recorded as failures, not used to claim numerical optimum parity.
  Successful estimated-scale nlm numerical outputs have separate references.
* EFS ancillary estimation requires a free smoothing parameter. Its captured
  all-fixed-SP source call errors; use Newton, BFGS or optim when smoothing
  parameters are fixed. EFS selection uses REML/LAML.
* Zero precision, positive ``min_sp``, and rank-deficient outer uncertainty
  retain the documented extended-family restrictions. These do not imply
  that every upstream composition or dataset is rejected.

The port retains source arithmetic and formulas, including interval
derivative expressions and the unusual ``aic`` hook, which is independent
of fitted location. Consequently reported source-compatible AIC and derived
log likelihood are not substitutes for an independently normalized logistic
likelihood. Simulation and quantile hooks are unavailable upstream.

Source: mgcv 1.9-4 ``R/efam.r::clog, estimate.theta`` and
``R/gam.fit4.r::gam.fit4, efsudr``. These focused checks establish the
listed scope; cross-platform CI and merge acceptance remain separate.


Censored Poisson family
-----------------------

``CensoredPoissonFamily(link="log")`` implements ``family="cpois"``.
The predictor models the latent Poisson mean, using a log, identity or
square-root link. There are no ancillary distribution parameters to estimate.
Use ``cbind(value, bound)`` for censoring: equal endpoints denote an exact
count, ``-Inf`` denotes left censoring and ``+Inf`` right censoring. A finite
unequal pair denotes the discrete event ``min(value,bound) < Y <= max(value,bound)``.
For example, ``[2, 4]`` includes counts three and four. Predictions return the
underlying mean, with standard errors for that mean.

The source likelihood, deviance, saturated likelihood and derivative hooks
ignore observation weights. They are not frequency-weighted Poisson hooks;
nonunit-weight reference fits retain this source convention. Density and tail
probabilities follow the scalar R Mathlib ``dpois``/``ppois`` branches, including
extreme-tail and degenerate-mean cases. Location derivatives through fourth
order, the discrete saturated-likelihood correction, residuals and null
calculations use those source probabilities.

Actual-mgcv fixtures cover exact/mixed responses, all three links, fixed
smoothing, ML/REML selection, positive external dispersion and observation weights,
coefficients, covariance, EDF, likelihood criteria, residuals and predictions/SE.
The alternative ``bfgs``, ``optim``, ``nlm`` and ``efs`` routes have direct
references. The nlm comparison uses matching tighter controls:
``control={"epsilon": 1e-10, "nlm": {"gradtol": 1e-10, "steptol": 1e-10}}``.
Source quantile and simulation hooks are unavailable. See
``examples/example_cpois_gam.py``. Source: mgcv 1.9-4 ``R/efam.r::cpois``
and the R Mathlib Poisson density/tail routines.


Censored Box-Cox Gaussian family
--------------------------------

``CensoredBoxCoxGaussianFamily(theta=None, link="identity")`` implements
``family="bcg"``. One predictor describes Gaussian location after a Box-Cox
response transformation; links are identity, log and square root.
``getTheta()`` returns lambda and log SD, and ``getTheta(trans=True)``
transforms the second entry to SD. The default estimates both parameters,
starting at working values ``[1, 0]``. Positive precision weights replace
log SD by ``logSD - log(weight)/2``.

Censoring uses explicit paired response data. Nonpositive second-column
values denote left censoring, positive unequal finite endpoints denote an
interval, and ``+Inf`` denotes right censoring. Source initialization uses the
input response scale. Predictions describe the transformed Gaussian location;
they do not apply an inverse Box-Cox mean correction.

The source fixed constructor has a specific failure: positive
``theta=[lambda, sd]`` stores three working entries,
``[lambda, log(lambda), log(sd)]``. Its fitted source call fails with an
``ls$lsth2`` indexing error. NAMpy rejects this malformed fitted state with
``ValueError``. The source-owned setter provides an explicit supported
working-state route:

.. code-block:: python

   import numpy as np
   from nampy.gam.families import CensoredBoxCoxGaussianFamily

   family = CensoredBoxCoxGaussianFamily(theta=[0.7, 0.8])
   family.putTheta([0.7, np.log(0.8)])

The family remains fixed, with a coherent two-entry working state; no automatic
constructor repair is performed. A nonpositive second constructor entry starts
estimation and retains that entry as the initial working log SD, following
source initialization order.

References cover the near-zero-lambda expansions, all censor blocks, fourth
location and second ancillary derivatives, exact and mixed responses,
precision weights, offsets, fixed/estimated ancillary parameters, and
fixed/ML/REML smoothing. Estimated-parameter vector fits cover all three links.
EFS uses the two-parameter source ``estimate.theta`` update inside PIRLS;
fixed/estimated ancillary vector and mixed-response endpoints have direct
references. Theta-update references also cover external dispersion one/two;
the captured scale-two source updates warn about step failure and retain their
returned state. Estimating ancillary parameters with EFS requires a free
smoothing parameter: the all-fixed-SP source call errors. Positive ``min_sp``
and rank-deficient outer-uncertainty combinations retain the extended-family
guards described above.

Source AIC omits scale/Box-Cox Jacobians and is not an independently normalized
Box-Cox density. Source saturated-likelihood derivative arrays are zero despite
the value's ancillary dependence. Exactly transformed censor endpoints can
produce source NaNs in signed density differences; those NaNs remain visible.
Quantile and simulation hooks are unavailable. See ``examples/example_bcg_gam.py``.
Source: mgcv 1.9-4 ``R/efam.r::bcg, estimate.theta`` and ``R/gam.fit4.r``.


Additional general families
----------------------------

The following constructors are exported from ``nampy.gam.families`` and
``nampy.gam.families.gamlss``. ``GAM`` accepts their names, option dictionaries
or instances; ``GAMLSS`` exposes natural distribution parameters through
``predict`` and raw additive predictors through ``predict_link`` or
``predict(raw=True)``. Source ``GAM.predict(type="response")`` parameterizations
are described below. Natural-parameter standard errors from
``GAMLSS.standard_errors(type="response")`` apply the analytic parameter
Jacobian, including cross-predictor covariance for multinomial probabilities.
The source GAM prediction hooks retain their own reported-SE conventions.
For fixed smoothing parameters, general-family AIC uses the conditional
effective degrees of freedom; smoothing-parameter uncertainty is included
only when the fitted route supplies its required derivatives.

All six source native likelihoods ignore prior observation weights. This does
not imply that residuals, initialization or null-deviance calculations ignore
weights. Each port has frozen actual-R likelihood/derivative and fitted-output
references; the tested routes are listed separately below. Their seven-stage
pipeline checks also cover public additive components and versioned save/load.
These focused references do not establish every smooth/optimizer composition.

``gumbls``
~~~~~~~~~~

``gumbls(link=("identity", "log"), b=-7)`` has identity-linked location and
log scale. Its source second ``"log"`` link lower-bounds log scale at ``b``;
an identity log-scale link is also accepted. ``GAMLSS.predict`` returns
``(location, scale)`` and ``predict_point`` returns
``location + EulerGamma * scale``. Source response prediction returns
``(location, log_scale)``, while source postprocessed fitted values contain
``(expectation, log_scale)``.

Both scale links have fixed/selected REML references with two smooths, offsets,
coefficient covariance, criterion/EDF, residuals and response predictions/SE.
Analytic derivatives through fourth order, source initialization and sampling
are implemented. Deviance, Pearson and response residuals follow source; null
deviance is undefined. The source scale-prediction SE uses its scalar ``ve[2]``
expression, which is preserved. Natural-parameter errors use each row's
Jacobian. Example: ``examples/example_gumbls_gam.py``.

``ziplss``
~~~~~~~~~~

``ziplss(link=("identity", "identity"))`` uses separate predictors for log
Poisson intensity and complementary-log-log presence. It is a hurdle model:
a positive response follows a zero-truncated Poisson distribution.
``GAMLSS.predict`` returns ``(lambda, presence)`` and ``predict_point`` returns
``presence * lambda / (1 - exp(-lambda))``. Source fitted values retain latent
predictors; source response prediction returns expected counts. Responses must
be nonnegative counts; the source rejects binary zero/one data.

Fixed/selected REML references cover two smooths, offsets, covariance,
criterion/EDF, initialization, residuals and predictions/SE. Analytic
likelihood derivatives through fourth order and simulation are supplied.
Source residuals support deviance and response types. Its expected-count SE
uses the first predictor variance twice and clips cross covariance to
nonnegative values. This source expression is preserved; GAMLSS natural
parameter errors use the individual parameter Jacobians. Example:
``examples/example_ziplss_gam.py``.

``multinom``
~~~~~~~~~~~~

``multinom(K=1)`` uses ``K`` identity-linked log odds against reference class
zero, for categories ``0,...,K``. ``GAM`` response prediction and
``GAMLSS.predict`` return ``K+1`` probabilities, ordered ``p0,...,pK``.
``predict_point`` selects the first most-probable category. Raw/link prediction
has ``K`` columns, named ``eta1,...,etaK``. Additive components expand these
logits to ``K+1`` columns with a zero baseline column, so their contributions
and intercept reconstruct exactly. The probability Jacobian has shape
``(n, K+1, K)``.

Kernel references cover every packed derivative pattern through fourth order
at ``K=1,2,3,4``. Fixed-SP fits at ``K=1,2,3`` and a selected REML fit at
``K=2`` compare coefficients, covariance, EDF, score, predictions/SE and
deviance residuals. The latter are positive for a correct classification and
negative otherwise; other residual types are unavailable.

The source baseline dummy is one, adding one per class-zero observation to
its fitting log likelihood. At fourth derivative order its loop variable
also overwrites the returned scalar likelihood with ``K``. Both source
behaviors remain explicit in the likelihood contract. The public normalized
categorical log density uses actual probabilities. Source GAM probability SEs
clip cross-predictor covariance at zero; GAMLSS natural errors retain the full
covariance in the delta method. Example: ``examples/example_multinom_gam.py``.

``gevlss``
~~~~~~~~~~

``gevlss(link=("identity", "identity", "logit"))`` has three predictors.
Location accepts identity/log, log scale accepts identity, and shape accepts
identity or a shifted logit confining it to ``(-1, 0.5)``.
Source response prediction returns ``(mu, log_sigma, xi)``;
``GAMLSS.predict`` returns ``(mu, sigma, xi)``. The point prediction uses the
same conditional-mean expression as source residuals.

All four link combinations have analytic first-through-fourth derivative,
initialization, density and residual references. Identity/log-location
fixed/selected REML fits have coefficient, covariance, criterion/EDF and
prediction/SE references. Source likelihood evaluation replaces tiny shapes
by signed ``+/-1e-7``; it has no separate zero-shape derivative branch.
Simulation uses the source inverse CDF and its ``1e-8`` shape cutoff.

The source residual mean is computed before its shape cutoff, retaining an
undefined ratio at exactly ``xi=0``. Its Pearson denominator retains the
source ``exp(rho)/xi`` sign. Deviance residuals retain source scaling without
an added factor two; null deviance is undefined. Public density is negative
infinity outside support; invalid source likelihood trials remain nonfinite
for the existing fit-step checks. Example: ``examples/example_gevlss_gam.py``.

``shash``
~~~~~~~~~

``shash(link=("identity", "logeb", "identity", "identity"), b=0.01,
phi_pen=0.001)`` models location, log scale, skewness and log tail shape.
These are the only supported links. ``sigma=exp(eta2)+b`` is bounded below by
positive ``b``. ``GAMLSS.predict`` returns ``(mu, sigma, epsilon, delta)``;
``predict_point`` is the source conditional median ``qf(0.5)``. Source quantile,
CDF/log-CDF and inverse-transform simulation hooks are present.
Source GAM response prediction returns ``(mu, log_sigma, epsilon, log_delta)``.

The fitting likelihood includes the source ridge
``-phi_pen * log(delta)**2`` (``phiPen`` is an accepted alias). Public normalized
density scoring omits that estimation ridge. All analytic derivatives through
fourth order are supplied. Fixed SP, Newton-selected REML and EFS references
cover initialization, covariance, criterion/EDF, AIC/logLik, predictions/SE,
residuals and summary tables.

Nonzero offsets are rejected. The source fitting likelihood's bin lookup
excludes a lower ``-Inf`` endpoint: an exactly zero standardized residual
therefore produces positive-infinite source likelihood. The normalized density
and residual expression stay finite there. Residual centering retains the
source epsilon-independent Bessel expression. Source residuals support
deviance and response types; no source null-deviance hook exists.
Example: ``examples/example_shash_gamlss.py``.

``twlss``
~~~~~~~~~

``twlss(link=("log", "identity", "identity"), a=1.01, b=1.99)`` models
Tweedie mean, transformed power and log dispersion, with ``1 < a < b < 2``.
The mean link also accepts identity or square root; the remaining links are
identity. Source response values are ``(mu, theta, rho)``;
``GAMLSS.predict`` returns ``(mu, p, phi)``, where ``phi=exp(rho)`` and the
source logistic power map confines ``p`` to the supplied bounds.
``predict_point`` returns mean ``mu``.

Source ``available.derivs=0`` supplies the analytic likelihood gradient and
Hessian; third/fourth likelihood arrays and a numerical outer-derivative
fallback are unavailable. Smoothing selection automatically uses EFS,
including when another optimizer was requested. Fixed-SP and selected-EFS
fits have direct references for covariance, criterion/EDF, AIC/logLik,
predictions/SE, all three residual types and summary tables. The source null
fit retains its independent bounds ``a=1.001, b=1.999``; null postprocessing
uses each observation's fitted dispersion. Prior weights enter deviance
residuals and null deviance, although the fitting likelihood ignores them.

The source formula-list setup rejects an external fixed-SP vector when a
parametric-only middle predictor precedes another smooth. Equivalent source
captures fix the parameters inside their ``s(..., sp=...)`` terms. All three
predictor offsets, zero/positive responses, custom power bounds and both source
initialization branches have kernel references.
Example: ``examples/example_twlss_gamlss.py``.

Source for these six ports: mgcv 1.9-4 ``R/gamlss.r`` family constructors,
``pen.reg``, ``gamlss.etamu``, ``gamlss.gH`` and ``gam.fit5``; Tweedie uses
``R/gam.fit3.r::ldTweedie`` and ``R/gamlss.r::tw.null.fit``. Cross-platform CI
and merge acceptance are tracked separately from these local reference scopes.
