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
