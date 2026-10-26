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
``smoothing_method="fixed"`` uses REML. ``LAML`` is a REML alias; ``GCV.Cp`` uses the upstream
extended-family REML route, not a separate GCV or UBRE objective. Joint
estimation with the ``efs`` optimizer is unsupported and raises an explicit
error. ``ML`` with zero observation weights is also explicitly guarded:
actual mgcv 1.9-4 reference runs crash in this combination. Zero weights are
covered for fixed smoothing and REML.
Positive ``min_sp`` values are unsupported for SCAT with either fixed or
estimated ancillary parameters. The fixed-penalty offset parameterization
has not been ported for joint fitting and smoothing-uncertainty corrections;
zero ``min_sp`` retains the ordinary supported path.

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
